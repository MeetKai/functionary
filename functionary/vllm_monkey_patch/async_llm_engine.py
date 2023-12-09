import asyncio
import json
import re
import time
from functools import partial
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Type, Union

from vllm.config import ModelConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm.engine.ray_utils import initialize_cluster, ray
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams

logger = init_logger(__name__)


class AsyncEngineDeadError(RuntimeError):
    pass


def _raise_exception_on_finish(
    task: asyncio.Task, request_tracker: "RequestTracker"
) -> None:
    msg = (
        "Task finished unexpectedly. This should never happen! "
        "Please open an issue on Github."
    )
    try:
        try:
            task.result()
        except asyncio.CancelledError:
            return
        except Exception as exc:
            raise AsyncEngineDeadError(
                msg + " See stack trace above for the actual cause."
            ) from exc
        raise AsyncEngineDeadError(msg)
    except Exception as exc:
        request_tracker.propagate_exception(exc)
        raise exc


class AsyncStream:
    """A stream of RequestOutputs for a request that can be
    iterated over asynchronously."""

    def __init__(self, request_id: str) -> None:
        self.request_id = request_id
        self._queue = asyncio.Queue()
        self._finished = False

    def put(self, item: RequestOutput) -> None:
        if self._finished:
            return
        self._queue.put_nowait(item)

    def finish(self) -> None:
        self._queue.put_nowait(StopIteration)
        self._finished = True

    @property
    def finished(self) -> bool:
        return self._finished

    def __aiter__(self):
        return self

    async def __anext__(self) -> RequestOutput:
        result = await self._queue.get()
        if result is StopIteration:
            raise StopAsyncIteration
        elif isinstance(result, Exception):
            raise result
        return result


class RequestTracker:
    """Synchronous abstraction for tracking requests."""

    def __init__(self) -> None:
        self._request_streams: Dict[str, AsyncStream] = {}
        self._finished_requests: asyncio.Queue[str] = asyncio.Queue()
        self._new_requests: asyncio.Queue[Tuple[AsyncStream, dict]] = asyncio.Queue()
        self.new_requests_event = None

    def __contains__(self, item):
        return item in self._request_streams

    def init_event(self):
        self.new_requests_event = asyncio.Event()

    def propagate_exception(
        self, exc: Exception, request_id: Optional[str] = None
    ) -> None:
        """Propagate an exception to request streams
        (all if request_id is None)."""
        if request_id is not None:
            self._request_streams[request_id].put(exc)
        else:
            for stream in self._request_streams.values():
                stream.put(exc)

    def process_request_output(
        self, request_output: RequestOutput, *, verbose: bool = False
    ) -> None:
        """Process a request output from the engine."""
        request_id = request_output.request_id

        self._request_streams[request_id].put(request_output)
        if request_output.finished:
            if verbose:
                logger.info(f"Finished request {request_id}.")
            self.abort_request(request_id)

    def add_request(self, request_id: str, **engine_add_request_kwargs) -> AsyncStream:
        """Add a request to be sent to the engine on the next background
        loop iteration."""
        if request_id in self._request_streams:
            raise KeyError(f"Request {request_id} already exists.")

        stream = AsyncStream(request_id)
        self._new_requests.put_nowait(
            (stream, {"request_id": request_id, **engine_add_request_kwargs})
        )

        self.new_requests_event.set()

        return stream

    def abort_request(self, request_id: str, *, verbose: bool = False) -> None:
        """Abort a request during next background loop iteration."""
        if verbose:
            logger.info(f"Aborted request {request_id}.")

        self._finished_requests.put_nowait(request_id)

        if (
            request_id not in self._request_streams
            or self._request_streams[request_id].finished
        ):
            # The request has already finished or been aborted.
            return

        self._request_streams[request_id].finish()

    def get_new_and_finished_requests(self) -> Tuple[List[dict], Set[str]]:
        """Get the new requests and finished requests to be
        sent to the engine."""
        new_requests: List[dict] = []
        finished_requests: Set[str] = set()

        while not self._finished_requests.empty():
            request_id = self._finished_requests.get_nowait()
            finished_requests.add(request_id)
            self._request_streams.pop(request_id, None)

        while not self._new_requests.empty():
            stream, new_request = self._new_requests.get_nowait()
            if stream.request_id in finished_requests:
                # The request has already been aborted.
                stream.finish()
                continue
            self._request_streams[stream.request_id] = stream
            new_requests.append(new_request)

        self.new_requests_event.clear()

        return new_requests, finished_requests

    async def wait_for_new_requests(self):
        await self.new_requests_event.wait()


class _AsyncLLMEngine(LLMEngine):
    """Extension of LLMEngine to add async methods."""

    # This is a dict mapping request_id to the list of tools/functions details
    tools_or_functions: dict = {}
    # This is a dict mapping request_id to the prompt_template_version
    prompt_template_versions: dict = {}

    def check_to_sample(self, suffix, prompt_template_version, request_id):
        """This function checks the state of the prompt on whether
        it is at a state where grammar-sampling is required:
        - When the model is generating a function/tool name
        - If reached the parameter generation stage, check whether the model
          is in the midst of generating a parameter name and not the value
        """
        # Check if the model is in the midst of generating a function name
        # This will always return True for incomplete function name
        if any(
            [
                tool_or_func["name"].startswith(suffix)
                and tool_or_func["name"] != suffix
                for tool_or_func in self.tools_or_functions[request_id]
            ]
        ):
            return True

        # Check if the model has reached the parameter generation stage
        if (prompt_template_version == "v1" and ":\n{" in suffix) or (
            prompt_template_version == "v2" and "\n<|content|>{" in suffix
        ):
            # Extract whatever parameters the model generated up to now
            if prompt_template_version == "v1":
                _, curr_params_str = suffix.split(":\n{")
            elif prompt_template_version == "v2":
                _, curr_params_str = suffix.split("n<|content|>{")

            # Use two stacks/counts to check if the model is currently generating
            # a parameter name and not a parameter value.
            # - `apostrophe_count` keeps track of the number of `"` in params_str
            # - `nested_obj_stack` keeps track of the number of `{` and `[` in params_str
            # If the model is in the midst of generating a parameter name,
            # `nested_obj_stack` must be 0 and `apostrophe_count` must be an odd number
            # nested_obj_stack, apostrophe_count = 0, 0
            # for char in curr_params_str:
            #     if char == '"':
            #         apostrophe_count += 1
            #     elif char in ["{", "["]:
            #         nested_obj_stack += 1
            #     elif char in ["}", "]"]:
            #         nested_obj_stack -= 1
            # if nested_obj_stack == 0 and apostrophe_count % 2 == 1:
            #     breakpoint()
            #     return True

            # Loop through curr_params_str from the back
            for i in range(len(curr_params_str) - 1, -1, -1):
                try:
                    full_params_json = json.loads("{" + curr_params_str[:i] + "}")
                    latest_param_str = curr_params_str[i:].lstrip(", ")

                    # Get the list of parameters for the curr_fn_name
                    if prompt_template_version == "v1":
                        curr_fn_name, curr_params_str = suffix.split(":\n{")
                    elif prompt_template_version == "v2":
                        curr_fn_name, curr_params_str = suffix.split("\n<|content|>{")
                    for tool_or_func in self.tools_or_functions[request_id]:
                        if tool_or_func["name"] == curr_fn_name:
                            parameter_options = list(
                                tool_or_func["parameters"]["properties"].keys()
                            )
                            break

                    # Check if the latest_param_str starts with `"{parameter_name}`
                    # If so, return False as the model has finished generating the
                    # parameter name.
                    if (
                        len(latest_param_str) == 0
                        or latest_param_str == "}"
                        or any(
                            [
                                latest_param_str.startswith(f'"{parameter_name}')
                                for parameter_name in parameter_options
                            ]
                        )
                    ):
                        return False
                    else:
                        return True
                except:
                    pass

        # Return False if the model is not in the midst of generating a function name
        # or in the parameter generation stage
        return False

    def check_end_of_sampling(self, suffix, request_id):
        """This function checks the state of the prompt on whether
        it is at a state where the complete function name is just
        generated in order to stop the model from hallucinating a
        longer function name (by selecting '\n' token)
        """

        return any(
            [
                tool_or_func["name"] == suffix
                for tool_or_func in self.tools_or_functions[request_id]
            ]
        )

    def sample(
        self,
        delta_token_ids,
        output_token_ids,
        suffix,
        prompt_template_version,
        request_id,
    ):
        """Applies grammar-sampling to the token generation and returns a
        newly sampled token.

        This function checks whether the model-sampled token helps towards
        forming one of the function names or parameter names. It loops through
        a list of token ids sorted in descending order by the log probabilities.
        It replaces the output token if the grammar-sampled token is different
        from the model-sampled token
        """
        # Check whether the model is in function name or parameter generation stage
        # If the model is in parameter generation stage
        if (prompt_template_version == "v1" and ":\n{" in suffix) or (
            prompt_template_version == "v2" and "\n<|content|>{" in suffix
        ):
            stage = "parameter"

            # Get the list of parameters for the curr_fn_name
            if prompt_template_version == "v1":
                curr_fn_name, curr_params_str = suffix.split(":\n{")
            elif prompt_template_version == "v2":
                curr_fn_name, curr_params_str = suffix.split("\n<|content|>{")
            for tool_or_func in self.tools_or_functions[request_id]:
                if tool_or_func["name"] == curr_fn_name:
                    parameter_options = list(
                        tool_or_func["parameters"]["properties"].keys()
                    )
                    break
        # If the model is in function name generation stage
        else:
            stage = "function"

            # Retrieve the list of function names by request_id
            func_options = [
                tool_or_func["name"]
                for tool_or_func in self.tools_or_functions[request_id]
            ]
            if prompt_template_version == "v2":
                func_options.append("all")

        # Loop through the list of token ids sorted in descending order
        for i, sampled_token_ind in enumerate(delta_token_ids):
            sampled_token = self.tokenizer.decode(
                [sampled_token_ind], add_special_tokens=False
            )

            # Form a mask made up of booleans where the index of the mask ==
            # index of function/parameter name in func_options/parameter_options.
            if stage == "function":
                # Form the function name with the current sampled token id
                new_curr_tokens_id = output_token_ids + [sampled_token_ind]
                new_curr_tokens = self.tokenizer.decode(new_curr_tokens_id)

                options_mask = [
                    True if option.startswith(new_curr_tokens.lstrip(" ")) else False
                    for option in func_options
                ]
            else:
                # Check which parameters are already generated and mask away those
                # while creating options_mask
                # Use regex to extract the keys
                key_pattern = r'"([^"]+)":'
                wellformed_params = re.findall(key_pattern, curr_params_str)

                # Form the parameter name with the current sampled token id
                new_curr_tokens = curr_params_str + self.tokenizer.decode(
                    [sampled_token_ind]
                )

                options_mask = []
                for option in parameter_options:
                    if option in wellformed_params or not option.startswith(
                        new_curr_tokens[new_curr_tokens.rfind('"') + 1 :]
                    ):
                        options_mask.append(False)
                    else:
                        options_mask.append(True)

            # The grammar-sampled token is valid if any element in options_mask
            # is True (The token helps in forming that function name).
            # Reject the whitespace (" ") and empty ("") tokens too
            if any(options_mask) and sampled_token.strip(" ") != "":
                return sampled_token_ind

        return None

    async def step_async(self) -> List[RequestOutput]:
        """Performs one decoding iteration and returns newly generated results.
        The workers are ran asynchronously if possible.

        This function performs one decoding iteration of the engine. It first
        schedules the sequences to be executed in the next iteration and the
        token blocks to be swapped in/out/copy. Then, it executes the model
        and updates the scheduler with the model outputs. Finally, it decodes
        the sequences and returns the newly generated results.
        """
        seq_group_metadata_list, scheduler_outputs, ignored = self._schedule()
        if scheduler_outputs.is_empty():
            return ignored

        # Execute the model.
        output = await self._run_workers_async(
            "execute_model",
            seq_group_metadata_list=seq_group_metadata_list,
            blocks_to_swap_in=scheduler_outputs.blocks_to_swap_in,
            blocks_to_swap_out=scheduler_outputs.blocks_to_swap_out,
            blocks_to_copy=scheduler_outputs.blocks_to_copy,
        )

        # Loop through all the output in the batch
        for i in range(len(output)):
            # Get all the required variables for grammar sampling
            model_sampled_token_id = output[i].samples[-1].output_token
            delta_token_id_by_logprobs = list(output[i].samples[-1].logprobs.keys())

            request_id = seq_group_metadata_list[i].request_id
            seq_data_id = list(seq_group_metadata_list[i].seq_data.keys())[0]
            prompt_token_ids = (
                seq_group_metadata_list[i].seq_data[seq_data_id].prompt_token_ids
                + seq_group_metadata_list[i].seq_data[seq_data_id].output_token_ids
            )
            prompt_str = self.tokenizer.decode(prompt_token_ids)

            prompt_template_version = self.prompt_template_versions[request_id]
            if prompt_template_version == "v1":
                start_token = "<|START_OF_FUNCTION_CALL|>"
            elif prompt_template_version == "v2":
                start_token = "<|recipient|>"

            # Get the suffix (whatever text after the start_token)
            suffix = prompt_str[
                prompt_str.rfind(start_token) + len(start_token) :
            ].lstrip()

            # Get the tokens generated in previous steps
            # Remove initial <|START_OF_FUNCTION_CALL|> for v1
            output_token_ids = (
                seq_group_metadata_list[i].seq_data[seq_data_id].output_token_ids
            )
            if prompt_template_version == "v1" and len(output_token_ids) > 0:
                output_token_ids = (
                    output_token_ids[1:]
                    if output_token_ids[0] == 32005
                    else output_token_ids
                )

            # Check whether to apply grammar sampling
            if start_token in prompt_str and self.check_to_sample(
                suffix=suffix,
                prompt_template_version=prompt_template_version,
                request_id=request_id,
            ):
                grammar_sampled_token_id = self.sample(
                    delta_token_ids=delta_token_id_by_logprobs,
                    output_token_ids=output_token_ids,
                    suffix=suffix,
                    prompt_template_version=prompt_template_version,
                    request_id=request_id,
                )

                # Replace the model-sampled token id only if the grammar-sampled
                # token id is different
                if (
                    grammar_sampled_token_id is not None
                    and grammar_sampled_token_id != model_sampled_token_id
                ):
                    output[i].samples[-1].output_token = grammar_sampled_token_id
            # This is to check if the complete function name is generated and replace
            # the intended model token to "\n", stopping the model from hallucinating a
            # function name that has the intended function name as prefix
            elif self.check_end_of_sampling(suffix=suffix, request_id=request_id):
                if prompt_template_version == "v1":
                    # encoding ":" will get 714 instead which has a preceding whitespace " "
                    # Thus, directly using ":" without any whitespace here
                    output[i].samples[-1].output_token = 28747
                else:
                    stopping_token = "\n"
                    output[i].samples[-1].output_token = self.tokenizer.encode(
                        stopping_token, add_special_tokens=False
                    )[-1]

        return self._process_model_outputs(output, scheduler_outputs) + ignored

    async def _run_workers_async(
        self,
        method: str,
        *args,
        get_all_outputs: bool = False,
        **kwargs,
    ) -> Any:
        """Runs the given method on all workers."""
        all_outputs = []
        for worker in self.workers:
            if self.parallel_config.worker_use_ray:
                executor = partial(worker.execute_method.remote, method)
            else:
                executor = getattr(worker, method)

            output = executor(*args, **kwargs)
            all_outputs.append(output)

        if self.parallel_config.worker_use_ray:
            all_outputs = await asyncio.gather(*all_outputs)

        if get_all_outputs:
            return all_outputs

        # Make sure all workers have the same results.
        output = all_outputs[0]
        for other_output in all_outputs[1:]:
            assert output == other_output
        return output


class AsyncLLMEngine:
    """An asynchronous wrapper for LLMEngine.

    This class is used to wrap the LLMEngine class to make it asynchronous. It
    uses asyncio to create a background loop that keeps processing incoming
    requests. The LLMEngine is kicked by the generate method when there
    are requests in the waiting queue. The generate method yields the outputs
    from the LLMEngine to the caller.

    NOTE: For the comprehensive list of arguments, see `LLMEngine`.

    Args:
        worker_use_ray: Whether to use Ray for model workers. Required for
            distributed execution. Should be the same as
            `parallel_config.worker_use_ray`.
        engine_use_ray: Whether to make LLMEngine a Ray actor. If so, the
            async frontend will be executed in a separate process as the
            model workers.
        log_requests: Whether to log the requests.
        start_engine_loop: If True, the background task to run the engine
            will be automatically started in the generate call.
        *args, *kwargs: Arguments for LLMEngine.
    """

    _engine_class: Type[_AsyncLLMEngine] = _AsyncLLMEngine

    def __init__(
        self,
        worker_use_ray: bool,
        engine_use_ray: bool,
        *args,
        log_requests: bool = True,
        max_log_len: Optional[int] = None,
        start_engine_loop: bool = True,
        **kwargs,
    ) -> None:
        self.worker_use_ray = worker_use_ray
        self.engine_use_ray = engine_use_ray
        self.log_requests = log_requests
        self.max_log_len = max_log_len
        self.engine = self._init_engine(*args, **kwargs)

        self.background_loop = None
        # We need to keep a reference to unshielded
        # task as well to prevent it from being garbage
        # collected
        self._background_loop_unshielded = None
        self.start_engine_loop = start_engine_loop
        self._request_tracker = RequestTracker()

    @property
    def is_running(self) -> bool:
        return self.background_loop is not None and not self.background_loop.done()

    def start_background_loop(self) -> None:
        """Start the background loop."""
        if self.is_running:
            raise RuntimeError("Background loop is already running.")
        self._request_tracker.init_event()

        self._background_loop_unshielded = asyncio.get_event_loop().create_task(
            self.run_engine_loop()
        )
        self._background_loop_unshielded.add_done_callback(
            partial(_raise_exception_on_finish, request_tracker=self._request_tracker)
        )
        self.background_loop = asyncio.shield(self._background_loop_unshielded)

    def _init_engine(self, *args, **kwargs) -> Union[_AsyncLLMEngine, "ray.ObjectRef"]:
        if not self.engine_use_ray:
            engine_class = self._engine_class
        elif self.worker_use_ray:
            engine_class = ray.remote(num_cpus=0)(self._engine_class).remote
        else:
            engine_class = ray.remote(num_gpus=1)(self._engine_class).remote
        return engine_class(*args, **kwargs)

    async def engine_step(self) -> bool:
        """Kick the engine to process the waiting requests.

        Returns True if there are in-progress requests."""

        (
            new_requests,
            finished_requests,
        ) = self._request_tracker.get_new_and_finished_requests()

        for new_request in new_requests:
            # Add the request into the vLLM engine's waiting queue.
            # TODO: Maybe add add_request_batch to reduce Ray overhead
            if self.engine_use_ray:
                await self.engine.add_request.remote(**new_request)
            else:
                self.engine.add_request(**new_request)

        if finished_requests:
            await self._engine_abort(finished_requests)

        if self.engine_use_ray:
            request_outputs = await self.engine.step.remote()
        else:
            request_outputs = await self.engine.step_async()

        # Put the outputs into the corresponding streams.
        for request_output in request_outputs:
            self._request_tracker.process_request_output(
                request_output, verbose=self.log_requests
            )

        return len(request_outputs) > 0

    async def _engine_abort(self, request_ids: Iterable[str]):
        if self.engine_use_ray:
            await self.engine.abort_request.remote(request_ids)
        else:
            self.engine.abort_request(request_ids)

    async def run_engine_loop(self):
        # Initialize the RequestTracker here so it uses the right event loop.
        has_requests_in_progress = False
        while True:
            if not has_requests_in_progress:
                await self._request_tracker.wait_for_new_requests()
            has_requests_in_progress = await self.engine_step()
            await asyncio.sleep(0)

    async def add_request(
        self,
        request_id: str,
        prompt: Optional[str],
        sampling_params: SamplingParams,
        prompt_token_ids: Optional[List[int]] = None,
        arrival_time: Optional[float] = None,
    ) -> AsyncStream:
        if self.log_requests:
            shortened_prompt = prompt
            shortened_token_ids = prompt_token_ids
            if self.max_log_len is not None:
                if shortened_prompt is not None:
                    shortened_prompt = shortened_prompt[: self.max_log_len]
                if shortened_token_ids is not None:
                    shortened_token_ids = shortened_token_ids[: self.max_log_len]
            logger.info(
                f"Received request {request_id}: "
                f"prompt: {shortened_prompt!r}, "
                f"sampling params: {sampling_params}, "
                f"prompt token ids: {shortened_token_ids}."
            )

        if not self.is_running:
            if self.start_engine_loop:
                self.start_background_loop()
            else:
                raise AsyncEngineDeadError(
                    "Background loop is not running. If it was running, "
                    "inspect the output to find the stacktrace of the "
                    "error that caused the background loop to stop "
                    "(AsyncEngineDeadError)."
                )

        stream = self._request_tracker.add_request(
            request_id,
            prompt=prompt,
            sampling_params=sampling_params,
            prompt_token_ids=prompt_token_ids,
            arrival_time=arrival_time,
        )

        return stream

    async def generate(
        self,
        prompt: Optional[str],
        sampling_params: SamplingParams,
        request_id: str,
        prompt_token_ids: Optional[List[int]] = None,
        tools_or_functions: Optional[List[dict]] = None,
        prompt_template_version: str = "v2",
    ) -> RequestOutput:
        """Generate outputs for a request.

        Generate outputs for a request. This method is a coroutine. It adds the
        request into the waiting queue of the LLMEngine and streams the outputs
        from the LLMEngine to the caller.

        Args:
            prompt: The prompt string. Can be None if prompt_token_ids is
                provided.
            sampling_params: The sampling parameters of the request.
            request_id: The unique id of the request.
            prompt_token_ids: The token IDs of the prompt. If None, we
                use the tokenizer to convert the prompts to token IDs.

        Yields:
            The output `RequestOutput` objects from the LLMEngine for the
            request.
        """
        # Preprocess the request.
        # This should not be used for logging, as it is monotonic time.
        arrival_time = time.monotonic()

        # Initialize the request_id entry of self.engine.tools_or_functions
        # and prompt_template_version at the start of generate method
        self.engine.tools_or_functions[request_id] = [
            tool_or_function for tool_or_function in tools_or_functions
        ]
        self.engine.prompt_template_versions[request_id] = prompt_template_version

        try:
            stream = await self.add_request(
                request_id,
                prompt,
                sampling_params,
                prompt_token_ids=prompt_token_ids,
                arrival_time=arrival_time,
            )

            async for request_output in stream:
                yield request_output
        except (Exception, asyncio.CancelledError) as e:
            # If there is an exception or coroutine is cancelled, abort the
            # request.
            self._abort(request_id)

            # Delete request_id from self.engine.tools_or_functions before raising error
            del self.engine.tools_or_functions[request_id]
            # Delete request_id from self.engine.prompt_template_versions before raising error
            del self.engine.prompt_template_versions[request_id]

            raise e

        # Delete the request_id from self.engine.tools_or_functions before finishing the request
        del self.engine.tools_or_functions[request_id]
        # Delete the request_id from self.engine.prompt_template_versions before finishing the request
        del self.engine.prompt_template_versions[request_id]

    async def abort(self, request_id: str) -> None:
        """Abort a request.

        Abort a submitted request. If the request is finished or not found,
        this method will be a no-op.

        Args:
            request_id: The unique id of the request.
        """
        if not self.is_running:
            raise AsyncEngineDeadError(
                "Background loop is not running. If it was running, "
                "inspect the output to find the stacktrace of the "
                "error that caused the background loop to stop "
                "(AsyncEngineDeadError)."
            )

        return self._abort(request_id)

    def _abort(self, request_id: str) -> None:
        """Abort a request.

        Abort a submitted request. If the request is finished or not found,
        this method will be a no-op.

        Args:
            request_id: The unique id of the request.
        """
        self._request_tracker.abort_request(request_id, verbose=self.log_requests)

    async def get_model_config(self) -> ModelConfig:
        """Get the model configuration of the vLLM engine."""
        if self.engine_use_ray:
            return await self.engine.get_model_config.remote()
        else:
            return self.engine.get_model_config()

    @classmethod
    def from_engine_args(
        cls, engine_args: AsyncEngineArgs, start_engine_loop: bool = True
    ) -> "AsyncLLMEngine":
        """Creates an async LLM engine from the engine arguments."""
        # Create the engine configs.
        engine_configs = engine_args.create_engine_configs()
        parallel_config = engine_configs[2]
        # Initialize the cluster.
        distributed_init_method, placement_group = initialize_cluster(
            parallel_config, engine_args.engine_use_ray
        )
        # Create the async LLM engine.
        engine = cls(
            engine_args.worker_use_ray,
            engine_args.engine_use_ray,
            *engine_configs,
            distributed_init_method,
            placement_group,
            log_requests=not engine_args.disable_log_requests,
            log_stats=not engine_args.disable_log_stats,
            max_log_len=engine_args.max_log_len,
            start_engine_loop=start_engine_loop,
        )
        return engine
        return engine
