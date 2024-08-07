import json
import time
from http import HTTPStatus
from typing import Any, AsyncGenerator, Dict, List, Literal, Optional, Tuple, Union

from fastapi import BackgroundTasks, Request
from fastapi.responses import JSONResponse, StreamingResponse
from vllm.entrypoints.openai.protocol import ErrorResponse
from vllm.inputs import TokensPrompt
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid

from functionary.inference_stream import generate_openai_format_from_stream_async
from functionary.inference_utils import analyze_tools_and_tool_choice
from functionary.openai_types import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
    Function,
    FunctionCall,
    StreamChoice,
    Tool,
    UsageInfo,
)
from functionary.prompt_template import get_prompt_template_from_tokenizer
from functionary.prompt_template.prompt_utils import (
    enforce_tool_choice,
    get_random_tool_call_id,
    prepare_messages_for_inference,
)


def create_error_response(
    status_code: HTTPStatus, message: str, param: Optional[str]
) -> JSONResponse:
    return JSONResponse(
        ErrorResponse(
            message=message,
            type="invalid_request_error",
            param=param,
            code=status_code.value,
        ).dict(),
        status_code=status_code.value,
    )


async def check_all_errors(request, served_model) -> Optional[JSONResponse]:
    if request.model != served_model:
        return create_error_response(
            status_code=HTTPStatus.NOT_FOUND,
            message=f"The model `{request.model}` does not exist.",
            param=None,
        )
    if request.tools and request.functions:
        return create_error_response(
            status_code=HTTPStatus.BAD_REQUEST,
            message="'functions' and 'tools' cannot both be provided. 'functions' are deprecated; use the 'tools' parameter instead.",
            param=None,
        )
    if isinstance(request.function_call, str) and request.function_call not in [
        "none",
        "auto",
    ]:
        return create_error_response(
            status_code=HTTPStatus.BAD_REQUEST,
            message=f"Invalid value: '{request.function_call}'. Supported values are: 'none' and 'auto'.",
            param="function_call",
        )
    if isinstance(request.tool_choice, str) and request.tool_choice not in [
        "none",
        "auto",
        "required",
    ]:
        return create_error_response(
            status_code=HTTPStatus.BAD_REQUEST,
            message=f"Invalid value: '{request.tool_choice}'. Supported values are: 'none', 'auto', and 'required'.",
            param="tool_choice",
        )
    if request.functions is None and request.function_call is not None:
        return create_error_response(
            status_code=HTTPStatus.BAD_REQUEST,
            message=f"Invalid value for 'function_call': 'function_call' is only allowed when 'functions' are specified.",
            param="function_call",
        )
    if request.tools is None and request.tool_choice is not None:
        return create_error_response(
            status_code=HTTPStatus.BAD_REQUEST,
            message=f"Invalid value for 'tool_choice': 'tool_choice' is only allowed when 'tools' are specified.",
            param="tool_choice",
        )
    return


async def check_length(request, input_ids, model_config):
    if hasattr(model_config.hf_config, "max_sequence_length"):
        context_len = model_config.hf_config.max_sequence_length
    elif hasattr(model_config.hf_config, "seq_length"):
        context_len = model_config.hf_config.seq_length
    elif hasattr(model_config.hf_config, "max_position_embeddings"):
        context_len = model_config.hf_config.max_position_embeddings
    elif hasattr(model_config.hf_config, "seq_length"):
        context_len = model_config.hf_config.seq_length
    else:
        context_len = 4096

    # Scale the context_len if rope scaling with "type" is provided
    # Currently only supports ["linear", "dynamic", "yarn"], not yet for "su"/"longrope"
    if (
        hasattr(model_config.hf_config, "rope_scaling")
        and model_config.hf_config.rope_scaling is not None
        and "type" in model_config.hf_config.rope_scaling
    ):
        # From vLLM's code, it seems like only YaRN requires
        # "original_max_position_embeddings" in rope_scaling dict
        # https://github.com/vllm-project/vllm/blob/main/vllm/config.py#L1458-L1460
        if model_config.hf_config.rope_scaling["type"] == "yarn":
            context_len = model_config.hf_config.rope_scaling[
                "original_max_position_embeddings"
            ]
        context_len *= model_config.hf_config.rope_scaling["factor"]

    token_num = len(input_ids)

    if token_num + request.max_tokens > context_len:
        return create_error_response(
            status_code=HTTPStatus.BAD_REQUEST,
            message=(
                f"This model's maximum context length is {context_len} tokens. "
                f"However, you requested {request.max_tokens + token_num} tokens "
                f"({token_num} in the messages, "
                f"{request.max_tokens} in the completion). "
                f"Please reduce the length of the messages or completion."
            ),
            param=None,
        )
    else:
        return None


async def process_chat_completion(
    request: ChatCompletionRequest,
    raw_request: Optional[Request],
    tokenizer: Any,
    served_model: str,
    engine_model_config: Any,
    enable_grammar_sampling: bool,
    engine: Any,
):
    error_check_ret = await check_all_errors(request, served_model)
    if error_check_ret is not None:
        return error_check_ret

    tools_or_functions, tool_func_choice = analyze_tools_and_tool_choice(request)

    prompt_token_ids = prepare_messages_for_inference(
        tokenizer=tokenizer,
        messages=request.messages,
        tools_or_functions=tools_or_functions,
        tool_choice=tool_func_choice,
    ).tolist()[0]

    error_check_ret = await check_length(request, prompt_token_ids, engine_model_config)
    if error_check_ret is not None:
        return error_check_ret

    model_name = request.model
    request_id = f"cmpl-{random_uuid()}"
    created_time = int(time.time())

    # compute stop_token_ids
    stop_token_ids = []
    prompt_template = get_prompt_template_from_tokenizer(tokenizer)
    for stop_tok in prompt_template.get_stop_tokens_for_generation():
        tok_ids = tokenizer.encode(stop_tok, add_special_tokens=False)
        stop_token_ids.append(tok_ids[-1])

    # In vLLM==0.4.1, SamplingParams.logprobs has a proportional effect on latency
    # We need to limit the size of SamplingParams.logprobs as a temporary fix first
    # while investigating this problem in vLLM
    if enable_grammar_sampling is False:
        logprobs = None
    else:
        logprobs = 200

    try:
        sampling_params = SamplingParams(
            n=request.n,
            presence_penalty=request.presence_penalty,
            frequency_penalty=request.frequency_penalty,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=request.stop,
            stop_token_ids=stop_token_ids,
            max_tokens=request.max_tokens,
            best_of=request.best_of,
            top_k=request.top_k,
            ignore_eos=request.ignore_eos,
            use_beam_search=request.use_beam_search,
            skip_special_tokens=False,
            logprobs=logprobs,
        )
    except ValueError as e:
        return create_error_response(HTTPStatus.BAD_REQUEST, str(e))

    if enable_grammar_sampling:
        result_generator = engine.generate(
            inputs=TokensPrompt(prompt_token_ids=prompt_token_ids),
            sampling_params=sampling_params,
            request_id=request_id,
            tools_or_functions=tools_or_functions,
            prompt_template_cls=prompt_template,
            tool_choice=tool_func_choice,
        )
    else:
        result_generator = engine.generate(
            inputs=TokensPrompt(prompt_token_ids=prompt_token_ids),
            sampling_params=sampling_params,
            request_id=request_id,
        )

    async def abort_request() -> None:
        await engine.abort(request_id)

    async def wrap_vllm_generator(
        tool_choice,
    ) -> AsyncGenerator[Tuple[str, Optional[str]], None]:
        previous_texts = ""
        async for res in result_generator:
            for output in res.outputs:
                delta_text = output.text[len(previous_texts) :]
                previous_texts = output.text
                finish_reason = output.finish_reason

                # If finish_reason is not None and delta_text is not empty,
                # the delta_text is the eos_token and just remove it
                if output.finish_reason is not None and len(delta_text) > 0:
                    delta_text = ""

                if (
                    delta_text.strip()
                    not in prompt_template.get_stop_tokens_for_generation()
                ):
                    # This part checks if delta_text is the first token and tool_choice is provided by user
                    # If so, it yields the prefix containing the tool_choice name first
                    if (
                        previous_texts == delta_text
                        and delta_text in prompt_template.fn_param_sep_token
                        and prompt_template.version != "v1"
                    ):
                        if tool_choice == "none":
                            yield "all" + prompt_template.fn_param_sep_token, finish_reason
                        elif isinstance(tool_choice, Tool):
                            yield tool_choice.function.name + prompt_template.fn_param_sep_token, finish_reason
                        elif isinstance(tool_choice, Function):
                            yield tool_choice.name + prompt_template.fn_param_sep_token, finish_reason
                    yield delta_text, finish_reason
        # yield "", "stop"

    async def completion_stream_generator(
        tool_choice, functions, tools_or_functions
    ) -> AsyncGenerator[str, None]:
        generator = wrap_vllm_generator(tool_choice=tool_choice)

        tool_call_count = 0
        async for response in generate_openai_format_from_stream_async(
            generator, prompt_template, tool_choice, tools_or_functions
        ):
            # Convert tool_calls to function_call if request.functions is provided
            if (
                functions
                and len(functions) > 0
                and "tool_calls" in response["delta"]
                and response["delta"]["tool_calls"]
                and len(response["delta"]["tool_calls"]) > 0
            ):
                tool_name = response["delta"]["tool_calls"][0]["function"]["name"]
                tool_args = response["delta"]["tool_calls"][0]["function"]["arguments"]
                response["delta"]["function_call"] = response["delta"]["tool_calls"][0][
                    "function"
                ]
                response["delta"]["tool_calls"] = None
                if tool_name and len(tool_name) > 0 and tool_args == "":
                    tool_call_count += 1
            # Return finish_reason after the first tool_call is streamed if functions is provided
            if functions and tool_call_count == 2:
                response["delta"] = {}
                response["finish_reason"] = "function_call"

            # Convert v1 from function_call to tool_calls if tools are provided instead of functions
            if prompt_template.version == "v1" and (
                functions is None or len(functions) == 0
            ):
                if "function_call" in response["delta"]:
                    response["delta"] = {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "function": response["delta"]["function_call"],
                                "id": get_random_tool_call_id(),
                                "type": "function",
                            }
                        ],
                    }
                if response["finish_reason"] == "function_call":
                    response["finish_reason"] = "tool_calls"
            chunk = StreamChoice(**response)
            result = ChatCompletionChunk(id=request_id, choices=[chunk])
            chunk_dic = result.dict(exclude_unset=True)
            chunk_data = json.dumps(chunk_dic, ensure_ascii=False)
            yield f"data: {chunk_data}\n\n"
            # Break from for loop after the first tool_call is streamed if functions is provided
            if functions and tool_call_count == 2:
                break
        yield "data: [DONE]\n\n"

    # Streaming response
    if request.stream:
        background_tasks = BackgroundTasks()
        # Abort the request if the client disconnects.
        background_tasks.add_task(abort_request)
        return StreamingResponse(
            completion_stream_generator(
                tool_choice=tool_func_choice,
                functions=request.functions,
                tools_or_functions=tools_or_functions,
            ),
            media_type="text/event-stream",
            background=background_tasks,
        )

    # Non-streaming response
    final_res: RequestOutput = None
    async for res in result_generator:
        if raw_request and await raw_request.is_disconnected():
            # Abort the request if the client disconnects.
            await abort_request()
            return create_error_response(HTTPStatus.BAD_REQUEST, "Client disconnected")
        final_res = res
    assert final_res is not None
    choices = []
    for output in final_res.outputs:
        text_response = output.text.strip()
        chat_mess = prompt_template.parse_assistant_response(
            llm_output=text_response,
            tool_choice=tool_func_choice,
        )  # parse_generated_content(text_response)

        # Convert tool_calls to function_call if request.functions is provided
        if (
            request.functions
            and "tool_calls" in chat_mess
            and chat_mess["tool_calls"] is not None
            and len(chat_mess["tool_calls"]) > 0
        ):
            chat_mess["function_call"] = {
                "name": chat_mess["tool_calls"][0]["function"]["name"],
                "arguments": chat_mess["tool_calls"][0]["function"]["arguments"],
            }
            chat_mess["tool_calls"] = None

        # Postprocess finish reason
        if "function_call" in chat_mess and chat_mess["function_call"]:
            output.finish_reason = "function_call"

        if "tool_calls" in chat_mess and chat_mess["tool_calls"]:
            output.finish_reason = "tool_calls"

        # Convert v1 from function_call to tool_calls if tools are provided instead of functions
        if (
            prompt_template.version == "v1"
            and output.finish_reason == "function_call"
            and (request.functions is None or len(request.functions) == 0)
        ):
            chat_mess = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "function": {
                            "name": chat_mess["function_call"]["name"],
                            "arguments": chat_mess["function_call"]["arguments"],
                        },
                        "id": get_random_tool_call_id(),
                        "type": "function",
                    }
                ],
            }
            output.finish_reason = "tool_calls"

        choice_data = ChatCompletionResponseChoice(
            index=output.index,
            message=ChatMessage(**chat_mess),
            finish_reason=output.finish_reason,
        )
        choices.append(choice_data)

    num_prompt_tokens = len(final_res.prompt_token_ids)
    num_generated_tokens = sum(len(output.token_ids) for output in final_res.outputs)
    usage = UsageInfo(
        prompt_tokens=num_prompt_tokens,
        completion_tokens=num_generated_tokens,
        total_tokens=num_prompt_tokens + num_generated_tokens,
    )
    response = ChatCompletionResponse(
        id=request_id,
        created=created_time,
        model=model_name,
        choices=choices,
        usage=usage,
    )

    if request.stream:
        # When user requests streaming but we don't stream, we still need to
        # return a streaming response with a single event.
        response_json = response.model_dump_json(exclude_unset=True)

        async def fake_stream_generator() -> AsyncGenerator[str, None]:
            yield f"data: {response_json}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            fake_stream_generator(), media_type="text/event-stream"
        )

    return response
