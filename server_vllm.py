# Adapted from
# https://github.com/vllm-project/vllm/blob/2bdea7ac110d3090d6a3c582aed36577ca480473/vllm/entrypoints/openai/api_server.py

# Copyright 2023 vLLM contributors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import asyncio
import json
import re
import time
from http import HTTPStatus
from typing import Any, AsyncGenerator, Dict, List, Literal, Optional, Tuple, Union

import fastapi
import uvicorn
from fastapi import BackgroundTasks, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.openai.protocol import (
    ErrorResponse,
    LogProbs,
    ModelCard,
    ModelList,
    ModelPermission,
)
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.transformers_utils.tokenizer import get_tokenizer
from vllm.utils import random_uuid

from functionary.inference import enforce_tool_choice, prepare_messages_for_inference
from functionary.inference_stream import generate_openai_format_from_stream_async
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
from functionary.prompt_template import (
    PredefinedFuncTypes,
    PromptTemplate,
    get_prompt_template_from_tokenizer,
)
from functionary.prompt_template.prompt_utils import get_random_tool_call_id

TIMEOUT_KEEP_ALIVE = 5  # seconds

logger = init_logger(__name__)
served_model = None
app = fastapi.FastAPI()


def create_error_response(status_code: HTTPStatus, message: str) -> JSONResponse:
    return JSONResponse(
        ErrorResponse(message=message, type="invalid_request_error").dict(),
        status_code=status_code.value,
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):  # pylint: disable=unused-argument
    return create_error_response(HTTPStatus.BAD_REQUEST, str(exc))


async def check_model(request) -> Optional[JSONResponse]:
    if request.model == served_model:
        return
    ret = create_error_response(
        HTTPStatus.NOT_FOUND,
        f"The model `{request.model}` does not exist.",
    )
    return ret


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

    token_num = len(input_ids)

    if token_num + request.max_tokens > context_len:
        return create_error_response(
            HTTPStatus.BAD_REQUEST,
            f"This model's maximum context length is {context_len} tokens. "
            f"However, you requested {request.max_tokens + token_num} tokens "
            f"({token_num} in the messages, "
            f"{request.max_tokens} in the completion). "
            f"Please reduce the length of the messages or completion.",
        )
    else:
        return None


@app.get("/v1/models")
async def show_available_models():
    """Show available models. Right now we only have one model."""
    model_cards = [
        ModelCard(id=served_model, root=served_model, permission=[ModelPermission()])
    ]
    return ModelList(data=model_cards)


def create_logprobs(
    token_ids: List[int],
    id_logprobs: List[Dict[int, float]],
    initial_text_offset: int = 0,
) -> LogProbs:
    """Create OpenAI-style logprobs."""
    logprobs = LogProbs()
    last_token_len = 0
    for token_id, id_logprob in zip(token_ids, id_logprobs):
        token = tokenizer.convert_ids_to_tokens(token_id)
        logprobs.tokens.append(token)
        logprobs.token_logprobs.append(id_logprob[token_id])
        if len(logprobs.text_offset) == 0:
            logprobs.text_offset.append(initial_text_offset)
        else:
            logprobs.text_offset.append(logprobs.text_offset[-1] + last_token_len)
        last_token_len = len(token)

        logprobs.top_logprobs.append(
            {tokenizer.convert_ids_to_tokens(i): p for i, p in id_logprob.items()}
        )
    return logprobs


@app.post("/v1/chat/completions")
async def create_chat_completion(raw_request: Request):
    """Completion API similar to OpenAI's API.
    See  https://platform.openai.com/docs/api-reference/chat/create
    for the API specification. This API mimics the OpenAI ChatCompletion API.
    NOTE: Currently we do not support the following features:
        - logit_bias (to be supported by vLLM engine)
    """
    request_json = await raw_request.json()
    request = ChatCompletionRequest(**request_json)

    logger.info(f"Received chat completion request: {request}")

    error_check_ret = await check_model(request)
    if error_check_ret is not None:
        return error_check_ret

    if request.logit_bias is not None and request.logit_bias:
        # TODO: support logit_bias in vLLM engine.
        return create_error_response(
            HTTPStatus.BAD_REQUEST, "logit_bias is not currently supported"
        )

    if request.tools:
        tools = enforce_tool_choice(
            tool_choice=request.tool_choice, tools=request.tools
        )
        tools_or_functions = [item.dict() for item in tools]
    elif request.functions:
        tools = None
        tools_or_functions = [item.dict() for item in request.functions]
    else:
        tools = None
        tools_or_functions = []

    prompt_token_ids = prepare_messages_for_inference(
        tokenizer=tokenizer,
        messages=request.messages,
        functions=request.functions,
        tools=tools,
        tool_choice=request.tool_choice,
    ).tolist()[0]

    # Remove any code_interpreter tools remaining
    if tools:
        tools = [tool for tool in tools if tool.type != "code_interpreter"]
        tools_or_functions = [
            tool for tool in tools_or_functions if tool["type"] != "code_interpreter"
        ]

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
    if args.grammar_sampling is False:
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

    if args.grammar_sampling:
        result_generator = engine.generate(
            prompt=None,
            sampling_params=sampling_params,
            request_id=request_id,
            prompt_token_ids=prompt_token_ids,
            tools_or_functions=tools_or_functions,
            prompt_template_cls=prompt_template,
            tool_choice=request.tool_choice,
        )
    else:
        result_generator = engine.generate(
            prompt=None,
            sampling_params=sampling_params,
            request_id=request_id,
            prompt_token_ids=prompt_token_ids,
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
                            yield prompt_template.get_predefined_function_names(
                                function_types=PredefinedFuncTypes.no_tool_call
                            )[0] + prompt_template.fn_param_sep_token, finish_reason
                        elif isinstance(tool_choice, Tool):
                            yield tool_choice.function.name + prompt_template.fn_param_sep_token, finish_reason
                    yield delta_text, finish_reason
        yield "", "stop"

    async def completion_stream_generator(
        tool_choice, functions
    ) -> AsyncGenerator[str, None]:
        generator = wrap_vllm_generator(tool_choice=tool_choice)
        async for response in generate_openai_format_from_stream_async(
            generator, prompt_template, tool_choice
        ):
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
        yield "data: [DONE]\n\n"

    # Streaming response
    if request.stream:
        background_tasks = BackgroundTasks()
        # Abort the request if the client disconnects.
        background_tasks.add_task(abort_request)
        return StreamingResponse(
            completion_stream_generator(
                tool_choice=request.tool_choice, functions=request.functions
            ),
            media_type="text/event-stream",
            background=background_tasks,
        )

    # Non-streaming response
    final_res: RequestOutput = None
    async for res in result_generator:
        if await raw_request.is_disconnected():
            # Abort the request if the client disconnects.
            await abort_request()
            return create_error_response(HTTPStatus.BAD_REQUEST, "Client disconnected")
        final_res = res
    assert final_res is not None
    choices = []
    for output in final_res.outputs:
        text_response = output.text.strip()
        chat_mess = prompt_template.parse_assistant_response(
            llm_output=text_response, tool_choice=request.tool_choice
        )  # parse_generated_content(text_response)

        # Postprocess finish reason
        if "function_call" in chat_mess and chat_mess["function_call"] is not None:
            output.finish_reason = "function_call"
        if "tool_calls" in chat_mess and chat_mess["tool_calls"] is not None:
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server."
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="host name")
    parser.add_argument("--port", type=int, default=8000, help="port number")
    parser.add_argument(
        "--allow-credentials", action="store_true", help="allow credentials"
    )
    parser.add_argument(
        "--allowed-origins", type=json.loads, default=["*"], help="allowed origins"
    )
    parser.add_argument(
        "--allowed-methods", type=json.loads, default=["*"], help="allowed methods"
    )
    parser.add_argument(
        "--allowed-headers", type=json.loads, default=["*"], help="allowed headers"
    )
    parser.add_argument(
        "--served-model-name",
        type=str,
        default=None,
        help="The model name used in the API. If not "
        "specified, the model name will be the same as "
        "the huggingface name.",
    )
    parser.add_argument(
        "--enable-grammar-sampling",
        dest="grammar_sampling",
        action="store_true",
        default=False,
        help="enable grammar sampling for function names",
    )

    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    pattern = r"v1.*$"
    if re.search(pattern, args.model):
        args.grammar_sampling = False

    if args.grammar_sampling:
        logger.info("Grammar sampling enabled.")
        from functionary.vllm_monkey_patch.async_llm_engine import AsyncLLMEngine
    else:
        from vllm.engine.async_llm_engine import AsyncLLMEngine

    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allowed_origins,
        allow_credentials=args.allow_credentials,
        allow_methods=args.allowed_methods,
        allow_headers=args.allowed_headers,
    )

    logger.info(f"args: {args}")

    if args.served_model_name is not None:
        served_model = args.served_model_name
    else:
        served_model = args.model

    engine_args = AsyncEngineArgs.from_cli_args(args)
    # A separate tokenizer to map token IDs to strings.
    tokenizer = get_tokenizer(
        engine_args.tokenizer, tokenizer_mode=engine_args.tokenizer_mode
    )
    # Overwrite vLLM's default ModelConfig.max_logprobs of 5
    engine_args.max_logprobs = len(tokenizer.vocab.keys())

    engine = AsyncLLMEngine.from_engine_args(engine_args)
    engine_model_config = asyncio.run(engine.get_model_config())

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
    )
