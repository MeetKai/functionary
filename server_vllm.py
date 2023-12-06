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
import time
from http import HTTPStatus
from typing import Any, AsyncGenerator, Dict, List, Literal, Optional, Tuple, Union

import fastapi
import uvicorn
from fastapi import BackgroundTasks, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.protocol import (
    ErrorResponse,
    LogProbs,
    ModelCard,
    ModelList,
    ModelPermission,
    UsageInfo,
)
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.transformers_utils.tokenizer import get_tokenizer
from vllm.utils import random_uuid

from functionary.grammar_sampler import GrammarSampler
from functionary.inference import prepare_messages_for_inference
from functionary.inference_stream import generate_openai_format_from_stream_async
from functionary.openai_types import (
    ChatCompletionChunk,
    ChatMessage,
    Function,
    FunctionCall,
    StreamChoice,
    Tool,
)
from functionary.prompt_template import (
    PromptTemplate,
    get_prompt_template_from_tokenizer,
)

TIMEOUT_KEEP_ALIVE = 5  # seconds

logger = init_logger(__name__)
served_model = None
app = fastapi.FastAPI()


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    functions: Optional[List[Function]] = None
    tools: Optional[List[Tool]] = None
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    max_tokens: Optional[int] = 256
    stop: Optional[Union[str, List[str]]] = Field(default_factory=list)
    stream: Optional[bool] = False
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    # Additional parameters supported by vLLM
    best_of: Optional[int] = None
    top_k: Optional[int] = -1
    ignore_eos: Optional[bool] = False
    use_beam_search: Optional[bool] = False


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[Literal["stop", "length", "function_call"]] = None


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{random_uuid()}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: UsageInfo


class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length", "function_call"]] = None


class ChatCompletionStreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{random_uuid()}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseStreamChoice]


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

    if request.logit_bias is not None:
        # TODO: support logit_bias in vLLM engine.
        return create_error_response(
            HTTPStatus.BAD_REQUEST, "logit_bias is not currently supported"
        )

    prompt_token_ids = prepare_messages_for_inference(
        tokenizer=tokenizer,
        messages=request.messages,
        functions=request.functions,
        tools=request.tools,
    ).tolist()[0]
    error_check_ret = await check_length(request, prompt_token_ids, engine_model_config)
    if error_check_ret is not None:
        return error_check_ret

    def create_request_id():
        return f"cmpl-{random_uuid()}"

    model_name = request.model
    request_id = create_request_id()
    created_time = int(time.time())

    # compute stop_token_ids
    stop_token_ids = []
    prompt_template = get_prompt_template_from_tokenizer(tokenizer)
    for stop_tok in prompt_template.get_stop_tokens_for_generation():
        tok_ids = tokenizer.encode(stop_tok, add_special_tokens=False)
        stop_token_ids.append(tok_ids[-1])

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
            logprobs=len(tokenizer.vocab.keys()),
        )
    except ValueError as e:
        return create_error_response(HTTPStatus.BAD_REQUEST, str(e))

    async def abort_request() -> None:
        await engine.abort(request_id)

    async def wrap_vllm_generator(
        prompt_token_ids, result_generator, functions, tools
    ) -> AsyncGenerator[Tuple[str, Optional[str]], None]:
        previous_texts = ""

        if prompt_template.version == "v1":
            fn_call_token_to_check = prompt_template.start_function
        elif prompt_template.version == "v2":
            fn_call_token_to_check = prompt_template.recipient_token
        else:
            raise NotImplementedError(
                "Grammar sampling not implemented for this prompt template version. Please check..."
            )

        grammar_sampling_to_break = False

        async for res in result_generator:
            for output in res.outputs:
                delta_text = output.text[len(previous_texts) :]
                text_to_check_grammar = tokenizer.decode(prompt_token_ids)
                previous_texts = output.text

                if grammar_sampler.check_to_sample(
                    text=text_to_check_grammar,
                    start_token=fn_call_token_to_check,
                    functions=functions,
                ):
                    delta_logprobs_token_ids = list(output.logprobs[-1].keys())
                    delta_logprobs = list(output.logprobs[-1].values())

                    grammar_sampled_token = grammar_sampler.sample(
                        functions=functions,
                        tools=tools,
                        delta_logprobs=delta_logprobs,
                        delta_token_ids=delta_logprobs_token_ids,
                        prompt_template_version=prompt_template.version,
                    )

                    if (
                        grammar_sampled_token is not None
                        and grammar_sampled_token != delta_text
                    ):
                        grammar_sampling_to_break = True
                        break

                finish_reason = output.finish_reason

                if (
                    delta_text.strip()
                    not in prompt_template.get_stop_tokens_for_generation()
                ):
                    yield delta_text, finish_reason

            if grammar_sampling_to_break:
                break

        if grammar_sampling_to_break:
            yield grammar_sampled_token, "grammar_sampling_stop"
        else:
            yield "", "stop"

    async def completion_stream_generator(
        prompt_token_ids,
    ) -> AsyncGenerator[str, None]:
        while True:
            result_generator = engine.generate(
                None,
                sampling_params,
                create_request_id(),
                prompt_token_ids=prompt_token_ids,
            )

            generator = wrap_vllm_generator(
                prompt_token_ids=prompt_token_ids,
                result_generator=result_generator,
                functions=request.functions,
                tools=request.tools,
            )

            async for response in generate_openai_format_from_stream_async(
                generator, prompt_template
            ):
                chunk = StreamChoice(**response)
                result = ChatCompletionChunk(id=request_id, choices=[chunk])
                chunk_dic = result.dict(exclude_unset=True)
                chunk_data = json.dumps(chunk_dic, ensure_ascii=False)
                if response["finish_reason"] == "grammar_sampling_stop":
                    break
                yield f"data: {chunk_data}\n\n"
            else:
                break

            prompt_token_ids = [
                i
                for i in tokenizer.encode(
                    tokenizer.decode(prompt_token_ids) + response["delta"]["content"],
                    add_special_tokens=False,
                )
                if i != 28705
            ]

        yield "data: [DONE]\n\n"

    # Streaming response
    if request.stream:
        background_tasks = BackgroundTasks()
        # Abort the request if the client disconnects.
        background_tasks.add_task(abort_request)
        return StreamingResponse(
            completion_stream_generator(prompt_token_ids=prompt_token_ids),
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
            text_response
        )  # parse_generated_content(text_response)
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

    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()

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
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    engine_model_config = asyncio.run(engine.get_model_config())

    # A separate tokenizer to map token IDs to strings.
    tokenizer = get_tokenizer(
        engine_args.tokenizer, tokenizer_mode=engine_args.tokenizer_mode
    )

    # A grammar-sampling module based on Guidance (https://github.com/guidance-ai/guidance)
    # that makes sure function name is always correct
    grammar_sampler = GrammarSampler(tokenizer=tokenizer)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
    )
