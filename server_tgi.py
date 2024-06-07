import argparse
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, Optional, Tuple

import docker
import requests
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from huggingface_hub import InferenceClient
from transformers import AutoTokenizer

from functionary.inference_stream import generate_openai_format_from_stream_async
from functionary.openai_types import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
    Function,
    StreamChoice,
    Tool,
    UsageInfo,
)
from functionary.prompt_template import get_prompt_template_from_tokenizer
from functionary.prompt_template.prompt_utils import enforce_tool_choice

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_health():
    try:
        response = requests.get(f"{args.tgi_endpoint}/health")
        if response.status_code == 200:
            return "healthy"
    except requests.RequestException:
        pass

    return "unhealthy"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Run TGI docker container at start up
    logger.info("Starting TGI docker container...")
    port = int(args.tgi_endpoint[args.tgi_endpoint.rindex(":") + 1 :])
    tgi_docker_client = docker.from_env()
    tgi_container = tgi_docker_client.containers.run(
        "ghcr.io/huggingface/text-generation-inference:2.0.4",
        f"--model-id {args.model} --max-total-tokens {args.tgi_max_total_tokens}",
        device_requests=[docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])],
        shm_size="1g",
        ports={"80/tcp": port},
        volumes={args.model_save_folder: {"bind": "/data", "mode": "rw"}},
        detach=True,
    )
    # Wait until model is loaded in TGI container
    sleep_time, elapsed_time = 1, 0
    for _ in range(args.tgi_startup_timeout):
        time.sleep(sleep_time)
        if get_health() != "healthy":
            elapsed_time += sleep_time
        else:
            break
    logger.info("TGI docker container started. Ready to go...")
    yield
    # Stop and remove TGI docker container at shutdown
    tgi_container.stop()
    tgi_container.remove()


app = FastAPI(title="Functionary TGI", lifespan=lifespan)


@app.post("/v1/chat/completions")
async def create_chat_completion(raw_request: dict):
    """Completion API similar to OpenAI's API.
    See  https://platform.openai.com/docs/api-reference/chat/create
    for the API specification. This API follows the OpenAI ChatCompletion API.
    """

    request = ChatCompletionRequest(**raw_request)

    request_id = f"cmpl-{str(uuid.uuid4().hex)}"
    created_time = int(time.time())

    tool_func_choice = request.tool_choice or "auto"

    # Create messages and tools/functions of type dicts
    dic_messages = [mess.dict() for mess in request.messages]
    dic_messages.append({"role": "assistant"})
    dic_messages = prompt_template.pre_process_messages_before_inference(dic_messages)
    tools_or_functions = request.tools if request.tools else request.functions
    tools_or_functions = enforce_tool_choice(
        choice=tool_func_choice, tools_or_functions=tools_or_functions
    )
    tools_or_functions = [
        tool_or_function.model_dump() for tool_or_function in tools_or_functions
    ]

    # Form prompt and suffix from tool_func_choice
    prompt = prompt_template.get_prompt_from_messages(
        messages=dic_messages, tools_or_functions=tools_or_functions
    )
    if tool_func_choice == "none":
        prompt += prompt_template.get_force_text_generation_prefix()
    elif isinstance(tool_func_choice, Tool) or isinstance(tool_func_choice, Function):
        prompt += prompt_template.get_force_function_call_prefix(
            tool_func_choice.function.name
            if isinstance(tool_func_choice, Tool)
            else tool_func_choice.name
        )
    elif tool_func_choice == "required" and hasattr(
        prompt_template, "function_separator"
    ):
        prompt += getattr(prompt_template, "function_separator")

    hyperparams = {
        "stream": request.stream,
        "best_of": request.best_of,
        "do_sample": True if request.temperature > 0 else False,
        "frequency_penalty": request.frequency_penalty,
        "max_new_tokens": request.max_tokens,
        "stop_sequences": request.stop,
        "temperature": request.temperature,
        "top_k": request.top_k if request.top_k > 0 else None,
        "top_p": request.top_p if 0 < request.top_p < 1.0 else None,
    }

    async def wrap_tgi_generator() -> AsyncGenerator[Tuple[str, Optional[str]], None]:
        for chunk in client.text_generation(prompt=prompt, details=True, **hyperparams):
            delta_text = chunk.token.text
            finish_reason = None
            if (
                delta_text.strip()
                not in prompt_template.get_stop_tokens_for_generation()
            ):
                yield delta_text, finish_reason
        yield "", "stop"

    async def completion_stream_generator(
        tool_choice, functions
    ) -> AsyncGenerator[str, None]:
        generator = wrap_tgi_generator()
        tool_call_count = 0
        async for response in generate_openai_format_from_stream_async(
            generator, prompt_template, tool_choice
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

            chunk = StreamChoice(**response)
            result = ChatCompletionChunk(id=request_id, choices=[chunk])
            chunk_dic = result.dict(exclude_unset=True)
            chunk_data = json.dumps(chunk_dic, ensure_ascii=False)
            yield f"data: {chunk_data}\n\n"
            # Break from for loop after the first tool_call is streamed if functions is provided
            if functions and tool_call_count == 2:
                break
        yield "data: [DONE]\n\n"

    if request.stream:
        return StreamingResponse(
            completion_stream_generator(
                tool_choice=tool_func_choice,
                functions=request.functions,
            ),
            media_type="text/event-stream",
        )
    else:
        response = client.text_generation(prompt=prompt, details=True, **hyperparams)
        # Transformers tokenizers is problematic with some special tokens such that they
        # are not reflected in `response.generated_text`. Use this hack temporarily first.
        # Issue: https://github.com/huggingface/text-generation-inference/issues/1984
        response_text = "".join([token.text for token in response.details.tokens])
        chat_mess = prompt_template.parse_assistant_response(
            llm_output=response_text, tool_choice=tool_func_choice
        )
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
        finish_reason = "stop"
        if "function_call" in chat_mess and chat_mess["function_call"]:
            finish_reason = "function_call"
        if "tool_calls" in chat_mess and chat_mess["tool_calls"]:
            finish_reason = "tool_calls"

        choices = [
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(**chat_mess),
                finish_reason=finish_reason,
            )
        ]

        num_prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
        num_generated_tokens = response.details.generated_tokens
        usage = UsageInfo(
            prompt_tokens=num_prompt_tokens,
            completion_tokens=num_generated_tokens,
            total_tokens=num_prompt_tokens + num_generated_tokens,
        )

        return ChatCompletionResponse(
            id=request_id,
            created=created_time,
            model=request.model,
            choices=choices,
            usage=usage,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Functionary TGI Server")
    parser.add_argument(
        "--model", type=str, default="meetkai/functionary-small-v2.5", help="Model name"
    )
    parser.add_argument(
        "--tgi_endpoint",
        type=str,
        default="http://127.0.0.1:8080",
        help="The http address for TGI server",
    )
    parser.add_argument(
        "--model_save_folder",
        type=str,
        default=f"{Path.home()}/data",
        help="The folder to save and cache the model weights",
    )
    parser.add_argument(
        "--tgi_max_total_tokens",
        type=int,
        default=8192,
        help="The context window of the model",
    )
    parser.add_argument(
        "--tgi_startup_timeout",
        type=int,
        default=120,
        help="How long this server waits for TGI server to load the model",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    args = parser.parse_args()

    client = InferenceClient(args.tgi_endpoint)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompt_template = get_prompt_template_from_tokenizer(tokenizer)
    uvicorn.run(app, host=args.host, port=args.port)
