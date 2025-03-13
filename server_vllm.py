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
import logging
import re
from typing import Any, AsyncGenerator, Dict, List, Literal, Optional, Tuple, Union

import fastapi
import uvicorn
import vllm.entrypoints.openai.api_server as vllm_api_server
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.api_server import mount_metrics
from vllm.entrypoints.openai.protocol import (
    LoadLoraAdapterRequest,
    ModelCard,
    ModelList,
    ModelPermission,
    UnloadLoraAdapterRequest,
)
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.transformers_utils.tokenizer import get_tokenizer
from vllm.utils import AtomicCounter

from functionary.openai_types import ChatCompletionRequest
from functionary.vllm_inference import (
    process_chat_completion,
    process_load_lora_adapter,
    process_unload_lora_adapter,
)

TIMEOUT_KEEP_ALIVE = 5  # seconds

# logger = init_logger(__name__)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


served_model = []
served_loras = []
lora_id_counter = AtomicCounter(0)
app = fastapi.FastAPI()


@app.get("/health")
async def _health():
    """Health check."""
    # vLLM's OpenAI server's health check is too heavy and also requires
    # creating engine_client here, so we just return 200 here.
    return Response(status_code=200)


@app.get("/v1/models")
async def show_available_models():
    """Show available models."""
    model_cards = []
    if isinstance(served_model, list):
        for model in served_model:
            model_cards.append(
                ModelCard(id=model, root=model, permission=[ModelPermission()])
            )
    else:
        model_cards.append(
            ModelCard(
                id=served_model, root=served_model, permission=[ModelPermission()]
            )
        )

    for lora in served_loras:
        parent = (
            lora.base_model_name
            if lora.base_model_name
            else (served_model[0] if isinstance(served_model, list) else served_model)
        )
        model_cards.append(
            ModelCard(
                id=lora.lora_name,
                root=lora.lora_path,
                parent=parent,
                permission=[ModelPermission()],
            )
        )

    return ModelList(data=model_cards)


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

    return await process_chat_completion(
        request=request,
        raw_request=raw_request,
        tokenizer=tokenizer,
        served_model=served_model,
        served_loras=served_loras,
        engine_model_config=engine_model_config,
        engine=engine,
    )


@app.post("/v1/load_lora_adapter")
async def load_lora_adapter(request: LoadLoraAdapterRequest):
    global served_loras

    error, served_loras = await process_load_lora_adapter(
        request, served_loras, lora_id_counter
    )
    if not isinstance(error, str):
        return error

    # `error` is the success message if it is a string
    return Response(status_code=200, content=error)


@app.post("/v1/unload_lora_adapter")
async def unload_lora_adapter(request: UnloadLoraAdapterRequest):
    global served_loras

    error, served_loras = await process_unload_lora_adapter(request, served_loras)
    if not isinstance(error, str):
        return error

    # `error` is the success message if it is a string
    return Response(status_code=200, content=error)


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
        "--lora-modules",
        nargs="*",
        type=str,
        help="LoRA modules in the format 'name=path name=path ...'",
        default=[],
    )

    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    mount_metrics(app)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allowed_origins,
        allow_credentials=args.allow_credentials,
        allow_methods=args.allowed_methods,
        allow_headers=args.allowed_headers,
    )

    logger.info(f"args: {args}")

    served_model = [args.model]

    if args.served_model_name is not None:
        served_model += args.served_model_name

    for lora_module in args.lora_modules:
        lora_name, lora_path = lora_module.split("=")
        served_loras.append(
            LoRARequest(
                lora_name=lora_name,
                lora_int_id=lora_id_counter.inc(1),
                lora_path=lora_path,
            )
        )

    engine_args = AsyncEngineArgs.from_cli_args(args)
    # A separate tokenizer to map token IDs to strings.
    tokenizer = get_tokenizer(
        engine_args.tokenizer, 
        tokenizer_mode=engine_args.tokenizer_mode,
        trust_remote_code=engine_args.trust_remote_code,
    )

    engine = AsyncLLMEngine.from_engine_args(engine_args)
    engine_model_config = asyncio.run(engine.get_model_config())

    # Adapt to vLLM's health endpoint
    vllm_api_server.async_engine_client = engine

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
    )
