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
from typing import Any, AsyncGenerator, Dict, List, Literal, Optional, Tuple, Union
import logging

import fastapi
import uvicorn
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.openai.protocol import ModelCard, ModelList, ModelPermission
from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer import get_tokenizer

from functionary.openai_types import ChatCompletionRequest
from functionary.vllm_inference import process_chat_completion

TIMEOUT_KEEP_ALIVE = 5  # seconds

#logger = init_logger(__name__)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


served_model = None
app = fastapi.FastAPI()


@app.get("/v1/models")
async def show_available_models():
    """Show available models. Right now we only have one model."""
    model_cards = [
        ModelCard(id=served_model, root=served_model, permission=[ModelPermission()])
    ]
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
        engine_model_config=engine_model_config,
        enable_grammar_sampling=args.grammar_sampling,
        engine=engine,
    )


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
        "--enable-grammar-sampling",
        dest="grammar_sampling",
        action="store_true",
        default=False,
        help="enable grammar sampling for function names",
    )

    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    v1_pattern = r"v1.*$"
    v31_pattern = r"v3.1$"
    if re.search(v1_pattern, args.model) or re.search(v31_pattern, args.model):
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
