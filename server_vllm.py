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
from pydantic import BaseModel
import fastapi
import uvicorn
from fastapi import Request, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.openai.protocol import ModelCard, ModelList, ModelPermission
from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer import get_tokenizer

from functionary.openai_types import ChatCompletionRequest
from functionary.vllm_inference import process_chat_completion
import requests

import torch

DEVICE = "auto"
# Check if CUDA is available
if torch.cuda.is_available():
    print(f"CUDA is available! Number of devices: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA is not available. Setting device to cpu")
    DEVICE = "cpu"


TIMEOUT_KEEP_ALIVE = 5  # seconds

logger = init_logger(__name__)
served_model: List[str] = []
app = fastapi.FastAPI()


@app.get("/healthz")
async def get_health_and_readiness():
    """
    Indicate service health and readiness.

    Returns:
        dict: A dictionary containing the vLLM inference service's readiness and health status.
              - "ready" (bool): Indicates if the service is ready to accept requests.
              - "health" (bool): Indicates if the service is healthy and operational.
    """
    return {"ready": True, "health": True}


@app.get("/v1/models")
async def show_available_models():
    """Show available models. Right now we only have one model."""
    if len(served_model) > 0:
        model_cards = []
        for model in served_model:
            model_cards.append(
                ModelCard(
                    id=served_model[0],
                    root=served_model[0],
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

    if request.images is not None:
        logger.info(f"Vision request: {repr(request.images)}")
    return await process_chat_completion(
        request=request,
        raw_request=raw_request,
        tokenizer=tokenizer,
        served_model=request.model,
        engine_model_config=engine_model_config,
        enable_grammar_sampling=args.grammar_sampling,
        engine=engine,
    )


class EmbeddingData(BaseModel):
    object: str = "embedding"
    index: int
    embedding: List[float]


class UsageData(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: UsageData


class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    model: str


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def get_embedding(request: EmbeddingRequest, authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")

    # Handle both single string and list of strings
    inputs = [request.input] if isinstance(request.input, str) else request.input

    url = "http://embeddings:8080/embed"
    headers = {"Content-Type": "application/json"}
    data = {"inputs": inputs}

    response = requests.post(url, headers=headers, json=data)
    embeddings = response.json()

    # Construct the response data
    data = [
        EmbeddingData(object="embedding", index=i, embedding=embeddings[i])
        for i in range(len(inputs))
    ]

    response = EmbeddingResponse(
        object="list",
        data=data,
        model=request.model,
        usage=UsageData(prompt_tokens=len(inputs) * 5, total_tokens=len(inputs) * 5),
    )

    return response


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Functionary Inference Service: 100% OpenAI-compatible vLLM inference RESTful API server incl. tools"
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="host name")
    # parser.add_argument(
    #     "--chat-template", type=str, default=None, help="chat template .jinja"
    # )
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

    args.device = DEVICE
    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allowed_origins,
        allow_credentials=args.allow_credentials,
        allow_methods=args.allowed_methods,
        allow_headers=args.allowed_headers,
    )

    if args.served_model_name is not None:
        served_model = args.served_model_name
    else:
        served_model = args.model
    engine_args = AsyncEngineArgs.from_cli_args(args)
    # A separate tokenizer to map token IDs to strings.
    tokenizer = get_tokenizer(
        engine_args.tokenizer,
        tokenizer_mode=engine_args.tokenizer_mode,
        trust_remote_code=True,
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
