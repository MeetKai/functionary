import os
import time
import uuid
from http import HTTPStatus
from typing import Annotated, Dict, List, Literal, Optional, Union

import modal
from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse

from functionary.openai_types import ChatCompletionRequest
from modal_server_config import Settings

app = modal.App("functionary_vllm")
fast_api_app = FastAPI(title="Functionary API")
settings = Settings()


def download_model_to_image(model_dir, model_name):
    from huggingface_hub import snapshot_download
    from transformers.utils import move_cache

    os.makedirs(model_dir, exist_ok=True)

    snapshot_download(
        model_name,
        local_dir=model_dir,
        ignore_patterns=["*.pt", "*.bin"],  # Using safetensors
    )
    move_cache()


def get_model():
    # this is lazy should be using the modal model class
    import asyncio

    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.transformers_utils.tokenizer import get_tokenizer

    if settings.enable_grammar_sampling:
        from functionary.vllm_monkey_patch.async_llm_engine import AsyncLLMEngine
    else:
        from vllm.engine.async_llm_engine import AsyncLLMEngine

    engine_args = AsyncEngineArgs(
        model=settings.storage_model_dir,
        tensor_parallel_size=settings.gpu_config.count,
        gpu_memory_utilization=settings.gpu_memory_utilization,
        enforce_eager=False,  # capture the graph for faster inference, but slower cold starts
        disable_log_stats=True,  # disable logging so we can stream tokens
        disable_log_requests=True,
        max_model_len=settings.max_model_length,
    )

    # A separate tokenizer to map token IDs to strings.
    tokenizer = get_tokenizer(
        engine_args.tokenizer, tokenizer_mode=engine_args.tokenizer_mode
    )

    # Overwrite vLLM's default ModelConfig.max_logprobs of 5
    engine_args.max_logprobs = len(tokenizer.vocab.keys())

    engine = AsyncLLMEngine.from_engine_args(engine_args)
    engine_model_config = asyncio.run(engine.get_model_config())

    return engine, tokenizer, engine_model_config


image = (
    modal.Image.debian_slim()
    .pip_install(
        "hf-transfer==0.1.6", "huggingface_hub==0.22.2", "pydantic-settings==2.2.1"
    )
    .pip_install_from_requirements("requirements.txt")
    .run_function(
        download_model_to_image,
        timeout=60 * 20,
        kwargs={"model_dir": settings.storage_model_dir, "model_name": settings.model},
    )
)


@app.cls(
    gpu=settings.gpu_config,
    timeout=settings.execution_timeout,
    container_idle_timeout=settings.container_idle_timeout,
    allow_concurrent_inputs=settings.batch_size_per_container,
    image=image,
)
class Model:
    @modal.enter()
    def start_engine(self):
        model, tokenizer, engine_model_config = get_model()
        self.model = model
        self.tokenizer = tokenizer
        self.engine_model_config = engine_model_config

    @modal.method()
    async def generate(self, request: ChatCompletionRequest):
        from functionary.vllm_inference import process_chat_completion

        return await process_chat_completion(
            request=request,
            raw_request=None,
            tokenizer=self.tokenizer,
            served_model=settings.model,
            engine_model_config=self.engine_model_config,
            enable_grammar_sampling=settings.enable_grammar_sampling,
            engine=self.model,
        )

    @modal.exit()
    def stop_engine(self):
        if settings.gpu_config.count > 1:
            import ray

            ray.shutdown()


@fast_api_app.post("/v1/chat/completions")
async def chat_endpoint(raw_request: Request):
    """Completion API similar to OpenAI's API.
    See  https://platform.openai.com/docs/api-reference/chat/create
    for the API specification. This API mimics the OpenAI ChatCompletion API.
    NOTE: Currently we do not support the following features:
        - logit_bias (to be supported by vLLM engine)
    """

    request_json = await raw_request.json()
    request = ChatCompletionRequest(**request_json)

    model = Model()

    return model.generate.remote(request)


@app.function(image=image)
@modal.asgi_app()
def fastapi_app():
    return fast_api_app
