import os
import threading
import time
import uuid
from http import HTTPStatus
from typing import Annotated, Dict, List, Literal, Optional, Union

import modal
from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse, Response
from pydantic_settings import BaseSettings, SettingsConfigDict

from functionary.openai_types import ChatCompletionRequest

app = modal.App("functionary_vllm")
fast_api_app = FastAPI(title="Functionary API")


class Settings(BaseSettings):
    model: str = "meetkai/functionary-small-v2.5"
    max_model_length: int = 8192
    gpu_type: Literal["l4", "t4", "a10g", "a100-40gb", "a100-80gb", "h100"] = "a10g"
    gpu_count: int = 1
    gpu_memory_utilization: float = 0.9
    storage_model_dir: str = "/model"
    execution_timeout: int = 1200  # Generation timeout in seconds
    container_idle_timeout: int = 1200  # In seconds
    batch_size_per_container: int = 10
    enable_grammar_sampling: bool = False

    model_config = SettingsConfigDict(env_prefix="modal_")


settings = Settings()


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
        model=settings.model,
        tensor_parallel_size=settings.gpu_count,
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
        "hf-transfer==0.1.8", "huggingface_hub==0.26.2", "pydantic-settings==2.6.1"
    )
    .pip_install_from_pyproject("pyproject.toml", optional_dependencies=["vllm"])
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_TOKEN": os.environ["HF_TOKEN"],
            "MODAL_MODEL": settings.model,
            "MODAL_MAX_MODEL_LENGTH": str(settings.max_model_length),
            "MODAL_GPU_TYPE": settings.gpu_type,
            "MODAL_GPU_COUNT": str(settings.gpu_count),
            "MODAL_GPU_MEMORY_UTILIZATION": str(settings.gpu_memory_utilization),
            "MODAL_STORAGE_MODEL_DIR": settings.storage_model_dir,
            "MODAL_EXECUTION_TIMEOUT": str(settings.execution_timeout),
            "MODAL_CONTAINER_IDLE_TIMEOUT": str(settings.container_idle_timeout),
            "MODAL_BATCH_SIZE_PER_CONTAINER": str(settings.batch_size_per_container),
            "MODAL_ENABLE_GRAMMAR_SAMPLING": str(
                settings.enable_grammar_sampling
            ).lower(),
        }
    )
)


class AtomicCounter:
    """An atomic, thread-safe counter"""

    def __init__(self, initial=0):
        """Initialize a new atomic counter to given initial value"""
        self._value = initial
        self._lock = threading.Lock()

    def inc(self, num=1):
        """Atomically increment the counter by num and return the new value"""
        with self._lock:
            self._value += num
            return self._value

    def dec(self, num=1):
        """Atomically decrement the counter by num and return the new value"""
        with self._lock:
            self._value -= num
            return self._value

    @property
    def value(self):
        return self._value


@app.cls(
    image=image,
    gpu=f"{settings.gpu_type}:{settings.gpu_count}",
    timeout=settings.execution_timeout,
    container_idle_timeout=settings.container_idle_timeout,
    allow_concurrent_inputs=settings.batch_size_per_container,
)
class Model:
    served_model = []
    served_loras = []
    lora_id_counter = AtomicCounter(0)

    @modal.build()
    def download_model(self):
        from huggingface_hub import snapshot_download
        from transformers.utils import move_cache

        snapshot_download(
            settings.model,
            ignore_patterns=["*.pt", "*.bin"],  # Using safetensors
        )
        move_cache()

    def _start_engine(self):
        model, tokenizer, engine_model_config = get_model()
        self.model = model
        self.tokenizer = tokenizer
        self.engine_model_config = engine_model_config

    @modal.method()
    async def generate(self, request: ChatCompletionRequest):
        from functionary.vllm_inference import process_chat_completion

        # Only start the engine when generate is called
        self._start_engine()

        return await process_chat_completion(
            request=request,
            raw_request=None,
            tokenizer=self.tokenizer,
            served_model=settings.model,
            served_loras=[],
            engine_model_config=self.engine_model_config,
            enable_grammar_sampling=settings.enable_grammar_sampling,
            engine=self.model,
        )

    @modal.method()
    async def modal_load_lora_adapter(self, request_json: dict):
        from vllm.entrypoints.openai.protocol import LoadLoraAdapterRequest

        from functionary.vllm_inference import process_load_lora_adapter

        request = LoadLoraAdapterRequest(**request_json)
        error, self.served_loras = await process_load_lora_adapter(
            request, self.served_loras, self.lora_id_counter
        )
        if not isinstance(error, str):
            return error

        return Response(status_code=200, content=error)

    @modal.method()
    async def modal_unload_lora_adapter(self, request_json: dict):
        from vllm.entrypoints.openai.protocol import UnloadLoraAdapterRequest

        from functionary.vllm_inference import process_unload_lora_adapter

        request = UnloadLoraAdapterRequest(**request_json)
        error, self.served_loras = await process_unload_lora_adapter(
            request, self.served_loras
        )
        if not isinstance(error, str):
            return error

        return Response(status_code=200, content=error)

    @modal.exit()
    def stop_engine(self):
        if settings.gpu_count > 1:
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


@fast_api_app.post("/v1/load_lora_adapter")
async def load_lora_adapter(raw_request: Request):
    """Dynamically load a LoRA adapter into the server."""

    request_json = await raw_request.json()

    model = Model()

    return model.modal_load_lora_adapter.remote(request_json)


@fast_api_app.post("/v1/unload_lora_adapter")
async def unload_lora_adapter(raw_request: Request):
    """Dynamically unload a LoRA adapter from the server."""

    request_json = await raw_request.json()

    model = Model()

    return model.modal_unload_lora_adapter.remote(request_json)


@app.function(image=image)
@modal.asgi_app()
def fastapi_app():
    return fast_api_app
