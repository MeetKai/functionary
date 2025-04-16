# Adapted from
# https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/server.py

# Copyright 2023-2024 SGLang Team

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
import atexit
import dataclasses
import json
import logging
import multiprocessing as mp
import os
import threading
import time
from http import HTTPStatus
from typing import AsyncIterator, Dict, List, Optional, Union, Any, Callable

import orjson

# Fix a bug of Python threading
setattr(threading, "_register_atexit", lambda *args, **kwargs: None)

import sglang as sgl
import uvicorn
import uvloop
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse, Response, StreamingResponse
from uvicorn.config import LOGGING_CONFIG
from sglang.lang.backend.runtime_endpoint import RuntimeEndpoint
from sglang.srt.managers.data_parallel_controller import (
    run_data_parallel_controller_process,
)
from sglang.srt.entrypoints.http_server import (
    _launch_subprocesses,
    set_uvicorn_logging_configs,
    _wait_and_warmup,
    enable_func_timer,
    add_prometheus_middleware,
    lifespan,
)
from sglang.srt.managers.detokenizer_manager import run_detokenizer_process
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.scheduler import run_scheduler_process
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.openai_api.adapter import load_chat_template_for_openai_api
from sglang.srt.openai_api.protocol import ModelCard, ModelList
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import (
    add_api_key_middleware,
    configure_logger,
    is_port_available,
    prepare_model_and_tokenizer,
)
from sglang.srt.utils import kill_process_tree
from functionary.sglang_inference import v1_chat_completions

logger = logging.getLogger(__name__)

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())


app = FastAPI(lifespan=lifespan)
tokenizer_manager = None
served_model = []

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> Response:
    """Check the health of the http server."""
    return Response(status_code=200)


@app.get("/health_generate")
async def health_generate(request: Request) -> Response:
    """Check the health of the inference server by generating one token."""
    gri = GenerateReqInput(
        text="s", sampling_params={"max_new_tokens": 1, "temperature": 0.7}
    )
    try:
        async for _ in tokenizer_manager.generate_request(gri, request):
            break
        return Response(status_code=200)
    except Exception as e:
        logger.exception(e)
        return Response(status_code=503)


@app.get("/get_model_info")
async def get_model_info():
    """Get the model information."""
    result = {
        "model_path": tokenizer_manager.model_path,
        "is_generation": tokenizer_manager.is_generation,
    }
    return result


@app.get("/get_server_args")
async def get_server_args():
    """Get the server arguments."""
    return dataclasses.asdict(tokenizer_manager.server_args)


@app.get("/flush_cache")
async def flush_cache():
    """Flush the radix cache."""
    tokenizer_manager.flush_cache()
    return Response(
        content="Cache flushed.\nPlease check backend logs for more details. "
        "(When there are running or waiting requests, the operation will not be performed.)\n",
        status_code=200,
    )


# fastapi implicitly converts json in the request to obj (dataclass)
async def generate_request(obj: GenerateReqInput, request: Request):
    """Handle a generate request."""
    if obj.stream:

        async def stream_results() -> AsyncIterator[bytes]:
            try:
                async for out in tokenizer_manager.generate_request(obj, request):
                    yield b"data: " + orjson.dumps(
                        out, option=orjson.OPT_NON_STR_KEYS
                    ) + b"\n\n"
            except ValueError as e:
                out = {"error": {"message": str(e)}}
                yield b"data: " + orjson.dumps(
                    out, option=orjson.OPT_NON_STR_KEYS
                ) + b"\n\n"
            yield b"data: [DONE]\n\n"

        return StreamingResponse(
            stream_results(),
            media_type="text/event-stream",
            background=tokenizer_manager.create_abort_task(obj),
        )
    else:
        try:
            ret = await tokenizer_manager.generate_request(obj, request).__anext__()
            return ret
        except ValueError as e:
            return ORJSONResponse(
                {"error": {"message": str(e)}}, status_code=HTTPStatus.BAD_REQUEST
            )


app.post("/generate")(generate_request)
app.put("/generate")(generate_request)


@app.post("/v1/chat/completions")
async def openai_v1_chat_completions(raw_request: Request):
    return await v1_chat_completions(tokenizer_manager, None, raw_request, served_model)


@app.get("/v1/models", response_class=ORJSONResponse)
def available_models():
    """Show available models."""
    model_cards = []
    if isinstance(served_model, list):
        for model in served_model:
            model_cards.append(ModelCard(id=model, root=model))
    else:
        model_cards.append(ModelCard(id=served_model, root=served_model))
    return ModelList(data=model_cards)


def launch_server(
    server_args: ServerArgs,
    pipe_finish_writer: Optional[mp.connection.Connection] = None,
    launch_callback: Optional[Callable[[], None]] = None,
):
    """
    Launch SRT (SGLang Runtime) Server.

    The SRT server consists of an HTTP server and an SRT engine.

    - HTTP server: A FastAPI server that routes requests to the engine.
    - The engine consists of three components:
        1. TokenizerManager: Tokenizes the requests and sends them to the scheduler.
        2. Scheduler (subprocess): Receives requests from the Tokenizer Manager, schedules batches, forwards them, and sends the output tokens to the Detokenizer Manager.
        3. DetokenizerManager (subprocess): Detokenizes the output tokens and sends the result back to the Tokenizer Manager.

    Note:
    1. The HTTP server, Engine, and TokenizerManager both run in the main process.
    2. Inter-process communication is done through IPC (each process uses a different port) via the ZMQ library.
    """
    global tokenizer_manager, scheduler_info
    tokenizer_manager, scheduler_info = _launch_subprocesses(server_args=server_args)

    # Add api key authorization
    if server_args.api_key:
        add_api_key_middleware(app, server_args.api_key)

    # Add prometheus middleware
    if server_args.enable_metrics:
        add_prometheus_middleware(app)
        enable_func_timer()

    # Send a warmup request - we will create the thread launch it
    # in the lifespan after all other warmups have fired.
    warmup_thread = threading.Thread(
        target=_wait_and_warmup,
        args=(
            server_args,
            pipe_finish_writer,
            "",
            launch_callback,
        ),
    )
    app.warmup_thread = warmup_thread

    try:
        # Update logging configs
        set_uvicorn_logging_configs()
        app.server_args = server_args
        # Listen for HTTP requests
        uvicorn.run(
            app,
            host=server_args.host,
            port=server_args.port,
            log_level=server_args.log_level_http or server_args.log_level,
            timeout_keep_alive=5,
            loop="uvloop",
        )
    finally:
        warmup_thread.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logfile",
        type=str,
        default=None,
        help="enable detailed request input/output logging by providing logfile",
    )
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args()

    served_model = [args.model_path]
    if args.served_model_name is not None:
        served_model += args.served_model_name

    server_args = ServerArgs.from_cli_args(args)

    try:
        launch_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
