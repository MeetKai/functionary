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
from typing import AsyncIterator, Dict, List, Optional, Union

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
from sglang.srt.managers.detokenizer_manager import run_detokenizer_process
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.scheduler import run_scheduler_process
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.openai_api.adapter import load_chat_template_for_openai_api
from sglang.srt.openai_api.protocol import ModelCard, ModelList
from sglang.srt.server import Runtime, _set_envs_and_config, _wait_and_warmup
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import (
    add_api_key_middleware,
    configure_logger,
    is_port_available,
    prepare_model_and_tokenizer,
)

from functionary.sglang_inference import v1_chat_completions

logger = logging.getLogger(__name__)

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())


app = FastAPI()
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
    global tokenizer_manager, backend

    # if not args.grammar_sampling:
    #     backend = None
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


def launch_engine(server_args: ServerArgs):
    """
    Launch the Tokenizer Manager in the main process, the Scheduler in a subprocess, and the Detokenizer Manager in another subprocess.
    """

    global tokenizer_manager

    # Configure global environment
    configure_logger(server_args)
    server_args.check_server_args()
    _set_envs_and_config(server_args)

    # Allocate ports for inter-process communications
    port_args = PortArgs.init_new(server_args)
    logger.info(f"{server_args=}")

    # If using model from www.modelscope.cn, first download the model.
    server_args.model_path, server_args.tokenizer_path = prepare_model_and_tokenizer(
        server_args.model_path, server_args.tokenizer_path
    )

    if server_args.dp_size == 1:
        # Launch tensor parallel scheduler processes
        scheduler_procs = []
        scheduler_pipe_readers = []
        tp_size_per_node = server_args.tp_size // server_args.nnodes
        tp_rank_range = range(
            tp_size_per_node * server_args.node_rank,
            tp_size_per_node * (server_args.node_rank + 1),
        )
        for tp_rank in tp_rank_range:
            reader, writer = mp.Pipe(duplex=False)
            gpu_id = tp_rank % tp_size_per_node
            proc = mp.Process(
                target=run_scheduler_process,
                args=(server_args, port_args, gpu_id, tp_rank, None, writer),
            )
            proc.start()
            scheduler_procs.append(proc)
            scheduler_pipe_readers.append(reader)

        if server_args.node_rank >= 1:
            # For other nodes, they do not need to run tokenizer or detokenizer,
            # so they can just wait here.
            while True:
                pass
    else:
        # Launch the data parallel controller
        reader, writer = mp.Pipe(duplex=False)
        scheduler_pipe_readers = [reader]
        proc = mp.Process(
            target=run_data_parallel_controller_process,
            args=(server_args, port_args, writer),
        )
        proc.start()

    # Launch detokenizer process
    detoken_proc = mp.Process(
        target=run_detokenizer_process,
        args=(
            server_args,
            port_args,
        ),
    )
    detoken_proc.start()

    # Launch tokenizer process
    if args.logfile is not None:
        from functionary.sglang_monkey_patch.tokenizer_manager import (
            MonkeyPatchTokenizerManager,
        )

        tokenizer_manager = MonkeyPatchTokenizerManager(
            server_args, port_args, args.logfile
        )
    else:
        tokenizer_manager = TokenizerManager(server_args, port_args)
    if server_args.chat_template:
        load_chat_template_for_openai_api(tokenizer_manager, server_args.chat_template)

    # Wait for model to finish loading
    for i in range(len(scheduler_pipe_readers)):
        scheduler_pipe_readers[i].recv()


def launch_server(
    server_args: ServerArgs,
    pipe_finish_writer: Optional[mp.connection.Connection] = None,
):
    """
    Launch SRT (SGLang Runtime) Server

    The SRT server consists of an HTTP server and the SRT engine.

    1. HTTP server: A FastAPI server that routes requests to the engine.
    2. SRT engine:
        1. Tokenizer Manager: Tokenizes the requests and sends them to the scheduler.
        2. Scheduler (subprocess): Receives requests from the Tokenizer Manager, schedules batches, forwards them, and sends the output tokens to the Detokenizer Manager.
        3. Detokenizer Manager (subprocess): Detokenizes the output tokens and sends the result back to the Tokenizer Manager.

    Note:
    1. The HTTP server and Tokenizer Manager both run in the main process.
    2. Inter-process communication is done through ICP (each process uses a different port) via the ZMQ library.
    """

    launch_engine(server_args=server_args)

    # Add api key authorization
    if server_args.api_key:
        add_api_key_middleware(app, server_args.api_key)

    # Send a warmup request
    t = threading.Thread(
        target=_wait_and_warmup, args=(server_args, pipe_finish_writer, os.getpid())
    )
    t.start()

    try:
        # Listen for HTTP requests
        LOGGING_CONFIG["formatters"]["default"][
            "fmt"
        ] = "[%(asctime)s] %(levelprefix)s %(message)s"
        LOGGING_CONFIG["formatters"]["default"]["datefmt"] = "%Y-%m-%d %H:%M:%S"
        LOGGING_CONFIG["formatters"]["access"][
            "fmt"
        ] = '[%(asctime)s] %(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s'
        LOGGING_CONFIG["formatters"]["access"]["datefmt"] = "%Y-%m-%d %H:%M:%S"
        uvicorn.run(
            app,
            host=server_args.host,
            port=server_args.port,
            log_level=server_args.log_level_http or server_args.log_level,
            timeout_keep_alive=5,
            loop="uvloop",
        )
    finally:
        t.join()


class FunctionaryRuntime(Runtime):
    """
    A wrapper for the server.
    This is used for launching the server in a python program without
    using the commond line interface.
    """

    def __init__(
        self,
        log_level: str = "error",
        *args,
        **kwargs,
    ):
        """See the arguments in server_args.py::ServerArgs"""
        self.server_args = ServerArgs(*args, log_level=log_level, **kwargs)

        # before python program terminates, call shutdown implicitly. Therefore, users don't have to explicitly call .shutdown()
        atexit.register(self.shutdown)

        # Pre-allocate ports
        for port in range(10000, 40000):
            if is_port_available(port):
                break
            port += 1
        self.server_args.port = port

        self.url = self.server_args.url()
        self.generate_url = self.url + "/generate"

        # NOTE: We store pid instead of proc to fix some issues during __delete__
        self.pid = None
        pipe_reader, pipe_writer = mp.Pipe(duplex=False)

        proc = mp.Process(
            target=launch_server,
            args=(self.server_args, pipe_writer),
        )
        proc.start()
        pipe_writer.close()
        self.pid = proc.pid

        try:
            init_state = pipe_reader.recv()
        except EOFError:
            init_state = ""

        if init_state != "ready":
            self.shutdown()
            raise RuntimeError(
                "Initialization failed. Please see the error messages above."
            )

        self.endpoint = RuntimeEndpoint(self.url)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logfile",
        type=str,
        default=None,
        help="enable detailed request input/output logging by providing logfile",
    )
    # parser.add_argument(
    #     "--enable-grammar-sampling",
    #     dest="grammar_sampling",
    #     action="store_true",
    #     default=False,
    #     help="enable grammar sampling for function names",
    # )
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args()

    served_model = [args.model_path]
    if args.served_model_name is not None:
        served_model += args.served_model_name

    server_args = ServerArgs.from_cli_args(args)

    launch_server(server_args)

    # if args.grammar_sampling:
    #     backend = FunctionaryRuntime(**vars(server_args))
    #     sgl.set_default_backend(
    #         sgl.RuntimeEndpoint(
    #             f"http://{backend.server_args.host}:{backend.server_args.port}"
    #         )
    #     )
    #     uvicorn.run(
    #         app,
    #         host=server_args.host,
    #         port=server_args.port,
    #         log_level=server_args.log_level_http or server_args.log_level,
    #         timeout_keep_alive=5,
    #         loop="uvloop",
    #     )
    # else:
    #     launch_server(server_args)
