"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
The entry point of inference server.
SRT = SGLang Runtime.
"""

import argparse
import asyncio
import dataclasses
import json
import logging
import multiprocessing as mp
import os
import sys
import threading
import time
from http import HTTPStatus
from typing import Dict, List, Optional, Union

# Fix a bug of Python threading
setattr(threading, "_register_atexit", lambda *args, **kwargs: None)

import requests
import uvicorn
import uvloop
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse
from sglang.srt.constrained import disable_cache
from sglang.srt.hf_transformers_utils import get_tokenizer
from sglang.srt.managers.controller_multi import (
    start_controller_process as start_controller_process_multi,
)
from sglang.srt.managers.controller_single import launch_tp_servers
from sglang.srt.managers.controller_single import (
    start_controller_process as start_controller_process_single,
)
from sglang.srt.managers.detokenizer_manager import start_detokenizer_process
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.openai_api.adapter import (
    load_chat_template_for_openai_api,
    v1_batches,
    v1_delete_file,
    v1_files_create,
    v1_retrieve_batch,
    v1_retrieve_file,
    v1_retrieve_file_content,
)
from sglang.srt.openai_api.protocol import ModelCard, ModelList
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import (
    add_api_key_middleware,
    allocate_init_ports,
    assert_pkg_version,
    enable_show_time_cost,
    maybe_set_triton_cache_manager,
    prepare_model,
    prepare_tokenizer,
    set_ulimit,
)
from sglang.utils import get_exception_traceback

from functionary.openai_types import ChatCompletionRequest
from functionary.sglang_inference import v1_chat_completions

logger = logging.getLogger(__name__)

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())


app = FastAPI()
tokenizer_manager = None


@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)


@app.get("/get_model_info")
async def get_model_info():
    result = {
        "model_path": tokenizer_manager.model_path,
        "is_generation": tokenizer_manager.is_generation,
    }
    return result


@app.get("/get_server_args")
async def get_server_args():
    return dataclasses.asdict(tokenizer_manager.server_args)


@app.get("/flush_cache")
async def flush_cache():
    tokenizer_manager.flush_cache()
    return Response(
        content="Cache flushed.\nPlease check backend logs for more details. "
        "(When there are running or waiting requests, the operation will not be performed.)\n",
        status_code=200,
    )


async def generate_request(obj: GenerateReqInput, request: Request):
    """Handle a generate request."""
    if obj.stream:

        async def stream_results():
            try:
                async for out in tokenizer_manager.generate_request(obj, request):
                    yield f"data: {json.dumps(out, ensure_ascii=False)}\n\n"
            except ValueError as e:
                out = {"error": {"message": str(e)}}
                yield f"data: {json.dumps(out, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

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
            return JSONResponse(
                {"error": {"message": str(e)}}, status_code=HTTPStatus.BAD_REQUEST
            )


app.post("/generate")(generate_request)
app.put("/generate")(generate_request)


@app.post("/v1/chat/completions")
async def openai_v1_chat_completions(raw_request: Request):
    if args.logfile is not None:
        request_json = await raw_request.json()
        logger.info(ChatCompletionRequest(**request_json).model_dump(mode="json"))
    return await v1_chat_completions(tokenizer_manager, raw_request)


@app.get("/v1/models")
def available_models():
    """Show available models."""
    served_model_names = [tokenizer_manager.served_model_name]
    model_cards = []
    for served_model_name in served_model_names:
        model_cards.append(ModelCard(id=served_model_name, root=served_model_name))
    return ModelList(data=model_cards)


@app.post("/v1/files")
async def openai_v1_files(file: UploadFile = File(...), purpose: str = Form("batch")):
    return await v1_files_create(
        file, purpose, tokenizer_manager.server_args.file_storage_pth
    )


@app.delete("/v1/files/{file_id}")
async def delete_file(file_id: str):
    # https://platform.openai.com/docs/api-reference/files/delete
    return await v1_delete_file(file_id)


@app.post("/v1/batches")
async def openai_v1_batches(raw_request: Request):
    return await v1_batches(tokenizer_manager, raw_request)


@app.get("/v1/batches/{batch_id}")
async def retrieve_batch(batch_id: str):
    return await v1_retrieve_batch(batch_id)


@app.get("/v1/files/{file_id}")
async def retrieve_file(file_id: str):
    # https://platform.openai.com/docs/api-reference/files/retrieve
    return await v1_retrieve_file(file_id)


@app.get("/v1/files/{file_id}/content")
async def retrieve_file_content(file_id: str):
    # https://platform.openai.com/docs/api-reference/files/retrieve-contents
    return await v1_retrieve_file_content(file_id)


def launch_server(
    server_args: ServerArgs,
    model_overide_args: Optional[dict] = None,
    pipe_finish_writer: Optional[mp.connection.Connection] = None,
):
    """Launch an HTTP server."""
    global tokenizer_manager

    logging.basicConfig(
        filename=args.logfile,
        filemode="a",
        level=getattr(logging, server_args.log_level.upper()),
        format="%(message)s",
    )

    server_args.check_server_args()
    _set_envs_and_config(server_args)

    # Allocate ports
    server_args.port, server_args.additional_ports = allocate_init_ports(
        server_args.port,
        server_args.additional_ports,
        server_args.dp_size,
    )
    ports = server_args.additional_ports
    port_args = PortArgs(
        tokenizer_port=ports[0],
        controller_port=ports[1],
        detokenizer_port=ports[2],
        nccl_ports=ports[3:],
    )
    logger.info(f"{server_args=}")

    # Use model from www.modelscope.cn, first download the model.
    server_args.model_path = prepare_model(server_args.model_path)
    server_args.tokenizer_path = prepare_tokenizer(server_args.tokenizer_path)

    # Launch processes for multi-node tensor parallelism
    if server_args.nnodes > 1:
        if server_args.node_rank != 0:
            tp_size_local = server_args.tp_size // server_args.nnodes
            gpu_ids = [
                i for _ in range(server_args.nnodes) for i in range(tp_size_local)
            ]
            tp_rank_range = list(
                range(
                    server_args.node_rank * tp_size_local,
                    (server_args.node_rank + 1) * tp_size_local,
                )
            )
            procs = launch_tp_servers(
                gpu_ids,
                tp_rank_range,
                server_args,
                ports[3],
                model_overide_args,
            )
            while True:
                pass

    # Launch processes
    tokenizer_manager = TokenizerManager(server_args, port_args, model_overide_args)
    if server_args.chat_template:
        load_chat_template_for_openai_api(tokenizer_manager, server_args.chat_template)
    pipe_controller_reader, pipe_controller_writer = mp.Pipe(duplex=False)
    pipe_detoken_reader, pipe_detoken_writer = mp.Pipe(duplex=False)

    if server_args.dp_size == 1:
        start_process = start_controller_process_single
    else:
        start_process = start_controller_process_multi
    proc_controller = mp.Process(
        target=start_process,
        args=(server_args, port_args, pipe_controller_writer, model_overide_args),
    )
    proc_controller.start()
    proc_detoken = mp.Process(
        target=start_detokenizer_process,
        args=(
            server_args,
            port_args,
            pipe_detoken_writer,
        ),
    )
    proc_detoken.start()

    # Wait for the model to finish loading
    controller_init_state = pipe_controller_reader.recv()
    detoken_init_state = pipe_detoken_reader.recv()

    if controller_init_state != "init ok" or detoken_init_state != "init ok":
        proc_controller.kill()
        proc_detoken.kill()
        print(
            f"Initialization failed. controller_init_state: {controller_init_state}",
            flush=True,
        )
        print(
            f"Initialization failed. detoken_init_state: {detoken_init_state}",
            flush=True,
        )
        sys.exit(1)
    assert proc_controller.is_alive() and proc_detoken.is_alive()

    # Add api key authorization
    if server_args.api_key:
        add_api_key_middleware(app, server_args.api_key)

    # Send a warmup request
    t = threading.Thread(
        target=_wait_and_warmup, args=(server_args, pipe_finish_writer)
    )
    t.start()

    # Listen for requests
    try:
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


def _set_envs_and_config(server_args: ServerArgs):
    # Set global environments
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["NCCL_CUMEM_ENABLE"] = "0"
    os.environ["NCCL_NVLS_ENABLE"] = "0"
    os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"

    # Set ulimit
    set_ulimit()

    # Enable show time cost for debugging
    if server_args.show_time_cost:
        enable_show_time_cost()

    # Disable disk cache
    if server_args.disable_disk_cache:
        disable_cache()

    # Fix triton bugs
    if server_args.tp_size * server_args.dp_size > 1:
        # FIXME: remove this after https://github.com/triton-lang/triton/pull/4295 is used as a dependency.
        maybe_set_triton_cache_manager()

    # Check flashinfer version
    if not server_args.disable_flashinfer:
        assert_pkg_version(
            "flashinfer",
            "0.1.5",
            "Please uninstall the old version and "
            "reinstall the latest version by following the instructions "
            "at https://docs.flashinfer.ai/installation.html.",
        )


def _wait_and_warmup(server_args, pipe_finish_writer):
    headers = {}
    url = server_args.url()
    if server_args.api_key:
        headers["Authorization"] = f"Bearer {server_args.api_key}"

    # Wait until the server is launched
    success = False
    for _ in range(120):
        time.sleep(1)
        try:
            res = requests.get(url + "/get_model_info", timeout=5, headers=headers)
            assert res.status_code == 200, f"{res}"
            success = True
            break
        except (AssertionError, requests.exceptions.RequestException) as e:
            last_traceback = get_exception_traceback()
            pass
    model_info = res.json()

    if not success:
        if pipe_finish_writer is not None:
            pipe_finish_writer.send(last_traceback)
        print(f"Initialization failed. warmup error: {last_traceback}", flush=True)
        sys.exit(1)

    # Send a warmup request
    request_name = "/generate" if model_info["is_generation"] else "/encode"
    max_new_tokens = 8 if model_info["is_generation"] else 1
    json_data = {
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": max_new_tokens,
        },
    }
    if server_args.skip_tokenizer_init:
        json_data["input_ids"] = [10, 11, 12]
    else:
        json_data["text"] = "The capital city of France is"

    try:
        for _ in range(server_args.dp_size):
            res = requests.post(
                url + request_name,
                json=json_data,
                headers=headers,
                timeout=600,
            )
            assert res.status_code == 200, f"{res}"
    except Exception as e:
        last_traceback = get_exception_traceback()
        if pipe_finish_writer is not None:
            pipe_finish_writer.send(last_traceback)
        print(f"Initialization failed. warmup error: {last_traceback}", flush=True)
        sys.exit(1)

    # Print warnings here
    if server_args.disable_radix_cache and server_args.chunked_prefill_size is not None:
        logger.warning(
            "You set both `--disable-radix-cache` and `--chunked-prefill-size`. "
            "This combination is an experimental feature and we noticed it can lead to "
            "wrong generation results. If you want to use chunked prefill, it is recommended "
            "not using `--disable-radix-cache`."
        )

    logger.info("The server is fired up and ready to roll!")
    if pipe_finish_writer is not None:
        pipe_finish_writer.send("init ok")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logfile", type=str, default=None, help="name of the file to log requests"
    )
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args()
    server_args = ServerArgs.from_cli_args(args)

    launch_server(server_args)
