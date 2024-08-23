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

"""TokenizerManager is a process that tokenizes the text."""

import asyncio
import dataclasses
import logging
from typing import Dict, List, Tuple, Union

import uvloop
from sglang.srt.managers.io_struct import EmbeddingReqInput, GenerateReqInput
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.server_args import PortArgs, ServerArgs

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

logger = logging.getLogger("tokenizer_logger")


@dataclasses.dataclass
class ReqState:
    """Store the state a request."""

    out_list: List
    finished: bool
    event: asyncio.Event


class MonkeyPatchTokenizerManager(TokenizerManager):
    """TokenizerManager is a process that tokenizes the text."""

    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        model_overide_args: Dict = None,
        logfile: str = "logfile.txt",
    ):
        super().__init__(server_args, port_args, model_overide_args)
        file_handler = logging.handlers.RotatingFileHandler(
            logfile, maxBytes=1024 * 1024 * 100, backupCount=10
        )
        logger.addHandler(file_handler)

    async def _wait_for_response(
        self,
        event: asyncio.Event,
        state: ReqState,
        obj: Union[GenerateReqInput, EmbeddingReqInput],
        rid: str,
        request,
        index: int = None,
        response_index: int = 0,
    ):
        while True:
            try:
                await asyncio.wait_for(event.wait(), timeout=4)
            except asyncio.TimeoutError:
                if request is not None and await request.is_disconnected():
                    for rid in [obj.rid] if obj.is_single else obj.rid:
                        self.abort_request(rid)
                    raise ValueError(f"Abort request {rid}")
                continue

            if self.is_generation:
                out = self.convert_logprob_style(
                    state.out_list[-1],
                    obj.return_logprob if index is None else obj.return_logprob[index],
                    (
                        obj.top_logprobs_num
                        if index is None
                        else obj.top_logprobs_num[index]
                    ),
                    obj.return_text_in_logprobs,
                )
            else:  # isinstance(obj, EmbeddingReqInput)
                out = state.out_list[-1]

            out["index"] = response_index

            # Log requests
            # if self.server_args.log_requests and state.finished:
            if state.finished:
                if obj.text is None and obj.input_ids is not None:
                    obj.text = self.tokenizer.decode(obj.input_ids)
                    obj.input_ids = None
                logger.info(dict(input=obj.__dict__, output=out))

            state.out_list = []
            if state.finished:
                del self.rid_to_state[rid]
                yield out
                break

            event.clear()
            yield out
