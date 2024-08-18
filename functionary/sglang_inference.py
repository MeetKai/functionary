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

"""Conversion between OpenAI APIs and native SRT APIs"""

import asyncio
import json
import os
import time
import uuid
from http import HTTPStatus
from typing import Dict, List, Optional

from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.openai_api.protocol import (
    BatchResponse,
    ChatCompletionTokenLogprob,
    ChoiceLogprobs,
    DeltaMessage,
    ErrorResponse,
    FileResponse,
    LogProbs,
    TopLogprob,
)

from functionary.inference_stream import generate_openai_format_from_stream_async
from functionary.inference_utils import analyze_tools_and_tool_choice
from functionary.openai_types import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
    StreamChoice,
    UsageInfo,
)
from functionary.prompt_template import get_prompt_template_from_tokenizer
from functionary.prompt_template.prompt_utils import prepare_messages_for_inference


class FileMetadata:
    def __init__(self, filename: str, purpose: str):
        self.filename = filename
        self.purpose = purpose


# In-memory storage for batch jobs and files
batch_storage: Dict[str, BatchResponse] = {}
file_id_request: Dict[str, FileMetadata] = {}
file_id_response: Dict[str, FileResponse] = {}
# map file id to file path in SGLang backend
file_id_storage: Dict[str, str] = {}


# backend storage directory
storage_dir = None


def format_finish_reason(finish_reason) -> Optional[str]:
    if finish_reason.startswith("None"):
        return None
    elif finish_reason.startswith("FINISH_MATCHED"):
        return "stop"
    elif finish_reason.startswith("FINISH_LENGTH"):
        return "length"
    elif finish_reason.startswith("FINISH_ABORT"):
        return "abort"
    else:
        return "unknown"


def create_error_response(
    message: str,
    err_type: str = "BadRequestError",
    status_code: HTTPStatus = HTTPStatus.BAD_REQUEST,
):
    error = ErrorResponse(message=message, type=err_type, code=status_code.value)
    return JSONResponse(content=error.model_dump(), status_code=error.code)


def create_streaming_error_response(
    message: str,
    err_type: str = "BadRequestError",
    status_code: HTTPStatus = HTTPStatus.BAD_REQUEST,
) -> str:
    error = ErrorResponse(message=message, type=err_type, code=status_code.value)
    json_str = json.dumps({"error": error.model_dump()})
    return json_str


def v1_chat_generate_request(all_requests, tokenizer_manager):
    input_ids = []
    sampling_params_list = []
    image_data_list = []
    return_logprobs = []
    top_logprobs_nums = []
    for request in all_requests:
        # Prep the data needed for the underlying GenerateReqInput:
        #  - prompt: The full prompt string.
        #  - stop: Custom stop tokens.
        #  - image_data: None or a list of image strings (URLs or base64 strings).
        #    None skips any image processing in GenerateReqInput.
        tools_or_functions, tool_func_choice = analyze_tools_and_tool_choice(
            request=request
        )
        if not isinstance(request.messages, str):
            # Apply chat template and its stop strings.
            prompt_ids = prepare_messages_for_inference(
                tokenizer=tokenizer_manager.tokenizer,
                messages=request.messages,
                tools_or_functions=tools_or_functions,
                tool_choice=tool_func_choice,
                device="cpu",
            ).tolist()[0]
            stop = request.stop
            image_data = None
        else:
            # Use the raw prompt and stop strings if the messages is already a string.
            prompt_ids = request.messages
            stop = request.stop
            image_data = None
        input_ids.append(prompt_ids)
        return_logprobs.append(request.logprobs)
        top_logprobs_nums.append(request.top_logprobs)
        sampling_params_list.append(
            {
                "temperature": request.temperature,
                "max_new_tokens": request.max_tokens,
                "min_new_tokens": request.min_tokens,
                "stop": stop,
                "stop_token_ids": request.stop_token_ids,
                "top_p": request.top_p,
                "presence_penalty": request.presence_penalty,
                "frequency_penalty": request.frequency_penalty,
                "repetition_penalty": request.repetition_penalty,
                "regex": request.regex,
                "n": request.n,
            }
        )
        image_data_list.append(image_data)
    if len(all_requests) == 1:
        input_ids = input_ids[0]
        if isinstance(input_ids, str):
            prompt_kwargs = {"text": input_ids}
        else:
            prompt_kwargs = {"input_ids": input_ids}
        sampling_params_list = sampling_params_list[0]
        image_data = image_data_list[0]
        return_logprobs = return_logprobs[0]
        top_logprobs_nums = top_logprobs_nums[0]
    else:
        if isinstance(input_ids[0], str):
            prompt_kwargs = {"text": input_ids}
        else:
            prompt_kwargs = {"input_ids": input_ids}
    adapted_request = GenerateReqInput(
        **prompt_kwargs,
        image_data=image_data,
        sampling_params=sampling_params_list,
        return_logprob=return_logprobs,
        top_logprobs_num=top_logprobs_nums,
        stream=all_requests[0].stream,
        return_text_in_logprobs=True,
    )
    if len(all_requests) == 1:
        return adapted_request, all_requests[0]
    return adapted_request, all_requests


def v1_chat_generate_response(request, prompt_template, ret):
    choices = []

    _, tool_func_choice = analyze_tools_and_tool_choice(request=request)

    for idx, ret_item in enumerate(ret):
        logprobs = False
        if isinstance(request, list) and request[idx].logprobs:
            logprobs = True
        elif (not isinstance(request, list)) and request.logprobs:
            logprobs = True
        if logprobs:
            logprobs = to_openai_style_logprobs(
                output_token_logprobs=ret_item["meta_info"]["output_token_logprobs"],
                output_top_logprobs=ret_item["meta_info"]["output_top_logprobs"],
            )
            token_logprobs = []
            for token, logprob in zip(logprobs.tokens, logprobs.token_logprobs):
                token_bytes = list(token.encode("utf-8"))
                top_logprobs = []
                if logprobs.top_logprobs:
                    for top_token, top_logprob in logprobs.top_logprobs[0].items():
                        top_token_bytes = list(top_token.encode("utf-8"))
                        top_logprobs.append(
                            TopLogprob(
                                token=top_token,
                                bytes=top_token_bytes,
                                logprob=top_logprob,
                            )
                        )
                token_logprobs.append(
                    ChatCompletionTokenLogprob(
                        token=token,
                        bytes=token_bytes,
                        logprob=logprob,
                        top_logprobs=top_logprobs,
                    )
                )

            choice_logprobs = ChoiceLogprobs(content=token_logprobs)
        else:
            choice_logprobs = None

        chat_mess = prompt_template.parse_assistant_response(
            llm_output=ret_item["text"], tool_choice=tool_func_choice
        )
        finish_reason = False

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
        if "function_call" in chat_mess and chat_mess["function_call"]:
            finish_reason = "function_call"

        if "tool_calls" in chat_mess and chat_mess["tool_calls"]:
            finish_reason = "tool_calls"

        if not finish_reason:
            finish_reason = format_finish_reason(ret_item["meta_info"]["finish_reason"])

        choice_data = ChatCompletionResponseChoice(
            index=idx,
            message=ChatMessage(**chat_mess),
            # logprobs=choice_logprobs,
            finish_reason=finish_reason,
        )

        choices.append(choice_data)

    prompt_tokens = sum(
        ret[i]["meta_info"]["prompt_tokens"] for i in range(0, len(ret), request.n)
    )
    completion_tokens = sum(item["meta_info"]["completion_tokens"] for item in ret)
    response = ChatCompletionResponse(
        id=ret[0]["meta_info"]["id"],
        model=request.model,
        choices=choices,
        usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )
    return response


async def v1_chat_completions(tokenizer_manager, raw_request: Request):
    request_json = await raw_request.json()
    all_requests = [ChatCompletionRequest(**request_json)]

    prompt_template = get_prompt_template_from_tokenizer(
        tokenizer=tokenizer_manager.tokenizer
    )
    tools_or_functions, tool_func_choice = analyze_tools_and_tool_choice(
        all_requests[0]
    )

    adapted_request, request = v1_chat_generate_request(all_requests, tokenizer_manager)

    if adapted_request.stream:

        async def wrap_sgl_generator():
            stream_buffer = ""
            async for content in tokenizer_manager.generate_request(
                adapted_request, raw_request
            ):
                prompt_tokens = content["meta_info"]["prompt_tokens"]
                completion_tokens = content["meta_info"]["completion_tokens"]
                text = content["text"]
                delta = text[len(stream_buffer) :]
                stream_buffer = stream_buffer + delta
                finish_reason = format_finish_reason(
                    content["meta_info"]["finish_reason"]
                )

                # If finish_reason is not None and delta_text is not empty,
                # the delta_text is the eos_token and just remove it
                if finish_reason is not None and len(delta) > 0:
                    delta = ""
                yield delta, finish_reason

        async def completion_stream_generator():
            generator = wrap_sgl_generator()

            tool_call_count = 0
            async for response in generate_openai_format_from_stream_async(
                generator, prompt_template, tool_func_choice, tools_or_functions
            ):
                # Convert tool_calls to function_call if request.functions is provided
                if (
                    request.functions
                    and len(request.functions) > 0
                    and "tool_calls" in response["delta"]
                    and response["delta"]["tool_calls"]
                    and len(response["delta"]["tool_calls"]) > 0
                ):
                    tool_name = response["delta"]["tool_calls"][0]["function"]["name"]
                    tool_args = response["delta"]["tool_calls"][0]["function"][
                        "arguments"
                    ]
                    response["delta"]["function_call"] = response["delta"][
                        "tool_calls"
                    ][0]["function"]
                    response["delta"]["tool_calls"] = None
                    if tool_name and len(tool_name) > 0 and tool_args == "":
                        tool_call_count += 1
                # Return finish_reason after the first tool_call is streamed if functions is provided
                if request.functions and tool_call_count == 2:
                    response["delta"] = {}
                    response["finish_reason"] = "function_call"

                chunk = StreamChoice(**response)
                result = ChatCompletionChunk(id=adapted_request.rid, choices=[chunk])
                chunk_dic = result.dict(exclude_unset=True)
                chunk_data = json.dumps(chunk_dic, ensure_ascii=False)
                yield f"data: {chunk_data}\n\n"
                # Break from for loop after the first tool_call is streamed if functions is provided
                if request.functions and tool_call_count == 2:
                    break
            yield "data: [DONE]\n\n"

        # async def generate_stream_resp():
        #     is_first = True

        #     stream_buffer = ""
        #     n_prev_token = 0
        #     try:
        #         async for content in tokenizer_manager.generate_request(
        #             adapted_request, raw_request
        #         ):
        #             prompt_tokens = content["meta_info"]["prompt_tokens"]
        #             completion_tokens = content["meta_info"]["completion_tokens"]
        #             if request.logprobs:
        #                 logprobs = to_openai_style_logprobs(
        #                     output_token_logprobs=content["meta_info"][
        #                         "output_token_logprobs"
        #                     ][n_prev_token:],
        #                     output_top_logprobs=content["meta_info"][
        #                         "output_top_logprobs"
        #                     ][n_prev_token:],
        #                 )

        #                 n_prev_token = len(
        #                     content["meta_info"]["output_token_logprobs"]
        #                 )
        #                 token_logprobs = []
        #                 for token, logprob in zip(
        #                     logprobs.tokens, logprobs.token_logprobs
        #                 ):
        #                     token_bytes = list(token.encode("utf-8"))
        #                     top_logprobs = []
        #                     if logprobs.top_logprobs:
        #                         for top_token, top_logprob in logprobs.top_logprobs[
        #                             0
        #                         ].items():
        #                             top_token_bytes = list(top_token.encode("utf-8"))
        #                             top_logprobs.append(
        #                                 TopLogprob(
        #                                     token=top_token,
        #                                     bytes=top_token_bytes,
        #                                     logprob=top_logprob,
        #                                 )
        #                             )
        #                     token_logprobs.append(
        #                         ChatCompletionTokenLogprob(
        #                             token=token,
        #                             bytes=token_bytes,
        #                             logprob=logprob,
        #                             top_logprobs=top_logprobs,
        #                         )
        #                     )

        #                 choice_logprobs = ChoiceLogprobs(content=token_logprobs)

        #             else:
        #                 choice_logprobs = None

        #             if is_first:
        #                 # First chunk with role
        #                 is_first = False
        #                 choice_data = ChatCompletionResponseStreamChoice(
        #                     index=0,
        #                     delta=DeltaMessage(role="assistant"),
        #                     finish_reason=format_finish_reason(
        #                         content["meta_info"]["finish_reason"]
        #                     ),
        #                     # logprobs=choice_logprobs,
        #                 )
        #                 chunk = ChatCompletionStreamResponse(
        #                     id=content["meta_info"]["id"],
        #                     choices=[choice_data],
        #                     model=request.model,
        #                 )
        #                 yield f"data: {chunk.model_dump_json()}\n\n"

        #             text = content["text"]
        #             delta = text[len(stream_buffer) :]
        #             stream_buffer = stream_buffer + delta
        #             choice_data = ChatCompletionResponseStreamChoice(
        #                 index=0,
        #                 delta=DeltaMessage(content=delta),
        #                 finish_reason=format_finish_reason(
        #                     content["meta_info"]["finish_reason"]
        #                 ),
        #                 logprobs=choice_logprobs,
        #             )
        #             chunk = ChatCompletionStreamResponse(
        #                 id=content["meta_info"]["id"],
        #                 choices=[choice_data],
        #                 model=request.model,
        #             )
        #             yield f"data: {chunk.model_dump_json()}\n\n"
        #         if request.stream_options and request.stream_options.include_usage:
        #             usage = UsageInfo(
        #                 prompt_tokens=prompt_tokens,
        #                 completion_tokens=completion_tokens,
        #                 total_tokens=prompt_tokens + completion_tokens,
        #             )

        #             final_usage_chunk = ChatCompletionStreamResponse(
        #                 id=str(uuid.uuid4().hex),
        #                 choices=[],
        #                 model=request.model,
        #                 usage=usage,
        #             )
        #             final_usage_data = final_usage_chunk.model_dump_json(
        #                 exclude_unset=True, exclude_none=True
        #             )
        #             yield f"data: {final_usage_data}\n\n"
        #     except ValueError as e:
        #         error = create_streaming_error_response(str(e))
        #         yield f"data: {error}\n\n"
        #     yield "data: [DONE]\n\n"

        return StreamingResponse(
            # generate_stream_resp(),
            completion_stream_generator(),
            media_type="text/event-stream",
            background=tokenizer_manager.create_abort_task(adapted_request),
        )

    # Non-streaming response.
    try:
        ret = await tokenizer_manager.generate_request(
            adapted_request, raw_request
        ).__anext__()
    except ValueError as e:
        return create_error_response(str(e))
    if not isinstance(ret, list):
        ret = [ret]

    response = v1_chat_generate_response(request, prompt_template, ret)

    return response


def to_openai_style_logprobs(
    input_token_logprobs=None,
    output_token_logprobs=None,
    input_top_logprobs=None,
    output_top_logprobs=None,
):
    ret_logprobs = LogProbs()

    def append_token_logprobs(token_logprobs):
        for logprob, _, token_text in token_logprobs:
            ret_logprobs.tokens.append(token_text)
            ret_logprobs.token_logprobs.append(logprob)

            # Not supported yet
            ret_logprobs.text_offset.append(-1)

    def append_top_logprobs(top_logprobs):
        for tokens in top_logprobs:
            if tokens is not None:
                ret_logprobs.top_logprobs.append(
                    {token[2]: token[0] for token in tokens}
                )
            else:
                ret_logprobs.top_logprobs.append(None)

    if input_token_logprobs is not None:
        append_token_logprobs(input_token_logprobs)
    if output_token_logprobs is not None:
        append_token_logprobs(output_token_logprobs)
    if input_top_logprobs is not None:
        append_top_logprobs(input_top_logprobs)
    if output_top_logprobs is not None:
        append_top_logprobs(output_top_logprobs)

    return ret_logprobs
