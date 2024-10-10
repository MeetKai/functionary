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
import re
import time
import uuid
from http import HTTPStatus
from typing import Any, Dict, List, Optional

import sglang as sgl
from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse
from outlines.fsm.json_schema import build_regex_from_schema
from sglang.lang.choices import greedy_token_selection
from sglang.lang.interpreter import ProgramState
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
    Function,
    StreamChoice,
    Tool,
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


# Choices sampling method for sgl.select
CHOICES_SAMPLING_METHOD = greedy_token_selection


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


def v1_chat_generate_request(all_requests, tokenizer):
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
                tokenizer=tokenizer,
                messages=request.messages,
                tools_or_functions=tools_or_functions,
                tool_choice=tool_func_choice,
                device="cpu",
            ).tolist()[0]
            stop = (
                request.stop
                + get_prompt_template_from_tokenizer(
                    tokenizer=tokenizer
                ).get_stop_tokens_for_generation()
            )
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
    tokenizer = tokenizer_manager.tokenizer

    prompt_template = get_prompt_template_from_tokenizer(
        tokenizer=tokenizer_manager.tokenizer
    )
    tools_or_functions, tool_func_choice = analyze_tools_and_tool_choice(
        all_requests[0]
    )

    adapted_request, request = v1_chat_generate_request(all_requests, tokenizer)

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
                result = ChatCompletionChunk(
                    id=adapted_request.rid, choices=[chunk], model=request.model
                )
                chunk_dic = result.dict(exclude_unset=True)
                chunk_data = json.dumps(chunk_dic, ensure_ascii=False)
                yield f"data: {chunk_data}\n\n"
                # Break from for loop after the first tool_call is streamed if functions is provided
                if request.functions and tool_call_count == 2:
                    break
            yield "data: [DONE]\n\n"

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


async def v1_chat_completions_grammar_sampling(backend, raw_request: Request):
    request_json = await raw_request.json()
    request = ChatCompletionRequest(**request_json)
    tokenizer = backend.get_tokenizer()
    request_id = f"cmpl-{uuid.uuid4().hex}"

    # Convert legacy functions to tools
    if request.functions is not None:
        request.tools = [
            Tool(type="function", function=function) for function in request.functions
        ]
    # Convert legacy function_call to tool_choice
    if request.function_call is not None:
        if isinstance(request.function_call, str) and (
            request.function_call == "none" or request.function_call == "auto"
        ):
            request.tool_choice = request.function_call
        if request.function_call and isinstance(request.function_call, Function):
            request.tool_choice = Tool(
                type="function", function=Function(name=request.function_call.name)
            )

    prompt_template = get_prompt_template_from_tokenizer(tokenizer=tokenizer)
    tools_or_functions, tool_func_choice = analyze_tools_and_tool_choice(request)

    gen_state = prompt_template.initialize_fsm_gen_state(
        tool_choice=tool_func_choice,
        curr_text="",
        curr_tokens=None,
        add_code_interpreter=(
            True
            if any(
                [
                    "type" in tool_or_func
                    and tool_or_func["type"] == "code_interpreter"
                    for tool_or_func in tools_or_functions
                ]
            )
            else False
        ),
    )
    prompt = prepare_messages_for_inference(
        tokenizer=tokenizer,
        messages=request.messages,
        tools_or_functions=tools_or_functions,
        tool_choice=tool_func_choice,
        return_text=True,
    )

    content_var = "content"
    completion_tokens = 0

    @sgl.function
    def generate_response(s: ProgramState, gen_state: Dict):
        nonlocal completion_tokens

        s += prompt

        # Form the options for the following stages
        tools = []
        for tool in tools_or_functions:
            if "type" in tool:
                if tool["type"] == "function":
                    tools.append(tool["function"])
            else:
                tools.append(tool)
        options = prompt_template.get_options_from_gen_state(
            gen_state=gen_state, tools_or_functions=tools
        )

        stop_tokens = prompt_template.get_stop_tokens_for_generation()
        function_call_token = prompt_template.get_start_of_function_call_token()

        def check_stop_condition():
            stop_match = s.get_meta_info(content_var)["finish_reason"]["matched"]
            if not isinstance(stop_match, str):
                stop_match = tokenizer.decode(stop_match)
            return stop_match in stop_tokens

        while True:
            if gen_state["stage"] == "function":
                choices = [
                    tool["function"]["name"]
                    for tool in tools_or_functions
                    if tool["type"] == "function"
                ]
                if gen_state["add_all_recipient"]:
                    choices.append("all")
                if gen_state["add_code_interpreter"]:
                    choices.append("python")
                s += sgl.select(
                    name=content_var,
                    choices=choices,
                    choices_method=CHOICES_SAMPLING_METHOD,
                )
                new_token = s[content_var]
                completion_tokens += len(
                    tokenizer.encode(s[content_var], add_special_tokens=False)
                )
            elif gen_state["stage"] == "pre-parameter":
                s += prompt_template.fn_param_sep_token
                new_token = prompt_template.fn_param_sep_token
            elif gen_state["stage"] == "parameter":
                tool = next(t for t in tools if t["name"] == gen_state["func_name"])
                regex = (
                    build_regex_from_schema(json.dumps(tool["parameters"]))
                    + f"({re.escape(function_call_token)})?"
                )
                s += sgl.gen(name=content_var, regex=regex, stop=function_call_token)
                new_token = s[content_var]
                completion_tokens += s.get_meta_info(content_var)["completion_tokens"]
                if check_stop_condition():
                    break
            elif gen_state["stage"] in ["text-gen", "code-interpreter"]:
                s += sgl.gen(name=content_var, stop=function_call_token)
                completion_tokens += s.get_meta_info(content_var)["completion_tokens"]
                if check_stop_condition():
                    break
                else:
                    s += function_call_token
                    new_token = s[content_var] + function_call_token
            elif gen_state["stage"] == "pre-function":
                s += function_call_token
                new_token = function_call_token

            gen_state = prompt_template.update_fsm_gen_state(
                gen_state=gen_state,
                new_token=new_token,
                new_token_id=None,
                options=options,
                tokenizer=tokenizer,
            )

    state = generate_response.run(
        gen_state=gen_state,
        max_new_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        frequency_penalty=request.frequency_penalty,
        presence_penalty=request.presence_penalty,
        stream=request.stream,
    )

    async def wrap_sgl_generator():
        for out in state.text_iter():
            if out.startswith(prompt):
                continue
            yield out, None
        yield "", "stop"

    async def completion_stream_generator(functions):
        generator = wrap_sgl_generator()

        tool_call_count = 0
        async for response in generate_openai_format_from_stream_async(
            generator, prompt_template, tool_func_choice, tools_or_functions
        ):
            # Convert tool_calls to function_call if request.functions is provided
            if (
                functions
                and len(functions) > 0
                and "tool_calls" in response["delta"]
                and response["delta"]["tool_calls"]
                and len(response["delta"]["tool_calls"]) > 0
            ):
                tool_name = response["delta"]["tool_calls"][0]["function"]["name"]
                tool_args = response["delta"]["tool_calls"][0]["function"]["arguments"]
                response["delta"]["function_call"] = response["delta"]["tool_calls"][0][
                    "function"
                ]
                response["delta"]["tool_calls"] = None
                if tool_name and len(tool_name) > 0 and tool_args == "":
                    tool_call_count += 1

            chunk = StreamChoice(**response)
            result = ChatCompletionChunk(
                id=request_id, choices=[chunk], model=request.model
            )
            chunk_dic = result.dict(exclude_unset=True)
            chunk_data = json.dumps(chunk_dic, ensure_ascii=False)
            yield f"data: {chunk_data}\n\n"
            # Break from for loop after the first tool_call is streamed if functions is provided
            if functions and tool_call_count == 2:
                break
        yield "data: [DONE]\n\n"

    if request.stream:
        return StreamingResponse(
            completion_stream_generator(functions=request.functions),
            media_type="text/event-stream",
        )

    chat_mess = prompt_template.parse_assistant_response(
        llm_output=state.text()[len(prompt) :], tool_choice=tool_func_choice
    )

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
    finish_reason = "stop"
    if "function_call" in chat_mess and chat_mess["function_call"]:
        finish_reason = "function_call"
    if "tool_calls" in chat_mess and chat_mess["tool_calls"]:
        finish_reason = "tool_calls"

    choices = [
        ChatCompletionResponseChoice(
            index=0,
            message=ChatMessage(**chat_mess),
            # logprobs=choice_logprobs,
            finish_reason=finish_reason,
        )
    ]

    prompt_tokens = len(tokenizer.encode(prompt))
    response = ChatCompletionResponse(
        id=state.get_meta_info(content_var)["id"],
        model=request.model,
        choices=choices,
        usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )
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
