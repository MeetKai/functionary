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
# Variable name for sgl frontend runtime generation
CONTENT_VAR = "content"


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


def convert_tool_calls_to_function_call(
    functions: Optional[List[Function]], chat_message: Dict
):
    if (
        functions
        and len(functions) > 0
        and "tool_calls" in chat_message
        and chat_message["tool_calls"] is not None
        and len(chat_message["tool_calls"]) > 0
    ):
        chat_message["function_call"] = {
            "name": chat_message["tool_calls"][0]["function"]["name"],
            "arguments": chat_message["tool_calls"][0]["function"]["arguments"],
        }
        chat_message["tool_calls"] = None

    return chat_message


def v1_chat_generate_request(
    request, tokenizer, tools_or_functions, tool_func_choice, return_text=False
):
    # Apply chat template and its stop strings.
    input_ids = prepare_messages_for_inference(
        tokenizer=tokenizer,
        messages=request.messages,
        tools_or_functions=tools_or_functions,
        tool_choice=tool_func_choice,
        device="cpu",
        return_text=return_text,
    )
    if not return_text:
        input_ids = input_ids.tolist()[0]

    stop = (
        request.stop
        + get_prompt_template_from_tokenizer(
            tokenizer=tokenizer
        ).get_stop_tokens_for_generation()
    )
    sampling_params = {
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

    if isinstance(input_ids, str):
        prompt_kwargs = {"text": input_ids}
    else:
        prompt_kwargs = {"input_ids": input_ids}

    adapted_request = GenerateReqInput(
        **prompt_kwargs,
        image_data=None,
        sampling_params=sampling_params,
        return_logprob=request.logprobs,
        top_logprobs_num=request.top_logprobs,
        stream=request.stream,
        return_text_in_logprobs=True,
        rid=f"cmpl-{uuid.uuid4().hex}",
    )

    return adapted_request, request


@sgl.function
def generate_sglang_srt_response(
    s: ProgramState,
    prompt: str,
    prompt_template,
    tools_or_functions,
    tool_func_choice,
    tokenizer,
):
    completion_tokens = 0
    stop_tokens = prompt_template.get_stop_tokens_for_generation()
    function_call_token = prompt_template.get_start_of_function_call_token()
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

    def check_stop_condition():
        stop_match = s.get_meta_info(CONTENT_VAR)["finish_reason"]["matched"]
        if not isinstance(stop_match, str):
            stop_match = tokenizer.decode(stop_match)
        return stop_match in stop_tokens

    s += prompt
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
                name=CONTENT_VAR,
                choices=choices,
                choices_method=CHOICES_SAMPLING_METHOD,
            )
            new_token = s[CONTENT_VAR]
            completion_tokens += len(
                tokenizer.encode(s[CONTENT_VAR], add_special_tokens=False)
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
            s += sgl.gen(name=CONTENT_VAR, regex=regex, stop=function_call_token)
            new_token = s[CONTENT_VAR]
            completion_tokens += s.get_meta_info(CONTENT_VAR)["completion_tokens"]
            if check_stop_condition():
                break
        elif gen_state["stage"] in ["text-gen", "code-interpreter"]:
            s += sgl.gen(name=CONTENT_VAR, stop=function_call_token)
            completion_tokens += s.get_meta_info(CONTENT_VAR)["completion_tokens"]
            if check_stop_condition():
                break
            else:
                s += function_call_token
                new_token = s[CONTENT_VAR] + function_call_token
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


async def wrap_sgl_generator(
    adapted_request,
    raw_request,
    request,
    tokenizer,
    tokenizer_manager,
    backend,
    prompt_template,
    tools_or_functions,
    tool_func_choice,
    frontend_state,
    grammar_sampling,
):
    if grammar_sampling:
        prompt = (
            adapted_request.text
            if adapted_request.text
            else tokenizer.decode(adapted_request.input_ids)
        )
        for out in frontend_state.text_iter():
            if out.startswith(prompt):
                continue
            yield out, None
        yield "", "stop"
    else:
        stream_buffer = ""
        async for content in tokenizer_manager.generate_request(
            adapted_request, raw_request
        ):
            text = content["text"]
            delta = text[len(stream_buffer) :]
            stream_buffer = stream_buffer + delta
            finish_reason = content["meta_info"]["finish_reason"]

            # If finish_reason is not None and delta_text is not empty,
            # the delta_text is the eos_token and just remove it
            if finish_reason is not None and len(delta) > 0:
                delta = ""
            yield delta, finish_reason


async def completion_stream_generator(
    adapted_request,
    raw_request,
    request,
    tokenizer,
    tokenizer_manager,
    backend,
    prompt_template,
    tools_or_functions,
    tool_func_choice,
    frontend_state,
    grammar_sampling,
):
    generator = wrap_sgl_generator(
        adapted_request,
        raw_request,
        request,
        tokenizer,
        tokenizer_manager,
        backend,
        prompt_template,
        tools_or_functions,
        tool_func_choice,
        frontend_state,
        grammar_sampling,
    )

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
            tool_args = response["delta"]["tool_calls"][0]["function"]["arguments"]
            response["delta"]["function_call"] = response["delta"]["tool_calls"][0][
                "function"
            ]
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


async def v1_chat_generate_completion(
    adapted_request,
    raw_request,
    request,
    tokenizer,
    tokenizer_manager,
    backend,
    prompt_template,
    tools_or_functions,
    tool_func_choice,
):
    grammar_sampling = True if backend else False
    if grammar_sampling:
        prompt = (
            adapted_request.text
            if adapted_request.text
            else tokenizer.decode(adapted_request.input_ids)
        )
        state = generate_sglang_srt_response.run(
            prompt=prompt,
            prompt_template=prompt_template,
            tools_or_functions=tools_or_functions,
            tool_func_choice=tool_func_choice,
            tokenizer=tokenizer,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            frequency_penalty=request.frequency_penalty,
            presence_penalty=request.presence_penalty,
            stream=request.stream,
        )
        if adapted_request.stream:
            return (
                StreamingResponse(
                    completion_stream_generator(
                        adapted_request,
                        raw_request,
                        request,
                        tokenizer,
                        tokenizer_manager,
                        backend,
                        prompt_template,
                        tools_or_functions,
                        tool_func_choice,
                        state,
                        grammar_sampling,
                    ),
                    media_type="text/event-stream",
                ),
                None,
            )
        else:
            return state.text()[len(prompt) :], None
    else:
        if adapted_request.stream:
            return (
                StreamingResponse(
                    completion_stream_generator(
                        adapted_request,
                        raw_request,
                        request,
                        tokenizer,
                        tokenizer_manager,
                        backend,
                        prompt_template,
                        tools_or_functions,
                        tool_func_choice,
                        None,
                        grammar_sampling,
                    ),
                    media_type="text/event-stream",
                    background=tokenizer_manager.create_abort_task(adapted_request),
                ),
                None,
            )
        else:
            try:
                ret = await tokenizer_manager.generate_request(
                    adapted_request, raw_request
                ).__anext__()
            except ValueError as e:
                return None, create_error_response(str(e))
            return ret["text"], None


def v1_chat_generate_response(
    adapted_request,
    raw_request,
    request,
    output_text,
    prompt_template,
    tokenizer,
    tool_func_choice,
):
    chat_mess = prompt_template.parse_assistant_response(
        llm_output=output_text, tool_choice=tool_func_choice
    )
    chat_mess = convert_tool_calls_to_function_call(
        functions=request.functions, chat_message=chat_mess
    )

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
            finish_reason=finish_reason,
        )
    ]
    prompt_tokens = (
        len(adapted_request.input_ids)
        if adapted_request.input_ids
        else len(tokenizer.encode(adapted_request.text))
    )
    completion_tokens = len(tokenizer.encode(output_text, add_special_tokens=False)) + 1

    response = ChatCompletionResponse(
        id=adapted_request.rid,
        model=request.model,
        choices=choices,
        usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )
    return response


async def v1_chat_completions(tokenizer_manager, backend, raw_request: Request):
    request_json = await raw_request.json()
    request = ChatCompletionRequest(**request_json)
    tokenizer = (
        tokenizer_manager.tokenizer if tokenizer_manager else backend.get_tokenizer()
    )

    prompt_template = get_prompt_template_from_tokenizer(tokenizer=tokenizer)
    tools_or_functions, tool_func_choice = analyze_tools_and_tool_choice(request)

    adapted_request, request = v1_chat_generate_request(
        request, tokenizer, tools_or_functions, tool_func_choice, return_text=False
    )

    output, error = await v1_chat_generate_completion(
        adapted_request=adapted_request,
        raw_request=raw_request,
        request=request,
        tokenizer=tokenizer,
        tokenizer_manager=tokenizer_manager,
        backend=backend,
        prompt_template=prompt_template,
        tools_or_functions=tools_or_functions,
        tool_func_choice=tool_func_choice,
    )
    if error:
        return error

    if adapted_request.stream:
        return output

    response = v1_chat_generate_response(
        adapted_request=adapted_request,
        raw_request=raw_request,
        request=request,
        output_text=output,
        prompt_template=prompt_template,
        tokenizer=tokenizer,
        tool_func_choice=tool_func_choice,
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
