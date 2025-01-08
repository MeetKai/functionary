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
from dataclasses import dataclass
from http import HTTPStatus
from typing import Dict, List, Optional, Tuple, Union

import sglang as sgl
from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse
from outlines.fsm.json_schema import build_regex_from_schema
from sglang.lang.choices import greedy_token_selection
from sglang.lang.interpreter import ProgramState
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.openai_api.protocol import ErrorResponse
from sglang.srt.server import Runtime
from transformers import AutoTokenizer

from functionary.inference_stream import generate_openai_format_from_stream_async
from functionary.inference_utils import (
    analyze_tools_and_tool_choice,
    check_all_errors,
    convert_tool_calls_to_function_call,
    create_error_response,
)
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
from functionary.prompt_template import (
    PromptTemplate,
    get_prompt_template_from_tokenizer,
)
from functionary.prompt_template.prompt_utils import prepare_messages_for_inference

# Choices sampling method for sgl.select
CHOICES_SAMPLING_METHOD = greedy_token_selection
# Variable name for sgl frontend runtime generation
CONTENT_VAR = "content"


@dataclass
class ChatCompletionParams:
    """Parameters and context used across various chat completion functions"""

    adapted_request: GenerateReqInput
    raw_request: Request
    request: ChatCompletionRequest
    tokenizer: AutoTokenizer
    tokenizer_manager: Optional[TokenizerManager]
    srt_backend: Optional[Runtime]
    prompt_template: PromptTemplate
    tools_or_functions: List[Dict]
    tool_func_choice: Optional[Union[str, Tool, Function]]
    frontend_state: Optional[ProgramState]
    grammar_sampling: bool


def v1_chat_generate_request(
    request: ChatCompletionRequest,
    tokenizer: AutoTokenizer,
    tools_or_functions: List[Dict],
    tool_func_choice: Optional[Union[str, Tool, Function]],
    return_text: bool = False,
) -> Tuple[GenerateReqInput, ChatCompletionRequest]:
    """
    Generate an adapted request that SGLang uses.

    This function prepares the input for SGLang inference by processing the chat completion request,
    applying the appropriate tokenization, and setting up the sampling parameters.

    Args:
        request (ChatCompletionRequest): The original chat completion request.
        tokenizer (AutoTokenizer): The tokenizer to use for encoding the text input, if any.
        tools_or_functions (List[Dict]): List of available tools or functions.
        tool_func_choice (Optional[Union[str, Tool, Function]]): The chosen tool or function, if any.
        return_text (bool, optional): Whether to return the input as text instead of token IDs. Defaults to False.

    Returns:
        Tuple[GenerateReqInput, ChatCompletionRequest]: A tuple containing:
            - The adapted request (GenerateReqInput) to be used by SGLang.
            - The original request (ChatCompletionRequest), NOT modified.

    Note:
        This function handles the conversion of the chat messages into a format suitable for SGLang,
        applies the chat template, sets up stopping criteria, and configures sampling parameters.
    """
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
        "skip_special_tokens": False,
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
        rid=f"chatcmpl-{uuid.uuid4().hex}",
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
    """
    Generate a response using SGLang Frontend Runtime (SRT).

    This function is used when grammar-sampling is enabled. It uses the SRT program
    state to update the specific prompt-template Finite State Machine (FSM) generation
    state. Constrained generation is performed at specific stages of the FSM.

    Args:
        s (ProgramState): The current program state in SGLang.
        prompt (str): The input prompt to generate a response for.
        prompt_template: The template used to structure the prompt and response.
        tools_or_functions (list): Available tools or functions for the model to use.
        tool_func_choice (str): The chosen tool or function choice.
        tokenizer: The tokenizer used for encoding and decoding text.

    Returns:
        ProgramState: The updated program state after generating the response.
    """
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
            regex = build_regex_from_schema(json.dumps(tool["parameters"]))
            s += sgl.gen(name=CONTENT_VAR, regex=regex, stop=function_call_token)
            new_token = s[CONTENT_VAR]
            completion_tokens += s.get_meta_info(CONTENT_VAR)["completion_tokens"]
            # Generate new token to determin if there is another tool call
            s += sgl.gen(name=CONTENT_VAR, stop=function_call_token)
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


async def wrap_sgl_generator(params: ChatCompletionParams):
    """
    This asynchronous generator function yields generated text chunks along
    with their finish reasons.

    Args:
        params (ChatCompletionParams): A dataclass containing all necessary
            parameters for the chat completion, including the request details,
            tokenizer, backend, and other configuration options.

    Yields:
        Tuple[str, Optional[str]]: A tuple containing:
            - str: The generated text chunk.
            - Optional[str]: The finish reason, if any (e.g., "stop", "length", etc.).
    """
    if params.grammar_sampling:
        prompt = (
            params.adapted_request.text
            if params.adapted_request.text
            else params.tokenizer.decode(params.adapted_request.input_ids)
        )
        # Iterates over the text generated by the SGLang Frontend Runtime
        for out in params.frontend_state.text_iter():
            if out.startswith(prompt):
                continue
            yield out, None
        yield "", "stop"
    else:
        # Iterates over the text generated by the tokenizer manager
        stream_buffer = ""
        async for content in params.tokenizer_manager.generate_request(
            params.adapted_request, params.raw_request
        ):
            text = content["text"]
            delta = text[len(stream_buffer) :]
            stream_buffer = stream_buffer + delta
            finish_reason = content["meta_info"]["finish_reason"]

            # If finish_reason is not None and delta_text is not empty,
            # the delta_text is the eos_token and just remove it
            if finish_reason is not None:
                finish_reason = finish_reason["type"]
                if len(delta) > 0:
                    delta = ""
            yield delta, finish_reason


async def completion_stream_generator(params: ChatCompletionParams):
    """
    This asynchronous generator function produces a stream of ChatCompletionChunk
    objects. It handles both grammar-sampling and regular generations,
    depending on the parameters provided.

    Args:
        params (ChatCompletionParams): A dataclass containing all necessary
            parameters for the chat completion, including the request details,
            tokenizer, backend, and other configuration options.

    Yields:
        str: JSON-formatted strings representing chunks of the chat completion
             response, including delta updates and finish reasons.

    Notes:
        - The function adapts its behavior based on whether grammar sampling
          is enabled or not.
        - It handles the conversion of tool calls to function calls when
          appropriate.
        - The stream is terminated with a "[DONE]" message.
    """
    # Initialize the text generator
    generator = wrap_sgl_generator(params)

    tool_call_count = 0
    # Generate the text in openai format
    async for response in generate_openai_format_from_stream_async(
        generator,
        params.prompt_template,
        params.tool_func_choice,
        params.tools_or_functions,
    ):
        # Convert tool_calls to function_call if request.functions is provided
        response = convert_tool_calls_to_function_call(
            functions=params.request.functions, chat_message=response
        )
        if response["delta"]["function_call"]:
            tool_name = response["delta"]["function_call"]["name"]
            tool_args = response["delta"]["function_call"]["arguments"]
            if tool_name and len(tool_name) > 0 and tool_args == "":
                tool_call_count += 1

            # Return finish_reason after the first tool_call is streamed if functions is provided
            if params.request.functions and tool_call_count == 2:
                response["delta"] = {}
                response["finish_reason"] = "function_call"

        chunk = StreamChoice(**response)
        result = ChatCompletionChunk(
            id=params.adapted_request.rid, choices=[chunk], model=params.request.model
        )
        chunk_dic = result.model_dump()
        chunk_data = json.dumps(chunk_dic, ensure_ascii=False)
        yield f"data: {chunk_data}\n\n"
        # Break from for loop after the first tool_call is streamed if functions is provided
        if params.request.functions and tool_call_count == 2:
            break
    yield "data: [DONE]\n\n"


async def v1_chat_generate_completion(
    params: ChatCompletionParams,
) -> Tuple[Union[StreamingResponse, str], Optional[JSONResponse]]:
    """
    Generate a text completion.

    This function handles both streaming and non-streaming responses for chat completions.
    It supports both regular and grammar-sampling generations.

    Args:
        params (ChatCompletionParams): A dataclass containing all necessary parameters and context
                                       for generating the text.

    Returns:
        Tuple[Union[StreamingResponse, str], Optional[JSONResponse]]:
            - If streaming is requested, returns a StreamingResponse object.
            - If non-streaming, returns the generated text as a string.
            - The second element is an optional JSONResponse for error cases.

    Note:
        - For grammar-sampling, it uses the SGLang Frontend Runtime.
        - For regular generation, it uses the tokenizer manager to generate the response.
        - Streaming responses are handled by the completion_stream_generator function.
    """
    # If streaming, return the StreamingResponse else return the text
    if params.grammar_sampling:
        # Form the text prompt and run the SGLang Frontend Runtime
        prompt = (
            params.adapted_request.text
            if params.adapted_request.text
            else params.tokenizer.decode(params.adapted_request.input_ids)
        )
        state = generate_sglang_srt_response.run(
            prompt=prompt,
            prompt_template=params.prompt_template,
            tools_or_functions=params.tools_or_functions,
            tool_func_choice=params.tool_func_choice,
            tokenizer=params.tokenizer,
            max_new_tokens=params.request.max_tokens,
            temperature=params.request.temperature,
            top_p=params.request.top_p,
            top_k=params.request.top_k,
            frequency_penalty=params.request.frequency_penalty,
            presence_penalty=params.request.presence_penalty,
            stream=params.request.stream,
        )

        if params.adapted_request.stream:
            params.frontend_state = state
            return (
                StreamingResponse(
                    completion_stream_generator(params),
                    media_type="text/event-stream",
                ),
                None,
            )
        else:
            return state.text()[len(prompt) :], None
    else:
        if params.adapted_request.stream:
            return (
                StreamingResponse(
                    completion_stream_generator(params),
                    media_type="text/event-stream",
                    background=params.tokenizer_manager.create_abort_task(
                        params.adapted_request
                    ),
                ),
                None,
            )
        else:
            try:
                ret = await params.tokenizer_manager.generate_request(
                    params.adapted_request, params.raw_request
                ).__anext__()
            except ValueError as e:
                return None, create_error_response(
                    status_code=HTTPStatus.BAD_REQUEST, message=str(e), param=None
                )
            return ret["text"], None


def v1_chat_generate_response(
    output_text: str, params: ChatCompletionParams
) -> ChatCompletionResponse:
    """
    Generate a ChatCompletionResponse from the output text and parameters.

    This function processes the output text, parses it according to the prompt template,
    and constructs a ChatCompletionResponse object.

    Args:
        output_text (str): The raw output text from SGLang inference.
        params (ChatCompletionParams): Parameters and context for the chat completion.

    Returns:
        ChatCompletionResponse: An OpenAI-compatible response containing the assistant's message,
        usage information, and other metadata.
    """
    # Parse the output text using the specific prompt template
    chat_mess = params.prompt_template.parse_assistant_response(
        llm_output=output_text, tool_choice=params.tool_func_choice
    )
    # Convert tool_calls to function_call if request.functions is provided
    chat_mess = convert_tool_calls_to_function_call(
        functions=params.request.functions, chat_message=chat_mess
    )

    # Postprocess finish reason
    finish_reason = "stop"
    if params.tool_func_choice is None or params.tool_func_choice in [
        "auto",
        "required",
    ]:
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
        len(params.adapted_request.input_ids)
        if params.adapted_request.input_ids
        else len(params.tokenizer.encode(params.adapted_request.text))
    )
    completion_tokens = (
        len(params.tokenizer.encode(output_text, add_special_tokens=False)) + 1
    )  # +1 for the eos token

    response = ChatCompletionResponse(
        id=params.adapted_request.rid,
        model=params.request.model,
        choices=choices,
        usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )
    return response


async def v1_chat_completions(
    tokenizer_manager: Optional[TokenizerManager],
    srt_backend: Optional[Runtime],
    raw_request: Request,
    served_model: List[str],
):
    """
    Handle chat completions for v1 of the API.

    This function processes the incoming request, prepares the necessary parameters,
    generates the chat completion, and returns the response. It supports both
    streaming and non-streaming responses.

    Args:
        tokenizer_manager (Optional[TokenizerManager]): Manager for tokenization tasks.
            None if grammar sampling is enabled.
        srt_backend (Optional[Runtime]): The SRT backend for processing.
            None if grammar sampling is disabled.
        raw_request (Request): The raw incoming request object.

    Returns:
        Union[ChatCompletionResponse, StreamingResponse, JSONResponse]:
            - ChatCompletionResponse for non-streaming successful responses.
            - StreamingResponse for streaming responses.
            - JSONResponse for error responses.

    Raises:
        No explicit raises, but may return error responses for various failure scenarios.
    """
    request_json = await raw_request.json()
    request = ChatCompletionRequest(**request_json)
    if type(request.temperature) is not float:
        request.temperature = 1e-5
        
    tokenizer = (
        tokenizer_manager.tokenizer
        if tokenizer_manager
        else srt_backend.get_tokenizer()
    )
    prompt_template = get_prompt_template_from_tokenizer(tokenizer=tokenizer)
    tools_or_functions, tool_func_choice = analyze_tools_and_tool_choice(request)

    # Check for errors
    error_check_ret = await check_all_errors(request, served_model)
    if error_check_ret is not None:
        return error_check_ret

    # Generate the adapted request
    adapted_request, request = v1_chat_generate_request(
        request, tokenizer, tools_or_functions, tool_func_choice, return_text=False
    )

    # Prepare the parameters for generate_completion and generate_response functions
    params = ChatCompletionParams(
        adapted_request=adapted_request,
        raw_request=raw_request,
        request=request,
        tokenizer=tokenizer,
        tokenizer_manager=tokenizer_manager,
        srt_backend=srt_backend,
        prompt_template=prompt_template,
        tools_or_functions=tools_or_functions,
        tool_func_choice=tool_func_choice,
        frontend_state=None,  # None first. Set later if needed
        grammar_sampling=True if srt_backend else False,
    )

    # Generate the text completion
    output, error = await v1_chat_generate_completion(params)
    if error:
        return error

    # If streaming, return the output(StreamingResponse) directly
    if adapted_request.stream:
        return output

    # Generate the API response
    response = v1_chat_generate_response(output_text=output, params=params)

    return response
