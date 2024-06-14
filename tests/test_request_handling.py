import json
import unittest
from http import HTTPStatus
from typing import Any, List, Optional, Union

from fastapi.responses import JSONResponse
from transformers import AutoTokenizer

from functionary.openai_types import (
    ChatCompletionRequest,
    ChatMessage,
    Function,
    FunctionCall,
    StreamChoice,
    Tool,
)
from functionary.prompt_template import (
    Llama3Template,
    PromptTemplate,
    PromptTemplateV2,
    Llama3TemplateV3,
    get_available_prompt_template_versions,
)
from functionary.prompt_template.prompt_utils import (
    enforce_tool_choice,
    prepare_messages_for_inference,
)


def create_error_response(status_code, message, param) -> JSONResponse:
    return JSONResponse(
        {
            "object": "error",
            "message": message,
            "type": "invalid_request_error",
            "param": param,
            "code": status_code.value,
        },
        status_code=status_code.value,
    )


async def check_all_errors(request, served_model):
    if request.model != served_model:
        return create_error_response(
            status_code=HTTPStatus.NOT_FOUND,
            message=f"The model `{request.model}` does not exist.",
            param=None,
        )
    if request.tools and request.functions:
        return create_error_response(
            status_code=HTTPStatus.BAD_REQUEST,
            message="'functions' and 'tools' cannot both be provided. 'functions' are deprecated; use the 'tools' parameter instead.",
            param=None,
        )
    if isinstance(request.function_call, str) and request.function_call not in [
        "none",
        "auto",
    ]:
        return create_error_response(
            status_code=HTTPStatus.BAD_REQUEST,
            message=f"Invalid value: '{request.function_call}'. Supported values are: 'none' and 'auto'.",
            param="function_call",
        )
    if isinstance(request.tool_choice, str) and request.tool_choice not in [
        "none",
        "auto",
        "required",
    ]:
        return create_error_response(
            status_code=HTTPStatus.BAD_REQUEST,
            message=f"Invalid value: '{request.tool_choice}'. Supported values are: 'none', 'auto', and 'required'.",
            param="tool_choice",
        )
    if request.functions is None and request.function_call is not None:
        return create_error_response(
            status_code=HTTPStatus.BAD_REQUEST,
            message=f"Invalid value for 'function_call': 'function_call' is only allowed when 'functions' are specified.",
            param="function_call",
        )
    if request.tools is None and request.tool_choice is not None:
        return create_error_response(
            status_code=HTTPStatus.BAD_REQUEST,
            message=f"Invalid value for 'tool_choice': 'tool_choice' is only allowed when 'tools' are specified.",
            param="tool_choice",
        )
    return


def convert_jsonresponse_to_json(obj):
    return json.loads(obj.body.decode())


def get_token_ids_from_request(
    request: ChatCompletionRequest, tokenizer: AutoTokenizer
):
    if request.tools:
        tools = enforce_tool_choice(
            choice=request.tool_choice, tools_or_functions=request.tools
        )
        tools_or_functions = [item.dict() for item in tools]
        tool_func_choice = request.tool_choice if request.tool_choice else "auto"
    elif request.functions:
        functions = enforce_tool_choice(
            choice=request.function_call, tools_or_functions=request.functions
        )
        tools_or_functions = [item.dict() for item in functions]
        tool_func_choice = request.function_call if request.function_call else "auto"
    else:
        tools_or_functions = []
        tool_func_choice = "none"

    prompt_token_ids = prepare_messages_for_inference(
        tokenizer=tokenizer,
        messages=request.messages,
        tools_or_functions=tools_or_functions,
        tool_choice=tool_func_choice,
    ).tolist()[0]

    return prompt_token_ids


def generate_raw_response(
    gen_text: bool,
    num_tool_calls: Optional[int],
    tool_func_choice: Union[str, Tool, Function],
    default_text_str: str,
    default_tool_call_name: str,
    default_tool_call_args: List[str],
    prompt_template: PromptTemplate,
):
    message = {"role": "assistant"}
    if gen_text:
        message["content"] = default_text_str
    if num_tool_calls is not None:
        tool_calls = []
        for tool_call_args in default_tool_call_args[:num_tool_calls]:
            tool_calls.append(
                {
                    "type": "function",
                    "function": {
                        "name": default_tool_call_name,
                        "arguments": tool_call_args,
                    },
                }
            )
        message["tool_calls"] = tool_calls

    return prompt_template.get_raw_response_from_assistant_message(
        message=message,
        tool_func_choice=tool_func_choice,
        default_tool_call_name=default_tool_call_name,
    )


class TestRequestHandling(unittest.IsolatedAsyncioTestCase):
    def __init__(self, *args, **kwargs):
        super(TestRequestHandling, self).__init__(*args, **kwargs)
        self.served_model = "meetkai/functionary-small-v2.5"
        self.default_messages = [{"role": "user", "content": "How are you?"}]
        self.default_functions = [
            {
                "name": "get_current_weather",
                "description": "Get the current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        }
                    },
                    "required": ["location"],
                },
            },
        ]
        self.default_tools = [
            {"type": "function", "function": self.default_functions[0]}
        ]
        self.test_prompt_templates = get_available_prompt_template_versions()
        self.prompt_template_to_tokenizer_name_mapping = {
            PromptTemplateV2: "meetkai/functionary-small-v2.4",
            Llama3Template: "meetkai/functionary-small-v2.5",
            Llama3TemplateV3: "meetkai/functionary-medium-v3.0",
        }
        self.default_text_str = "Normal text generation"
        self.default_tool_call_name = "get_weather"
        self.default_tool_call_args = [
            '{"location": "Istanbul"}',
            '{"location": "Singapore"}',
        ]
        self.request_handling_test_cases = [
            {
                "test_aim": 'Normal text gen with "auto"',
                "tools_or_functions": "tools",
                "tool_func_choice": "auto",
                "gen_text": True,
                "num_tool_calls": None,
                "expected_result": ChatMessage(
                    role="assistant", content=self.default_text_str
                ),
                "expected_finish_reason": "stop",
            },
            {
                "test_aim": 'Single tool_calls with "auto"',
                "tools_or_functions": "tools",
                "tool_func_choice": "auto",
                "gen_text": False,
                "num_tool_calls": 1,
                "expected_result": ChatMessage(
                    role="assistant",
                    tool_calls=[
                        {
                            "type": "function",
                            "function": {
                                "name": self.default_tool_call_name,
                                "arguments": self.default_tool_call_args[0],
                            },
                        }
                    ],
                ),
                "expected_finish_reason": "tool_calls",
            },
            {
                "test_aim": 'Single function_call with "auto"',
                "tools_or_functions": "functions",
                "tool_func_choice": "auto",
                "gen_text": False,
                "num_tool_calls": 1,
                "expected_result": ChatMessage(
                    role="assistant",
                    function_call={
                        "name": self.default_tool_call_name,
                        "arguments": self.default_tool_call_args[0],
                    },
                ),
                "expected_finish_reason": "function_call",
            },
            {
                "test_aim": 'Parallel tool_calls with "auto"',
                "tools_or_functions": "tools",
                "tool_func_choice": "auto",
                "gen_text": False,
                "num_tool_calls": 2,
                "expected_result": ChatMessage(
                    role="assistant",
                    tool_calls=[
                        {
                            "type": "function",
                            "function": {
                                "name": self.default_tool_call_name,
                                "arguments": self.default_tool_call_args[0],
                            },
                        },
                        {
                            "type": "function",
                            "function": {
                                "name": self.default_tool_call_name,
                                "arguments": self.default_tool_call_args[1],
                            },
                        },
                    ],
                ),
                "expected_finish_reason": "tool_calls",
            },
            {
                "test_aim": 'Parallel function_calls with "auto"',
                "tools_or_functions": "functions",
                "tool_func_choice": "auto",
                "gen_text": False,
                "num_tool_calls": 2,
                "expected_result": ChatMessage(
                    role="assistant",
                    function_call={
                        "name": self.default_tool_call_name,
                        "arguments": self.default_tool_call_args[0],
                    },
                ),
                "expected_finish_reason": "function_call",
            },
            {
                "test_aim": 'Normal text gen + tool_calls with "auto"',
                "tools_or_functions": "tools",
                "tool_func_choice": "auto",
                "gen_text": True,
                "num_tool_calls": 1,
                "expected_result": ChatMessage(
                    role="assistant",
                    content=self.default_text_str,
                    tool_calls=[
                        {
                            "type": "function",
                            "function": {
                                "name": self.default_tool_call_name,
                                "arguments": self.default_tool_call_args[0],
                            },
                        }
                    ],
                ),
                "expected_finish_reason": "tool_calls",
            },
            {
                "test_aim": 'Normal text gen with "none"',
                "tools_or_functions": "tools",
                "tool_func_choice": "none",
                "gen_text": True,
                "num_tool_calls": None,
                "expected_result": ChatMessage(
                    role="assistant",
                    content=self.default_text_str,
                ),
                "expected_finish_reason": "stop",
            },
            {
                "test_aim": "tool_calls with tool_choice",
                "tools_or_functions": "tools",
                "tool_func_choice": Tool(
                    function=Function(name=self.default_tool_call_name)
                ),
                "gen_text": False,
                "num_tool_calls": 1,
                "expected_result": ChatMessage(
                    role="assistant",
                    tool_calls=[
                        {
                            "type": "function",
                            "function": {
                                "name": self.default_tool_call_name,
                                "arguments": self.default_tool_call_args[0],
                            },
                        }
                    ],
                ),
                "expected_finish_reason": "tool_calls",
            },
            {
                "test_aim": "function_call with function_call",
                "tools_or_functions": "functions",
                "tool_func_choice": Function(name=self.default_tool_call_name),
                "gen_text": False,
                "num_tool_calls": 1,
                "expected_result": ChatMessage(
                    role="assistant",
                    function_call={
                        "name": self.default_tool_call_name,
                        "arguments": self.default_tool_call_args[0],
                    },
                ),
                "expected_finish_reason": "function_call",
            },
            {
                "test_aim": 'parallel tool_calls with "required"',
                "tools_or_functions": "tools",
                "tool_func_choice": "required",
                "gen_text": False,
                "num_tool_calls": 2,
                "expected_result": ChatMessage(
                    role="assistant",
                    tool_calls=[
                        {
                            "type": "function",
                            "function": {
                                "name": self.default_tool_call_name,
                                "arguments": self.default_tool_call_args[0],
                            },
                        },
                        {
                            "type": "function",
                            "function": {
                                "name": self.default_tool_call_name,
                                "arguments": self.default_tool_call_args[1],
                            },
                        },
                    ],
                ),
                "expected_finish_reason": "tool_calls",
            },
        ]

    async def test_edge_cases(self):
        # When model name in request != served_model
        request = ChatCompletionRequest(
            model="meetkai/functionary-small-v0",
            messages=self.default_messages,
        )
        response = convert_jsonresponse_to_json(
            await check_all_errors(request=request, served_model=self.served_model)
        )
        self.assertEqual(
            response["message"],
            "The model `meetkai/functionary-small-v0` does not exist.",
            "Edge case handling failed: request.model different from served_model",
        )

        # When both tools and functions are provided
        request = ChatCompletionRequest(
            model=self.served_model,
            messages=self.default_messages,
            functions=self.default_functions,
            tools=self.default_tools,
        )
        response = convert_jsonresponse_to_json(
            await check_all_errors(request=request, served_model=self.served_model)
        )
        self.assertEqual(
            response["message"],
            "'functions' and 'tools' cannot both be provided. 'functions' are deprecated; use the 'tools' parameter instead.",
            "Edge case handling failed: Both tools and functions are present in request",
        )

        # When wrong function_call is provided
        request = ChatCompletionRequest(
            model=self.served_model,
            messages=self.default_messages,
            functions=self.default_functions,
            function_call="required",
        )
        response = convert_jsonresponse_to_json(
            await check_all_errors(request=request, served_model=self.served_model)
        )
        self.assertEqual(
            response["message"],
            "Invalid value: 'required'. Supported values are: 'none' and 'auto'.",
            "Edge case handling failed: Wrong function_call value",
        )

        # When wrong tool_choice is provided
        request = ChatCompletionRequest(
            model=self.served_model,
            messages=self.default_messages,
            tools=self.default_tools,
            tool_choice="requiredd",
        )
        response = convert_jsonresponse_to_json(
            await check_all_errors(request=request, served_model=self.served_model)
        )
        self.assertEqual(
            response["message"],
            "Invalid value: 'requiredd'. Supported values are: 'none', 'auto', and 'required'.",
            "Edge case handling failed: Wrong tool_choice value",
        )

        # When function_call is provided and functions is not provided
        request = ChatCompletionRequest(
            model=self.served_model,
            messages=self.default_messages,
            tools=self.default_tools,
            function_call="none",
        )
        response = convert_jsonresponse_to_json(
            await check_all_errors(request=request, served_model=self.served_model)
        )
        self.assertEqual(
            response["message"],
            "Invalid value for 'function_call': 'function_call' is only allowed when 'functions' are specified.",
            "Edge case handling failed: function_call provided without providing functions",
        )

        # When tool_choice is provided and tools is not provided
        request = ChatCompletionRequest(
            model=self.served_model,
            messages=self.default_messages,
            functions=self.default_functions,
            tool_choice="none",
        )
        response = convert_jsonresponse_to_json(
            await check_all_errors(request=request, served_model=self.served_model)
        )
        self.assertEqual(
            response["message"],
            "Invalid value for 'tool_choice': 'tool_choice' is only allowed when 'tools' are specified.",
            "Edge case handling failed: tool_choice provided without providing tools",
        )

    async def test_prompt_template_to_tokenizer(self):
        # Test whether all prompt templates are included in template_to_tokenizer_mapping yet
        for prompt_template in self.test_prompt_templates:
            self.assertIn(
                type(prompt_template),
                self.prompt_template_to_tokenizer_name_mapping.keys(),
                f"Prompt template `{type(prompt_template)}` is not included in template_to_tokenizer_mapping yet.",
            )

    async def test_request_handling(self):
        for prompt_template in self.test_prompt_templates:
            for test_case in self.request_handling_test_cases:
                raw_response = generate_raw_response(
                    gen_text=test_case["gen_text"],
                    num_tool_calls=test_case["num_tool_calls"],
                    tool_func_choice=test_case["tool_func_choice"],
                    default_text_str=self.default_text_str,
                    default_tool_call_name=self.default_tool_call_name,
                    default_tool_call_args=self.default_tool_call_args,
                    prompt_template=prompt_template,
                )
                chat_mess = prompt_template.parse_assistant_response(
                    llm_output=raw_response, tool_choice=test_case["tool_func_choice"]
                )
                # Convert id of tool_calls to None for the sake of unittests
                if (
                    "tool_calls" in chat_mess
                    and chat_mess["tool_calls"] is not None
                    and len(chat_mess["tool_calls"]) > 0
                ):
                    for i in range(len(chat_mess["tool_calls"])):
                        chat_mess["tool_calls"][i]["id"] = None

                # Convert tool_calls to function_call if functions are provided
                if (
                    test_case["tools_or_functions"] == "functions"
                    and "tool_calls" in chat_mess
                    and chat_mess["tool_calls"] is not None
                    and len(chat_mess["tool_calls"]) > 0
                ):
                    chat_mess["function_call"] = {
                        "name": chat_mess["tool_calls"][0]["function"]["name"],
                        "arguments": chat_mess["tool_calls"][0]["function"][
                            "arguments"
                        ],
                    }
                    chat_mess["tool_calls"] = None

                if "tool_calls" in chat_mess and chat_mess["tool_calls"] is not None:
                    finish_reason = "tool_calls"
                elif (
                    "function_call" in chat_mess
                    and chat_mess["function_call"] is not None
                ):
                    finish_reason = "function_call"
                else:
                    finish_reason = "stop"

                self.assertEqual(
                    ChatMessage(**chat_mess),
                    test_case["expected_result"],
                    f"Wrong ChatMessage for version: {prompt_template.version} | test: `{test_case['test_aim']}`|result={chat_mess}|expect:%s"
                    % test_case["expected_result"],
                )
                self.assertEqual(
                    finish_reason,
                    test_case["expected_finish_reason"],
                    f"Wrong finish reason for version: {prompt_template.version} | test: `{test_case['test_aim']}`",
                )

    async def test_request_handling_streaming(self):
        async def wrap_generator(tokenizer, raw_response, test_case):
            tokenized_response = tokenizer.encode(
                raw_response, add_special_tokens=False
            )
            # Append None to the end to produce finish_reason
            tokenized_response.append(None)

            prev_ids = []
            for token_id in tokenized_response:
                if token_id is not None:
                    offset = tokenizer.decode(prev_ids + [token_id])[
                        len(tokenizer.decode(prev_ids)) :
                    ]
                    prev_ids.append(token_id)
                    yield offset, None
                else:
                    yield "", test_case["expected_finish_reason"]

        for prompt_template in self.test_prompt_templates:

            tokenizer = AutoTokenizer.from_pretrained(
                self.prompt_template_to_tokenizer_name_mapping[type(prompt_template)]
            )
            special_tokens = {
                "additional_special_tokens": prompt_template.get_additional_tokens()
            }
            tokenizer.add_special_tokens(special_tokens)

            for test_case in self.request_handling_test_cases:
                raw_response = generate_raw_response(
                    gen_text=test_case["gen_text"],
                    num_tool_calls=test_case["num_tool_calls"],
                    tool_func_choice=test_case["tool_func_choice"],
                    default_text_str=self.default_text_str,
                    default_tool_call_name=self.default_tool_call_name,
                    default_tool_call_args=self.default_tool_call_args,
                    prompt_template=prompt_template,
                )
                # Use tokenizer to split raw response into chunks
                generator = wrap_generator(tokenizer, raw_response, test_case)

                content = ""
                tool_calls = []
                tool_call_count = 0
                chunks_list = []
                state = {}
                async for delta_text, finish_reason in generator:
                    state, responses = (
                        prompt_template.update_response_state_from_delta_text(
                            current_state=state,
                            delta_text=delta_text,
                            finish_reason=finish_reason,
                            tool_choice=test_case["tool_func_choice"],
                        )
                    )

                    if responses is not None:
                        if type(responses) is not list:
                            responses = [responses]
                    else:
                        responses = []

                    for response in responses:
                        chunk = StreamChoice(**response)
                        if chunk.delta.content is not None:
                            content += chunk.delta.content
                        if chunk.delta.tool_calls is not None:
                            name = chunk.delta.tool_calls[0].function.name
                            arguments = chunk.delta.tool_calls[0].function.arguments
                            # Convert chunk's tool_calls into function_call
                            if test_case["expected_finish_reason"] == "function_call":
                                chunk.delta.function_call = FunctionCall(
                                    name=name, arguments=arguments
                                )
                                chunk.delta.tool_calls = None
                                if name is not None and len(name) > 0:
                                    tool_call_count += 1
                                # Do not process the second tool call onwards for function_call
                                if tool_call_count == 2:
                                    continue
                            if name is not None:
                                tool_calls.append(
                                    {"name": name, "arguments": arguments}
                                )
                            else:
                                try:
                                    tool_calls[-1]["arguments"] += arguments
                                except:
                                    breakpoint()
                        chunks_list.append(chunk)

                # Test content
                label_content = self.default_text_str if test_case["gen_text"] else ""
                self.assertEqual(
                    content,
                    label_content,
                    f"Wrong streamed content for version: {prompt_template.version} | test: `{test_case['test_aim']}`",
                )
                # Test tool_calls
                if test_case["expected_finish_reason"] == "tool_calls":
                    for tool_call in tool_calls:
                        self.assertEqual(
                            tool_call["name"],
                            self.default_tool_call_name,
                            f"Wrong streamed tool call name for version: {prompt_template.version} | test: `{test_case['test_aim']}`",
                        )
                        self.assertIn(
                            tool_call["arguments"],
                            self.default_tool_call_args,
                            f"Wrong streamed tool call args for version: {prompt_template.version} | test: `{test_case['test_aim']}`",
                        )
                # Test function_call
                elif test_case["expected_finish_reason"] == "function_call":
                    self.assertEqual(
                        len(tool_calls),
                        1,
                        f"More than 1 tool call for function_call for version: {prompt_template.version} | test: `{test_case['test_aim']}`",
                    )
                    self.assertEqual(
                        tool_calls[0]["name"],
                        self.default_tool_call_name,
                        f"Wrong streamed fn call name for version: {prompt_template.version} | test: `{test_case['test_aim']}`",
                    )
                    self.assertIn(
                        tool_calls[0]["arguments"],
                        self.default_tool_call_args,
                        f"Wrong streamed fn call args for version: {prompt_template.version} | test: `{test_case['test_aim']}`",
                    )
                # Test finish_reason at last chunk and None for all preceding chunks
                self.assertNotEqual(
                    chunks_list[-1].finish_reason,
                    None,
                    f"Wrong streamed finish_reason for version: {prompt_template.version} | test: `{test_case['test_aim']}`",
                )
                for chunk in chunks_list[:-1]:
                    self.assertEqual(
                        chunk.finish_reason,
                        None,
                        f"finish_reason should be None for all chunks except last for version: {prompt_template.version} | test: `{test_case['test_aim']}`",
                    )


if __name__ == "__main__":
    unittest.main()
