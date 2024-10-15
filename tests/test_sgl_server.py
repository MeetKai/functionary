import json
import subprocess
import time
import unittest
from typing import Dict, List, Optional

import psutil
import requests
from openai import OpenAI
from openai.types.chat.chat_completion_message import FunctionCall
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
)
from rich import print

DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH = 600


def popen_launch_sgl_server(
    model: str,
    base_url: str,
    timeout: float,
    context_length: int,
    grammar_sampling: bool,
    env: Optional[dict] = None,
    return_stdout_stderr: bool = False,
):
    _, host, port = base_url.split(":")
    host = host[2:]

    command = [
        "python3",
        "server_sglang.py",
        "--model",
        model,
        "--host",
        host,
        "--port",
        str(port),
        "--context-length",
        str(context_length),
    ]
    if grammar_sampling:
        command += ["--enable-grammar-sampling"]

    if return_stdout_stderr:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
        )
    else:
        process = subprocess.Popen(command, stdout=None, stderr=None, env=env)

    start_time = time.time()
    api_key = "test"
    while time.time() - start_time < timeout:
        try:
            headers = {
                "Content-Type": "application/json; charset=utf-8",
                "Authorization": f"Bearer {api_key}",
            }
            response = requests.get(f"{base_url}/health", headers=headers)
            if response.status_code == 200:
                return process
        except requests.RequestException:
            pass
        time.sleep(5)
    raise TimeoutError("Server failed to start within the timeout period.")


def kill_child_process(pid, including_parent=True, skip_pid=None):
    """Kill the process and all its children process."""
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return

    children = parent.children(recursive=True)
    for child in children:
        if child.pid == skip_pid:
            continue
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass

    if including_parent:
        try:
            parent.kill()
        except psutil.NoSuchProcess:
            pass


def call_openai_api(
    test_case: Dict,
    client: OpenAI,
    model: str,
    default_tools: List,
    python_tool: Dict,
    default_functions: List,
    stream: bool = False,
):
    if test_case["call_mode"] == "tools":
        if test_case["code_interpreter"]:
            if model.startswith("meetkai"):
                tools = default_tools + [{"type": "code_interpreter"}]
            else:
                tools = default_tools + [python_tool]
        else:
            tools = default_tools
        response = client.chat.completions.create(
            model=model,
            messages=test_case["messages"],
            tools=tools,
            tool_choice=test_case["choice"],
            temperature=0.0,
            stream=stream,
        )
    else:
        response = client.chat.completions.create(
            model=model,
            messages=test_case["messages"],
            functions=default_functions,
            function_call=test_case["choice"],
            temperature=0.0,
            stream=stream,
        )
    return response


class TestSglServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.openai_model = "gpt-4o-mini-2024-07-18"
        cls.base_url = "http://127.0.0.1:8000"
        cls.default_functions = [
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
        cls.default_tools = [{"type": "function", "function": cls.default_functions[0]}]
        cls.python_tool = {
            "type": "function",
            "function": {
                "name": "python",
                "description": "Generate Python code",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "The Python code to execute",
                        }
                    },
                    "required": ["code"],
                },
            },
        }
        cls.request_handling_test_cases = [
            {
                "test_aim": 'Normal text gen with "auto"',
                "messages": [{"role": "user", "content": "How are you?"}],
                "call_mode": "tools",
                "code_interpreter": False,
                "choice": "auto",
            },
            {
                "test_aim": 'Single tool_calls with "auto"',
                "messages": [
                    {"role": "user", "content": "What is the weather in Istanbul?"}
                ],
                "call_mode": "tools",
                "code_interpreter": False,
                "choice": "auto",
            },
            {
                "test_aim": 'Single function_call with "auto"',
                "messages": [
                    {"role": "user", "content": "What is the weather in Istanbul?"}
                ],
                "call_mode": "functions",
                "code_interpreter": False,
                "choice": "auto",
            },
            {
                "test_aim": 'Parallel tool_calls with "auto"',
                "messages": [
                    {
                        "role": "user",
                        "content": "What is the weather in Istanbul and Singapore respectively?",
                    }
                ],
                "call_mode": "tools",
                "code_interpreter": False,
                "choice": "auto",
            },
            {
                "test_aim": 'Parallel function_calls with "auto"',
                "messages": [
                    {
                        "role": "user",
                        "content": "What is the weather in Istanbul and Singapore respectively?",
                    }
                ],
                "call_mode": "functions",
                "code_interpreter": False,
                "choice": "auto",
            },
            {
                "test_aim": 'Normal text gen + tool_calls with "auto"',
                "messages": [
                    {
                        "role": "user",
                        "content": "What is the weather in Istanbul? Answer this question: 'How are you?', before checking the weather.",
                    }
                ],
                "call_mode": "tools",
                "code_interpreter": False,
                "choice": "auto",
            },
            {
                "test_aim": 'Normal text gen with "none"',
                "messages": [
                    {"role": "user", "content": "What is the weather in Istanbul?"}
                ],
                "call_mode": "tools",
                "code_interpreter": False,
                "choice": "none",
            },
            {
                "test_aim": "tool_calls with tool_choice",
                "messages": [{"role": "user", "content": "How are you?"}],
                "call_mode": "tools",
                "code_interpreter": False,
                "choice": {
                    "type": "function",
                    "function": {"name": cls.default_functions[0]["name"]},
                },
            },
            {
                "test_aim": "function_call with function_call",
                "messages": [{"role": "user", "content": "How are you?"}],
                "call_mode": "functions",
                "code_interpreter": False,
                "choice": {"name": cls.default_functions[0]["name"]},
            },
            {
                "test_aim": 'parallel tool_calls with "required"',
                "messages": [
                    {
                        "role": "user",
                        "content": "What is the weather in Istanbul and Singapore respectively?",
                    }
                ],
                "call_mode": "tools",
                "code_interpreter": False,
                "choice": "required",
            },
            {
                "test_aim": 'code generation using "python" tool',
                "messages": [
                    {
                        "role": "user",
                        "content": "Use the Python tool to write a Python function that adds 2 integers.",
                    }
                ],
                "call_mode": "tools",
                "code_interpreter": True,
                "choice": "auto",
            },
            {
                "test_aim": 'Normal text generation (CoT) + code generation using "python" tool',
                "messages": [
                    {
                        "role": "user",
                        "content": "Write a Python function that adds 2 integers. Answer this question: 'How are you?', before using the python tool.",
                    }
                ],
                "call_mode": "tools",
                "code_interpreter": True,
                "choice": "auto",
            },
        ]
        cls.client = OpenAI()
        for i, test_case in enumerate(cls.request_handling_test_cases):
            response = call_openai_api(
                test_case=test_case,
                client=cls.client,
                model=cls.openai_model,
                default_tools=cls.default_tools,
                python_tool=cls.python_tool,
                default_functions=cls.default_functions,
            )
            tool_calls = response.choices[0].message.tool_calls
            if tool_calls and len(tool_calls) > 0:
                for j in range(len(tool_calls)):
                    if tool_calls[j].function.name == "python":
                        response.choices[0].message.tool_calls[j].function.arguments = (
                            json.loads(tool_calls[j].function.arguments)["code"]
                        )
            cls.request_handling_test_cases[i]["label"] = response

            response = call_openai_api(
                test_case=test_case,
                client=cls.client,
                model=cls.openai_model,
                default_tools=cls.default_tools,
                python_tool=cls.python_tool,
                default_functions=cls.default_functions,
                stream=True,
            )
            chunks = [chunk for chunk in response]
            cls.request_handling_test_cases[i]["stream_label"] = chunks

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_child_process(cls.process.pid)

    def __init__(self, *args, **kwargs):
        super(TestSglServer, self).__init__(*args, **kwargs)
        self.served_models = [
            # "meetkai/functionary-small-v2.4",
            # "meetkai/functionary-small-v2.5",
            "meetkai/functionary-small-v3.1",
            # "meetkai/functionary-small-v3.2",
        ]

    def _check_nonstreaming_response(self, pred, label):
        # Check if both label.id and pred.id start with the same prefix
        assert pred.id.startswith(label.id[: label.id.index("-")])
        # Check if objects are equal
        assert pred.object == label.object
        pred_content = pred.choices[0].message.content
        label_content = label.choices[0].message.content
        pred_tool_calls = pred.choices[0].message.tool_calls
        label_tool_calls = label.choices[0].message.tool_calls
        pred_fn_call = pred.choices[0].message.function_call
        label_fn_call = label.choices[0].message.function_call
        # Check if content is equal
        assert (pred_content is None) == (label_content is None)
        # Check if tool_calls are equal
        assert (pred_tool_calls is None) == (label_tool_calls is None)
        if label_tool_calls is not None:
            assert len(pred_tool_calls) == len(label_tool_calls)
            for pred_tool_call, label_tool_call in zip(
                pred_tool_calls, label_tool_calls
            ):
                assert isinstance(pred_tool_call, ChatCompletionMessageToolCall)
                assert pred_tool_call.id.startswith(
                    "call_"
                ) and label_tool_call.id.startswith("call_")
                assert pred_tool_call.type == label_tool_call.type
                assert pred_tool_call.function.name == label_tool_call.function.name
                assert pred_tool_call.function.arguments is not None
        # Check if function_calls are equal
        assert (pred_fn_call is None) == (label_fn_call is None)
        if label_fn_call is not None:
            assert isinstance(pred_fn_call, FunctionCall)
            assert pred_fn_call.name == label_fn_call.name
            assert pred_fn_call.arguments is not None
        # Check finish_reason
        assert pred.choices[0].finish_reason == label.choices[0].finish_reason

    def _check_streaming_response(self, pred, label):
        tool_call_id = -1
        for i, chunk in enumerate(pred):
            # Check if both label.id and pred.id start with the same prefix
            assert chunk.id.startswith(label[0].id[: label[0].id.index("-")])
            # Check if objects are equal
            assert chunk.object == label[0].object
            # Check if the assistant turn is in the first chunk only
            if i == 0:
                assert chunk.choices[0].delta.role == "assistant"
            else:
                assert chunk.choices[0].delta.role is None
            # Check if the finish_reason is in the last chunk only
            if i == len(pred) - 1:
                assert chunk.choices[0].finish_reason is not None
            else:
                assert chunk.choices[0].finish_reason is None
            # Check if only one of content, function_call or tool_calls is not None
            non_none_fields = [
                chunk.choices[0].delta.content is not None,
                chunk.choices[0].delta.function_call is not None,
                chunk.choices[0].delta.tool_calls is not None,
            ]
            if i == len(pred) - 1:
                assert sum(non_none_fields) == 0
            else:
                assert sum(non_none_fields) == 1
            # Check tool_calls
            if chunk.choices[0].delta.tool_calls is not None:
                call_type = chunk.choices[0].delta.tool_calls[0].type
                name = chunk.choices[0].delta.tool_calls[0].function.name
                args = chunk.choices[0].delta.tool_calls[0].function.arguments
                # Check name, arguments, call_type and index
                assert args is not None
                if len(args) == 0:
                    assert name is not None
                    assert call_type == "function"
                    tool_call_id += 1
                    assert chunk.choices[0].delta.tool_calls[0].index == tool_call_id
                else:
                    assert name is None
                    assert call_type is None
            # Check function_call
            if chunk.choices[0].delta.function_call is not None:
                name = chunk.choices[0].delta.function_call.name
                args = chunk.choices[0].delta.function_call.arguments
                assert args is not None
                if len(args) == 0:
                    assert name is not None
                else:
                    assert name is None

    def test_sgl_server(self):
        for model in self.served_models:
            self.process = popen_launch_sgl_server(
                model=model,
                base_url=self.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                context_length=4096,
                grammar_sampling=False,
            )

            self.client = OpenAI(base_url=f"{self.base_url}/v1", api_key="test")
            try:
                for test_case in self.request_handling_test_cases:
                    pred = call_openai_api(
                        test_case=test_case,
                        client=self.client,
                        model=model,
                        default_tools=self.default_tools,
                        python_tool=self.python_tool,
                        default_functions=self.default_functions,
                    )
                    label = test_case["label"]
                    self._check_nonstreaming_response(pred, label)
                    pred = call_openai_api(
                        test_case=test_case,
                        client=self.client,
                        model=model,
                        default_tools=self.default_tools,
                        python_tool=self.python_tool,
                        default_functions=self.default_functions,
                        stream=True,
                    )
                    pred = [chunk for chunk in pred]
                    label = test_case["stream_label"]
                    self._check_streaming_response(pred, label)
            except AssertionError:
                raise
            finally:
                if self.process:
                    kill_child_process(self.process.pid)
