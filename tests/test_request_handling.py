import asyncio
import json
import unittest

from functionary.openai_types import ChatCompletionRequest
from functionary.vllm_inference import check_all_errors


def convert_jsonresponse_to_json(obj):
    return json.loads(obj.body.decode())


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
            function_call="required"
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
            tool_choice="requiredd"
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
            function_call="none"
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
            tool_choice="none"
        )
        response = convert_jsonresponse_to_json(
            await check_all_errors(request=request, served_model=self.served_model)
        )
        self.assertEqual(
            response["message"],
            "Invalid value for 'tool_choice': 'tool_choice' is only allowed when 'tools' are specified.",
            "Edge case handling failed: tool_choice provided without providing tools",
        )


if __name__ == "__main__":
    unittest.main()
