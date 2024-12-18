import datetime
import json
import re
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from functionary.openai_types import Function, Tool
from functionary.prompt_template import prompt_utils
from functionary.prompt_template.llama31_prompt_template import Llama31Template


def return_multi_tool_use():
    return {
        "type": "function",
        "function": {
            "name": "multi_tool_use",
            "description": "This tool serves as a wrapper for utilizing multiple tools. Each tool that can be used must be specified in the tool sections.\nEnsure that the parameters provided to each tool are valid according to that tool's specification.\nUse this function to run multiple tools simultaneously, but only if they can operate in parallel.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tool_uses": {
                        "type": "array",
                        "description": "The tools to be executed in parallel. NOTE: only functions tools are permitted",
                        "items": {
                            "type": "object",
                            "properties": {
                                "recipient_name": {
                                    "type": "string",
                                    "description": "The name of the tool to use. The format should either be just the name of the tool, or in the format namespace.function_name for plugin and function tools.",
                                },
                                "parameters": {
                                    "type": "object",
                                    "additionalProperties": True,
                                    "description": "The parameters to pass to the tool. Ensure these are valid according to the tool's own specifications.",
                                },
                            },
                            "required": ["recipient_name", "parameters"],
                        },
                    },
                },
                "required": ["tool_uses"],
            },
        },
    }


def merge_tool_calls(tool_calls: list[dict]) -> dict:
    tool_uses = []
    for tool_call in tool_calls:
        tool_uses.append(
            {
                "recipient_name": tool_call["function"]["name"],
                "parameters": json.loads(tool_call["function"]["arguments"]),
            }
        )
    return {
        "type": "function",
        "function": {
            "name": "multi_tool_use",
            "arguments": json.dumps({"tool_uses": tool_uses}, ensure_ascii=False),
        },
    }


def convert_parallel_to_multi_tool_use_example(
    messages: List[Dict], tools: List[Dict]
) -> Tuple[List[Dict], List[Dict]]:
    # add multi_tool_use tool
    all_tools = tools + [return_multi_tool_use()]
    merged_messages = []
    for message in messages:
        tool_calls = message.get("tool_calls", []) or []
        if len(tool_calls) > 0:
            if len(tool_calls) > 1:
                # print("mesage 0: ", messages[0]["content"])
                merged_tool_call = merge_tool_calls(tool_calls)
                merged_messages.append(
                    {
                        "role": "assistant",
                        "content": message.get("content", None),
                        "tool_calls": [merged_tool_call],
                    }
                )
            else:
                merged_messages.append(message)
        else:
            merged_messages.append(message)

    return all_tools, merged_messages


class MultiToolUseLlama31Template(Llama31Template):
    version = "v3-llama3.1-multi-tool-use"

    def get_prompt_from_messages(
        self,
        messages: List[Dict],
        tools_or_functions: Optional[List[Dict]] = None,
        bos_token: Optional[str] = "",
        add_generation_prompt: bool = False,
    ) -> str:
        """This function is used to get the complete prompt for list of messages

        Args:
            messages (List[Dict]): List of messages
            tools_or_functions (Optional[List[Dict]], optional): List of tools or functions. Defaults to None.

        Returns:
            str: the prompt for inference/training
        """
        if not tools_or_functions:
            all_tools, merged_messages = [], messages
        else:
            all_tools, merged_messages = convert_parallel_to_multi_tool_use_example(
                messages, tools_or_functions
            )
        return super().get_prompt_from_messages(
            merged_messages, all_tools, bos_token, add_generation_prompt
        )

    def parse_assistant_response(
        self, llm_output: str, tool_choice: Any = None
    ) -> Dict:
        assistant_response = super().parse_assistant_response(llm_output, tool_choice)
        tool_calls = assistant_response.get("tool_calls", [])
        n_tool_calls = []
        if tool_calls:
            for tool_call in tool_calls:
                if tool_call["function"]["name"] == "multi_tool_use":
                    sub_tool_calls = []
                    tool_use_list = json.loads(tool_call["function"]["arguments"])
                    for tool_use in tool_use_list:
                        sub_tool_calls.append(
                            {
                                "id": prompt_utils.get_random_tool_call_id(),
                                "type": "function",
                                "function": {
                                    "name": tool_use["recipient_name"],
                                    "arguments": json.dumps(
                                        tool_use["parameters"], ensure_ascii=False
                                    ),
                                },
                            }
                        )
                    n_tool_calls.extend(sub_tool_calls)
                else:
                    n_tool_calls.append(tool_call)
        return {
            "role": "assistant",
            "content": assistant_response.get("content", None),
            "tool_calls": n_tool_calls,
        }
