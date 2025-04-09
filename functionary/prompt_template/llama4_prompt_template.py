from functionary.prompt_template.base_template import PromptTemplate
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from functionary.prompt_template import prompt_utils
import copy 
import json 


class LLama4PromptTemplate(PromptTemplate):
    version = "llama4"
    
    def get_additional_tokens(self) -> List[str]:
        return []

    def get_assistant_prefixes(self) -> List[str]:
        return [f"<|header_start|>assistant<|header_end|>\n\n"]

    def pre_process_messages_before_inference(self, messages: List[Dict]) -> List[Dict]:
        """Order the tool results by the order of tool call ids

        Args:
            messages (List[Dict]): List of messages

        Returns:
            List[Dict]: List of messages
        """
        return prompt_utils.reorder_tool_messages_by_tool_call_ids(messages)

    def get_stop_tokens_for_generation(self) -> List[str]:
        return ["<|eot|>"]
    
    def get_force_function_call_prefix(self, function_name: str):
        return '<|python_start|><|python_end|>{"name": "' + function_name + '"}'

    def get_force_text_generation_prefix(self):
        return f""

    def get_tool_choice_required_prefix(self):
        return '<|python_start|>'
    
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
        # handle code_interpreter
        _tools = []
        if tools_or_functions:
            for tool in tools_or_functions:
                if tool["type"] == "code_interpreter":
                    _tools.append(
                        {
                            "type": "function",
                            "function": {
                                "name": "python",
                                "description": "This tool is used to execute python code. Code will be executed in a stateful Jupyter notebook environment. Python will respond with the output of the execution or time out after 60.0 seconds. The drive at '/mnt/data' can be used to save and persist user files.",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "code": {
                                            "type": "string",
                                            "description": "The python code to run",
                                        }
                                    },
                                },
                            },
                        }
                    )
                else:
                    _tools.append(tool)

        # find the assistant message that tool_call is python
        _messages = []
        for message in messages:
            n_message = copy.deepcopy(message)
            tool_calls = n_message.get("tool_calls", []) or []
            if len(tool_calls) > 0:
                for tool_call in tool_calls:
                    if tool_call["function"]["name"] == "python":
                        arguments = tool_call["function"][
                            "arguments"
                        ]  # currently the code is in string format
                        # check if argument is a valid JSON string or python code
                        try:  # if this is a valid JSON string --> no need to change anything
                            json.loads(arguments)
                        except:
                            tool_call["function"]["arguments"] = json.dumps(
                                {"code": arguments}, ensure_ascii=False
                            )
            _messages.append(n_message)
        prompt = super().get_prompt_from_messages(
            messages=_messages,
            tools_or_functions=_tools,
            bos_token=bos_token,
            add_generation_prompt=add_generation_prompt,
        )
        return prompt