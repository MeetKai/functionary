from functionary.prompt_template.base_template import PromptTemplate
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from functionary.prompt_template import prompt_utils
import json
import copy


class Qwen25PromptTemplate(PromptTemplate):
    version = "qwen2.5"
    start_of_turn = "<|im_start|>"
    end_of_turn = "<|im_end|>"

    def get_chat_template_jinja(self) -> str:
        path_prefix = "./functionary/prompt_template/jinja_templates/"
        with open(f"{path_prefix}{self.version}.txt", "r") as f:
            template = f.read()

        return template

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
                        code = tool_call["function"]["arguments"] # currently the code is in string format
                        tool_call["function"]["arguments"] = json.dumps({"code": code}, ensure_ascii=False)
            _messages.append(n_message)
            

        prompt = super().get_prompt_from_messages(
            messages=_messages,
            tools_or_functions=_tools,
            bos_token=bos_token,
            add_generation_prompt=add_generation_prompt,
        )
        return prompt

    def get_additional_tokens(self) -> List[str]:
        return [self.function_separator]

    def get_assistant_prefixes(self) -> List[str]:
        return [f"{self.start_of_turn}assistant\n"]

    def get_stop_tokens_for_generation(self) -> List[str]:
        return [self.end_of_turn]

    def get_force_function_call_prefix(self, function_name: str):
        return (
            """<tool_call>
{"name": "%s}"""
            % function_name
        )

    def pre_process_messages_before_inference(self, messages: List[Dict]) -> List[Dict]:
        """Order the tool results by the order of tool call ids

        Args:
            messages (List[Dict]): List of messages

        Returns:
            List[Dict]: List of messages
        """
        return prompt_utils.reorder_tool_messages_by_tool_call_ids(messages)

    def parse_assistant_response(
        self, llm_output: str, tool_choice: Any = None
    ) -> Dict:
        # first remove stop tokens if there exists
        for stop in self.get_stop_tokens_for_generation():
            if llm_output.endswith(stop):
                llm_output = llm_output[: -len(stop)]

        # add forced-function from tool_choice if exists
        if type(tool_choice) is not str and tool_choice is not None:
            llm_output = (
                self.get_force_function_call_prefix(tool_choice.function.name)
                + llm_output
            )
        elif tool_choice == "required":
            llm_output = self.function_separator + llm_output

        text_content = ""
        tool_call_strs = []

        # Split on tool call tags
        parts = llm_output.split("<tool_call>")

        if len(parts) > 0:
            # First part is the text content
            text_content = parts[0].strip()

            # Process remaining parts as tool calls
            for part in parts[1:]:
                if "</tool_call>" in part:
                    tool_call = part.split("</tool_call>")[0].strip()
                    if tool_call:
                        tool_call_strs.append(tool_call)
        tool_calls = []
        for tool_call_str in tool_call_strs:
            tool_call_dic = json.loads(tool_call_str)
            tool_calls.append(
                {
                    "type": "function",
                    "id": prompt_utils.get_random_tool_call_id(),
                    "function": {
                        "name": tool_call_dic["name"],
                        "arguments": json.dumps(
                            tool_call_dic["arguments"], ensure_ascii=False
                        ),
                    },
                }
            )

        return {
            "role": "assistant",
            "content": text_content if len(text_content) > 0 else None,
            "tool_calls": tool_calls,
        }
