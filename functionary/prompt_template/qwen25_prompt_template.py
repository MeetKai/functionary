import json
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from functionary.openai_types import Function, Tool
from functionary.prompt_template import prompt_utils
from functionary.prompt_template.base_template import PromptTemplate
import copy


class Qwen25Template(PromptTemplate):
    version = "qwen2.5"
    start_of_turn = "<|im_start|>"
    end_of_turn = "<|im_end|>"
    chat_template = None

    def get_additional_tokens(self) -> List[str]:
        return []

    def get_assistant_prefixes(self) -> List[str]:
        return [f"{self.start_of_turn}assistant\n"]

    def pre_process_messages_before_inference(self, messages: List[Dict]) -> List[Dict]:
        """Order the tool results by the order of tool call ids

        Args:
            messages (List[Dict]): List of messages

        Returns:
            List[Dict]: List of messages
        """
        return prompt_utils.reorder_tool_messages_by_tool_call_ids(messages)

    def get_stop_tokens_for_generation(self) -> List[str]:
        return [self.end_of_turn]

    def parse_assistant_response(
        self, llm_output: str, tool_choice: Any = None
    ) -> Dict:
        # first remove stop tokens if there exists
        for stop in self.get_stop_tokens_for_generation():
            if llm_output.endswith(stop):
                llm_output = llm_output[: -len(stop)]

        # add forced-function from tool_choice if exists
        llm_output = (
            self.get_generation_prefix_for_tool_choice(tool_choice) + llm_output
        )

        tool_calls = []
        text_response = ""

        while len(llm_output) > 0:
            start_tool_call_index = llm_output.find("<tool_call>")
            if start_tool_call_index >= 0:
                end_index = llm_output.find("</tool_call>", start_tool_call_index)
                if end_index >= 0:
                    json_between = llm_output[
                        start_tool_call_index + len("<tool_calls>") : end_index
                    ]
                    func_call = json.loads(json_between)
                    tool_calls.append(
                        {
                            "type": "function",
                            "id": prompt_utils.get_random_tool_call_id(),
                            "function": {
                                "name": func_call["name"],
                                "arguments": json.dumps(
                                    func_call["arguments"], ensure_ascii=False
                                ),
                            },
                        }
                    )
                    index = end_index + len("</tool_call>")

                    text_response += llm_output[:start_tool_call_index].strip()
                    llm_output = llm_output[index:]
                else:  # cannot find </tool_call> at the end
                    text_response += llm_output
                    llm_output = ""
            else:  # cannot find <tool_call>
                text_response += llm_output
                llm_output = ""

        if not text_response:
            text_response = None
        elif len(text_response.strip()) == 0:
            text_response = None

        if not tool_calls:
            tool_calls = None

        return {"role": "assistant", "content": text_response, "tool_calls": tool_calls}

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
        # qwen 2.5 use transformers chat template, need to convert argument string --> dictionary, this is noted in: https://huggingface.co/docs/transformers/main/en/chat_templating#a-complete-tool-use-example
        # If you’re familiar with the OpenAI API, you should pay attention to an important difference here - the tool_call is a dict, but in the OpenAI API it’s a JSON string. Passing a string may cause errors or strange model behaviour!
        new_messages = copy.deepcopy(messages)
        for message in new_messages:
            tool_calls = message.get("tool_calls", [])
            if tool_calls:
                for tool_call in tool_calls:
                    if type(tool_call["function"]["arguments"]) is str:
                        if tool_call["function"]["name"] != "python":
                            tool_call["function"]["arguments"] = json.loads(
                                tool_call["function"]["arguments"]
                            )
                        else:
                            tool_call["function"] = {
                                "name": "python",
                                "arguments": {
                                    "code": tool_call["function"]["arguments"]
                                },
                            }
        # check if contain code_interpreter, replace with python
        new_tools = copy.deepcopy(tools_or_functions)
        if tools_or_functions is not None and len(tools_or_functions) == 0:
            new_tools = None

        if new_tools:
            for tool in new_tools:
                if tool["type"] == "code_interpreter":
                    tool["type"] = "function"
                    tool["function"] = {
                        "name": "python",
                        "description": 'When you send a message containing Python code to python, it will be executed in a stateful Jupyter notebook environment. python will respond with the output of the execution or time out after 60.0 seconds. The drive at "/mnt/data" can be used to save and persist user files.',
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "code": {"type": "string", "description": "Python code"}
                            },
                            "required": ["code"],
                        },
                    }

        prompt = self._jinja_template.render(
            messages=new_messages,
            tools=new_tools,
            bos_token=bos_token,
            add_generation_prompt=add_generation_prompt,
        )

        return prompt
    
    def get_chat_template_jinja(self) -> str:
        if self.chat_template is None:
            jinja_template_file = "./functionary/prompt_template/jinja_templates/qwen2.5.txt"
            with open(jinja_template_file, "r") as f:
                self.chat_template = f.read()
        return self.chat_template