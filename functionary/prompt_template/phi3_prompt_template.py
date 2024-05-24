from functionary.prompt_template.base_template import PromptTemplate
from functionary.prompt_template.llama3_prompt_template import Llama3Template
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from functionary.prompt_template import prompt_utils


class Phi3Template(Llama3Template):
    version = "v2.phi3"
    function_separator = "<|placeholder6|>"

    def get_additional_tokens(self) -> List[str]:
        return []

    def get_assistant_prefixes(self) -> List[str]:
        return ["<|assistant|>\n"]

    def get_stop_tokens_for_generation(self) -> List[str]:
        return ["<|end|>"]

    def pre_process_messages_before_inference(self, messages: List[Dict]) -> List[Dict]:
        """Order the tool results by the order of tool call ids

        Args:
            messages (List[Dict]): List of messages

        Returns:
            List[Dict]: List of messages
        """
        return prompt_utils.reorder_tool_messages_by_tool_call_ids(messages)

    def convert_message_to_prompt(self, message: Dict) -> str:
        role = message["role"]
        content = message.get("content", None)
        if role == "tool":
            tool_name = message["name"]
            content = f"name={tool_name}\n{content}"

        prompt_template = "<|%s|>\n{text}<|end|>\n" % role

        if role in ["user", "system", "tool"]:
            return prompt_template.format(text=content)

        assert role == "assistant"
        tool_calls = message.get("tool_calls", [])
        if tool_calls is None:
            tool_calls = []

        if content is None and len(tool_calls) == 0:
            return f"<|assistant|>\n"

        if content is not None and len(tool_calls) == 0:  # text-only
            return prompt_template.format(text=content)

        tool_call_prompts = []
        for tool_call in tool_calls:
            arguments = tool_call["function"]["arguments"]
            tool_name = tool_call["function"]["name"]
            tool_prompt = f"{tool_name}\n{arguments}"
            tool_call_prompts.append(tool_prompt)

        if (
            content is None and len(tool_calls) > 0
        ):  # function call only (<sep>fc1<sep>fc2<sep>)
            tool_call_content = self.function_separator + self.function_separator.join(
                tool_call_prompts
            )
            return prompt_template.format(text=tool_call_content)

        # Here is the case contains both text-response and tool_calls (content<sep>fc1<sep>fc2<sep>)
        total_content = (
            content
            + self.function_separator
            + self.function_separator.join(tool_call_prompts)
        )
        return prompt_template.format(text=total_content)
