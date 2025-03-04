import datetime
import json
import re
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from functionary.openai_types import Function, Tool
from functionary.prompt_template import prompt_utils
from functionary.prompt_template.qwen25_text_only_template import (
    Qwen25TextOnlyPromptTemplate,
)
import copy


class R1DistilledQwen(Qwen25TextOnlyPromptTemplate):
    version = "r1_distilled_qwen"
    chat_template = None

    def get_additional_tokens(self) -> List[str]:
        return []

    def get_assistant_prefixes(self) -> List[str]:
        return [f"<｜Assistant｜>"]

    def pre_process_messages_before_inference(self, messages: List[Dict]) -> List[Dict]:
        """Order the tool results by the order of tool call ids

        Args:
            messages (List[Dict]): List of messages

        Returns:
            List[Dict]: List of messages
        """
        return prompt_utils.reorder_tool_messages_by_tool_call_ids(messages)

    def get_stop_tokens_for_generation(self) -> List[str]:
        return ["<｜end▁of▁sentence｜>"]

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

    def get_chat_template_jinja(self) -> str:
        if self.chat_template is None:
            jinja_template_file = (
                "./functionary/prompt_template/jinja_templates/r1_distilled_qwen.txt"
            )
            with open(jinja_template_file, "r") as f:
                self.chat_template = f.read()
        return self.chat_template
