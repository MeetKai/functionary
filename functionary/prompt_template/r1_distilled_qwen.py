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

    def get_chat_template_jinja(self) -> str:
        if self.chat_template is None:
            jinja_template_file = (
                "./functionary/prompt_template/jinja_templates/r1_distilled_qwen.txt"
            )
            with open(jinja_template_file, "r") as f:
                self.chat_template = f.read()
        return self.chat_template
