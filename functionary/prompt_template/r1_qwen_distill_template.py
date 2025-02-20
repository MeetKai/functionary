import datetime
import json
import re
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from functionary.openai_types import Function, Tool
from functionary.prompt_template import prompt_utils
from functionary.prompt_template.base_template import PromptTemplate


class R1QwenDistillTemplate(PromptTemplate):
    version = "r1-qwen-distill"

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