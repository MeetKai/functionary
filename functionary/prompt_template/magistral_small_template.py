from functionary.prompt_template.base_template import PromptTemplate
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from functionary.prompt_template import prompt_utils
from functionary.openai_types import Function, Tool
import json
import copy
import math
import re


class MagistralSmallPromptTemplate(PromptTemplate):
    version = "magistral_small"
    
    def get_chat_template_jinja(self) -> str:
        path_prefix = "./functionary/prompt_template/jinja_templates/"
        with open(f"{path_prefix}{self.version}.txt", "r") as f:
            template = f.read()

        return template
    
    def get_additional_tokens(self) -> List[str]:
        return []

    def get_assistant_prefixes(self) -> List[str]:
        return [f"[/INST]"]

    def get_stop_tokens_for_generation(self) -> List[str]:
        return ["</s>"]

    def parse_assistant_response(self, llm_output: str, tool_choice: Any | None) -> Dict:
        return {
            "role": "assistant",
            "content": llm_output,
        }
