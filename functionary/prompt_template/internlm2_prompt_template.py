import json
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from functionary.openai_types import Function, Tool
from functionary.prompt_template import prompt_utils
from functionary.prompt_template.llama3_prompt_template_v3 import Llama3TemplateV3
from functionary.schema import generate_schema_from_functions


class InternLMChat(Llama3TemplateV3):
    version = "internlm2-chat"
    img_token = "<img>"
    start_of_turn = "<|im_start|>"
    eos_token = "<|im_end|>"
    function_separator = ">>>"

    def get_assistant_prefixes(self) -> List[str]:
        return [f"{self.start_of_turn}assistant\n{self.function_separator}"]

    def get_stop_tokens_for_generation(self) -> List[str]:
        return [self.eos_token]
