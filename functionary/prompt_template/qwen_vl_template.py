from typing import Any, Dict, List

from functionary.prompt_template import prompt_utils
from functionary.prompt_template.base_template import PromptTemplate
from functionary.prompt_template.llama3_prompt_template_v3 import Llama3TemplateV3


class Qwen2VLTemplate(Llama3TemplateV3):
    version = "qwen2-vl"
    function_separator = ">>>"

    def get_additional_tokens(self) -> List[str]:
        return []

    def get_assistant_prefixes(self) -> List[str]:
        return [f"<|im_start|>assistant\n{self.function_separator}"]

    def get_stop_tokens_for_generation(self) -> List[str]:
        return ["<|im_end|>"]