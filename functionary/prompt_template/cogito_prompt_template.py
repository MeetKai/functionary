from functionary.prompt_template.base_template import PromptTemplate
from functionary.prompt_template.qwen25_text_only_template import (
    Qwen25TextOnlyPromptTemplate,
)
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from functionary.prompt_template import prompt_utils
from functionary.openai_types import Function, Tool


class CogitoPromptTemplate(Qwen25TextOnlyPromptTemplate):
    version = "cogito"

    def get_chat_template_jinja(self) -> str:
        path_prefix = "./functionary/prompt_template/jinja_templates/"
        with open(f"{path_prefix}cogito.txt", "r") as f:
            template = f.read()
        return template
