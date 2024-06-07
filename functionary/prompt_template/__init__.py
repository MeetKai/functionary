from typing import Any

from functionary.prompt_template.base_template import SYSTEM_MESSAGE, PromptTemplate
from functionary.prompt_template.llama3_prompt_template import Llama3Template
from functionary.prompt_template.prompt_template_v1 import PromptTemplateV1
from functionary.prompt_template.prompt_template_v2 import PromptTemplateV2
from functionary.prompt_template.qwen2_prompt_template import Qwen2PromptTemplate


def get_default_prompt_template() -> PromptTemplate:
    """Return default prompt template to be used

    Returns:
        _type_: _description_
    """
    return PromptTemplateV2.get_prompt_template()


def get_prompt_template_by_version(version: str) -> PromptTemplate:
    if version == "v2.llama3":
        return Llama3Template.get_prompt_template()

    if version == "v1":
        return PromptTemplateV1.get_prompt_template()
    
    if version == "v2.qwen2":
        return Qwen2PromptTemplate.get_prompt_template()
    
    assert version == "v2"
    return PromptTemplateV2.get_prompt_template()


def get_prompt_template_from_tokenizer(tokenizer: Any) -> PromptTemplate:
    """This function will determine the prompt template based on tokenizer.
    Under the hood, this function will check if tokenizer contains some special tokens from template or not

    Args:
        tokenizer (Any): Tokenizer

    Returns:
        _type_: _description_
    """
    p1 = PromptTemplateV1.get_prompt_template()
    p2 = PromptTemplateV2.get_prompt_template()
    p3 = Llama3Template.get_prompt_template()
    p4 = Qwen2PromptTemplate.get_prompt_template()
    
    token_ids = tokenizer.encode(p4.function_separator, add_special_tokens=False)
    if len(token_ids) == 1:
        return p4
    
    token_ids = tokenizer.encode(p3.function_separator, add_special_tokens=False)
    if len(token_ids) == 1:
        return p3

    token_ids = tokenizer.encode(p1.start_function, add_special_tokens=False)
    if token_ids[0] in [29871, 28705]:
        token_ids = token_ids[1:]
    if len(token_ids) == 1:
        return p1
    return p2
