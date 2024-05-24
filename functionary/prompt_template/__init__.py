from typing import Any

from functionary.prompt_template.base_template import SYSTEM_MESSAGE, PromptTemplate
from functionary.prompt_template.llama3_prompt_template import Llama3Template
from functionary.prompt_template.prompt_template_v1 import PromptTemplateV1
from functionary.prompt_template.prompt_template_v2 import PromptTemplateV2
from functionary.prompt_template.phi3_prompt_template import Phi3Template

_prompt_templates = [PromptTemplateV1(), PromptTemplateV2(), Llama3Template(), Phi3Template()]
_prompt_dic = {}
for prompt_template in _prompt_templates:
    _prompt_dic[prompt_template.version] = prompt_template


def get_default_prompt_template() -> PromptTemplate:
    """Return default prompt template to be used

    Returns:
        _type_: _description_
    """
    return PromptTemplateV2.get_prompt_template()


def get_prompt_template_by_version(version: str) -> PromptTemplate:
    return _prompt_dic[version]


def get_prompt_template_from_tokenizer(tokenizer: Any) -> PromptTemplate:
    """This function will determine the prompt template based on tokenizer.
    Under the hood, this function will check if tokenizer contains some special tokens from template or not

    Args:
        tokenizer (Any): Tokenizer

    Returns:
        _type_: _description_
    """
    token_ids = tokenizer.encode(Llama3Template.function_separator, add_special_tokens=False)
    if len(token_ids) == 1:
        return _prompt_dic[Llama3Template.version]
    
    token_ids = tokenizer.encode(Phi3Template.function_separator, add_special_tokens=False)
    if len(token_ids) == 1:
        return _prompt_dic[Phi3Template.version]

    token_ids = tokenizer.encode(PromptTemplateV1.start_function, add_special_tokens=False)
    if token_ids[0] in [29871, 28705]:
        token_ids = token_ids[1:]
    if len(token_ids) == 1:
        return _prompt_dic[PromptTemplateV1.version]
    return _prompt_dic[PromptTemplateV2.version]
