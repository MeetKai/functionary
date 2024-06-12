from typing import Any, List

from functionary.prompt_template.base_template import SYSTEM_MESSAGE, PromptTemplate
from functionary.prompt_template.llama3_prompt_template import Llama3Template
from functionary.prompt_template.prompt_template_v1 import PromptTemplateV1
from functionary.prompt_template.prompt_template_v2 import PromptTemplateV2
from functionary.prompt_template.qwen2_prompt_template import Qwen2PromptTemplate
from functionary.prompt_template.qwen2_prompt_template_v2 import Qwen2PromptTemplateV2
from functionary.prompt_template.llama3_prompt_plate_v3 import Llama3TemplateV3


def get_default_prompt_template() -> PromptTemplate:
    """Return default prompt template to be used

    Returns:
        _type_: _description_
    """
    return PromptTemplateV2.get_prompt_template()


def get_prompt_template_by_version(version: str) -> PromptTemplate:
    if version == "v3.llama3":
        return Llama3TemplateV3.get_prompt_template()
    
    if version == "v2.llama3":
        return Llama3Template.get_prompt_template()

    if version == "v1":
        return PromptTemplateV1.get_prompt_template()
    
    if version == "v2.qwen2":
        return Qwen2PromptTemplate.get_prompt_template()
    
    if version == "v2.qwen2_v2":
        return Qwen2PromptTemplateV2.get_prompt_template()
    
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
    p5 = Qwen2PromptTemplateV2.get_prompt_template()
    p6 = Llama3TemplateV3.get_prompt_template()
    
    token_ids = tokenizer.encode(p4.function_separator, add_special_tokens=False)
    if len(token_ids) == 1:
        return p4
    
    token_ids = tokenizer.encode(p3.function_separator, add_special_tokens=False)
    if len(token_ids) == 1 and token_ids[0] == 128254: # based on llam3
        if p3.function_separator in tokenizer.chat_template:
            return p3
        else:
            return p6
    
    token_ids = tokenizer.encode(p2.from_token, add_special_tokens=False)
    if len(token_ids) == 1:
        return p2

    token_ids = tokenizer.encode(p1.start_function, add_special_tokens=False)
    if token_ids[0] in [29871, 28705]:
        token_ids = token_ids[1:]
    if len(token_ids) == 1:
        return p1
    
    return p5



def get_available_prompt_template_versions() -> List[PromptTemplate]:
    """This function will get all the available prompt templates in the module.

    Returns:
        List[PromptTemplate]: All the prompt template objects
    """

    all_templates_cls = PromptTemplate.__subclasses__()
    # Remove PromptTemplateV1 as it is deprecated and not needed
    all_templates_cls.remove(PromptTemplateV1)

    all_templates_obj = [
        template_cls.get_prompt_template() for template_cls in all_templates_cls
    ]

    return all_templates_obj
