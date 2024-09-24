import json
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from functionary.prompt_template import prompt_utils
from functionary.prompt_template.llama3_prompt_template_v3 import Llama3TemplateV3


class LlavaLlama(Llama3TemplateV3):
    version = "v3.llava_llama"
    # This token will be replaced with image_token_id (-200) after we tokenize the text
    image_token = "<|reserved_special_token_250|>"
