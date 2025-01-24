from functionary.prompt_template.llama3_prompt_template_v3 import Llama3TemplateV3
from typing import List


class DeepSeekR1Template(Llama3TemplateV3):
    version = "v3-deepseek-r1"
    eos_token = "<｜end▁of▁sentence｜>"

    def get_stop_tokens_for_generation(self) -> List[str]:
        return [self.eos_token]
