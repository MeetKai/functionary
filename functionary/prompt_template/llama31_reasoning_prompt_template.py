from typing import Dict, List

from functionary.prompt_template.llama31_prompt_template import Llama31Template


class Llama31ReasoningTemplate(Llama31Template):
    version = "v3-llama3.1-reasoning"

    def get_prompt_from_messages(self, messages: List[Dict], tools_or_functions: List[Dict] | None = None, bos_token: str | None = "", add_generation_prompt: bool = False) -> str:
        reasoning_tool = {"type": "reasoning"}
        return super().get_prompt_from_messages(messages, tools_or_functions + [reasoning_tool], bos_token, add_generation_prompt)