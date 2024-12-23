from typing import Dict, List, Optional

from functionary.prompt_template.llama31_prompt_template import Llama31Template


class Llama31ReasoningTemplate(Llama31Template):
    version = "v3-llama3.1-reasoning"

    def get_prompt_from_messages(
        self,
        messages: List[Dict],
        tools_or_functions: Optional[List[Dict]] = None,
        bos_token: Optional[str] = "",
        add_generation_prompt: bool = False,
    ) -> str:
        reasoning_tool = {"type": "reasoning"}
        return super().get_prompt_from_messages(messages, tools_or_functions + [reasoning_tool], bos_token, add_generation_prompt)