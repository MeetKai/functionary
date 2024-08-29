import json
from typing import Dict, List, Optional

from functionary.prompt_template.llama3_prompt_template_v3 import (
    Llama3TemplateV3,
    SYSTEM_CONTENT,
    PYTHON_RUN_SYS_MSG,
)


def get_functions_in_json_schema(functions):
    if len(functions) == 0:
        return "No functions provided"
    result = []
    for function in functions:
        result.append(json.dumps(function, ensure_ascii=False) + "\n------------------")
    return "\n".join(result)
    
    
class Llama3JsonSchema(Llama3TemplateV3):
    version = "v3.json"

    def inject_system_messages_based_on_tools(
        self, messages: List[Dict], tools_or_functions: Optional[List[Dict]] = None
    ) -> List[Dict]:
        """This will be used to add Default system message, code-interpreter system message if needed

        Args:
            messages (List[Dict]): List of messages
            tools_or_functions (Optional[List[Dict]], optional): List of tools, functions. Defaults to None.

        Returns:
            List[Dict]: _description_
        """
        messages_clone = messages.copy()  # To avoid modifying the original list

        functions = []
        is_code_interpreter = False
        if tools_or_functions is not None:
            for item in tools_or_functions:
                if (
                    "function" in item and item["function"] is not None
                ):  #  new data format: tools: [{"type": xx, "function": xxx}]
                    functions.append(item["function"])
                elif "type" in item and item["type"] == "code_interpreter":
                    is_code_interpreter = True
                else:
                    functions.append(item)  #  old format

        messages_clone.insert(
            0,
            {
                "role": "system",
                "content": SYSTEM_CONTENT + json.dumps(functions, ensure_ascii=False)
            },
        )
        if is_code_interpreter:
            messages_clone.insert(1, {"role": "system", "content": PYTHON_RUN_SYS_MSG})

        return messages_clone
