from functionary.prompt_template.base_template import PromptTemplate, PYTHON_RUN_SYS_MSG
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from functionary.prompt_template import prompt_utils
from functionary.schema import generate_schema_from_functions

SYSTEM_CONTENT = """You are capable of executing available function(s) if required.
Only execute function(s) when absolutely necessary.
Ask for the required input to:recipient==all
Use JSON for function arguments.
Respond in this format:
>>>${recipient}
${content}
Available functions:
"""


class Qwen2PromptTemplateV2(PromptTemplate):
    version = "v2.qwen2_v2"
    function_separator = ">>>"
    start_of_turn = "<|im_start|>"
    end_of_turn = "<|im_end|>"


    def get_additional_tokens(self) -> List[str]:
        return [">>>"]

    def get_assistant_prefixes(self) -> List[str]:
        return [f"{self.start_of_turn}assistant\n{self.function_separator}"]

    def get_stop_tokens_for_generation(self) -> List[str]:
        return [self.end_of_turn]
    
    def get_force_function_call_prefix(self, function_name: str):
        return f"{self.function_separator}{function_name}\n"

    def pre_process_messages_before_inference(self, messages: List[Dict]) -> List[Dict]:
        """Order the tool results by the order of tool call ids

        Args:
            messages (List[Dict]): List of messages

        Returns:
            List[Dict]: List of messages
        """
        return prompt_utils.reorder_tool_messages_by_tool_call_ids(messages)

    def convert_message_to_prompt(self, message: Dict) -> str:
        role = message["role"]
        content = message.get("content", None)
        if role == "tool":
            tool_name = message["name"]
            content = f"name={tool_name}\n{content}"

        prompt_template = f"{self.start_of_turn}{role}\n" + "{text}" + f"{self.end_of_turn}\n"

        if role in ["user", "system", "tool"]:
            return prompt_template.format(text=content)

        assert role == "assistant"
        tool_calls = message.get("tool_calls", [])
        if tool_calls is None:
            tool_calls = []

        if content is None and len(tool_calls) == 0: # at inference
            return f"{self.start_of_turn}assistant\n{self.function_separator}"

        if content is not None:
            tool_calls = [{"function": {"arguments": content, "name": "all"}}] + tool_calls
        

        tool_call_prompts = []
        for tool_call in tool_calls:
            arguments = tool_call["function"]["arguments"]
            tool_name = tool_call["function"]["name"]
            tool_prompt = f"{tool_name}\n{arguments}"
            tool_call_prompts.append(tool_prompt)

        # Here is the case contains both text-response and tool_calls (content<sep>fc1<sep>fc2<sep>)
        total_content = self.function_separator + self.function_separator.join(tool_call_prompts)
        return prompt_template.format(text=total_content)

    def parse_assistant_response(
        self, llm_output: str, tool_choice: Any = None
    ) -> Dict:
        # first remove stop tokens if there exists
        for stop in self.get_stop_tokens_for_generation():
            if llm_output.endswith(stop):
                llm_output = llm_output[: -len(stop)]

        # add forced-function from tool_choice if exists
        if type(tool_choice) is not str and tool_choice is not None:
            llm_output = (
                self.get_force_function_call_prefix(tool_choice.function.name)
                + llm_output
            )
        elif tool_choice == "required":
            llm_output = self.function_separator + llm_output

        chunks = llm_output.split(self.function_separator)
        chunks = [chunk.strip() for chunk in chunks if len(chunk.strip()) > 0]

        tool_calls = []
        text_content = None

        for chunk in chunks:
            # format: function_name\narguments<end_of_functioncall>
            index = chunk.find("\n")
            func_name = chunk[:index].strip()
            arguments = chunk[index + 1 :].strip()
            if func_name == "all":
                text_content = arguments
            else:
                tool_calls.append(
                    {
                        "function": {"name": func_name, "arguments": arguments},
                        "id": prompt_utils.get_random_tool_call_id(),
                        "type": "function",
                    }
                )
        if len(tool_calls) == 0:
            tool_calls = None
            
        return {"role": "assistant", "content": text_content, "tool_calls": tool_calls}

    
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
            0, {"role": "system", "content": SYSTEM_CONTENT + generate_schema_from_functions(functions)}
        )
        if is_code_interpreter:
            messages_clone.insert(1, {"role": "system", "content": PYTHON_RUN_SYS_MSG})

        return messages_clone