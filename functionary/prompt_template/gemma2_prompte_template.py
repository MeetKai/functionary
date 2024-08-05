import json
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from functionary.openai_types import Function, Tool
from functionary.prompt_template import prompt_utils
from functionary.prompt_template.base_template import PYTHON_RUN_SYS_MSG, PromptTemplate
from functionary.schema import generate_schema_from_functions


class Gemma2PromptTemplate(PromptTemplate):
    version = "v3.gemma2"
    function_separator = ">>>"
    start_header = "<start_of_turn>"
    eos_token = "<end_of_turn>"
    # This token splits between function name and parameters
    fn_param_sep_token = "\n"

    def get_additional_tokens(self) -> List[str]:
        return []

    def get_assistant_prefixes(self) -> List[str]:
        return [
            f"{self.start_header}model\n{self.function_separator}"
        ]

    def get_stop_tokens_for_generation(self) -> List[str]:
        return [self.eos_token]

    def get_force_function_call_prefix(self, function_name: str):
        return f"{function_name}\n"

    def get_start_of_function_call_token(self) -> str:
        return self.function_separator

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

        # comment this as currently the Llama-70b was trained using this
        # if role == "tool":
        #     tool_name = message["name"]
        #     content = f"name={tool_name}\n{content}"

        prompt_template = (
            f"{self.start_header}" + "{role}\n"
            + "{text}"
            + self.eos_token + "\n"
        )

        if role in ["user", "system", "tool"]:
            return prompt_template.format(text=content, role=role)

        assert role == "assistant", f"role must be assistant, but: {role}"
        tool_calls = message.get("tool_calls", [])
        if tool_calls is None:
            tool_calls = []

        if content is None and len(tool_calls) == 0:  # inference time
            return f"{self.start_header}model\n{self.function_separator}"

        if content is not None:  # text-only
            tool_calls = [
                {"function": {"name": "all", "arguments": content}}
            ] + tool_calls

        # list of text representing function calls: {function_name}\n{arguments}
        tool_call_prompts = []
        for tool_call in tool_calls:
            arguments = tool_call["function"]["arguments"]
            tool_name = tool_call["function"]["name"]
            tool_prompt = f"{tool_name}\n{arguments}"
            tool_call_prompts.append(tool_prompt)

        # join all function calls
        total_content = self.function_separator + self.function_separator.join(
            tool_call_prompts
        )
        return prompt_template.format(role="model", text=total_content)

    def parse_assistant_response(
        self, llm_output: str, tool_choice: Any = None
    ) -> Dict:
        # first remove stop tokens if there exists
        for stop in self.get_stop_tokens_for_generation():
            if llm_output.endswith(stop):
                llm_output = llm_output[: -len(stop)]

        llm_output = (
            self.get_generation_prefix_for_tool_choice(tool_choice) + llm_output
        )

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

    def get_force_text_generation_prefix(self):
        return f"all\n"
