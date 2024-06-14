from functionary.prompt_template.base_template import PromptTemplate, PYTHON_RUN_SYS_MSG
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from functionary.prompt_template import prompt_utils
from functionary.schema import generate_schema_from_functions
from functionary.openai_types import Function, Tool

SYSTEM_CONTENT = """You are capable of executing available function(s) if required.
Only execute function(s) when absolutely necessary.
Ask for the required input to:recipient==all
Use JSON for function arguments.
Respond in this format:
>>>${recipient}
${content}
Available functions:
"""


class Llama3TemplateV3(PromptTemplate):
    version = "v3.llama3"
    function_separator = ">>>"
    start_header = "<|start_header_id|>"
    end_header = "<|end_header_id|>"
    eos_token = "<|eot_id|>"

    def get_additional_tokens(self) -> List[str]:
        return []

    def get_assistant_prefixes(self) -> List[str]:
        return [
            f"{self.start_header}assistant{self.end_header}\n\n{self.function_separator}"
        ]

    def get_stop_tokens_for_generation(self) -> List[str]:
        return [self.eos_token, "<|end_of_text|>"]

    # ["<|eot_id|>", "<|end_of_text|>"]

    def get_force_function_call_prefix(self, function_name: str):
        return f"{function_name}\n"

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
            f"{self.start_header}{role}{self.end_header}\n\n"
            + "{text}"
            + self.eos_token
        )

        if role in ["user", "system", "tool"]:
            return prompt_template.format(text=content)

        assert role == "assistant", f"role must be assistant, but: {role}"
        tool_calls = message.get("tool_calls", [])
        if tool_calls is None:
            tool_calls = []

        if content is None and len(tool_calls) == 0:  # inference time
            return f"{self.start_header}{role}{self.end_header}\n\n{self.function_separator}"

        if content is not None:  # text-only
            tool_calls = [
                {"function": {"name": "all", "arguments": content}}
            ] + tool_calls

        tool_call_prompts = []
        for tool_call in tool_calls:
            arguments = tool_call["function"]["arguments"]
            tool_name = tool_call["function"]["name"]
            tool_prompt = f"{tool_name}\n{arguments}"
            tool_call_prompts.append(tool_prompt)

        total_content = self.function_separator + self.function_separator.join(
            tool_call_prompts
        )
        return prompt_template.format(text=total_content)

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
                "content": SYSTEM_CONTENT + generate_schema_from_functions(functions),
            },
        )
        if is_code_interpreter:
            messages_clone.insert(1, {"role": "system", "content": PYTHON_RUN_SYS_MSG})

        return messages_clone

    def get_force_text_generation_prefix(self):
        return f"all\n"

    def update_state_for_function(self, current_state: Dict, func_name: str):
        """update the state when a function is going to be called

        Args:
            current_state (_type_): _description_
        """
        current_state["func_name"] = func_name
        current_state["func_index"] += 1
        current_state["call_id"] = prompt_utils.get_random_tool_call_id()
        current_state["first_time_func"] = True

    def update_response_state_from_delta_text(
        self,
        *,
        current_state: Dict[str, Any],
        delta_text: str,
        finish_reason: Optional[str],
        tool_choice: Any,
    ) -> Tuple[Dict[str, Any], Union[None, Dict, List[Dict]]]:
        """This function is used for streaming

        Args:
            current_state (Dict[str, Any]):  a dictionary containing the state of the streaming: such as current function_name,
            delta_text: new token generated
            finish_reason: if finished or not

        Returns:
            Tuple[Dict[str, Any], Optional[Dict]]: updated state, response: can be None, a dictionary: {} or a list of dictionary: [{}, ..., {}]
        """
        state_gen_function_name = "gen_function_name"
        state_gen_text = "gen_text"
        state_gen_arguments = "gen_arguments"

        if len(current_state) == 0:  # empty dict, at the first_time
            current_state = {
                "state_name": state_gen_function_name,  # can be all or a function_name
                "current_text": "",  # the concatenation of all tokens so far
                "func_name": None,  # function_name of the current tool, if the response requires to use tool
                "response_type": None,  # response_type=text(text response)/function (using tool)
                "func_index": -1,  # index of the tool in tool_calls
                "call_id": None,  # call_id of the current tool
                "gen_empty_text": True,  # if first_time we return an tempty delta with role=assistant
                "first_time_func": True,
            }
            if tool_choice == "none":
                current_state["state_name"] = state_gen_text

            elif type(tool_choice) is not str and tool_choice is not None:
                current_state["state_name"] = state_gen_arguments
                function_name = (
                    tool_choice.function.name
                    if isinstance(tool_choice, Tool)
                    else tool_choice.name
                )
                self.update_state_for_function(current_state, function_name)

        current_state["current_text"] += delta_text

        if finish_reason is not None:  # handle if finish
            if current_state["response_type"] == "function":
                finish_reason = "tool_calls"
            return current_state, prompt_utils.get_text_delta_response(
                None, False, finish_reason
            )

        if current_state["state_name"] == state_gen_function_name:
            if current_state["current_text"].endswith("\n"):
                func_name = current_state["current_text"].strip()
                if func_name == "all":  # start gen_text
                    current_state["state_name"] = state_gen_text
                    return current_state, None
                else:  # start gen function
                    current_state["state_name"] = state_gen_arguments
                    self.update_state_for_function(current_state, func_name)
                    return current_state, None
            else:
                return current_state, None

        elif current_state["state_name"] == state_gen_text:
            if delta_text == self.function_separator:
                current_state["state_name"] = state_gen_function_name
                current_state["current_text"] = ""
                return current_state, None
            else:
                responses = []
                if current_state["gen_empty_text"]:
                    empty_response = prompt_utils.get_text_delta_response(
                        "", True, finish_reason
                    )
                    current_state["gen_empty_text"] = False
                    responses.append(empty_response)
                responses.append(
                    prompt_utils.get_text_delta_response(
                        delta_text, True, finish_reason
                    )
                )
                return current_state, responses

        elif current_state["state_name"] == state_gen_arguments:
            if delta_text == self.function_separator:  # change to another function
                current_state["state_name"] = state_gen_function_name
                current_state["current_text"] = ""
                return current_state, None
            else:
                responses = []
                if current_state["first_time_func"]:
                    current_state["first_time_func"] = False
                    first_function_response = prompt_utils.get_function_delta_response(
                        current_state, "", True, False, finish_reason
                    )
                    responses.append(first_function_response)
                responses.append(
                    prompt_utils.get_function_delta_response(
                        current_state, delta_text, False, False, finish_reason
                    )
                )
                return current_state, responses
