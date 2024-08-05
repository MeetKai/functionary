import datetime
import json
import re
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from functionary.openai_types import Function, Tool
from functionary.prompt_template import prompt_utils
from functionary.prompt_template.base_template import PromptTemplate


def get_system_prompt_for_custom_tools(custom_tools: List) -> str:
    custom_tool_params = ""
    for t in custom_tools:
        custom_tool_params += get_instruction_string(t) + "\n"
        custom_tool_params += get_parameters_string(t) + "\n\n"

    content = f"""
You have access to the following functions:

{custom_tool_params}
Think very carefully before calling functions.
If a you choose to call a function ONLY reply in the following format:
<{{start_tag}}={{function_name}}>{{parameters}}{{end_tag}}
where

start_tag => `<function`
parameters => a JSON dict with the function argument name as key and function argument value as value.
end_tag => `</function>`

Here is an example,
<function=example_function_name>{{"example_name": "example_value"}}</function>

Reminder:
- If looking for real time information use relevant functions before falling back to brave_search
- Function calls MUST follow the specified format, start with <function= and end with </function>
- Required parameters MUST be specified
- Only call one function at a time
- Put the entire function call reply on one line

"""
    return content


def get_instruction_string(custom_tool_definition) -> str:
    name, description = (
        custom_tool_definition["name"],
        custom_tool_definition["description"],
    )
    return f"Use the function '{name}' to '{description}'"


def get_parameters_string(custom_tool_definition) -> str:
    return json.dumps(custom_tool_definition)


def get_system_message_for_tools(tools: List[Dict], use_code_interpreter) -> List[Dict]:
    content = ""
    if use_code_interpreter:
        content += "Environment: ipython\n"

    current_date = datetime.datetime.now()
    formatted_date = current_date.strftime("%d %B %Y")
    date_str = f"""
Cutting Knowledge Date: December 2023\n\n"""
    content += date_str

    if tools:
        custom_message = get_system_prompt_for_custom_tools(tools)
        content += custom_message

    return {"role": "system", "content": content}


def parse_function_call_from_text(function_call_text: str) -> Optional[Dict]:
    index = function_call_text.find(">")
    if index >= 0:
        func_name = function_call_text[:index].strip()
        arguments = function_call_text[index + 1 :].strip()
        return {"name": func_name, "arguments": arguments}
    return None


class Llama31Template(PromptTemplate):
    version = "v3-llama3.1"
    function_separator = "<function="
    start_header = "<|start_header_id|>"
    end_header = "<|end_header_id|>"
    eos_token = "<|eot_id|>"
    eof_message = "<|eom_id|>"
    fn_param_sep_token = '>{"'

    def get_additional_tokens(self) -> List[str]:
        return []

    def get_assistant_prefixes(self) -> List[str]:
        return [f"{self.start_header}assistant{self.end_header}\n\n"]

    def pre_process_messages_before_inference(self, messages: List[Dict]) -> List[Dict]:
        """Order the tool results by the order of tool call ids

        Args:
            messages (List[Dict]): List of messages

        Returns:
            List[Dict]: List of messages
        """
        return prompt_utils.reorder_tool_messages_by_tool_call_ids(messages)

    def get_stop_tokens_for_generation(self) -> List[str]:
        return [self.eos_token, "<|end_of_text|>", self.eof_message]

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

        tools_system_message = get_system_message_for_tools(
            functions, is_code_interpreter
        )
        messages_clone.insert(0, tools_system_message)

        return messages_clone

    def convert_message_to_prompt(self, message: Dict) -> str:
        role = message["role"]
        if role == "tool":
            role = "ipython"
        content = message.get("content", None)

        prompt_template = (
            f"{self.start_header}{role}{self.end_header}\n\n" + "{text}{eot_content}"
        )
        eot_content = self.eos_token

        if role in ["user", "system", "ipython"]:
            return prompt_template.format(text=content, eot_content=eot_content)

        assert role == "assistant", f"role must be assistant, but: {role}"

        tool_calls = message.get("tool_calls", [])
        if tool_calls is None:
            tool_calls = []

        if content is None and len(tool_calls) == 0:  # inference time
            return f"{self.start_header}{role}{self.end_header}\n\n"

        total_content = content if content else ""

        # list of text representing function calls: {function_name}\n{arguments}
        tool_call_prompts = []
        for tool_call in tool_calls:
            arguments = tool_call["function"]["arguments"]
            assert isinstance(arguments, str)
            tool_name = tool_call["function"]["name"]
            if tool_name == "python":
                tool_prompt = f"<|python_tag|>{arguments}"
            else:
                tool_prompt = f"<function={tool_name}>{arguments}</function>"
            tool_call_prompts.append(tool_prompt)

        # join all function calls
        if tool_call_prompts:
            total_content += "".join(tool_call_prompts)
            eot_content = self.eof_message
        return prompt_template.format(text=total_content, eot_content=eot_content)

    def parse_assistant_response(
        self, llm_output: str, tool_choice: Any = None
    ) -> Dict:
        # first remove stop tokens if there exists
        for stop in self.get_stop_tokens_for_generation():
            if llm_output.endswith(stop):
                llm_output = llm_output[: -len(stop)]

        # add forced-function from tool_choice if exists
        llm_output = (
            self.get_generation_prefix_for_tool_choice(tool_choice) + llm_output
        )

        tool_calls = []
        text_response = ""

        func_prefix = "<function="
        end_func = "</function>"
        python_tag = "<|python_tag|>"

        while len(llm_output) > 0:
            if llm_output.startswith(python_tag):  # check if use code interpreter
                code = llm_output[len(python_tag) :]
                function_call = {
                    "name": "python",
                    "arguments": code,
                }

                tool_calls.append(
                    {
                        "type": "function",
                        "id": prompt_utils.get_random_tool_call_id(),
                        "function": function_call,
                    }
                )
                llm_output = ""
            elif llm_output.startswith(func_prefix):  # Check if function_call
                end_index = llm_output.find(end_func)
                if end_index >= 0:
                    function_call_text = llm_output[len(func_prefix) : end_index]
                    function_call = parse_function_call_from_text(function_call_text)

                    tool_calls.append(
                        {
                            "type": "function",
                            "id": prompt_utils.get_random_tool_call_id(),
                            "function": function_call,
                        }
                    )
                    llm_output = llm_output[end_index + len(end_func) :]
                else:
                    # TODO cannot find close function call
                    text_response += llm_output
                    break
            else:  # If text-response
                text_response += llm_output[0]
                llm_output = llm_output[1:]

        if not text_response:
            text_response = None
        elif len(text_response.strip()) == 0:
            text_response = None

        if not tool_calls:
            tool_calls = None

        return {"role": "assistant", "content": text_response, "tool_calls": tool_calls}

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
        state_gen_preliminary = "gen_preliminary"
        state_gen_function_name = "gen_function_name"
        state_gen_text = "gen_text"
        state_gen_code = "gen_code"
        state_gen_arguments = "gen_arguments"

        if len(current_state) == 0:
            current_state = {
                "state_name": state_gen_preliminary,  # can be normal text or a function call
                "current_text": "",  # the concatenation of all tokens so far
                "func_name": None,  # function_name of the current tool, if the response requires to use tool
                "func_index": -1,  # index of the tool in tool_calls
                "call_id": None,  # call_id of the current tool
                "gen_empty_text": True,  # if first_time we return an tempty delta with role=assistant
                "first_time_func": True,
                "text_to_func_buffer": [],
            }
            if tool_choice == "none":
                current_state["state_name"] = state_gen_text
            elif tool_choice == "required":
                current_state["state_name"] = state_gen_function_name
                current_state["current_text"] = "<function="
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
            if current_state["state_name"] in [state_gen_arguments, state_gen_code]:
                finish_reason = "tool_calls"
            return current_state, prompt_utils.get_text_delta_response(
                None, False, finish_reason
            )

        if current_state["state_name"] == state_gen_preliminary:
            if current_state["current_text"].startswith("<"):
                if current_state["current_text"] == "<|python_tag|>":
                    current_state["state_name"] = state_gen_code
                else:
                    current_state["state_name"] = state_gen_function_name
                return current_state, None
            else:
                responses = []
                if current_state["gen_empty_text"]:
                    empty_response = prompt_utils.get_text_delta_response(
                        "", True, finish_reason
                    )
                    current_state["gen_empty_text"] = False
                    responses.append(empty_response)
                if delta_text != current_state["current_text"]:
                    responses.append(
                        prompt_utils.get_text_delta_response(
                            current_state["current_text"][: -len(delta_text)],
                            True,
                            finish_reason,
                        )
                    )
                responses.append(
                    prompt_utils.get_text_delta_response(
                        delta_text, True, finish_reason
                    )
                )
                current_state["state_name"] = state_gen_text
                return current_state, responses
        elif current_state["state_name"] == state_gen_function_name:
            pattern = r"<function=[^>]+>"
            match = re.search(pattern, current_state["current_text"])
            if match:
                func_name = match.group(0).removeprefix("<function=").removesuffix(">")
                self.update_state_for_function(current_state, func_name)
                current_state["state_name"] = state_gen_arguments
                delta_args = current_state["current_text"].removeprefix(match.group(0))
                current_state["current_text"] = delta_args
            return current_state, None
        elif current_state["state_name"] == state_gen_arguments:
            if "</" in current_state["current_text"]:
                if "</" in delta_text:
                    delta_args = delta_text.removesuffix("</")
                    if len(delta_args) > 0:
                        return current_state, prompt_utils.get_function_delta_response(
                            current_state, delta_args, False, False, finish_reason
                        )
                    else:
                        return current_state, None
                else:
                    if "</function>" in current_state["current_text"]:
                        current_state["state_name"] = state_gen_preliminary
                        current_state["current_text"] = current_state["current_text"][
                            current_state["current_text"].rindex("</function>")
                            + len("</function>") :
                        ]
                    return current_state, None
            else:
                responses = []
                if current_state["first_time_func"]:
                    current_state["first_time_func"] = False
                    first_function_response = prompt_utils.get_function_delta_response(
                        current_state, "", True, False, finish_reason
                    )
                    responses.append(first_function_response)
                    delta_args = current_state["current_text"].removesuffix(delta_text)
                    first_arg_response = prompt_utils.get_function_delta_response(
                        current_state, delta_args, False, False, finish_reason
                    )
                    responses.append(first_arg_response)
                responses.append(
                    prompt_utils.get_function_delta_response(
                        current_state, delta_text, False, False, finish_reason
                    )
                )
                return current_state, responses
        elif current_state["state_name"] == state_gen_code:
            responses = []
            if current_state["first_time_func"]:
                self.update_state_for_function(current_state, "python")
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
        else:
            responses = []
            text_in_buffer = "".join(
                current_state["text_to_func_buffer"] + [delta_text]
            )
            if "<" in text_in_buffer and "<function".startswith(
                text_in_buffer[text_in_buffer.index("<") :]
            ):
                if text_in_buffer[text_in_buffer.index("<") :] == "<function":
                    current_state["state_name"] = state_gen_function_name
                    current_state["current_text"] = "<function"
                else:
                    current_state["text_to_func_buffer"].append(delta_text)
                return current_state, None
            elif text_in_buffer == "<|python_tag|>":
                current_state["state_name"] = state_gen_code
                current_state["current_text"] = "<|python_tag|>"
                return current_state, None
            else:
                while len(current_state["text_to_func_buffer"]) > 0:
                    delta_text_to_stream = current_state["text_to_func_buffer"][0]
                    responses.append(
                        prompt_utils.get_text_delta_response(
                            delta_text_to_stream, True, finish_reason
                        )
                    )
                    current_state["text_to_func_buffer"] = current_state[
                        "text_to_func_buffer"
                    ][1:]
            responses.append(
                prompt_utils.get_text_delta_response(delta_text, True, finish_reason)
            )
            return current_state, responses

    def get_chat_template_jinja(self):
        return super().get_chat_template_jinja()

    def get_force_function_call_prefix(self, function_name: str):
        return f"<function={function_name}>"

    def get_force_text_generation_prefix(self):
        return f""

    def get_tool_choice_required_prefix(self):
        return "<function="
