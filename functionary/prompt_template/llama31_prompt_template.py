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

    def initialize_fsm_gen_state(
        self,
        tool_choice: Union[str, Tool],
        curr_text: str,
        curr_tokens: Optional[List[int]],
        add_code_interpreter: Optional[bool],
    ) -> Dict:
        func_name = None
        # To force a text response ("tool_choice"="none")
        if tool_choice == "none":
            stage = "text-gen"
        # Normal generation (function name first with <function= generated) (tool_choice="returned")
        elif tool_choice == "required":
            stage = "function"
            curr_text = "<function="
        # To force a function call (tool_choice={"type": "function", "function": {...}})
        elif not isinstance(tool_choice, str):
            stage = "parameter"
            func_name = (
                tool_choice.function.name
                if isinstance(tool_choice, Tool)
                else tool_choice.name
            )
        # Normal generation (text gen or function name) (tool_choice="auto")
        else:
            stage = "pre-function"

        gen_state = {
            "stage": stage,
            "curr_tokens": curr_tokens,
            "curr_text": curr_text,
            "func_name": func_name,
            "func_index": -1,  # index of the tool in tool_calls
            "call_id": None,  # call_id of the current tool
            "gen_empty_text": True,  # if first_time we return an empty delta with role=assistant
            "first_time_func": True,
            "text_to_func_buffer": [],
            "clear_buffer": False,
            "add_code_interpreter": add_code_interpreter,
        }

        return (
            self._update_gen_state_for_fn_call(gen_state, func_name)
            if func_name is not None
            else gen_state
        )

    def stream_delta_text(
        self,
        gen_state: Dict,
        delta_text: str,
        finish_reason: Optional[str],
        tools_or_functions: List[Dict],
        tool_choice: Any,
    ) -> Tuple[Dict, Optional[Union[Dict, List[Dict]]]]:
        if finish_reason is not None:  # handle if finish
            if gen_state["stage"] in ["parameter", "code-interpreter"]:
                finish_reason = "tool_calls"
            return gen_state, prompt_utils.get_text_delta_response(
                None, False, finish_reason
            )

        responses = []
        options = self.get_options_from_gen_state(
            gen_state=gen_state, tools_or_functions=tools_or_functions
        )

        if gen_state["stage"] == "text-gen":
            if gen_state["gen_empty_text"]:
                responses.append(
                    prompt_utils.get_text_delta_response("", True, finish_reason)
                )
                gen_state["gen_empty_text"] = False
                responses.append(
                    prompt_utils.get_text_delta_response(
                        gen_state["curr_text"], True, finish_reason
                    )
                )
            text_in_buffer = "".join(gen_state["text_to_func_buffer"] + [delta_text])
            if delta_text != "<|python_tag|>" and not (
                "<" in text_in_buffer
                and "<function".startswith(text_in_buffer[text_in_buffer.index("<") :])
            ):
                while len(gen_state["text_to_func_buffer"]) > 0:
                    delta_text_to_stream = gen_state["text_to_func_buffer"][0]
                    responses.append(
                        prompt_utils.get_text_delta_response(
                            delta_text_to_stream, True, finish_reason
                        )
                    )
                    gen_state["text_to_func_buffer"] = gen_state["text_to_func_buffer"][
                        1:
                    ]
                responses.append(
                    prompt_utils.get_text_delta_response(
                        delta_text, True, finish_reason
                    )
                )
            else:
                gen_state["text_to_func_buffer"].append(delta_text)
        elif gen_state["stage"] == "parameter":
            if gen_state["first_time_func"]:
                gen_state["first_time_func"] = False
                responses.append(
                    prompt_utils.get_function_delta_response(
                        gen_state, "", True, False, finish_reason
                    )
                )
                responses.append(
                    prompt_utils.get_function_delta_response(
                        gen_state, gen_state["curr_text"], False, False, finish_reason
                    )
                )

            if "</" in delta_text:
                delta_args = delta_text.removesuffix("</")
                if len(delta_args) > 0:
                    responses.append(
                        prompt_utils.get_function_delta_response(
                            gen_state, delta_args, False, False, finish_reason
                        )
                    )
            elif "</" in gen_state["curr_text"] and (
                "</function>".startswith(
                    gen_state["curr_text"][gen_state["curr_text"].rindex("</") :]
                    + delta_text
                )
                or "</function>" in gen_state["curr_text"] + delta_text
            ):
                pass
            else:
                responses.append(
                    prompt_utils.get_function_delta_response(
                        gen_state, delta_text, False, False, finish_reason
                    )
                )
        elif gen_state["stage"] == "code-interpreter":
            if gen_state["first_time_func"]:
                gen_state["first_time_func"] = False
                first_function_response = prompt_utils.get_function_delta_response(
                    gen_state, "", True, False, finish_reason
                )
                responses.append(first_function_response)
            responses.append(
                prompt_utils.get_function_delta_response(
                    gen_state, delta_text, False, False, finish_reason
                )
            )

        gen_state = self.update_fsm_gen_state(
            gen_state=gen_state,
            new_token=delta_text,
            new_token_id=None,
            options=options,
            tokenizer=None,
        )

        return gen_state, responses

    def update_fsm_gen_state(
        self,
        gen_state: Dict,
        new_token: Optional[str],
        new_token_id: Optional[str],
        options: Optional[List],
        tokenizer: Any,
    ) -> Dict:
        if gen_state["curr_tokens"] is not None:
            # Update curr_tokens and curr_text
            gen_state["curr_tokens"].append(new_token_id)
            gen_state["curr_text"] = tokenizer.decode(gen_state["curr_tokens"])
        else:
            gen_state["curr_text"] += new_token

        if gen_state["stage"] == "pre-function":
            if gen_state["curr_text"].startswith("<"):
                if gen_state["curr_text"] == "<|python_tag|>":
                    self._update_gen_state_for_fn_call(
                        gen_state=gen_state, func_name="python"
                    )
                    gen_state["stage"] = "code-interpreter"
                else:
                    gen_state["stage"] = "function"
            else:
                gen_state["stage"] = "text-gen"
        elif gen_state["stage"] == "text-gen":
            if gen_state["curr_text"].endswith("<function"):
                gen_state["stage"] = "function"
                gen_state["curr_text"] = "<function"
                gen_state["curr_tokens"] = (
                    tokenizer.encode(gen_state["curr_text"], add_special_tokens=False)
                    if gen_state["curr_tokens"] is not None
                    else None
                )
                gen_state["text_to_func_buffer"] = []
            elif gen_state["curr_text"].endswith("<|python_tag|>"):
                gen_state["stage"] = "code-interpreter"
                gen_state = self._update_gen_state_for_fn_call(
                    gen_state=gen_state, func_name="python"
                )
                gen_state = self._reset_fsm_curr_text_and_tokens(gen_state=gen_state)
                gen_state["text_to_func_buffer"] = []
        elif gen_state["stage"] == "function":
            pattern = r"<function=[^>]+>"
            match = re.search(pattern, gen_state["curr_text"])
            if match:
                func_name = match.group(0).removeprefix("<function=").removesuffix(">")
                gen_state = self._update_gen_state_for_fn_call(
                    gen_state=gen_state, func_name=func_name
                )
                gen_state["stage"] = "parameter"
                delta_args = gen_state["curr_text"].removeprefix(match.group(0))
                gen_state["curr_text"] = delta_args
                gen_state["curr_tokens"] = (
                    tokenizer.encode(gen_state["curr_text"], add_special_tokens=False)
                    if gen_state["curr_tokens"] is not None
                    else None
                )
        elif gen_state["stage"] == "parameter":
            if "</function>" in gen_state["curr_text"]:
                gen_state["stage"] = "pre-function"
                gen_state["curr_text"] = gen_state["curr_text"][
                    gen_state["curr_text"].rindex("</function>") + len("</function>") :
                ]
                gen_state["curr_tokens"] = (
                    tokenizer.encode(gen_state["curr_text"], add_special_tokens=False)
                    if gen_state["curr_tokens"] is not None
                    else None
                )

        return gen_state

    def get_options_from_gen_state(self, gen_state: Dict, tools_or_functions: List):
        return []

    def get_chat_template_jinja(self):
        return super().get_chat_template_jinja()

    def get_force_function_call_prefix(self, function_name: str):
        return f"<function={function_name}>"

    def get_force_text_generation_prefix(self):
        return f""

    def get_tool_choice_required_prefix(self):
        return "<function="
