from functionary.prompt_template.base_template import PromptTemplate
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from functionary.prompt_template import prompt_utils
from functionary.openai_types import Function, Tool
import copy
import json
import re


class LLama4PromptTemplate(PromptTemplate):
    version = "llama4"

    def get_additional_tokens(self) -> List[str]:
        return []

    def get_assistant_prefixes(self) -> List[str]:
        return [f"<|header_start|>assistant<|header_end|>\n\n"]

    def pre_process_messages_before_inference(self, messages: List[Dict]) -> List[Dict]:
        """Order the tool results by the order of tool call ids

        Args:
            messages (List[Dict]): List of messages

        Returns:
            List[Dict]: List of messages
        """
        return prompt_utils.reorder_tool_messages_by_tool_call_ids(messages)

    def get_stop_tokens_for_generation(self) -> List[str]:
        return ["<|eot|>"]

    def get_force_function_call_prefix(self, function_name: str):
        return (
            '<|python_start|><|python_end|>{"name": "'
            + function_name
            + '", "parameters'
        )

    def get_force_text_generation_prefix(self):
        return f""

    def get_tool_choice_required_prefix(self):
        return "<|python_start|><|python_end|>"

    def get_prompt_from_messages(
        self,
        messages: List[Dict],
        tools_or_functions: Optional[List[Dict]] = None,
        bos_token: Optional[str] = "",
        add_generation_prompt: bool = False,
    ) -> str:
        """This function is used to get the complete prompt for list of messages

        Args:
            messages (List[Dict]): List of messages
            tools_or_functions (Optional[List[Dict]], optional): List of tools or functions. Defaults to None.

        Returns:
            str: the prompt for inference/training
        """
        # handle code_interpreter
        _tools = []
        if tools_or_functions:
            for tool in tools_or_functions:
                if tool["type"] == "code_interpreter":
                    _tools.append(
                        {
                            "type": "function",
                            "function": {
                                "name": "python",
                                "description": "This tool is used to execute python code. Code will be executed in a stateful Jupyter notebook environment. Python will respond with the output of the execution or time out after 60.0 seconds. The drive at '/mnt/data' can be used to save and persist user files.",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "code": {
                                            "type": "string",
                                            "description": "The python code to run",
                                        }
                                    },
                                },
                            },
                        }
                    )
                else:
                    _tools.append(tool)

        # find the assistant message that tool_call is python
        _messages = []
        for message in messages:
            n_message = copy.deepcopy(message)
            tool_calls = n_message.get("tool_calls", []) or []
            if len(tool_calls) > 0:
                for tool_call in tool_calls:
                    if tool_call["function"]["name"] == "python":
                        arguments = tool_call["function"][
                            "arguments"
                        ]  # currently the code is in string format
                        # check if argument is a valid JSON string or python code
                        try:  # if this is a valid JSON string --> no need to change anything
                            json.loads(arguments)
                        except:
                            tool_call["function"]["arguments"] = json.dumps(
                                {"code": arguments}, ensure_ascii=False
                            )
            _messages.append(n_message)
        prompt = super().get_prompt_from_messages(
            messages=_messages,
            tools_or_functions=_tools,
            bos_token=bos_token,
            add_generation_prompt=add_generation_prompt,
        )
        return prompt

    def parse_assistant_response(
        self, llm_output: str, tool_choice: Any | None
    ) -> Dict:
        for stop in self.get_stop_tokens_for_generation():
            if llm_output.endswith(stop):
                llm_output = llm_output[: -len(stop)]

        # add forced-function from tool_choice if exists
        llm_output = (
            self.get_generation_prefix_for_tool_choice(tool_choice) + llm_output
        )
        # if there is a tool call, it will be in the format: <|python_start|><|python_end|>{"name": "python", "parameters": JSON_STRING}
        python_start_index = llm_output.find("<|python_start|>")
        tool_calls = []
        content = None
        if python_start_index >= 0:
            python_end_index = llm_output.find("<|python_end|>", python_start_index)
            if python_end_index >= 0:
                content = llm_output[
                    python_start_index + len("<|python_start|>") : python_end_index
                ].strip()
                tool_call_text = llm_output[
                    python_end_index + len("<|python_end|>") :
                ].strip()
                for line in tool_call_text.split("\n"):
                    if line.strip():
                        tool_call_item = json.loads(line)
                        tool_calls.append(
                            {
                                "id": prompt_utils.get_random_tool_call_id(),
                                "type": "function",
                                "function": {
                                    "name": tool_call_item["name"],
                                    "arguments": json.dumps(
                                        tool_call_item["parameters"], ensure_ascii=False
                                    ),
                                },
                            }
                        )
            else:
                content = llm_output
        else:
            content = llm_output
        return {
            "role": "assistant",
            "content": content if content else None,
            "tool_calls": tool_calls if tool_calls else None,
        }

    def initialize_fsm_gen_state(
        self,
        tool_choice: Union[str, Tool],
        curr_text: str,
        curr_tokens: Optional[List[int]],
        add_code_interpreter: Optional[bool],
    ) -> Dict:
        """Initializes FSM state for both streaming and grammar sampling

        Args:
            tool_choice (str): tool_choice provided by user
            curr_text (str): Text to initialize in gen_state
            curr_tokens (List[int]): Corresponding tokens of curr_text
            add_code_interpreter (bool): Flag indicating whether to add "python" tool in options in "function" stage.
        Returns:
            Dict: generation state
        """
        result = {
            "stage": "start",
            "func_index": -1,
            "curr_text": curr_text,
            "curr_tokens": curr_tokens,
            "add_code_interpreter": add_code_interpreter,
        }
        if tool_choice == "required":  # a tool must be used
            result["stage"] = "function_name"
            result["curr_text"] = ""
            result["func_index"] += 1
            result["call_id"] = prompt_utils.get_random_tool_call_id()

        elif not isinstance(tool_choice, str):  # a predefined tool is used
            func_name = (
                tool_choice.name
                if hasattr(tool_choice, "name")
                else tool_choice.function.name
            )
            result["stage"] = "function_name"
            result["curr_text"] = '{"name": "' + func_name + '", "parameters'
            result["func_index"] += 1
            result["call_id"] = prompt_utils.get_random_tool_call_id()

        return result

    def stream_delta_text(
        self,
        gen_state: Dict,
        delta_text: str,
        finish_reason: Optional[str],
        tools_or_functions: List[Dict],
        tool_choice: Any,
    ) -> Tuple[Dict, Optional[Union[Dict, List[Dict]]]]:
        if finish_reason is not None:  # handle if finish
            if gen_state["stage"] not in ["text-gen"]:
                finish_reason = "tool_calls"

            end_response = prompt_utils.get_text_delta_response(
                None, False, finish_reason
            )
            last_response = None
            if "buffer" in gen_state and len(gen_state["buffer"]) > 0:
                last_response = get_the_last_chunk_from_buffer(gen_state, None)

            return gen_state, (
                [last_response, end_response] if last_response else [end_response]
            )

        current_text = gen_state["curr_text"] + delta_text
        gen_state["curr_text"] = current_text

        if gen_state["stage"] == "start":
            if delta_text == "<|python_start|>":  # this is the tool_call
                gen_state["stage"] = "function-text-gen"
                gen_state["current_text"] = ""
                gen_state["buffer"] = []
                gen_state["first_chunk"] = True
            else:
                gen_state["stage"] = "text-gen"
                responses = [
                    prompt_utils.get_text_delta_response("", True, finish_reason)
                ]
                responses.append(
                    prompt_utils.get_text_delta_response(
                        delta_text, False, finish_reason
                    )
                )
                return gen_state, responses
        elif gen_state["stage"] == "text-gen":
            return gen_state, prompt_utils.get_text_delta_response(
                delta_text, False, finish_reason
            )
        elif gen_state["stage"] == "function-text-gen":
            if (
                delta_text == "<|python_end|>"
            ):  # no reasoning the next stage is function_call
                gen_state["stage"] = "function_name"
                gen_state["curr_text"] = ""
                gen_state["func_index"] += 1
                gen_state["call_id"] = prompt_utils.get_random_tool_call_id()
            else:  # contain the reasoning;
                responses = []
                if gen_state["first_chunk"]:  # the first chunk
                    gen_state["first_chunk"] = False
                    responses.append(
                        prompt_utils.get_text_delta_response("", True, finish_reason)
                    )
                responses.append(
                    prompt_utils.get_text_delta_response(
                        delta_text, False, finish_reason
                    )
                )
                return gen_state, responses
        elif gen_state["stage"] == "function_name":
            # wait until we get '{"name": "func_name", "parameters": {'
            pattern = (
                r'\s*{"name"\s*:\s*"(?P<function_name>.*)"\s*,\s*"parameters"\s*:\s*{'
            )
            match = re.search(pattern, current_text)
            if match:
                _, end_ind = match.start(), match.end()
                new_delta = current_text[end_ind - 1 :]
                gen_state["curr_text"] = new_delta  # -1 to retain "{"
                gen_state["func_name"] = match.group("function_name")
                gen_state["stage"] = "function_arguments"
                # if gen_state["func_name"] != "python"
                # else "python"
                responses = [
                    prompt_utils.get_function_delta_response(
                        gen_state, "", True, True, finish_reason
                    )
                ]  # the chunk containing function_name only
                gen_state["buffer"] = []
                # if gen_state["func_name"] != "python":
                gen_state["buffer"].append(new_delta)

                return gen_state, responses

        elif gen_state["stage"] == "function_arguments":
            # check if current function is python, we need to stream the code string inside, not a json
            if (
                "\n" in delta_text
            ):  # finish the current function call --> next function call
                index = delta_text.find("\n")
                last_chunk = delta_text[:index]
                if len(last_chunk) > 0:
                    gen_state["buffer"].append(last_chunk)
                remaining_chunk = delta_text[index + 1 :]
                # the last chunk of the current function call
                response = get_the_last_chunk_from_buffer(gen_state, finish_reason)
                # this is the next function call
                gen_state["stage"] = "function_name"
                gen_state["curr_text"] = remaining_chunk
                gen_state["func_index"] += 1
                gen_state["call_id"] = prompt_utils.get_random_tool_call_id()
                return gen_state, response
            elif delta_text == "<|eot|>":
                response = get_the_last_chunk_from_buffer(gen_state, finish_reason)
                return gen_state, response
            else:
                gen_state["buffer"].append(delta_text)
                if len(gen_state["buffer"]) >= 4:
                    delta_text_item = gen_state["buffer"].pop(0)
                    return gen_state, prompt_utils.get_function_delta_response(
                        gen_state, delta_text_item, False, False, finish_reason
                    )
        return gen_state, None


def get_the_last_chunk_from_buffer(gen_state, finish_reason):
    buffer_str = "".join(gen_state["buffer"]).rstrip()  # remove \n at the end
    if buffer_str.endswith("}}"):
        buffer_str = buffer_str[:-1]  # remove the last "}\n"

    # the last chunk of the current function call
    response = prompt_utils.get_function_delta_response(
        gen_state, buffer_str, False, False, finish_reason
    )
    return response
