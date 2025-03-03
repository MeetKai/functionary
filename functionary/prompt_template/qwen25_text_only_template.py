from functionary.prompt_template.base_template import PromptTemplate
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from functionary.prompt_template import prompt_utils
from functionary.openai_types import Function, Tool
import json
import copy
import math
import re


class Qwen25TextOnlyPromptTemplate(PromptTemplate):
    version = "qwen2.5-text-only"
    start_of_turn = "<|im_start|>"
    end_of_turn = "<|im_end|>"

    def get_chat_template_jinja(self) -> str:
        path_prefix = "./functionary/prompt_template/jinja_templates/"
        with open(f"{path_prefix}{self.version}.txt", "r") as f:
            template = f.read()

        return template

    def get_tool_choice_required_prefix(self) -> str:
        return "<tool_call>\n"

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

    def get_additional_tokens(self) -> List[str]:
        return []

    def get_assistant_prefixes(self) -> List[str]:
        return [f"{self.start_of_turn}assistant\n"]

    def get_stop_tokens_for_generation(self) -> List[str]:
        return [self.end_of_turn]

    def get_force_function_call_prefix(self, function_name: str):
        return """<tool_call>
{"name": "{function_name}", "arguments""".replace(
            "{function_name}", function_name
        )

    def pre_process_messages_before_inference(self, messages: List[Dict]) -> List[Dict]:
        """Order the tool results by the order of tool call ids

        Args:
            messages (List[Dict]): List of messages

        Returns:
            List[Dict]: List of messages
        """
        return prompt_utils.reorder_tool_messages_by_tool_call_ids(messages)

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

        # print(f"+++LLM_OUTPUT: {llm_output}")
        llm_output = post_process_llm_output(llm_output)
        # print(f"+++LLM_OUTPUT after post-processing: {llm_output}")
        text_content = ""
        tool_call_strs = []

        # Split on tool call tags
        parts = llm_output.split("<tool_call>")

        if len(parts) > 0:
            # First part is the text content
            text_content = parts[0].strip()

            # Process remaining parts as tool calls
            for part in parts[1:]:
                if "</tool_call>" in part:
                    tool_call = part.split("</tool_call>")[0].strip()
                    if tool_call:
                        tool_call_strs.append(tool_call)
        tool_calls = []
        for tool_call_str in tool_call_strs:
            tool_call_dic = json.loads(tool_call_str)
            if tool_call_dic["name"] == "python":
                arguments = tool_call_dic["arguments"]["code"]
            else:
                arguments = json.dumps(tool_call_dic["arguments"], ensure_ascii=False)
            tool_calls.append(
                {
                    "type": "function",
                    "id": prompt_utils.get_random_tool_call_id(),
                    "function": {
                        "name": tool_call_dic["name"],
                        "arguments": arguments,
                    },
                }
            )

        return {
            "role": "assistant",
            "content": text_content if len(text_content) > 0 else None,
            "tool_calls": None if len(tool_calls) == 0 else tool_calls,
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
            result["curr_text"] = '{"name": "{function_name}", "arguments'.replace(
                "{function_name}", func_name
            )
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
            if gen_state["stage"] not in ["text_gen"]:
                finish_reason = "tool_calls"

            end_response = prompt_utils.get_text_delta_response(
                None, False, finish_reason
            )
            last_response = None
            # still need to check if there is st in buffer
            if "buffer" in gen_state and len(gen_state["buffer"]) > 0:
                if gen_state["stage"] == "text_gen":
                    buffer_str = "".join(
                        gen_state["buffer"]
                    ).rstrip()  # remove \n at the end
                    last_response = prompt_utils.get_text_delta_response(
                        buffer_str, False, None
                    )

                elif gen_state["stage"] == "function_arguments":
                    buffer_str = "".join(
                        gen_state["buffer"]
                    ).rstrip()  # remove \n at the end
                    if buffer_str.endswith("}}"):
                        buffer_str = buffer_str[:-1]  # remove the last "}"

                    if len(buffer_str) > 0:
                        last_response = prompt_utils.get_function_delta_response(
                            gen_state, buffer_str, False, False, None
                        )
                elif gen_state["stage"] == "python":
                    last_response = return_all_code_from_buffer(gen_state)

            if last_response is not None:
                return gen_state, [last_response, end_response]
            else:
                return gen_state, [end_response]

        current_text = gen_state["curr_text"] + delta_text
        gen_state["curr_text"] = current_text
        if gen_state["stage"] == "start":
            if (
                gen_state.get("end_of_prev_function_call", False) and delta_text == "\n"
            ):  # ignore \n
                gen_state["end_of_prev_function_call"] = False
                gen_state["curr_text"] = ""

            elif delta_text == "<tool_call>":
                # print(f"delta text: {delta_text}; go to function_name")
                gen_state["stage"] = "function_name"
                gen_state["curr_text"] = ""
                gen_state["func_index"] += 1
                gen_state["call_id"] = prompt_utils.get_random_tool_call_id()
            else:
                # print(f"delta text: {delta_text}; go to text_gen")
                gen_state["stage"] = "text_gen"
                gen_state["curr_text"] = current_text
                gen_state["buffer"] = (
                    []
                )  # put to buffer before we return because we need to check the last item
                responses = [
                    prompt_utils.get_text_delta_response("", True, finish_reason)
                ]
                if len(delta_text) > 0:
                    gen_state["buffer"].append(delta_text)
                return gen_state, responses

        elif gen_state["stage"] == "function_name":
            # wait until we get '{"name": "func_name", "arguments": {'
            # print(f"current_text: {current_text}")
            pattern = (
                r'\s*{"name"\s*:\s*"(?P<function_name>.*)"\s*,\s*"arguments"\s*:\s*{'
            )
            match = re.search(pattern, current_text)
            if match:
                _, end_ind = match.start(), match.end()
                new_delta = current_text[end_ind - 1 :]
                gen_state["curr_text"] = new_delta  # -1 to retain "{"
                gen_state["func_name"] = match.group("function_name")
                gen_state["stage"] = (
                    "function_arguments"
                    if gen_state["func_name"] != "python"
                    else "python"
                )
                responses = [
                    prompt_utils.get_function_delta_response(
                        gen_state, "", True, True, finish_reason
                    )
                ]  # the chunk containing function_name only
                gen_state["buffer"] = []
                if gen_state["func_name"] != "python":
                    gen_state["buffer"].append(new_delta)

                return gen_state, responses
            else:
                return gen_state, None

        elif gen_state["stage"] == "text_gen":
            if delta_text == "<tool_call>":  # start a tool call
                # print("start a tool call after reasoning")
                gen_state["stage"] = "function_name"
                gen_state["curr_text"] = ""
                gen_state["func_index"] += 1
                gen_state["call_id"] = prompt_utils.get_random_tool_call_id()
                buffer_str = "".join(
                    gen_state["buffer"]
                ).rstrip()  # remove \n at the end
                if len(buffer_str) > 0:
                    return gen_state, prompt_utils.get_text_delta_response(
                        buffer_str, False, finish_reason
                    )
            else:
                gen_state["buffer"].append(delta_text)
                if len(gen_state["buffer"]) >= 2:
                    delta_text_item = gen_state["buffer"].pop(0)
                    return gen_state, prompt_utils.get_text_delta_response(
                        delta_text_item, False, finish_reason
                    )

        elif gen_state["stage"] == "function_arguments":
            # check if current function is python, we need to stream the code string inside, not a json
            if delta_text == "</tool_call>":
                gen_state["stage"] = "start"
                gen_state["curr_text"] = ""
                gen_state["end_of_prev_function_call"] = True
                # return all in the buffer but need to strip and remove the last "}"
                buffer_str = "".join(
                    gen_state["buffer"]
                ).rstrip()  # remove \n at the end
                if buffer_str.endswith("}}\n"):
                    buffer_str = buffer_str[:-2]  # remove the last "}\n"
                elif buffer_str.endswith("}}"):
                    buffer_str = buffer_str[:-1]  # remove the last "}"

                return gen_state, prompt_utils.get_function_delta_response(
                    gen_state, buffer_str, False, False, finish_reason
                )
            else:
                gen_state["buffer"].append(delta_text)
                if len(gen_state["buffer"]) >= 4:
                    delta_text_item = gen_state["buffer"].pop(0)
                    return gen_state, prompt_utils.get_function_delta_response(
                        gen_state, delta_text_item, False, False, finish_reason
                    )

        elif gen_state["stage"] == "python":
            return streamining_python_code(gen_state, delta_text)
        return gen_state, None


def return_all_code_from_buffer(gen_state: Dict) -> Optional[Union[Dict, List[Dict]]]:
    buffer_str = "".join(gen_state["buffer"]).rstrip()  # remove \n at the end
    if len(buffer_str) > 0:
        return prompt_utils.get_function_delta_response(
            gen_state, buffer_str, False, False, None
        )
    return None


def streamining_python_code(
    gen_state: Dict, delta_text: str
) -> Tuple[Dict, Optional[Union[Dict, List[Dict]]]]:
    if "current_code" not in gen_state:
        gen_state["current_code"] = ""
        return gen_state, None

    current_text = gen_state["curr_text"]
    current_code = gen_state["current_code"]
    # try extracting the latest code from current_text
    try:
        if delta_text == "</tool_call>":  # end of code
            full_code_arg_str = current_text.rstrip("</tool_call>").strip()
            if full_code_arg_str.endswith("}}"):
                full_code_arg_str = full_code_arg_str[:-1]
            new_code = json.loads(full_code_arg_str)["code"]
        else:
            new_code = json.loads(current_text + '"}')["code"]
            delta_code = new_code[len(current_code) :]

        gen_state["buffer"].append(delta_code)
        gen_state["current_code"] = new_code
    except:  # nothing changed
        return gen_state, None

    if delta_text == "</tool_call>":
        return gen_state, return_all_code_from_buffer(gen_state)
    else:
        if len(gen_state["buffer"]) >= 4:
            delta_text_item = gen_state["buffer"].pop(0)
            return gen_state, prompt_utils.get_function_delta_response(
                gen_state, delta_text_item, False, False, None
            )
    return gen_state, None


def match_pattern(pattern: str, text: str) -> Tuple[int, int]:
    match = re.search(pattern, text)
    if match:
        return match.start(), match.end()
    return -1, -1