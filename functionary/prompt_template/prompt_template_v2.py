import json
import random
import string
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from functionary.openai_types import Tool
from functionary.prompt_template.base_template import (
    PredefinedFuncTypes,
    PromptTemplate,
)
from functionary.prompt_template import prompt_utils


class PromptTemplateV2(PromptTemplate):
    from_token = "<|from|>"
    recipient_token = "<|recipient|>"
    content_token = "<|content|>"
    stop_token = "<|stop|>"
    version = "v2"
    # This token splits between function name and parameters
    fn_param_sep_token = "\n<|content|>"
    # This maps the predefined function type to its str name
    predefined_func_names = {
        PredefinedFuncTypes.no_tool_call: "all",
        PredefinedFuncTypes.code_interpreter: "python",
    }

    def get_start_of_function_call_token(self) -> str:
        return self.recipient_token

    def get_predefined_function_names(self, function_types: Any) -> List[str]:
        if function_types == "all":
            return [func_name for func_name in self.predefined_func_names.values()]

        if not isinstance(function_types, list):
            function_types = [function_types]

        predefined_function_names = []
        for function_type in function_types:
            predefined_function_names.append(self.predefined_func_names[function_type])

        return predefined_function_names

    def initialize_grammar_sampling_gen_state(
        self, tool_choice: str, curr_text: str, curr_tokens: List[int]
    ) -> Dict:
        # To force a text response ("tool_choice"="none")
        if tool_choice == "none":
            add_predefined_fns = False
            stage = "no-tool-call"
        # Normal generation (function name first without "all") (tool_choice="returned")
        elif tool_choice == "required":
            add_predefined_fns = False
            stage = "function"
        # To force a function call (tool_choice={"type": "function", "function": {...}})
        elif tool_choice != "":
            add_predefined_fns = False
            stage = "parameter"
        # Normal generation (function name first) (tool_choice="auto")
        else:
            add_predefined_fns = True
            stage = "function"

        return {
            "stage": stage,
            "curr_tokens": curr_tokens,
            "curr_text": curr_text,
            "func_name": tool_choice,
            "param_names": [],
            "add_predefined_fns": add_predefined_fns,
        }

    def get_additional_tokens(self) -> List[str]:
        return [
            self.from_token,
            self.recipient_token,
            self.content_token,
            self.stop_token,
        ]

    def convert_message_to_prompt(self, message: Dict) -> str:
        role = message["role"]
        content = message.get("content", None)

        if role in [
            "system",
            "user",
        ]:  # <|from|>system\n<|recipient|>all\n<|content|>xxx
            return f"{self.from_token}{role}\n{self.recipient_token}all\n{self.content_token}{content}\n"

        if role == "tool":  # <|from|>tool_name\n<|recipient|>all\n<|content|>xxx
            tool_name = message["name"]
            return f"{self.from_token}{tool_name}\n{self.recipient_token}all\n{self.content_token}{content}\n"

        assert role == "assistant"
        tool_calls = message.get("tool_calls", [])
        if tool_calls is None:
            tool_calls = []
        if (
            len(tool_calls) == 0 and content is None
        ):  # for inference: <|from|> assistant\n<|recipient|>
            return f"{self.from_token}{role}\n{self.recipient_token}"

        if len(tool_calls) == 0:  # <|from|>assistant\n<|recipient|>all\n<|content|>xxx
            return f"{self.from_token}{role}\n{self.recipient_token}all\n{self.content_token}{content}{self.stop_token}\n"

        result = ""
        if content is not None:  # both text-response and function_call
            result += f"{self.from_token}{role}\n{self.recipient_token}all\n{self.content_token}{content}\n"

        for tool in tool_calls:
            func_name = tool["function"]["name"]
            arguments = tool["function"]["arguments"]
            #  <|from|>assistant\n<|recipient|>func_name\n<|content|>xxxx
            result += f"{self.from_token}{role}\n{self.recipient_token}{func_name}\n{self.content_token}{arguments}\n"

        result = result.strip() + f"{self.stop_token}\n"
        return result

    def get_stop_tokens_for_generation(self) -> List[str]:
        return [self.stop_token]

    def get_assistant_prefixes(self) -> List[str]:
        return [f"{self.from_token}assistant\n{self.recipient_token}"]

    def parse_assistant_response(
        self, llm_output: str, tool_choice: Optional[Any] = None
    ) -> Dict:
        for stop in self.get_stop_tokens_for_generation():
            if llm_output.endswith(stop):
                llm_output = llm_output[: -len(stop)]

        recipient_to_fill = ""
        if tool_choice is not None:
            if tool_choice == "none":
                recipient_to_fill = (
                    self.get_predefined_function_names(
                        function_types=PredefinedFuncTypes.no_tool_call
                    )[0]
                    + self.fn_param_sep_token
                )
            elif isinstance(tool_choice, Tool):
                recipient_to_fill = tool_choice.function.name + self.fn_param_sep_token

        llm_output = (
            f"{self.from_token}assistant\n{self.recipient_token}"
            + recipient_to_fill
            + llm_output
        )
        responses = llm_output.split(self.from_token)
        responses = [response.strip() for response in responses]

        tool_calls = []
        text_response = None
        for response in responses:
            if len(response) == 0:
                continue
            # response = assistant<|recipient|>xxxx\n<|content|>yyy
            recipient_index = response.find(self.recipient_token)
            content_index = response.find(self.content_token)
            recipient = response[
                recipient_index + len(self.recipient_token) : content_index
            ].strip()
            content = response[content_index + len(self.content_token) :].strip()
            # print(f"recipient: {recipient}, content={content}")
            if recipient == "all":
                text_response = content
            else:
                tool_calls.append(
                    {
                        "function": {"name": recipient, "arguments": content},
                        "id": prompt_utils.get_random_tool_call_id(),
                        "type": "function",
                    }
                )

        return {"role": "assistant", "content": text_response, "tool_calls": tool_calls}

    def pre_process_messages_before_inference(self, messages: List[Dict]) -> List[Dict]:
        """re-order the messages where role = tool to match the order in tool_calls by tool_call_id
        Args:
            messages (List[Dict]): list of messages containing: tool_call_id

        Returns:
            List[Dict]: _description_
        """
        return prompt_utils.reorder_tool_messages_by_tool_call_ids(messages)

    def get_recipient(self, current_text: str) -> str:
        """Get recipient from the llm_output

        Args:
            current_text (str): _description_

        Returns:
            str: _description_
        """
        recipient_index = current_text.find(self.recipient_token)
        start_index = 0
        if recipient_index >= 0:
            start_index = recipient_index + len(self.recipient_token)

        end_index = current_text.find(f"\n{self.content_token}")
        return current_text[start_index:end_index].strip()

    def get_chat_template_jinja(self) -> str:
        chat_template = """{% for message in messages %}
        {% if message['role'] == 'user' or message['role'] == 'system' %}
            {{ '<|from|>' + message['role'] + '\n<|recipient|>all\n<|content|>' + message['content'] + '\n' }}<br>
        {% elif message['role'] == 'tool' %}
            {{ '<|from|>' + message['name'] + '\n<|recipient|>all\n<|content|>' + message['content'] + '\n' }}<br>
        {% else %}
            {% set contain_content='no'%}
            {% if message['content'] is not none %}
                {{ '<|from|>assistant\n<|recipient|>all\n<|content|>' + message['content'] }}<br>
                {% set contain_content='yes'%}
            {% endif %}
            {% if 'tool_calls' in message and message['tool_calls'] is not none %}
                {% for tool_call in message['tool_calls'] %}
                    {% set prompt='<|from|>assistant\n<|recipient|>' + tool_call['function']['name'] + '\n<|content|>' + tool_call['function']['arguments'] %}
                    {% if loop.index == 1 and contain_content == "no" %}
                        {{ prompt }}<br>
                    {% else %}
                        {{ '\n' + prompt}}<br>
                    {% endif %}
                {% endfor %}
            {% endif %}
            {{ '<|stop|>\n' }}<br>
        {% endif %}
        {% endfor %}
        {% if add_generation_prompt %}{{ '<|from|>assistant\n<|recipient|>' }}{% endif %}
        """
        chat_template = chat_template.replace("    ", "")
        chat_template = chat_template.replace("<br>\n", "")
        chat_template = chat_template.strip()
        return chat_template

    def update_response_state_from_delta_text(
        self,
        *,
        current_state: Dict[str, Any],
        delta_text: str,
        finish_reason: Optional[str],
        tool_choice: Any,
    ) -> Tuple[Dict[str, Any], Union[None, Dict, List[Dict]]]:
        func_name = None
        response_type = None
        skip_until_reach = ""

        if len(current_state) == 0:  # empty dict, at the first_time
            if tool_choice == "auto":
                response_type, skip_until_reach = None, self.content_token
                func_name = None
            elif tool_choice == "none":
                response_type, skip_until_reach = "text", ""
                func_name = self.predefined_func_names[PredefinedFuncTypes.no_tool_call]
            elif type(tool_choice) is not str:
                response_type, skip_until_reach = "function", ""
                func_name = tool_choice.function.name

            current_state = {
                "current_text": "",  # the concatenation of all tokens so far
                "func_name": func_name,  # function_name of the current tool, if the response requires to use tool
                "response_type": response_type,  # response_type=text(text response)/function (using tool)
                "func_index": -1,  # index of the tool in tool_calls
                "call_id": None,  # call_id of the current tool
                # skip_until_reach we skip new tokens until we reach certain token. This is used when we hit special tokens
                "skip_until_reach": skip_until_reach,  # at first we don't need to skip as we are generating function
                "first_time": True,  # if first_time we return an tempty delta with role=assistant
            }
        # check if previous token is <|content|>, there might be a space between this token and next token (ex, <|content|> Hello)
        if current_state["current_text"].endswith(self.content_token):
            if delta_text[0] == " ":
                delta_text = delta_text[1:]

        current_state["current_text"] += delta_text
        if finish_reason is not None:
            if current_state["response_type"] == "function":
                finish_reason = "tool_calls"

            return current_state, prompt_utils.get_text_delta_response(
                None, False, finish_reason
            )

        skip_until_reach = current_state.get("skip_until_reach", "")
        if skip_until_reach:
            if delta_text != skip_until_reach:
                return current_state, None
            else:
                current_state["skip_until_reach"] = ""  # once hit, no need to skip
                recipient = self.get_recipient(current_state["current_text"])
                first_time = current_state["first_time"]
                current_state["first_time"] = False

                if recipient == "all":
                    current_state["response_type"] = "text"
                    return current_state, prompt_utils.get_text_delta_response(
                        "", True, finish_reason
                    )
                else:
                    current_state["response_type"] = "function"
                    current_state["func_name"] = recipient
                    current_state["call_id"] = prompt_utils.get_random_tool_call_id()
                    current_state["func_index"] += 1

                    return current_state, prompt_utils.get_function_delta_response(
                        current_state, "", True, False, finish_reason
                    )
        else:
            assert current_state["response_type"] is not None

            first_time = current_state["first_time"]
            if (
                delta_text == self.from_token
            ):  # skip until reach <content> to check type of response
                current_state["current_text"] = ""
                current_state["skip_until_reach"] = self.content_token
                current_state["response_type"] = None
                return current_state, None

            else:
                if current_state["response_type"] == "function":
                    if first_time:
                        current_state["call_id"] = (
                            prompt_utils.get_random_tool_call_id()
                        )
                        current_state["func_index"] += 1
                        responses = []
                        responses.append(
                            prompt_utils.get_function_delta_response(
                                current_state, "", first_time, False, finish_reason
                            )
                        )
                        current_state["first_time"] = False
                        responses.append(
                            prompt_utils.get_function_delta_response(
                                current_state, delta_text, False, False, finish_reason
                            )
                        )
                    else:
                        responses = prompt_utils.get_function_delta_response(
                            current_state, delta_text, False, False, finish_reason
                        )
                    return current_state, responses
                else:  # response_type=text
                    responses = []
                    if first_time:
                        current_state["first_time"] = False
                        responses.append(
                            prompt_utils.get_text_delta_response(
                                "", True, finish_reason
                            )
                        )
                    responses.append(
                        prompt_utils.get_text_delta_response(
                            delta_text, True, finish_reason
                        )
                    )

                    return current_state, responses

    def get_force_function_call_prefix(self, function_name: str):
        return f"{function_name}{self.fn_param_sep_token}"
