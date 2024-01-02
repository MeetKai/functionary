import json
import random
import string
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from functionary.prompt_template.base_template import PromptTemplate


class PromptTemplateV2(PromptTemplate):
    from_token = "<|from|>"
    recipient_token = "<|recipient|>"
    content_token = "<|content|>"
    stop_token = "<|stop|>"
    version = "v2"
    # This token splits between function name and parameters
    fn_param_sep_token = "\n<|content|> {"

    def get_start_of_function_call_token(self) -> str:
        return self.recipient_token

    def get_stop_token_for_function_parameter(
        self, stage: Literal["function", "parameter"]
    ) -> int:
        if stage == "function":
            return "\n"  # 13
        else:
            return '":'  # 1264

    def get_predefined_function_names(self) -> List[str]:
        return ["all"]

    def initialize_grammar_sampling_gen_state(self, tool_choice: Optional[Any]) -> Dict:
        if tool_choice is not None:
            if isinstance(tool_choice, str):
                add_predefined_fns = True
            else:
                add_predefined_fns = False
        else:
            add_predefined_fns = True

        return {
            "stage": "function",
            "curr_tokens": [],
            "curr_text": "",
            "func_name": "",
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

    def parse_assistant_response(self, llm_output: str) -> Dict:
        for stop in self.get_stop_tokens_for_generation():
            if llm_output.endswith(stop):
                llm_output = llm_output[: -len(stop)]

        llm_output = f"{self.from_token}assistant\n{self.recipient_token}" + llm_output
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
                        "id": get_random_tool_call_id(),
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
        result = []
        index = 0
        while index < len(messages):
            message = messages[index]
            tool_calls = message.get("tool_calls", None)

            result.append(message)
            if message["role"] == "assistant" and tool_calls:
                num_calls = len(tool_calls)
                if (
                    tool_calls[0].get("id", None) is not None
                ):  # if tool_call contains "id" for mapping
                    tool_call_ids = [item["id"] for item in tool_calls]

                    tool_messages = [messages[index + 1 + j] for j in range(num_calls)]
                    id_2_tool_messages = {
                        item["tool_call_id"]: item for item in tool_messages
                    }
                    new_messages = [id_2_tool_messages[cid] for cid in tool_call_ids]

                    result.extend(new_messages)
                    index += num_calls + 1
                else:
                    index += 1
            else:
                index += 1
        return result

    def get_function_delta_response(
        self,
        current_state: Dict,
        delta_text: str,
        first_call: bool,
        return_role: bool,
        finish_reason: Optional[str],
    ) -> Dict:
        """Return delta for tool_call in streaming

        Args:
            current_state (Dict): _description_
            delta_text (str): _description_
            first_call (bool): _description_
            return_role (bool): _description_
            finish_reason (Optional[str]): _description_

        Returns:
            Dict: _description_
        """
        return {
            "delta": {
                "content": None,
                "function_call": None,
                "role": None if not return_role else "assistant",
                "tool_calls": [
                    {
                        "index": current_state["func_index"],
                        "id": current_state["call_id"]
                        if first_call
                        else None,  # only return call_id at the first time
                        "function": {
                            "arguments": delta_text,
                            "name": current_state["func_name"] if first_call else None,
                        },
                        "type": "function" if first_call else None,
                    }
                ],
            },
            "finish_reason": finish_reason,
            "index": 0,
        }

    def get_text_delta_response(
        self, delta_text: Optional[str], return_role: bool, finish_reason: Optional[str]
    ) -> Dict:
        """Return delta for text_response in streaming

        Args:
            delta_text (Optional[str]): _description_
            return_role (bool): _description_
            finish_reason (Optional[str]): _description_

        Returns:
            Dict: _description_
        """
        return {
            "delta": {
                "content": delta_text,
                "function_call": None,
                "role": None if not return_role else "assistant",
                "tool_calls": None,
            },
            "finish_reason": finish_reason,
            "index": 0,
        }

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
    ) -> Tuple[Dict[str, Any], Union[None, Dict, List[Dict]]]:
        if len(current_state) == 0:  # empty dict, at the first_time
            current_state = {
                "current_text": "",  # the concatenation of all tokens so far
                "func_name": None,  # function_name of the current tool, if the response requires to use tool
                "response_type": None,  # response_type=text(text response)/function (using tool)
                "func_index": -1,  # index of the tool in tool_calls
                "call_id": None,  # call_id of the current tool
                # skip_until_reach we skip new tokens until we reach certain token. This is used when we hit special tokens
                "skip_until_reach": self.content_token,  # at first we will skip until reach <|content|>
                "first_time": True,  # if first_time we return an tempty delta with role=assistant
            }
        current_state["current_text"] += delta_text

        if finish_reason is not None:
            if current_state["response_type"] == "function":
                finish_reason = "tool_calls"
            return current_state, self.get_text_delta_response(
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
                    return current_state, self.get_text_delta_response(
                        "", True, finish_reason
                    )
                else:
                    current_state["response_type"] = "function"
                    current_state["func_name"] = recipient
                    current_state["call_id"] = get_random_tool_call_id()
                    current_state["func_index"] += 1

                    responses = []
                    if (
                        first_time
                    ):  # first chunk of function_call is a message where all fields are None, except role
                        responses.append(
                            self.get_text_delta_response(None, True, finish_reason)
                        )
                    responses.append(
                        self.get_function_delta_response(
                            current_state, "", True, False, finish_reason
                        )
                    )
                    return current_state, responses
        else:
            assert current_state["response_type"] is not None
            if (
                delta_text == self.from_token
            ):  # skip until reach <content> to check type of response
                current_state["current_text"] = ""
                current_state["skip_until_reach"] = self.content_token
                current_state["response_type"] = None
                return current_state, None

            else:
                if current_state["response_type"] == "function":
                    return current_state, self.get_function_delta_response(
                        current_state, delta_text, False, False, finish_reason
                    )
                else:  # response_type=text
                    return current_state, self.get_text_delta_response(
                        delta_text, True, finish_reason
                    )


def get_random_tool_call_id():
    return "call_" + "".join(
        [random.choice(string.ascii_letters + string.digits) for _ in range(24)]
    )
