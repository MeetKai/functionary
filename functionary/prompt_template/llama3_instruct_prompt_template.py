import json
import random
import string
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from functionary.prompt_template import prompt_utils
from functionary.prompt_template.base_template import PromptTemplate


def convert_to_llama3_messages(messages: List[Dict]) -> List[Dict]:
    result = []
    index = 0
    while index < len(messages):
        if messages[index]["role"] in ["user", "system"]:
            result.append(messages[index])
            index += 1
        else:
            if messages[index]["role"] == "assistant":
                tool_calls = messages[index].get("tool_calls", [])
                if len(tool_calls) == 0:
                    result.append(messages[index])
                else:
                    messages


class Llam3InstructTemplate(PromptTemplate):
    function_separator = "<|reserved_special_token_249|>"
    version = "v2.llama3_instruct"

    def get_additional_tokens(self) -> List[str]:
        return []

    def get_assistant_prefixes(self) -> List[str]:
        # return [f"{self.from_token}assistant\n{self.recipient_token}"]
        return ["<|start_header_id|>assistant<|end_header_id|>\n\n"]

    def get_stop_tokens_for_generation(self) -> List[str]:
        return ["<|eot_id|>", "<|end_of_text|>"]

    def parse_assistant_response(
        self, llm_output: str, tool_choice: Any | None
    ) -> Dict:
        # first remove stop tokens if there exists
        for stop in self.get_stop_tokens_for_generation():
            if llm_output.endswith(stop):
                llm_output = llm_output[: -len(stop)]

        chunks = llm_output.split(self.function_separator)
        chunks = [chunk.strip() for chunk in chunks if len(chunk.strip()) > 0]

        tool_calls = []
        text_content = None
        tool_call_start_index = 0

        if not llm_output.startswith(
            self.function_separator
        ):  # if first token is not function_call
            text_content = chunks[0]
            tool_call_start_index = 1

        for chunk in chunks[tool_call_start_index:]:
            # format: function_name\narguments<end_of_functioncall>
            index = chunk.find("\n")
            func_name = chunk[:index].strip()
            arguments = chunk[index + 1 :].strip()
            tool_calls.append(
                {
                    "function": {"name": func_name, "arguments": arguments},
                    "id": prompt_utils.get_random_tool_call_id(),
                    "type": "function",
                }
            )
        return {"role": "assistant", "content": text_content, "tool_calls": tool_calls}

    def convert_message_to_prompt(self, message: Dict) -> str:
        role = message["role"]
        content = message.get("content", None)
        if role == "tool":
            tool_name = message["name"]
            content = f"name={tool_name}\n{content}"

        prompt_template = (
            "<|start_header_id|>%s<|end_header_id|>\n\n{text}<|eot_id|>" % role
        )

        if role in ["user", "system", "tool"]:
            return prompt_template.format(text=content)

        assert role == "assistant"
        tool_calls = message.get("tool_calls", [])
        if tool_calls is None:
            tool_calls = []

        if content is None and len(tool_calls) == 0:
            return f"<|start_header_id|>{role}<|end_header_id|>\n\n"

        if content is not None and len(tool_calls) == 0:  # text-only
            return prompt_template.format(text=content)

        tool_call_prompts = []
        for tool_call in tool_calls:
            arguments = tool_call["function"]["arguments"]
            tool_name = tool_call["function"]["name"]
            tool_prompt = f"{tool_name}\n{arguments}"
            tool_call_prompts.append(tool_prompt)

        if (
            content is None and len(tool_calls) > 0
        ):  # function call only (<sep>fc1<sep>fc2<sep>)
            tool_call_content = self.function_separator + self.function_separator.join(
                tool_call_prompts
            )
            return prompt_template.format(text=tool_call_content)

        # Here is the case contains both text-response and tool_calls (content<sep>fc1<sep>fc2<sep>)
        total_content = (
            content
            + self.function_separator
            + self.function_separator.join(tool_call_prompts)
        )
        return prompt_template.format(text=total_content)

    def pre_process_messages_before_inference(self, messages: List[Dict]) -> List[Dict]:
        return prompt_utils.reorder_tool_messages_by_tool_call_ids(messages)

    def get_chat_template_jinja(self):
        return "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"

    def update_state_for_function(self, current_state):
        current_state["response_type"] = "function"
        current_state["skip_until_reach"] = "\n"
        current_state["current_text"] = ""
        current_state["func_index"] += 1
        current_state["call_id"] = prompt_utils.get_random_tool_call_id()

    def update_response_state_from_delta_text(
        self,
        *,
        current_state: Dict[str, Any],
        delta_text: str,
        finish_reason: Optional[str],
    ) -> Tuple[Dict[str, Any], Union[None, Dict, List[Dict]]]:
        """This function is used for streaming

        Args:
            current_state (Dict[str, Any]):  a dictionary containing the state of the streaming: such as current function_name,
            delta_text: new token generated
            finish_reason: if finished or not

        Returns:
            Tuple[Dict[str, Any], Optional[Dict]]: updated state, response: can be None, a dictionary: {} or a list of dictionary: [{}, ..., {}]
        """
        if len(current_state) == 0:  # empty dict, at the first_time
            current_state = {
                "current_text": "",  # the concatenation of all tokens so far
                "func_name": None,  # function_name of the current tool, if the response requires to use tool
                "response_type": None,  # response_type=text(text response)/function (using tool)
                "func_index": -1,  # index of the tool in tool_calls
                "call_id": None,  # call_id of the current tool
                # skip_until_reach we skip new tokens until we reach certain token. This is used when we hit special tokens
                "skip_until_reach": None,  # at first we will skip until reach <|content|>
                "first_time": True,  # if first_time we return an tempty delta with role=assistant
            }

        current_state["current_text"] += delta_text

        if finish_reason is not None:  # handle if finish
            if current_state["response_type"] == "function":
                finish_reason = "tool_calls"
            return current_state, prompt_utils.get_text_delta_response(
                None, False, finish_reason
            )

        skip_until_reach = current_state["skip_until_reach"]
        if skip_until_reach:  # if have to wait
            if (
                delta_text != current_state["skip_until_reach"]
            ):  # wait for until reach: "\n" to get function_name
                return current_state, None
            else:  # finally we get the end of function
                skip_until_reach = None
                current_state["skip_until_reach"] = None
                current_state["func_name"] = current_state["current_text"].strip()
                return current_state, prompt_utils.get_function_delta_response(
                    current_state, "", True, False, finish_reason
                )

        if (
            current_state["response_type"] is None
        ):  # Check if first token is function_separator
            if delta_text == self.function_separator:  # if first we call a function
                self.update_state_for_function(current_state)
                # first chunk of function_call is a message where all fields are None, except role
                return current_state, prompt_utils.get_text_delta_response(
                    None, True, finish_reason
                )
            else:
                current_state["response_type"] = "text"
                # The first chunk is always empty delta
                empty_response = prompt_utils.get_text_delta_response(
                    "", True, finish_reason
                )
                first_chunk = prompt_utils.get_text_delta_response(
                    delta_text, True, finish_reason
                )
                return current_state, [empty_response, first_chunk]
        else:  # not the first time
            if (
                delta_text == self.function_separator
            ):  # end of current text_response or function
                self.update_state_for_function(current_state)
                # only first call contains empty delta
                return current_state, None
            else:  # not starting to call a function
                if current_state["response_type"] == "text":
                    return current_state, prompt_utils.get_text_delta_response(
                        delta_text, True, finish_reason
                    )
                else:  # response_type = function
                    return current_state, prompt_utils.get_function_delta_response(
                        current_state, delta_text, False, False, finish_reason
                    )

    def get_chat_template_jinja2(self) -> str:
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
