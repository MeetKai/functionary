import json
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from functionary.prompt_template.base_template import PromptTemplate


class PromptTemplateV1(PromptTemplate):
    start_function = "<|START_OF_FUNCTION_CALL|>"
    end_system = "<|END_OF_SYSTEM|>"
    end_user = "<|END_OF_USER|>"
    end_assistant = "<|END_OF_ASSISTANT|>"
    end_function = "<|END_OF_FUNCTION_RESULT|>"
    end_function_call = "<|END_OF_FUNCTION_CALL|>"
    version = "v1"
    # This token splits between function name and parameters
    fn_param_sep_token = ":\n{"

    def get_end_token_from_message(self, message: Dict) -> str:
        """this function is used for getting the end token for each message.
        For example, if message["role"] == "user" --> return EndToken.user
        if message["role"] == "assistant" and "function_call" in message --> EndTOken.function_call

        Args:
            message (Dict): A dictionary containing: role, content, function_call(optional)

        Returns:
            EndToken: End Token for this message, this will be appended to the end of the prompt for this message
        """
        role = message["role"]
        if role == "user":
            return self.end_user
        elif role == "system":
            return self.end_system
        elif role == "function":
            return self.end_function
        else:  # role = assistant
            if message.get("function_call", None) is not None:
                # if "function_call" in message and message["function_call"] is not None:
                return self.end_function_call
            else:
                return self.end_assistant

    def get_start_of_function_call_token(self) -> str:
        return self.start_function

    def initialize_grammar_sampling_gen_state(self) -> Dict:
        return {
            "stage": "pre-function",
            "curr_tokens": [],
            "curr_text": "",
            "func_name": "",
            "param_names": [],
        }

    def get_additional_tokens(self) -> List[str]:
        return [
            self.start_function,
            self.end_system,
            self.end_user,
            self.end_assistant,
            self.end_function,
            self.end_function_call,
        ]

    def convert_message_to_prompt(self, message: Dict) -> str:
        """convert a message to a string to be included in the prompt
        Args:
            message (Dict): A dictionary in OpenAI format (containing: role, content, function_call (optional))

        Returns:
            str: the string used in the final prompt of this message
        """
        end_token = self.get_end_token_from_message(message)
        content = message.get("content", None)

        if message["role"] == "system":
            text = f"system:\n{content}{end_token}\n"

        elif message["role"] in ["function", "tool"]:
            func_name = message.get("name", "")
            text = f"function name={func_name}:\n{content}{end_token}\n"

        elif message["role"] == "user" and content is None:
            text = "user:\n"

        elif message["role"] == "user":
            text = f"user:\n{content}{end_token}\n"

        elif message["role"] == "assistant":
            if (
                message.get("function_call", None) is not None
            ):  # format of openai: {"role": assistant, "function_call": {"name": xxx, "arguments": xxx}}
                function = message["function_call"]["name"]
                arguments = message["function_call"]["arguments"] + end_token
                if content is not None:
                    text = f"assistant:\n{content}\n{self.start_function}{function}:\n{arguments}\n"
                else:
                    text = (
                        f"assistant:\n{self.start_function}{function}:\n{arguments}\n"
                    )
            elif content is not None:  # this is text content
                text = f"assistant:\n{content}{end_token}\n"
            else:  # if no function call and content is None --> this is used at inference
                text = "assistant:"

        return text

    def get_stop_tokens_for_generation(self) -> List[str]:
        return [self.end_assistant, self.end_function_call]

    def get_assistant_prefixes(self) -> List[str]:
        result = []
        for item in [self.end_user, self.end_function]:
            prefix = f"{item}\nassistant:"
            result.append(prefix)
        return result

    def parse_assistant_response(
        self, llm_output: str, tool_choice: Optional[Any] = None
    ) -> Dict:
        generated_content = llm_output.strip()

        for endtoken in self.get_stop_tokens_for_generation():
            if generated_content.endswith(endtoken):
                generated_content = generated_content[: -len(endtoken)].strip()

        # First we need to check if llm_output contains start_token or not
        start_function_index = generated_content.find(self.start_function)
        text_content = generated_content
        result = {"role": "assistant", "content": None}

        if start_function_index >= 0:
            func_info = generated_content[
                start_function_index + len(self.start_function) :
            ].strip()
            index = func_info.find(":")
            func_name = func_info[:index].strip()
            arguments = func_info[index + 1 :].strip()

            text_content = generated_content[:start_function_index].strip()
            result["function_call"] = {
                "name": func_name,
                "arguments": arguments,
            }  # FunctionCall(name=func_name, arguments=arguments)
        if len(text_content) > 0:
            result["content"] = text_content
        return result

    def update_response_state_from_delta_text(
        self,
        *,
        current_state: Dict[str, Any],
        delta_text: str,
        finish_reason: Optional[str],
        tool_choice: Any,
    ) -> Tuple[Dict[str, Any], Optional[Dict]]:
        if len(current_state) == 0:
            current_state = {
                "response_type": None,  # the type of current response text (text_response)/function (function_call)
                "func_name": None,  # if response_type=function, this is the function_name
                "current_text": "",  # the concatenation of generated tokens so far
            }
        current_state["current_text"] += delta_text
        cur_text = current_state["current_text"]

        response: Optional[Dict[str, Any]] = None
        if current_state["response_type"] is None:
            if cur_text.strip().startswith(self.start_function):  # if function_call
                if cur_text.endswith(":"):
                    f_index = cur_text.find(self.start_function)
                    func_name = cur_text[
                        f_index + len(self.start_function) : -1
                    ].strip()
                    response = {
                        "delta": {
                            "role": "assistant",
                            "content": None,
                            "function_call": {"arguments": "", "name": func_name},
                        },
                        "finish_reason": None,
                    }
                    current_state["response_type"] = "function"
            else:  # if text_response
                current_state["response_type"] = "text"
                response = {
                    "delta": {"content": "", "role": "assistant"},
                    "finish_reason": None,
                    "index": 0,
                }

        elif current_state["response_type"] == "function":
            if finish_reason is None:
                response = {
                    "delta": {
                        "role": "assistant",
                        "function_call": {"arguments": delta_text},
                    },  # format of openAI at the second return, don't need to add function_name
                    "finish_reason": None,
                    "index": 0,
                }
            else:
                response = {
                    "delta": {},
                    "finish_reason": "function_call",
                    "index": 0,
                }  # format of openAI at the end, delta must be empty

        elif current_state["response_type"] == "text":
            if finish_reason is None:
                # need to check if call a function or not
                if cur_text.endswith(self.start_function):  # if call another function
                    print("call another function in the mean time")
                    cur_text = self.start_function
                    current_state["current_text"] = self.start_function
                    current_state["response_type"] = None
                else:
                    response = {
                        "delta": {"content": delta_text, "role": "assistant"},
                        "finish_reason": None,
                        "index": 0,
                    }
            else:  # finish generating
                response = {
                    "delta": {},
                    "finish_reason": finish_reason,
                    "index": 0,
                }  # format of openAI at the end, delta must be empty
        return current_state, response

    def get_chat_template_jinja(self) -> str:
        chat_template = """{% for message in messages %}
        {% if message['role'] == 'user' %}
            {{ message['role'] + ':\n' + message['content'] + '<|END_OF_USER|>' + '\n' }}<br>
        {% elif message['role'] == 'system' %}
            {{ message['role'] + ':\n' + message['content'] + '<|END_OF_SYSTEM|>' + '\n' }}<br>
        {% elif message['role'] == 'function' %}
            {{ 'function name=' + message['name'] + ':\n' + message['content']+ '<|END_OF_FUNCTION_RESULT|>\n' }}<br>
        {% elif message['role'] == 'assistant' %}
            {% if 'function_call' in message and message['function_call'] is not none %}
                {% if message['content'] is not none %}
                    {{ 'assistant:\n' + message['content'] + '\n<|START_OF_FUNCTION_CALL|>' + message['function_call']['name'] + ':\n' + message['function_call']['arguments'] + '<|END_OF_FUNCTION_CALL|>\n' }}<br>
                {% else %}
                    {{ 'assistant:\n<|START_OF_FUNCTION_CALL|>' + message['function_call']['name'] +  ':\n' + message['function_call']['arguments'] + '<|END_OF_FUNCTION_CALL|>\n' }}<br>
                {% endif %}
            {% else %}
                {{ 'assistant:\n' + message['content'] + '<|END_OF_ASSISTANT|>' + '\n' }}<br>
            {% endif %}
        {% endif %}
        {% endfor %}
        {% if add_generation_prompt %}{{ 'assistant:' }}{% endif %}
        """
        chat_template = chat_template.replace("    ", "")
        chat_template = chat_template.replace("<br>\n", "")
        chat_template = chat_template.strip()
        return chat_template
