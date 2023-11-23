from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import re

import string
import random
from abc import ABC, abstractmethod

from functionary.schema import generate_schema_from_functions

SYSTEM_MESSAGE = """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. The assistant calls functions with appropriate input when necessary"""


class PromptTemplate:
    _instance = None

    @abstractmethod
    def get_additional_tokens(self) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def get_text_from_message(self, message: Dict) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_stop_tokens_for_generation(self) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def get_assistant_prefixes(self) -> List[str]:
        raise NotImplementedError

    def pre_process_messages_before_inference(self, messages: List[Dict]) -> List[Dict]:
        return messages

    def get_prompt_from_messages(
        self, messages: List[Dict], tools_or_functions: Optional[List[Dict]] = None
    ) -> str:
        messages_clone = messages.copy()  # To avoid modifying the original list

        functions = []
        if tools_or_functions is not None:
            for item in tools_or_functions:
                if (
                    "function" in item
                ):  #  new data format: tools: [{"type": xx, "function": xxx}]
                    functions.append(item["function"])
                else:
                    functions.append(item)  #  old format

        messages_clone.insert(
            0, {"role": "system", "content": generate_schema_from_functions(functions)}
        )
        messages_clone.insert(1, {"role": "system", "content": SYSTEM_MESSAGE})

        full_text = ""
        for message in messages_clone:
            full_text += self.get_text_from_message(message)
        return full_text.strip()

    def get_end_token_to_token_id(self, tokenizer: Any) -> Dict[str, int]:
        """return a dictionary mapping from end_token --> token_id

        Args:
            tokenizer (Any): tokenizer in transformers

        Returns:
            Dict[int, EndToken]: the mapping from token_id --> end_token
        """
        result = {}
        for item in self.get_stop_tokens_for_generation():
            tok_ids = tokenizer.encode(item.value, add_special_tokens=False)
            assert len(tok_ids) <= 2
            if len(tok_ids) == 2:
                assert (
                    tok_ids[0] == 29871
                )  # Llama tokenizer adds this token intentionally
            result[item] = tok_ids[-1]
        return result

    @abstractmethod
    def parse_assistant_response(self, llm_ouput: str) -> Dict:
        raise NotImplementedError

    @abstractmethod
    def update_response_state_from_delta_text(
        self,
        *,
        current_state: Dict[str, Any],
        delta_text: str,
        finish_reason: Optional[str],
    ) -> Tuple[Dict[str, Any], Union[None, Dict, List[Dict]]]:
        """This function is used for streaming

        Args:
            current_state (Dict[str, Any]): a dictionary:
                + func_name: Optional[str],
                + response_type: Optional[str] text/function
            current_text: the llm_output until now
            delta_text: new token generated
            finish_reason: if finished or not

        Returns:
            Tuple[Dict[str, Any], Optional[Dict]]: {func_name, response_type}, response
        """
        raise NotImplementedError

    @classmethod
    def get_prompt_template(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


class PromptTemplateV1(PromptTemplate):
    start_function = "<|START_OF_FUNCTION_CALL|>"
    end_system = "<|END_OF_SYSTEM|>"
    end_user = "<|END_OF_USER|>"
    end_assistant = "<|END_OF_ASSISTANT|>"
    end_function = "<|END_OF_FUNCTION_RESULT|>"
    end_function_call = "<|END_OF_FUNCTION_CALL|>"

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

    def get_additional_tokens(self) -> List[str]:
        return [
            self.start_function,
            self.end_system,
            self.end_user,
            self.end_assistant,
            self.end_function,
            self.end_function_call,
        ]

    def get_text_from_message(self, message: Dict) -> str:
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

        elif message["role"] == "function":
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

    def parse_assistant_response(self, llm_ouput: str) -> Dict:
        generated_content = llm_ouput.strip()
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
    ) -> Tuple[Dict[str, Any], Optional[Dict]]:
        cur_text += delta_text
        response_type = current_state["response_type"]
        func_name = current_state["func_name"]

        response: Optional[Dict[str, Any]] = None
        if response_type is None:
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
                    response_type = "function"
            else:  # if text_response
                response_type = "text"
                response = {
                    "delta": {"content": "", "role": "assistant"},
                    "finish_reason": None,
                }

        elif response_type == "function":
            if finish_reason is None:
                response = {
                    "delta": {
                        "role": "assistant",
                        "function_call": {"arguments": delta_text},
                    },  # format of openAI at the second return, don't need to add function_name
                    "finish_reason": None,
                }
            else:
                response = {
                    "delta": {},
                    "finish_reason": "function_call",
                }  # format of openAI at the end, delta must be empty

        elif response_type == "text":
            if finish_reason is None:
                # need to check if call a function or not
                if cur_text.endswith(self.start_function):  # if call another function
                    print("call another function in the mean time")
                    cur_text = self.start_function
                    response_type = None
                else:
                    response = {
                        "delta": {"content": delta_text, "role": "assistant"},
                        "finish_reason": None,
                    }
            else:  # finish generating
                response = {
                    "delta": {},
                    "finish_reason": finish_reason,
                }  # format of openAI at the end, delta must be empty
        return {"response_type": response_type, "func_name": func_name}, response


class PromptTemplateV2(PromptTemplate):
    from_token = "<|from|>"
    recipient_token = "<|recipient|>"
    content_token = "<|content|>"
    stop_token = "<|stop|>"

    def get_additional_tokens(self) -> List[str]:
        return [
            self.from_token,
            self.recipient_token,
            self.content_token,
            self.stop_token,
        ]

    def get_text_from_message(self, message: Dict) -> str:
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

    def parse_assistant_response(self, llm_ouput: str) -> Dict:
        for stop in self.get_stop_tokens_for_generation():
            if llm_ouput.endswith(stop):
                llm_ouput = llm_ouput[: -len(stop)]
        print("---------------------------")
        llm_ouput = f"{self.from_token}assistant\n{self.recipient_token}" + llm_ouput
        print(llm_ouput)
        responses = llm_ouput.split(self.from_token)
        responses = [response.strip() for response in responses]
        functions = []
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
                functions.append({"name": recipient, "arguments": content})

        tool_calls = []
        for func in functions:
            tool_calls.append(
                {"function": func, "id": get_random_tool_call_id(), "type": "function"}
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
        return result

    def get_function_delta_response(
        self,
        current_state: Dict,
        delta_text: str,
        first_call: bool,
        return_role: bool,
        finish_reason: Optional[str],
    ) -> Dict:
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
        recipient_index = current_text.find(self.recipient_token)
        start_index = 0
        if recipient_index >= 0:
            start_index = recipient_index + len(self.recipient_token)
        end_index = current_text.find(f"\n{self.content_token}")
        return current_text[start_index:end_index].strip()

    def update_response_state_from_delta_text(
        self,
        *,
        current_state: Dict[str, Any],
        delta_text: str,
        finish_reason: Optional[str],
    ) -> Tuple[Dict[str, Any], Union[None, Dict, List[Dict]]]:
        if len(current_state) == 0:  # empty dict, at the first_time
            current_state = {
                "current_text": "",
                "func_name": None,
                "response_type": None,
                "func_index": -1,
                "call_id": None,
                "skip_until_reach": self.content_token,  # at first we will skip until reach <|content|>
                "first_time": True,
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


def get_default_prompt_template() -> PromptTemplate:
    """Return default prompt template to be used

    Returns:
        _type_: _description_
    """
    return PromptTemplateV2.get_prompt_template()


def get_prompt_template(version: int) -> PromptTemplate:
    if version == "1":
        return PromptTemplateV1.get_prompt_template()
    return PromptTemplateV2.get_prompt_template()
