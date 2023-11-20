from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import re

import torch
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

    def get_prompt_from_messages(
        self, messages: List[Dict], tools_or_functions: Optional[List[Dict]] = None
    ) -> str:
        messages_clone = messages.copy()  # To avoid modifying the original list

        functions = []
        if tools_or_functions is not None:
            for item in tools_or_functions:
                if "function" in item: #  new data format: tools: [{"type": xx, "function": xxx}]
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
        llm_ouput = f"{self.from_token}assistant\n{self.recipient_token}" + llm_ouput
        responses = llm_ouput.split(f"{self.from_token}assistant")
        responses = [response.strip() for response in responses]

        functions = []
        text_response = None
        for response in responses:
            if len(response) == 0:
                continue
            # <|recipient|>xxxx\n<|content|>yyy
            parts = response.split(self.content_token)
            recipient = parts[0][len(self.recipient_token) :].strip()
            content = parts[1].strip()

            if recipient == "all":
                text_response = content
            else:
                functions.append({"name": recipient, "arguments": content})

        tool_calls = []
        for func in functions:
            tool_calls.append({"function": func})
        return {"role": "assistant", "content": text_response, "tool_calls": tool_calls}


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
