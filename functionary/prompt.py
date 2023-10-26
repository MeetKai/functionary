from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import torch

from functionary.schema import generate_schema_from_functions

SYSTEM_MESSAGE = """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. The assistant calls functions with appropriate input when necessary"""


class EndToken(str, Enum):
    system = "<|END_OF_SYSTEM|>"
    user = "<|END_OF_USER|>"
    assistant = "<|END_OF_ASSISTANT|>"
    function = "<|END_OF_FUNCTION_RESULT|>"
    function_call = "<|END_OF_FUNCTION_CALL|>"

    @classmethod
    def from_message(cls, message: Dict) -> EndToken:
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
            return EndToken.user
        elif role == "system":
            return EndToken.system
        elif role == "function":
            return EndToken.function
        else:  # role = assistant
            if "function_call" in message and message["function_call"] is not None:
                return EndToken.function_call
            else:
                return EndToken.assistant


class StartToken(str, Enum):
    function = "<|START_OF_FUNCTION_CALL|>"


def get_additional_tokens() -> List[str]:
    """Return list of addional tokens, at training, you should use this function to get list of added tokens

    Returns:
        List[str]: List of added tokens
    """
    end_tokens = [e.value for e in EndToken]
    start_tokens = [e.value for e in StartToken]
    return start_tokens + end_tokens


def get_text_from_message(message: Dict) -> str:
    """convert a message to a string to be included in the prompt

    Args:
        message (Dict): A dictionary in OpenAI format (containing: role, content, function_call (optional))

    Returns:
        str: the string used in the final prompt of this message
    """
    end_token = EndToken.from_message(message).value
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
                text = f"assistant:\n{content}\n{StartToken.function.value}{function}:\n{arguments}\n"
            else:
                text = f"assistant:\n{StartToken.function.value}{function}:\n{arguments}\n"
        elif content is not None:  # this is text content
            text = f"assistant:\n{content}{end_token}\n"
        else:  # if no function call and content is None --> this is used at inference
            text = "assistant:"

    return text


def convert_old_format_to_openai_format(message: Dict) -> Dict:
    """convert {"to": xxx, "content": xxx} --> {"function_call": {"name": xxx, "arguments": xxx}}

    Args:
        message (Dict): dictionary in old format ({"to": xxx, "content": xxx})

    Returns:
        Dict: dictionary in openai format
    """
    if "to" not in message:
        return message
    return {
        "role": message["role"],
        "content": None,
        "function_call": {"name": message["to"], "arguments": message["content"]},
    }


def get_prompt_from_messages(messages: List[Dict], functions: Optional[List[Dict]] = []) -> str:
    """return the final prompt that will be used.
    Args:
        messages (List[Dict]): list of messages where each message is in the format of OpenAI
        functions (Optional[List[Dict]]): list of functions where each function is in the format of OpenAI

    Returns:
        str: the final prompt that will be used.
    """
    messages_clone = messages.copy()  # To avoid modifying the original list

    if functions is None:
        functions = []

    if len(messages_clone) > 0 and messages_clone[0]["role"] != "system":
        messages_clone.insert(0, {"role": "system", "content": generate_schema_from_functions(functions)})
        messages_clone.insert(1, {"role": "system", "content": SYSTEM_MESSAGE})

    full_text = ""
    for message in messages_clone:
        full_text += get_text_from_message(message)
    return full_text.strip()


def get_end_token_to_token_id(tokenizer: Any) -> Dict[EndToken, int]:
    """return a dictionary mapping from end_token --> token_id

    Args:
        tokenizer (Any): tokenizer in transformers

    Returns:
        Dict[int, EndToken]: the mapping from token_id --> end_token
    """
    result = {}
    for item in EndToken:
        tok_ids = tokenizer.encode(item.value, add_special_tokens=False)
        assert len(tok_ids) <= 2
        if len(tok_ids) == 2:
            assert tok_ids[0] == 29871  # Llama tokenizer adds this token intentionally
        result[item] = tok_ids[-1]
    return result


def get_number_of_tokens_of_prefix_assistant(tokenizer: Any) -> int:
    """This function is used to compute the number of tokens of "\nassistant" in full prompt

    Args:
        tokenizer (_type_): The tokenizer used for tokenizing

    Returns:
        _type_: number of tokens of "\nassistant" in full prompt
    """
    text1 = "<|END_OF_USER|>\nassistant"
    tok_ids1 = tokenizer.encode(text1, add_special_tokens=False)
    text2 = "<|END_OF_USER|>"
    tok_ids2 = tokenizer.encode(text2, add_special_tokens=False)
    return len(tok_ids1) - len(tok_ids2)
