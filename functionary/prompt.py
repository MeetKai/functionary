from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import torch


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
            if "function_call" in message:
                return EndToken.function_call
            else:
                return EndToken.assistant


def get_text_from_message(message: Dict) -> str:
    """convert a message to a string to be included in the prompt

    Args:
        message (Dict): A dictionary in OpenAI format (containing: role, content, function_call (optional))

    Returns:
        str: the string used in the final prompt of this message
    """
    stop_token = EndToken.from_message(message).value
    content = message.get("content", "")

    if content is not None:
        content = f"{content}{stop_token}"

    if message["role"] == "system":
        text = "system:\n{content}\n".format(content=content)

    elif message["role"] == "function":
        text = "function name={name}:\n{content}\n".format(
            name=message.get("name", ""), content=content
        )

    elif message["role"] == "user" and content is None:
        text = "user:\n"

    elif message["role"] == "user":
        text = "user:\n{content}\n".format(content=content)

    elif message["role"] == "assistant":
        function = None
        arguments = None
        if (
            "function_call" in message
        ):  # format of openai: {"role": assistant, "function_call": {"name": xxx, "arguments": xxx}}
            function = message["function_call"]["name"]
            arguments = message["function_call"]["arguments"] + stop_token
            text = f"assistant to={function}:\n{arguments}\n"
        elif content is not None:  # this is text content
            text = f"assistant:\n{content}\n"
        else:  # if no function call and content is None --> this is used at inference
            text = "assistant"

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


def get_prompt_from_messages(messages: List[Dict]) -> str:
    """return the final prompt that will be used.
    Args:
        messages (List[Dict]): list of messages where each message is in the format of OpenAI

    Returns:
        str: the final prompt that will be used.
    """
    result = ""
    for mess in messages:
        result += get_text_from_message(mess)
    return result


def get_token_id_to_end_token(tokenizer: Any) -> Dict[int, EndToken]:
    """return a dictionary mapping from token_id --> end_token

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
        result[tok_ids[-1]] = item
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
