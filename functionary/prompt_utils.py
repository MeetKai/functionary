from typing import List, Dict, Any, Optional, Tuple
import torch
from enum import Enum


class EndToken(str, Enum):
    system = "<|END_OF_SYSTEM|>"
    user = "<|END_OF_USER|>"
    assistant = "<|END_OF_ASSISTANT|>"
    function = "<|END_OF_FUNCTION_RESULT|>"
    function_call = "<|END_OF_FUNCTION_CALL|>"


def get_end_token_for_message(message: Dict) -> EndToken:
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
    stop_token = get_end_token_for_message(message).value
    content = message.get("content", "")

    if content is not None:
        content = f"{content}{stop_token}"

    if message["role"] == "system":
        text = "system:\n{content}\n".format(content=content)

    elif message["role"] == "function":
        text = "function name={name}:\n{content}\n".format(name=message.get("name", ""), content=content)

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


def get_number_token_of_assistant(tokenizer: Any) -> int:
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


def prepare_training_inputs(
    messages: List[Dict],
    tokenizer: Any,
    padding: str = "max_length",
    max_length: Optional[int] = None,
    return_tensor: bool = True,
    verbose=False,
) -> Tuple[str, Dict]:
    """This function is used for when you want to get a dictionary input for the model.forward.
    The dictionary will contain: input_ids, attention_maks, labels.
    labels is like input_ids except that content from user, system, function will be set as -100, only content from assistant remains

    Args:
        messages (List[Dict]): List of messages in openAI format (containing: role, content and function_call (optional))
        tokenizer (Any): tokenizer from transformers
        padding (str, optional): type of padding (longest, max_length), this is passed to tokenizer(). Defaults to "max_length".
        max_length (Optional[int], optional): maximum number of tokens allowed in prompt. Defaults to None.
        return_tensor (bool, optional): if true, the input_dic will be dictionary[str, Tensor] else dictionary[str, List[int]]. Defaults to True.
        verbose (bool, optional): to print some useful information or not. Defaults to False.

    Returns:
        Tuple[str, Dict]: (final_prompt_str, inputs)
            final_prompt_str: the final prompt to be used,
            inputs: a dictionary containing: input_ids, attention_mask, labels. This will be used in model.forward(**inputs)
    """
    # a dictionary mapping from token_id --> end_token
    id_to_endtoken = get_token_id_to_end_token(tokenizer)
    prompt_str = get_prompt_from_messages(messages)  # prompt_str is the concatenation of all prompts from messages
    max_length = max_length if max_length is not None else tokenizer.model_max_length

    input_dic = tokenizer(prompt_str, padding=padding, max_length=max_length, truncation=True)
    input_token_ids = input_dic["input_ids"]
    # first we initialize labels with all positions as -100,
    # then we will fill in positions where role=assistant as we only include these in computing the loss
    labels = [-100 for _ in range(len(input_token_ids))]
    start = 0
    # Now we find the positions where role=assistant and copy the value from input_token_ids
    # this is done by finding the chunk (previous_stop_token_index + 1, current_stop_token_index + 1)
    # where current_stop_token is EndToken.assistant or EndToken.function_call
    assistant_tok_len = get_number_token_of_assistant(
        tokenizer
    )  # get the number of tokens for "\nassistant" in full prompt
    for index, tok_id in enumerate(input_token_ids):
        if tok_id in id_to_endtoken:  # Find position of end_token in final_prompt
            end_token = id_to_endtoken[tok_id]
            if (
                end_token == EndToken.assistant or end_token == EndToken.function_call
            ):  # only compute loss from tokens of assistant
                for i in range(
                    start + assistant_tok_len, index + 1
                ):  # The reason for start + assistant_tok_len is to ignore: "\nassistant" in computing loss
                    labels[i] = input_token_ids[i]  # overwrite -100 to compute the loss
                if verbose:
                    chunk = input_token_ids[start + 2 : index + 1]
                    print("----------------------------")
                    print("+++ chunk assistant to compute loss: ", tokenizer.decode(chunk))
                    print("chunk tokens: ", chunk)
            start = index + 1

    input_dic["labels"] = labels
    assert len(labels) == len(input_dic["input_ids"]) == len(input_dic["attention_mask"])

    if return_tensor:
        for key in input_dic:
            input_dic[key] = torch.tensor(input_dic[key])
    return prompt_str, input_dic
