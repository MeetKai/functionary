import random
import string
from typing import Dict, List, Optional, Union
from PIL import Image
from io import BytesIO
import os
import base64
import requests
import torch
from transformers import LlamaTokenizer

from functionary.openai_types import ChatMessage, Function, Tool


def enforce_tool_choice(
    choice: Union[str, Tool, Function],
    tools_or_functions: Optional[List[Union[Tool, Function]]],
) -> Optional[List[Tool]]:
    """This function is used to enforce tool_choice in the list of tools if it is provided by the user

    Args:
        choice: (Union[str, Tool]): either "auto", "none" or Tool/Function object
        tools_or_functions (Optional[List[Tool, Function]]): the existing list of tools passed in from user

    Returns:
        List[Tool, Function]: the modified tools_or_functions based on tool_choice
    """

    if choice == "none":
        return []
    elif isinstance(choice, Tool):
        if choice.function.description == "" and choice.function.parameters is None:
            tools_or_functions = [
                tool
                for tool in tools_or_functions
                if tool.type == "function"
                and tool.function.name == choice.function.name
            ]
            assert (
                len(tools_or_functions) > 0
            ), f"Invalid value for 'tool_choice': no function named {choice.function.name} was specified in the 'tools' parameter"
        else:
            tools_or_functions = [choice]
    elif isinstance(choice, Function):
        tools_or_functions = [
            function for function in tools_or_functions if function.name == choice.name
        ]
        assert (
            len(tools_or_functions) > 0
        ), f"Invalid value for 'function_call': no function named {choice.name} was specified in the 'functions' parameter"

    return tools_or_functions


def prepare_messages_for_inference(
    *,
    tokenizer: LlamaTokenizer,
    messages: List[ChatMessage],
    tools_or_functions: List[Dict],
    tool_choice: Optional[Union[str, Tool, Function]] = None,
    device="cuda:0",
) -> torch.Tensor:
    """This function receives the messages and generates the final prompt tokenized by the
    tokenizer.

    Args:
        tokenizer (LlamaTokenizer): The tokenizer object
        messages (List[ChatMessage]): The list of messages for the conversation
        tools_or_functions (List[Dict]): list of tools or functions
        tool_choice (Optional[Union[str, Tool, Function]], optional): tool_choice provided by the user. Defaults to None.
        device (str, optional): device for the tokenized tensor. Defaults to "cuda:0".

    Returns:
        torch.Tensor: The tokenized tensor
    """

    # Import function in this function to prevent circular imports
    from functionary.prompt_template import get_prompt_template_from_tokenizer

    prompt_template = get_prompt_template_from_tokenizer(tokenizer)

    dic_messages = [mess.dict() for mess in messages]
    dic_messages.append({"role": "assistant"})

    dic_messages = prompt_template.pre_process_messages_before_inference(dic_messages)

    # This also checks for code_interpreter and adds python default system message instead
    # default system message
    final_prompt = prompt_template.get_prompt_from_messages(
        dic_messages, tools_or_functions=tools_or_functions
    )

    # add prefix based on tool-choice
    final_prompt += prompt_template.get_generation_prefix_for_tool_choice(tool_choice)
    input_ids = tokenizer(final_prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(device)
    return input_ids


def get_function_delta_response(
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
                    "id": (
                        current_state["call_id"] if first_call else None
                    ),  # only return call_id at the first time
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
    delta_text: Optional[str], return_role: bool, finish_reason: Optional[str]
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


def get_random_tool_call_id():
    return "call_" + "".join(
        [random.choice(string.ascii_letters + string.digits) for _ in range(24)]
    )


def reorder_tool_messages_by_tool_call_ids(messages: List[Dict]) -> List[Dict]:
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


def stringify_content_with_images(content: List[Dict], image_token: str) -> str:
    result = ""
    for item in content:
        if item["type"] == "text":
            result += item["text"]
        elif item["type"] == "image_url":
            result += image_token
    return result


def extract_images_from_messages(messages: List[Dict]) -> List:
    result = []
    for message in messages:
        if message["role"] == "user":
            if type(message["content"]) is list:
                result.extend(extract_images_from_content(message["content"]))
    return result


def extract_images_from_content(content: List[Dict]) -> List:
    result = []
    for item in content:
        if item["type"] == "image_url":
            img = download_image_from_image_url(item["image_url"]["url"])
            result.append(img)
    return result


def download_image_from_image_url(image_url: str):
    base64_prefix = "data:image/jpg;base64,"
    file_prefix = "file://"
    url_prefix = "url://"
    if image_url.startswith(base64_prefix):
        encoded_data = image_url[len(base64_prefix) :]
        return Image.open(BytesIO(base64.b64decode(encoded_data)))

    elif image_url.startswith(file_prefix):
        img_path = image_url[len(file_prefix) :].strip()
        # image = Image.open(image_file).convert('RGB')
        return Image.open(open(img_path, "rb"))

    elif image_url.startswith(url_prefix):
        url = image_url[len(url_prefix) :].strip()
        return Image.open(requests.get(url, stream=True).raw)

    else:
        raise (
            f"image not found, image_url must startswith one of: '{base64_prefix}'; '{file_prefix}', '{url_prefix}'"
        )


def inject_image_token(inputs, img_token_id):
    """This function will replace the last token with image_token at the end of: input_ids and also modify attention_mask, labels accordingly

    Args:
        inputs (_type_): input_dic containing: input_ids, labels, attention_mask
        img_token_id (_type_): _description_

    Returns:
        _type_: _description_
    """
    input_ids = inputs["input_ids"]
    label_ids = inputs["labels"]
    attention_mask = inputs["attention_mask"]
    # make sure that virtual token is always appended at the end --> won't influence the attention_scores for other tokens
    input_ids[-1] = img_token_id
    label_ids[-1] = -100  # make sure that
    attention_mask[-1] = (
        1  # allow attend so prepare_inputs_labels_for_multimodal will work
    )

    return {
        "input_ids": input_ids,
        "labels": label_ids,
        "attention_mask": attention_mask,
    }
