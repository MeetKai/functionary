import json
from typing import Any, Dict, List, Optional, Tuple

import torch
import transformers
from torch.utils.data import Dataset

from functionary.prompt import (
    EndToken,
    get_number_of_tokens_of_prefix_assistant,
    get_prompt_from_messages,
    get_token_id_to_end_token,
)
from functionary.schema import generate_schema_from_functions


def split_data(raw_data, input_file, percentage):
    """Splits the raw data into train and validation sets. Saves both into new jsonl files too."""
    # Calculate the split index
    split_idx = int(len(raw_data) * percentage)
    # Split the data into training and validation sets
    train_data, val_data = raw_data[:split_idx], raw_data[split_idx:]
    # Write the training data to a new JSONL file
    with open(input_file.rstrip(".jsonl") + "_train.jsonl", "w") as f:
        for item in train_data:
            json.dump(item, f)
            f.write("\n")
    # Write the validation data to a new JSONL file
    with open(input_file.rstrip(".jsonl") + "_val.jsonl", "w") as f:
        for item in val_data:
            json.dump(item, f)
            f.write("\n")
    if torch.distributed.get_rank() == 0:
        print(
            f"Data split into training (size: {len(train_data)}) and validation (size: {len(val_data)}) sets."
        )
    return train_data, val_data


def create_target_tensors(input_ids, ignore_from=None, ignore_to=None):
    """Creates target tensors based on the ignoring range."""
    targets = input_ids.clone()
    if ignore_from is not None:
        targets[ignore_from:] = -100  # OR LabelSmoother.ignore_index
    if ignore_to is not None:
        targets[:ignore_to] = -100  # OR LabelSmoother.ignore_index
    return targets


def prepare_message_for_model_llama2chat(messages, tokenizer):
    """Prepares given messages for the model for llama2-chat models"""

    system_content = ""
    conversation_content = ""
    inst_flag = False  # Flag to check if a new turn has started

    for message in messages:
        if message["role"] == "system":
            content = message.get("content", "")
            system_content += "<<SYS>>\n{content}\n<</SYS>>\n".format(content=content)

        elif message["role"] == "user" and message.get("content") is not None:
            if inst_flag:  # Check if a new turn should start
                conversation_content += "</s><s>[INST]"
            else:
                conversation_content += "[INST]"
            content = message.get("content", "")
            conversation_content += "{content}[/INST] ".format(content=content)
        ## this condition must be before 'elif message["role"] == 'assistant' and message.get("content") is not None:'
        elif message["role"] == "assistant" and message.get("to"):
            fn_call = "to={to}:\n{content}".format(
                to=message.get("to", ""), content=message.get("content", "")
            )
            print(fn_call)
            conversation_content += "{content}".format(content=fn_call)
            inst_flag = True

        elif message["role"] == "assistant" and message.get("content") is not None:
            content = message.get("content", "")
            conversation_content += "{content}".format(content=content)
            inst_flag = True

        elif message["role"] == "function":
            text = "function name={name}:\n{content}\n".format(
                name=message.get("name", ""), content=message.get("content", "")
            )
            if inst_flag:  # Check if a new turn should start
                conversation_content += "</s><s>[INST]".format(content=text)
            else:
                conversation_content += "[INST]"
            conversation_content += "{content}[/INST] ".format(content=text)

    # Check if the last turn was not closed
    if inst_flag:
        conversation_content += "</s>"

    text = "<s>[INST]{system_content}{conversation_content}".format(
        system_content=system_content, conversation_content=conversation_content
    )
    return text


def prepare_message_for_model(message, tokenizer):
    """Prepares a given message for the model by tokenizing the content and determining target tokens."""

    if message["role"] == "system":
        text = "system:\n{content}\n".format(content=message.get("content", ""))
        input_ids = tokenizer(
            text, add_special_tokens=False, return_tensors="pt"
        ).input_ids
        targets = create_target_tensors(
            input_ids, ignore_from=0, ignore_to=len(input_ids[0])
        )

    elif message["role"] == "function":
        text = "function name={name}:\n{content}\n".format(
            name=message.get("name", ""), content=message.get("content", "")
        )
        input_ids = tokenizer(
            text, add_special_tokens=False, return_tensors="pt"
        ).input_ids
        targets = create_target_tensors(
            input_ids, ignore_from=0, ignore_to=len(input_ids[0])
        )

    elif message["role"] == "user" and message.get("content") is None:
        text = "user:\n</s>"
        input_ids = tokenizer(
            text, add_special_tokens=False, return_tensors="pt"
        ).input_ids
        targets = create_target_tensors(input_ids)

    elif message["role"] == "user":
        text = "user:\n</s>{content}\n".format(content=message.get("content", ""))
        input_ids = tokenizer(
            text, add_special_tokens=False, return_tensors="pt"
        ).input_ids
        targets = create_target_tensors(input_ids, ignore_from=4)

    elif message["role"] == "assistant" and message.get("to") is not None:
        text = "assistant to={to}:\n{content}</s>".format(
            to=message.get("to", ""), content=message.get("content", "")
        )
        input_ids = tokenizer(
            text, add_special_tokens=False, return_tensors="pt"
        ).input_ids
        targets = create_target_tensors(input_ids)

    elif message["role"] == "assistant" and message.get("content") is None:
        text = "assistant"
        input_ids = tokenizer(
            text, add_special_tokens=False, return_tensors="pt"
        ).input_ids
        targets = create_target_tensors(input_ids)

    elif message["role"] == "assistant":
        text = "assistant:\n{content}\n".format(content=message.get("content", ""))
        input_ids = tokenizer(
            text, add_special_tokens=False, return_tensors="pt"
        ).input_ids
        targets = create_target_tensors(input_ids)

    return text, input_ids, targets


def prepare_messages_for_model(messages, tokenizer):
    """Prepares a list of messages for the model by calling `prepare_message_for_model` function on each of them and
    concatenating the returned input_ids and targets. Also, the function merges the text of the messages.
    """
    all_texts = []
    all_input_ids = []
    all_targets = []

    for message in messages:
        text, input_ids, targets = prepare_message_for_model(message, tokenizer)
        all_texts.append(text)
        all_input_ids.append(input_ids.squeeze(0))
        all_targets.append(targets.squeeze(0))

    input_ids_tensor = torch.cat(all_input_ids, dim=-1)
    targets_tensor = torch.cat(all_targets, dim=-1)
    merged_text = "".join(all_texts)

    prepared_input = tokenizer.prepare_for_model(
        input_ids_tensor.tolist(),
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
    )

    prepared_targets = tokenizer.prepare_for_model(
        targets_tensor.tolist(),
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
    )

    return dict(
        input_ids=prepared_input["input_ids"],
        labels=prepared_targets["input_ids"],
        attention_mask=prepared_input["attention_mask"],
    )


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
    prompt_str = (
        "system:\n"
        + generate_schema_from_functions(functions=messages["functions"])
        + "\nsystem:\nA chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. The assistant calls functions with appropriate input when necessary\n"
    )
    prompt_str += get_prompt_from_messages(
        messages["messages"]
    )  # prompt_str is the concatenation of all prompts from messages
    max_length = max_length if max_length is not None else tokenizer.model_max_length

    input_dic = tokenizer(
        prompt_str, padding=padding, max_length=max_length, truncation=True
    )
    input_token_ids = input_dic["input_ids"]
    # first we initialize labels with all positions as -100,
    # then we will fill in positions where role=assistant as we only include these in computing the loss
    labels = [-100 for _ in range(len(input_token_ids))]
    start = 0
    # Now we find the positions where role=assistant and copy the value from input_token_ids
    # this is done by finding the chunk (previous_stop_token_index + 1, current_stop_token_index + 1)
    # where current_stop_token is EndToken.assistant or EndToken.function_call
    assistant_tok_len = get_number_of_tokens_of_prefix_assistant(
        tokenizer
    )  # get the number of tokens for "\nassistant" in full prompt, these tokens remain -100
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
                    print(
                        "+++ chunk assistant to compute loss: ", tokenizer.decode(chunk)
                    )
                    print("chunk tokens: ", chunk)
            start = index + 1

    input_dic["labels"] = labels
    assert (
        len(labels) == len(input_dic["input_ids"]) == len(input_dic["attention_mask"])
    )

    if return_tensor:
        for key in input_dic:
            input_dic[key] = torch.tensor(input_dic[key])
    return prompt_str, input_dic


class CustomDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(CustomDataset, self).__init__()
        self.tokenizer = tokenizer

        self.raw_data = raw_data
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = prepare_training_inputs(self.raw_data[i], self.tokenizer)
        ret = {
            "input_ids": ret[1]["input_ids"],
            "labels": ret[1]["labels"],
            "attention_mask": ret[1]["attention_mask"],
        }
        self.cached_data_dict[i] = ret
        return ret
