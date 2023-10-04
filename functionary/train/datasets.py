import json
from typing import Any, Dict, List, Optional, Tuple, Union

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


def prepare_training_inputs(
    messages: Dict[str, List],
    tokenizer: Any,
    padding: str = "max_length",
    max_length: Optional[int] = None,
    return_tensor: bool = True,
    verbose=False,
) -> Dict[str, Union[str, Dict]]:
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
        Dict[str, Union[str, Dict]]: {"final_prompt": str, "inputs": Dict}
            final_prompt: the final prompt to be used,
            inputs: a dictionary containing: input_ids, attention_mask, labels. This will be used in model.forward(**inputs)
    """
    # a dictionary mapping from token_id --> end_token
    id_to_endtoken = get_token_id_to_end_token(tokenizer)
    prompt_str = get_prompt_from_messages(
        messages["messages"], messages["functions"]
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

    return dict(final_prompt=prompt_str, inputs=input_dic)


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
            "input_ids": ret["inputs"]["input_ids"],
            "labels": ret["inputs"]["labels"],
            "attention_mask": ret["inputs"]["attention_mask"],
        }
        self.cached_data_dict[i] = ret
        return ret
