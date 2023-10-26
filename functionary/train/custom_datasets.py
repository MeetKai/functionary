import json
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import transformers
from torch.utils.data import Dataset

from functionary.prompt import (
    EndToken,
    get_number_of_tokens_of_prefix_assistant,
    get_prompt_from_messages,
    get_end_token_to_token_id,
)


def get_prefix_assistant_token_ids(tokenizer: Any):
    result = []
    for e in EndToken:
        prefix = f"{e.value}\nassistant:"
        token_ids = tokenizer.encode(prefix, add_special_tokens=False)
        if token_ids[0] == 29871:
            token_ids = token_ids[1:]
        result.append(token_ids)
    return result


def get_matching_prefix(prefix_tokens, sequence_ids):
    for prefix in prefix_tokens:
        if len(sequence_ids) >= len(prefix):
            if sequence_ids[: len(prefix)] == prefix:
                return prefix
    return None


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
    endtoken_2_id = get_end_token_to_token_id(tokenizer)
    prompt_str = get_prompt_from_messages(
        messages["messages"], messages["functions"]
    )  # prompt_str is the concatenation of all prompts from messages
    max_length = max_length if max_length is not None else tokenizer.model_max_length

    input_dic = tokenizer(prompt_str, padding=padding, max_length=max_length, truncation=True)
    input_token_ids = input_dic["input_ids"]
    # first we initialize labels with all positions as -100,
    # then we will fill in positions where role=assistant as we only include these in computing the loss
    labels = [-100 for _ in range(len(input_token_ids))]
    start = 0

    # now we will unmask labels by positions that was from assistant
    # we will find the chunks: "<endtoken>assistant ...(<end_of_function>|<end_of_assistant>) from input_token_ids
    # and unmask: this part: "...(<end_of_function>|<end_of_assistant>"
    # find token_ids of: "<endtoken>assistant"
    prefix_token_ids = get_prefix_assistant_token_ids(tokenizer)
    if verbose:
        print("prefix_token_ids: ", prefix_token_ids)
    index = 0
    total_input_leng = len(input_token_ids)
    while index < total_input_leng:
        # finding the index that start with: "<endtoken>assistant" --> we will unmask labels from this position
        matched_prefix = get_matching_prefix(prefix_token_ids, input_token_ids[index:])
        if matched_prefix is not None:
            end_index = -1
            # unmask until reach <end_of_function> or <end_of_assistant>
            for i in range(index + len(matched_prefix), total_input_leng):
                tok_id = input_token_ids[i]
                if tok_id in [
                    endtoken_2_id[EndToken.assistant],
                    endtoken_2_id[EndToken.function_call],
                ]:  # check if this is end of turn
                    labels[i] = input_token_ids[i]  # unmask labels at this position
                    end_index = i
                    break
                else:
                    labels[i] = input_token_ids[i]  # unmask labels at this position
            if verbose:
                print("------------------------")
                start = index + len(matched_prefix)
                chunk_ids = input_token_ids[start : end_index + 1] if end_index > -1 else input_token_ids[start:]
                print("chunk_ids: ", chunk_ids)
                print(
                    "longer chunk: ",
                    input_token_ids[index : end_index + 1] if end_index > 1 else input_token_ids[index:],
                )
                print(f"chunk:{tokenizer.decode(chunk_ids)}")
                print("-------------------")
            if (
                end_index == -1
            ):  # if at the end, cannot find EndToken.assistant or EndToken.function_call --> this data point was truncated
                break
            index = end_index
        else:
            index += 1

    input_dic["labels"] = labels
    assert len(labels) == len(input_dic["input_ids"]) == len(input_dic["attention_mask"])

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
