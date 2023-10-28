import json
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import transformers
from torch.utils.data import Dataset
import datetime

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
    padding: Optional[str] = "max_length",
    max_length: Optional[int] = None,
    return_tensor: bool = True,
    verbose=False,
) -> Dict[str, Union[str, Dict]]:
    batch_result = prepare_training_inputs_batch([messages], tokenizer, padding, max_length, return_tensor, verbose)
    return dict(final_prompt=batch_result["batch_prompts"][0], inputs=batch_result["batch_inputs"][0])


def get_masked_labels(input_token_ids: List[int], tokenizer: Any, endtoken_2_id: Dict, verbose: bool = False):
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
    return labels


def prepare_training_inputs_batch(
    batch_messages: Dict[str, List],
    tokenizer: Any,
    padding: Optional[str] = "max_length",
    max_length: Optional[int] = None,
    return_tensor: bool = True,
    verbose=False,
) -> List[Dict[str, Union[str, Dict]]]:
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
    prompt_str_list = []
    for messages in batch_messages:
        prompt_str = get_prompt_from_messages(
            messages["messages"], messages["functions"]
        )  # prompt_str is the concatenation of all prompts from messages
        prompt_str_list.append(prompt_str)
    max_length = max_length if max_length is not None else tokenizer.model_max_length

    input_dic = tokenizer(prompt_str_list, padding=padding, max_length=max_length, truncation=True)
    #input_token_ids = input_dic["input_ids"]
    batch_labels = []
    for input_token_ids in input_dic["input_ids"]:
        labels = get_masked_labels(input_token_ids, tokenizer, endtoken_2_id, verbose=verbose)
        batch_labels.append(labels)
        assert len(labels) == len(input_token_ids)

    input_dic["labels"] = batch_labels
    assert len(input_dic["labels"]) == len(input_dic["input_ids"]) == len(input_dic["attention_mask"]) == len(batch_messages)
    
    batch_inputs = []
    for i in range(len(input_dic["input_ids"])):
        inputs = {}
        for key in ["labels", "input_ids", "attention_mask"]:
            inputs[key] = input_dic[key][i]
            if return_tensor:
                inputs[key] = torch.tensor(inputs[key])
        batch_inputs.append(inputs)

    return dict(batch_prompts=prompt_str_list, batch_inputs=batch_inputs)


class CustomDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, batch_size: int = 2000):
        super(CustomDataset, self).__init__()
        self.tokenizer = tokenizer
        self.processed_data = []
        data_size = len(raw_data)
        t1 = datetime.datetime.now()
        for i in range(data_size // batch_size + 1):
            start = i * batch_size
            end = i * batch_size + batch_size
            if end > data_size:
                end = data_size
            if end > start:
                batch_result = prepare_training_inputs_batch(raw_data[start: end], tokenizer, return_tensor=True)
                assert len(batch_result["batch_inputs"]) == len(raw_data[start: end])
                for item in batch_result["batch_inputs"]:
                    self.processed_data.append(item)
            t2 = datetime.datetime.now()
            avg_time = (t2 - t1).total_seconds() / len(self.processed_data)
            remaining_time = avg_time * (data_size - len(self.processed_data))
            print(f"{len(self.processed_data)}/{data_size}, avg_time per 1000 data points: {avg_time * 1000}, remaining time: {remaining_time}")
        assert len(self.processed_data) == data_size

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return self.processed_data[i]


def convert_input_dic_to_int(input_dic: Dict) -> Dict:
    input_ids = input_dic["input_ids"]
    attention_mask = input_dic["attention_mask"]
    labels = input_dic["labels"]
    leng = sum(attention_mask)
    return {"input_ids": input_ids[: leng].tolist(), "labels": labels[: leng].tolist()}


def merge_data_points_by_length(lengths: List[int], max_length: int):
    items = [{"length": length, "index": i} for i, length in enumerate(lengths)]
    items = sorted(items, key=lambda x: x["index"])
    merges = []
    current_sum = 0
    current_list = []
    for i in range(len(items)):
        cur_length = items[i]["length"]
        if cur_length + current_sum <= max_length:
            current_sum += items[i]["length"]
            current_list.append(i)
        else:
            merges.append(current_list)
            current_list = [i]
            current_sum = cur_length
    if len(current_list) > 0:
        merges.append(current_list)
    result = []
    for merge in merges:
        sub_items = [items[index]["index"] for index in merge]
        result.append(sub_items)
    return result


def get_causal_mask(length, m_value):
    mask = torch.full((length, length), m_value)
    mask_cond = torch.arange(mask.size(-1))
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    return mask

    
def create_mask_from_lengths(lengths, max_length, m_value):
    result = torch.full((max_length, max_length), m_value)
    acc_leng = 0
    for length in lengths:
        x = get_causal_mask(length, m_value)
        result[acc_leng: acc_leng + length, acc_leng: acc_leng + length] = x
        acc_leng += length
    pad_length = max_length - sum(lengths)
    if pad_length > 0:
        result[-pad_length: , :] = 0
        result[:, -pad_length: ] = m_value
    return result


def merge_data_points(data_points: List[Dict], pad_token_id, max_sequence_length) -> Dict:
    input_ids = []
    lengths = []
    label_ids = []
    for item in data_points:
        input_ids += item["input_ids"]
        assert item["labels"][0] == -100 # This is to make sure that the first token won't be included in computing loss
        label_ids += item["labels"]
        lengths.append(len(item["input_ids"]))
    new_masks = create_mask_from_lengths(lengths, max_sequence_length, float("-inf"))
    pad_leng = max_sequence_length - len(input_ids)
    input_ids += [pad_token_id for _ in range(pad_leng)]
    label_ids += [-100 for _ in range(pad_leng)]
    assert len(input_ids) == len(label_ids) == new_masks.size(0)
    return {
        "input_ids": torch.tensor(input_ids), 
        "labels": torch.tensor(label_ids), 
        "attention_mask": torch.unsqueeze(new_masks, 0)  # This is because the shape is: B x 1 x N x N
    }


class PackedDataset(Dataset):
    def __init__(self, normal_dataset: CustomDataset, max_sequence_length: int, pad_token_id: int) -> None:
        super(PackedDataset, self).__init__()
        data_size = len(normal_dataset)
        original_datapoints = []
        lengths = []
        t1 = datetime.datetime.now()
        for i in range(data_size):
            item = normal_dataset[i]
            n_item = convert_input_dic_to_int(item)
            lengths.append(len(n_item["input_ids"]))
            original_datapoints.append(n_item)
            t2 = datetime.datetime.now()
            avg_time = (t2 - t1).total_seconds() / (i + 1)
            remaining_time = avg_time * (data_size - i - 1)
            if i % 500 == 0:
                print(f"{i}/{data_size} avg-time: {avg_time}, remaining_time: {remaining_time}")
        print("time for loading CUSTOMIZED DATA: ", (t2 - t1).total_seconds())
        groups = merge_data_points_by_length(lengths, max_sequence_length)
        print("number of groups after merging: ", len(groups))
        self.data_points = []
        t1 = datetime.datetime.now()
        for group in groups:
            group_data_points = [original_datapoints[index] for index in group]
            self.data_points.append(merge_data_points(group_data_points, pad_token_id, max_sequence_length))
        t2 = datetime.datetime.now()
        print("time for converting: ", (t2 - t1).total_seconds())
    
    def __len__(self):
        return len(self.data_points)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return self.data_points[i]


