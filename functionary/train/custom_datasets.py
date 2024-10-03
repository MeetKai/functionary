import datetime
import json
import os
import pickle
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import transformers
from torch.utils.data import Dataset

from functionary.prompt_template import (
    PromptTemplate,
    get_prompt_template_from_tokenizer,
)
import string
from pathlib import Path


def get_batch_indices(size: int, batch_size: int) -> List[Tuple[int, int]]:
    """Split indices into batchs
    Ex, size = 10, batch_size=3 --> split: [[0, 1, 2, ..., 9] --> [0, 1, 2], [3, 4, 5], [6,7,8], [9]]
    Args:
        size (int): total number of indices
        batch_size (int): _description_

    Returns:
        List[Tuple[int, int]]: _description_
    """
    result = []
    for i in range(size // batch_size + 1):
        start = i * batch_size
        end = i * batch_size + batch_size
        if end > size:
            end = size
        if end > start:
            result.append((start, end))
    return result


def get_prefix_assistant_token_ids(
    prompt_template: PromptTemplate, tokenizer: Any
) -> List[List[int]]:
    """Get prefix assistant token_ids for masking labels.
    In message where role=assistant, content of assistant always start with a prefix, such as: "Assistant:" or "<|from|>assistant"
    We convert these prefixs to token_ids, so we can detect this in the input_ids of the final prompt
    Args:
        prompt_template (PromptTemplate): Template to use
        tokenizer (Any): Tokenizer

    Returns:
        List[List[int]]: List of token_ids of assistant prefixs
    """
    result = []
    for prefix in prompt_template.get_assistant_prefixes():
        token_ids = tokenizer.encode(prefix, add_special_tokens=False)
        if token_ids[0] == 29871:
            token_ids = token_ids[1:]
        result.append(token_ids)
    return result


def get_matching_prefix(
    prefix_tokens: List[List[int]], sequence_ids: List[int]
) -> List[int]:
    """This function is used to check if sequence_ids starts with any prefix

    Args:
        prefix_tokens (List[List[int]]): _description_
        sequence_ids (List[int]): _description_

    Returns:
        List[int]: _description_
    """
    for prefix in prefix_tokens:
        if len(sequence_ids) >= len(prefix):
            if sequence_ids[: len(prefix)] == prefix:
                return prefix
    return None


def get_longest_matching_prefix(masking_tuples: List[Tuple[List[int], List[int], bool]], sequence_ids: List[int]) -> Tuple[Tuple[List[int], List[int], bool], int]:
    result = None
    for masking_tuple in masking_tuples:
        matched_tokens = get_matching_prefix(masking_tuple[0], sequence_ids)
        if matched_tokens is not None:
            if result is None:
                result = masking_tuple
            else:
                if len(result[0]) < len(masking_tuple[0]):
                    result = masking_tuple
    return result


def get_cached_folder(data_path, model_path):
    current_folder = os.path.dirname(os.path.abspath(__file__))
    cached_folder = os.path.join(current_folder, "_data_cached")

    # file name = concatenation of data_path and model_path exluding non-digits and non ascii_letters
    cached_data_folder_name = "".join(
        [
            ch
            for ch in data_path + model_path
            if ch in string.digits + string.ascii_letters
        ]
    )
    cached_data_folder = os.path.join(cached_folder, cached_data_folder_name)

    return cached_data_folder


def read_dataset(model_path, data_args, training_args, tokenizer, ds_type):
    """This function is used to read dataset for training

    Args:
        data_args (_type_): _description_
        training_args (_type_): _description_
        tokenizer (_type_): _description_
        ds_type (_type_): one of: "train"/"validation"

    Returns:
        _type_: _description_
    """
    data_path = (
        data_args.train_data_path if ds_type == "train" else data_args.eval_data_path
    )

    data_ratio = (
        data_args.training_ratio if ds_type == "train" else data_args.eval_ratio
    )

    # Do not unmask assistant prefix for validation ds.
    if ds_type == "train":
        keep_assistant_prefix = training_args.keep_assistant_prefix
    else:
        keep_assistant_prefix = False

    if not data_args.packing:
        with open(data_path, "r") as file:
            raw_data = [json.loads(line) for line in file]
            if data_ratio < 1:
                raw_data = raw_data[: int(data_ratio * len(raw_data))]
        ds = LazyPreprocessDataset(
            raw_data, tokenizer, keep_assistant_prefix=keep_assistant_prefix
        )
        return ds

    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    # The way we read dataset is:
    # Rank 0 will process the dataset and save the result to cached_folder, other ranks will read from the cached_folder

    pack_length = data_args.pack_length if data_args.pack_length > 0 else None

    cached_folder = get_cached_folder(data_path, model_path)

    if (
        training_args.local_rank > 0
    ):  # If this is not rank 0, stay here, wait for rank 0 to process the data
        print(
            f"process: {local_rank} wait for main process to prepare the training data"
        )
        torch.distributed.barrier()
    else:  # rank 0 process the data and save to cached_folder

        Path(cached_folder).mkdir(
            parents=True, exist_ok=True
        )  # create folder by process 0 if doesn't exist

        with open(data_path, "r") as file:
            raw_train_data = [json.loads(line) for line in file]
            if data_ratio < 1:
                raw_train_data = raw_train_data[: int(data_ratio * len(raw_train_data))]

        print(f"{ds_type} size: : {len(raw_train_data)}")
        # ignore_cached=True to ignore the cached if exist, rank 0 will always process the data
        ds = PackedDataset(
            raw_train_data,
            tokenizer,
            cached_folder=cached_folder,
            ignore_cached=False,
            keep_assistant_prefix=False,
            use_flash_attention=True,
            pack_length=pack_length,
            max_packed_size=data_args.max_packed_size,
        )
        print(f"process: {local_rank} finish processing data")
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        if world_size > 1:
            torch.distributed.barrier()  # allow other ranks to execute

    # All ranks will read the processed data from cached_path created by rank 0
    ds = PackedDataset(
        None,
        tokenizer,
        cached_folder=cached_folder,
        ignore_cached=False,
        use_flash_attention=True,
        pack_length=pack_length,
        max_packed_size=data_args.max_packed_size,
    )
    if local_rank == 0:
        ds.stat()  #  print some statistics about the dataset
    return ds


def prepare_training_inputs(
    *,
    messages: Dict[str, List],
    tokenizer: Any,
    padding: Optional[str] = "max_length",
    max_length: Optional[int] = None,
    return_tensor: bool = True,
    keep_assistant_prefix: bool = False,
    verbose=False,
) -> Dict[str, Union[str, Dict]]:
    """This function is used to convert a data point into input that is ready for training.
    The inputs is of format: {"input_ids": xxx, "labels": xxx, "attention_mask": xxx}

    Args:
        messages (Dict[str, List]): List of messages in OpenAI format
        tokenizer (Any): tokenizer
        padding (Optional[str], optional): _description_. Defaults to "max_length".
        max_length (Optional[int], optional): _description_. Defaults to None.
        return_tensor (bool, optional): _description_. Defaults to True.
        keep_assistant_prefix (bool, optional): _description_. Defaults to False.
        verbose (bool, optional): _description_. Defaults to False.

    Returns:
        Dict[str, Union[str, Dict]]: _description_
    """
    batch_result = prepare_training_inputs_batch(
        batch_messages=[messages],
        tokenizer=tokenizer,
        padding=padding,
        max_length=max_length,
        return_tensor=return_tensor,
        keep_assistant_prefix=keep_assistant_prefix,
        verbose=verbose,
    )
    return dict(
        final_prompt=batch_result["batch_prompts"][0],
        inputs=batch_result["batch_inputs"][0],
    )


def get_masked_labels(
    *,
    input_token_ids: List[int],
    tokenizer: Any,
    keep_assistant_prefix: bool = False,
    verbose: bool = False,
    masked_assistant_indices: Optional[List[int]] = None,
):
    """This function is used to mask labels.
    This will retain only chunks: (prefix assistant tokens) CHUNK_TO_UNMASK (stop tokens) for computing loss

    Args:
        input_token_ids (List[int]): input_token_ids
        tokenizer (Any): _description_
        assistant_prefix_tokens (List[List[int]]): _description_
        assistant_stop_tokens (List[int]): _description_
        keep_assistant_prefix (bool, optional): _description_. Defaults to False.
        verbose (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    # first we initialize labels with all positions as -100,
    # then we will fill in positions where role=assistant as we only include these in computing the loss
    labels = [-100 for _ in range(len(input_token_ids))]
    start = 0
    # now we will unmask labels by positions that was from assistant
    # we will find the chunks: "<endtoken>assistant ...(<end_of_function>|<end_of_assistant>) from input_token_ids
    # and unmask: this part: "...(<end_of_function>|<end_of_assistant>"
    # find token_ids of: "<endtoken>assistant"
    # prefix_token_ids = get_prefix_assistant_token_ids(tokenizer)
    # if verbose:
    #    print("prefix_token_ids: ", prefix_token_ids)
    prompt_template = get_prompt_template_from_tokenizer(tokenizer)
    masking_tupples = prompt_template.get_assistant_masking_start_end_tuples(tokenizer)
    
    index = 0
    total_input_leng = len(input_token_ids)
    assistant_index = -1
    while index < total_input_leng:
        # finding the index that start with: "<endtoken>assistant" --> we will unmask labels from this position
        matched_tuple = get_longest_matching_prefix(
            masking_tupples, input_token_ids[index:]
        )
        if matched_tuple is not None:
            assistant_index += 1
            end_index = -1
            # unmask until reach <end_of_function> or <end_of_assistant>
            start_masked_index = index + len(matched_tuple[0])
            if keep_assistant_prefix or matched_tuple[-1]:  # unmask prefix of assistant
                start_masked_index = index

            unmask_this = True
            if masked_assistant_indices and (
                assistant_index in masked_assistant_indices
            ):
                unmask_this = False

            for i in range(start_masked_index, total_input_leng):
                matched_prefix = get_matching_prefix(matched_tuple[1], input_token_ids[i: ])
                if matched_prefix:  # check if this is end of turn
                    if unmask_this:  # if we include this to training
                        labels[i: i + len(matched_prefix)] = input_token_ids[i: i + len(matched_prefix)]  # unmask labels at this position
                    end_index = i + len(matched_prefix)
                    break
                else:
                    if unmask_this:  # if we include this to training
                        labels[i] = input_token_ids[i]  # unmask labels at this position

            if verbose and unmask_this:
                print("------------------------")
                start = start_masked_index  # index + len(matched_prefix)
                chunk_ids = (
                    input_token_ids[start : end_index + 1]
                    if end_index > -1
                    else input_token_ids[start:]
                )
                print("chunk_ids: ", chunk_ids)
                print(
                    "longer chunk: ",
                    (
                        input_token_ids[index : end_index + 1]
                        if end_index > 1
                        else input_token_ids[index:]
                    ),
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


def get_assistant_stop_token_ids(prompt_template, tokenizer: Any) -> Dict[str, int]:
    """return a dictionary mapping from end_token --> token_id

    Args:
        tokenizer (Any): tokenizer in transformers

    Returns:
        Dict[int, EndToken]: the mapping from token_id --> end_token
    """
    result = []
    for stop_token in prompt_template.get_stop_tokens_for_generation():
        tok_ids = tokenizer.encode(stop_token, add_special_tokens=False)
        assert len(tok_ids) <= 2, f"stop token: {stop_token} is not added"
        if len(tok_ids) == 2:
            assert tok_ids[0] in [
                29871,
                28705,
            ], f"stop token: {stop_token} is not added"  # Llama tokenizer adds this token intentionally
        result.append(tok_ids[-1])
    return result


def get_masked_indices_of_assistant_messages(messages: List[Dict]) -> List[int]:
    masked_assistant_indices = []
    assistant_messages = []
    for message in messages:
        if message["role"] == "assistant":
            assistant_messages.append(message)

    for message_index, message in enumerate(assistant_messages):
        metadata = message.get("metadata", {})
        if metadata.get("masked", False):
            masked_assistant_indices.append(message_index)
    return masked_assistant_indices


def prepare_training_inputs_batch(
    *,
    batch_messages: Dict[str, List],
    tokenizer: Any,
    padding: Optional[str] = "max_length",
    max_length: Optional[int] = None,
    return_tensor: bool = True,
    keep_assistant_prefix: bool = False,
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
    # a dictionary mapping from end_token_ --> end_token
    prompt_template = get_prompt_template_from_tokenizer(tokenizer)
    prompt_str_list = []
    for messages in batch_messages:
        # old format: functions, new format: tools
        tools_or_functions = (
            messages["tools"] if "tools" in messages else messages.get("functions", [])
        )

        prompt_str = prompt_template.get_prompt_from_messages(
            messages["messages"], tools_or_functions
        )  # prompt_str is the concatenation of all prompts from messages
        prompt_str_list.append(prompt_str)
    max_length = max_length if max_length is not None else tokenizer.model_max_length

    input_dic = tokenizer(
        prompt_str_list, padding=padding, max_length=max_length, truncation=True
    )
    # input_token_ids = input_dic["input_ids"]
    batch_labels = []
    for index, input_token_ids in enumerate(input_dic["input_ids"]):
        # index of assistant message that will be masked, note that index here is not index from messages
        # Assume messages=[user1, assistant1, tool1, assistant2, tool2, assitant3, tool3] and we masked assitant1, assitant2
        # --> masked_assistant_indices = [0, 1]
        masked_assistant_indices = get_masked_indices_of_assistant_messages(
            batch_messages[index]["messages"]
        )
        labels = get_masked_labels(
            input_token_ids=input_token_ids,
            tokenizer=tokenizer,
            keep_assistant_prefix=keep_assistant_prefix,
            verbose=verbose,
            masked_assistant_indices=masked_assistant_indices,
        )

        batch_labels.append(labels)
        assert len(labels) == len(input_token_ids)

    input_dic["labels"] = batch_labels
    assert (
        len(input_dic["labels"])
        == len(input_dic["input_ids"])
        == len(input_dic["attention_mask"])
        == len(batch_messages)
    )

    batch_inputs = []
    for i in range(len(input_dic["input_ids"])):
        inputs = {}
        for key in ["labels", "input_ids", "attention_mask"]:
            inputs[key] = input_dic[key][i]
            if return_tensor:
                inputs[key] = torch.tensor(inputs[key])
        batch_inputs.append(inputs)

    return dict(batch_prompts=prompt_str_list, batch_inputs=batch_inputs)


def map_raw_data_to_input_dic(
    *,
    raw_data: List[Dict],
    tokenizer: Any,
    padding: str,
    batch_size: int = 5000,
    keep_assistant_prefix: bool = False,
) -> List[Dict]:
    """This function is used to map list of raw_data to list of processed data points for packing
    Args:
        raw_data (List[Dict]): data points from train_file/evaluation_file
        tokenizer (Any): _description_
        padding (str): _description_
        batch_size (int, optional): _description_. Defaults to 5000.
        keep_assistant_prefix (bool, optional): if we unmask assistant prefix in computing loss. Defaults to False.

    Returns:
        List[Dict]: _description_
    """
    invalid_count = 0
    data_size = len(raw_data)
    data_points = []
    t1 = datetime.datetime.now()
    for start, end in get_batch_indices(data_size, batch_size):
        batch_result = prepare_training_inputs_batch(
            batch_messages=raw_data[start:end],
            tokenizer=tokenizer,
            padding=padding,
            return_tensor=False,
            keep_assistant_prefix=keep_assistant_prefix,
        )

        assert len(batch_result["batch_inputs"]) == len(raw_data[start:end])
        for item in batch_result["batch_inputs"]:
            if is_valid_labels(item["labels"]):
                data_points.append(item)
            else:
                invalid_count += 1

        t2 = datetime.datetime.now()
        avg_time = (t2 - t1).total_seconds() / len(data_points)
        remaining_time = avg_time * (data_size - len(data_points))
        print(
            f"{len(data_points)}/{data_size}, avg_time per 1000 data points: {avg_time * 1000}, remaining time: {remaining_time}"
        )
    if invalid_count > 0:
        print(
            f"*****WARNING: invalid data points: {invalid_count} because of labels=-100 all the time"
        )
    assert len(data_points) == data_size - invalid_count
    return data_points


def pack_data_points_by_length(
    lengths: List[int], max_length: int, max_size: int = -1
) -> List[List[int]]:
    """given lengths of data points, we merge consecutive data points into a new data point, as long as the concatenated length is less than max_length
    Args:
        lengths (List[int]): List of lengths of data points
        max_length (int): the concatenated length must be less than or equal max_length
        max_size: if != -1; the maximum number of consecutive items being merged; max_size: -1 --> no limit for number of items being merged

    max_size: the maximum number of data points being merged
    For example, lengths=[1, 3, 2, 2, 6, 4, 2, 6, 5]; max_length=10
    if max_size=-1 --> [[0,1,2,3], [4, 5], [6,7], [8]]
    if max_size=3 --> [[0,1,2], [3,4], [5, 6], [7], [8]]

    Returns:
        _type_: groups of indices: [[index1, index2, ...], [], ...]
    """
    result = []
    current_concatenated_length = 0
    current_list = []
    for i in range(len(lengths)):
        cur_length = lengths[i]
        if cur_length + current_concatenated_length <= max_length and (
            max_size == -1 or len(current_list) < max_size
        ):
            current_concatenated_length += cur_length
            current_list.append(i)
        else:  # current_list is done, create a new one
            if len(current_list) > 0:
                result.append(current_list)
            current_list = [i]
            current_concatenated_length = cur_length

    if len(current_list) > 0:
        result.append(current_list)

    # assert to make sure no indices were missing
    assert sum([len(indices) for indices in result]) == len(lengths)
    return result


def get_causal_mask(length: int, m_value: float) -> torch.tensor:
    """Return causal mask filling with m_value

    Args:
        length (int): _description_
        m_value (float): _description_

    Returns:
        torch.tensor: _description_
    """
    mask = torch.full((length, length), m_value)
    mask_cond = torch.arange(mask.size(-1))
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    return mask


def create_mask_from_lengths(
    lengths: List[int], pack_length: int, m_value: float
) -> torch.tensor:
    """create attention_mask: N x N where masked value = m_value
    Args:
        lengths (List[int]): length of data points
        tokenizer (Any): _description_
        m_value (float): _description_

    Returns:
        torch.tensor: _description_
    """
    max_length = pack_length
    result = torch.full((max_length, max_length), m_value)
    acc_leng = 0
    for length in lengths:
        # mask for a data point with length
        x = get_causal_mask(length, m_value)
        result[acc_leng : acc_leng + length, acc_leng : acc_leng + length] = x
        acc_leng += length

    pad_length = max_length - sum(lengths)
    if pad_length > 0:
        result[-pad_length:, :] = 0
        result[:, -pad_length:] = m_value
    return result


def pack_data_points(data_points: List[Dict], tokenizer: Any, pack_length: int) -> Dict:
    """This method is used to pack multiple data points into a single data point used for Normal Attention (vs FlashAttention)

    Args:
        data_points (List[Dict]): _description_
        tokenizer (Any): _description_

    Returns:
        Dict: _description_
    """
    input_ids = []
    lengths = []
    label_ids = []
    for item in data_points:
        input_ids += item["input_ids"]
        # assert item["labels"][0] == -100 # This is to make sure that the first token won't be included in computing loss
        labels = list(item["labels"])
        labels[0] = -100
        label_ids += labels
        lengths.append(len(item["input_ids"]))

    attention_mask = create_mask_from_lengths(lengths, pack_length, float("-inf"))
    pad_leng = pack_length - len(input_ids)  # padding to model_max_length

    if tokenizer.padding_side == "right":
        input_ids = input_ids + [tokenizer.pad_token_id for _ in range(pad_leng)]
        label_ids = label_ids + [-100 for _ in range(pad_leng)]
    else:
        input_ids = [tokenizer.pad_token_id for _ in range(pad_leng)] + input_ids
        label_ids = [-100 for _ in range(pad_leng)] + label_ids

    assert len(input_ids) == len(label_ids) == attention_mask.size(0) == pack_length

    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(label_ids),
        "attention_mask": torch.unsqueeze(
            attention_mask, 0
        ),  # unsqueeze <-- because the shape is: B x 1 x N x N
    }


def pack_data_points_FA(data_points: List[Dict], tokenizer: Any) -> Dict:
    """This method is used to pack multiple data_points into a single data point usable for Flash Attention

    For example, we want to pack 2 inputs with padding_size=right:
    input1= {"input_ids": token_ids1, "labels": label_ids1}
    input2= {"input_ids": token_ids2, "labels": label_ids2}
    --> output would be:

    output = {"input_ids": token_ids1 + token_ids + [pad_token, ...]} padding to tokenizer.model_max_length
    output["labels"] =  label_ids1 + label_ids2 + [-100, -100, ...]
    output["attention_mask"] = [1,...,1, 2,...,2, 0...0]
        number of 1s = len(input_ids1)
        number of 2s = len(input_ids2)
        number of 0s = padding_length

    Args:
        data_points (List[Dict]): List of data points to pack: [{"input_ids": xxx, "labels": xxx}, ...]
        tokenizer (Any): _description_

    Returns:
        Dict: final single data point
    """
    input_ids = []
    lengths = []
    label_ids = []
    attention_mask = []

    for index, item in enumerate(data_points):
        input_ids += item["input_ids"]
        # assert item["labels"][0] == -100 # This is to make sure that the first token won't be included in computing loss
        labels = list(item["labels"])
        labels[0] = -100
        label_ids += labels
        lengths.append(len(item["input_ids"]))
        attention_mask += [index + 1 for _ in range(len(item["input_ids"]))]

    pad_leng = tokenizer.model_max_length - len(
        input_ids
    )  # padding to model_max_length

    if tokenizer.padding_side == "right":
        input_ids = input_ids + [tokenizer.pad_token_id for _ in range(pad_leng)]
        label_ids = label_ids + [-100 for _ in range(pad_leng)]
        attention_mask = attention_mask + [0 for _ in range(pad_leng)]
    else:
        input_ids = [tokenizer.pad_token_id for _ in range(pad_leng)] + input_ids
        label_ids = [-100 for _ in range(pad_leng)] + label_ids
        attention_mask = [0 for _ in range(pad_leng)] + attention_mask

    assert (
        len(input_ids)
        == len(label_ids)
        == len(attention_mask)
        == tokenizer.model_max_length
    )
    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(label_ids),
        "attention_mask": torch.tensor(
            attention_mask
        ),  # unsqueeze <-- because the shape is: B x 1 x N x N
    }


def is_valid_labels(labels: Union[List[int], torch.Tensor]) -> bool:
    """by setting max_length, there might be the case that the labels are all -100 -> loss=nan
    Args:
        labels (Union[List[int], torch.Tensor]): _description_

    Returns:
        bool: _description_
    """
    if type(labels) is list:
        non_mask_count = 0
        for label in labels:
            if label != -100:
                non_mask_count += 1

        if non_mask_count == 0:
            return False
        return True
    else:
        if sum(labels + 100) == 0:
            return False
        return True


def remove_invalid_label_items(data_points: List[Dict]) -> List[Dict]:
    """Remove data points where labels are all -100

    Args:
        data_points (List[Dict]): _description_

    Returns:
        _type_: _description_
    """
    result = []
    for dp in data_points:
        if is_valid_labels(dp["labels"]):
            result.append(dp)
    return result


class CachedDataset(Dataset):
    """This class implements a dataset that can be cached in a folder

    Args:
        Dataset (_type_): _description_
    """

    def __init__(
        self, tokenizer: Any, cached_folder: str, ignore_cached: bool = False
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.data_points = []
        self.load_from_cache = False
        if cached_folder is not None and not ignore_cached:
            data_path = self.get_data_point_path(cached_folder)
            if os.path.exists(data_path):
                print(f"cached found, load from cached: {cached_folder}")
                self.load(cached_folder)
                self.load_from_cache = True

    def __len__(self):
        return len(self.data_points)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return self.data_points[i]

    def create_meta_info(self):
        return {
            "max_length": self.tokenizer.model_max_length,
            "size": len(self.data_points),
        }

    def load(self, folder: str):
        t1 = datetime.datetime.now()
        with open(self.get_data_point_path(folder), "rb") as file:
            self.data_points = pickle.load(file)
        t2 = datetime.datetime.now()
        print("time for loading cached data: ", (t2 - t1).total_seconds())

    def get_data_point_path(self, folder: str) -> str:
        return os.path.join(folder, "data_points.pkl")

    def get_metainfo_path(self, folder: str) -> str:
        return os.path.join(folder, "meta_info.json")

    def dump(self, folder: str):
        t1 = datetime.datetime.now()
        if not os.path.exists(folder):
            os.mkdir(folder)

        with open(self.get_data_point_path(folder), "wb") as file:
            pickle.dump(self.data_points, file)

        with open(self.get_metainfo_path(folder), "w") as f:
            f.write(json.dumps(self.create_meta_info()))
        t2 = datetime.datetime.now()
        print("time for dumping data: ", (t2 - t1).total_seconds())

    def stat(self):
        print(json.dumps(self.create_meta_info()))


class CustomDataset(CachedDataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        raw_data: List[Dict],
        tokenizer: transformers.PreTrainedTokenizer,
        cached_folder: Optional[str] = None,
        ignore_cached: bool = False,
        batch_size: int = 5000,
        keep_assistant_prefix: bool = False,
    ):
        super().__init__(tokenizer, cached_folder, ignore_cached)

        if not self.load_from_cache:  # if not loaded from cached
            self.data_points = map_raw_data_to_input_dic(
                raw_data=raw_data,
                tokenizer=tokenizer,
                padding="max_length",
                batch_size=batch_size,
                keep_assistant_prefix=keep_assistant_prefix,
            )
            if cached_folder is not None:
                print(f"dump data to cached: {cached_folder}")
                self.dump(cached_folder)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        dp = self.data_points[i]
        result = {}
        for key in dp:
            result[key] = torch.tensor(dp[key])
        return result


class LazyPreprocessDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        raw_data,
        tokenizer: transformers.PreTrainedTokenizer,
        keep_assistant_prefix: bool = False,
    ):
        super().__init__()
        self.tokenizer = tokenizer

        self.raw_data = raw_data
        self.cached_data_dict = {}
        self.keep_assistant_prefix = keep_assistant_prefix

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = prepare_training_inputs(
            messages=self.raw_data[i],
            tokenizer=self.tokenizer,
            keep_assistant_prefix=self.keep_assistant_prefix,
        )
        ret = {
            "input_ids": ret["inputs"]["input_ids"],
            "labels": ret["inputs"]["labels"],
            "attention_mask": ret["inputs"]["attention_mask"],
        }
        self.cached_data_dict[i] = ret
        return ret


class PackedDataset(CachedDataset):
    """This class is used for Packing without Flash Attention"""

    def __init__(
        self,
        raw_data: List[Dict],
        tokenizer: transformers.PreTrainedTokenizer,
        cached_folder: Optional[str] = None,
        ignore_cached: bool = False,
        batch_size: int = 5000,
        keep_assistant_prefix: bool = False,
        use_flash_attention: bool = True,
        pack_length: Optional[int] = None,
        max_packed_size: Optional[int] = -1,
    ):
        super().__init__(tokenizer, cached_folder, ignore_cached)
        self.use_flash_attention = use_flash_attention
        self.pack_length = pack_length if pack_length else tokenizer.model_max_length
        self.max_packed_size = max_packed_size
        print("self.pack_length: ", self.pack_length)
        if not self.load_from_cache:
            self.data_points = map_raw_data_to_input_dic(
                raw_data=raw_data,
                tokenizer=tokenizer,
                padding="do_not_pad",
                batch_size=batch_size,
                keep_assistant_prefix=keep_assistant_prefix,
            )
            self.update_packing_info()
            if cached_folder is not None:
                print(f"dump data to cached: {cached_folder}")
                self.dump(cached_folder)
        else:  # update packing
            self.update_packing_info()

    def update_packing_info(self):
        self.lengths = [len(item["input_ids"]) for item in self.data_points]
        self.groups = pack_data_points_by_length(
            self.lengths, self.pack_length, self.max_packed_size
        )

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        group = self.groups[i]
        group_data_points = [self.data_points[index] for index in group]
        if not self.use_flash_attention:
            return pack_data_points(group_data_points, self.tokenizer, self.pack_length)
        return pack_data_points_FA(group_data_points, self.tokenizer)

    def stat(self):
        print(
            f"number of original data points:{len(self.data_points)}; packed to: {len(self.groups)} data points"
        )
        original_avg_length = sum(self.lengths) / len(self.lengths)
        packed_lengths = []
        for group in self.groups:
            lengths = [self.lengths[index] for index in group]
            packed_lengths.append(sum(lengths))
        avg_packed_length = sum(packed_lengths) / len(packed_lengths)
        print(
            f"original avg length: {original_avg_length}; avg packed length: {avg_packed_length}"
        )

    def length_statistics(self):
        count_dic = {}
        for length in self.lengths:
            count_dic[length] = count_dic.get(length, 0) + 1

        thresholds = [2048 * i for i in [1, 2, 4, 8, 16, 32]]
        for threshold in thresholds:
            greater_count = 0
            for key in count_dic:
                if key >= threshold:
                    greater_count += count_dic[key]
            print(f"number of data points > {threshold}: {greater_count}")
