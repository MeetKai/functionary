from typing import List, Dict, Any
import torch
from torch.utils.data import Dataset


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


def pack_data_points_FA(
    data_points: List[Dict], tokenizer: Any, model_max_length: int
) -> Dict:
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

    pad_leng = model_max_length - len(input_ids)  # padding to model_max_length

    if tokenizer.padding_side == "right":
        input_ids = input_ids + [tokenizer.pad_token_id for _ in range(pad_leng)]
        label_ids = label_ids + [-100 for _ in range(pad_leng)]
        attention_mask = attention_mask + [0 for _ in range(pad_leng)]
    else:
        input_ids = [tokenizer.pad_token_id for _ in range(pad_leng)] + input_ids
        label_ids = [-100 for _ in range(pad_leng)] + label_ids
        attention_mask = [0 for _ in range(pad_leng)] + attention_mask

    assert len(input_ids) == len(label_ids) == len(attention_mask) == model_max_length
    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(label_ids),
        "attention_mask": torch.tensor(
            attention_mask
        ),  # unsqueeze <-- because the shape is: B x 1 x N x N
    }


class PackedDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        tokenizer: Any,
        max_input_length: int,
        pack_length: int = -1,
        max_packed_size: int = -1,
    ) -> None:
        """This class is used to convert regular dataset to packed dataset
        Args:
            dataset (Dataset): regular dataset that implements 2 methods: __len__ and __getitem__
            tokenizer (Any): The tokenizer used to tokenize the dataset
            max_input_length (int): max sequence length
            pack_length (int, optional): The maximum length of packed data points, if = 1 --> value = max_input_length. Defaults to -1.
            max_packed_size (int, optional): Maximum number of data points that can be packed. If value = -1, there is no limit for this, as long as the length of packed data point < pack_length. Defaults to -1.
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length

        self.pack_length = pack_length
        if pack_length == -1:
            self.pack_length = self.max_input_length

        self.lengths = []
        self.data_points = []
        size = len(dataset)

        for i in range(size):
            data_point = dataset[i]
            input_length = torch.sum(data_point["attention_mask"]).item()
            n_data_point = {}
            n_data_point["input_ids"] = (
                data_point["input_ids"][:input_length]
                if tokenizer.padding_side == "right"
                else data_point["input_ids"][-input_length:]
            )

            if "labels" not in data_point:  # create labels if not existed
                labels = n_data_point["input_ids"].clone()
                labels[labels == tokenizer.pad_token_id] = -100  # mask pad_token
                n_data_point["labels"] = labels.tolist()
            else:
                n_data_point["labels"] = (
                    data_point["labels"][:input_length]
                    if tokenizer.padding_side == "right"
                    else data_point["labels"][-input_length:]
                )

            self.data_points.append(n_data_point)
            self.lengths.append(input_length)

        max_input_length = max(self.lengths)
        assert self.pack_length >= max(
            self.lengths
        ), f"pack_length must be >= max(input lengths), found pack_length={self.pack_length}, max_input_length={max_input_length}"
        self.groups = pack_data_points_by_length(
            self.lengths, self.pack_length, max_packed_size
        )

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        group = self.groups[i]
        group_data_points = [self.data_points[index] for index in group]
        return pack_data_points_FA(
            group_data_points, self.tokenizer, self.max_input_length
        )

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
