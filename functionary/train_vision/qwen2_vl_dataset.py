from typing import Any, Dict, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler
from typing import List, Tuple, Union

import transformers
from functionary.prompt_template import prompt_utils, get_prompt_template_from_tokenizer
from functionary.train.custom_datasets import prepare_training_inputs
from transformers import AutoProcessor, Qwen2VLImageProcessor, Qwen2VLProcessor
from functionary.prompt_template.base_template import PromptTemplate
from functionary.train import custom_datasets
from functionary.train_vision.base_datasets import CustomCollator, VisionDataset
import numpy as np
import datetime
import json


class Qwen2VLCollator(CustomCollator):
    def __call__(self, features: Any) -> Any:
        result = {}
        first = features[0]
        for k, v in first.items():
            if k in ["input_ids", "attention_mask", "labels"]:
                if isinstance(v, torch.Tensor):
                    result[k] = torch.stack([f[k] for f in features])
                elif isinstance(v, np.ndarray):
                    result[k] = torch.tensor(np.stack([f[k] for f in features]))
                else:
                    result[k] = torch.tensor([f[k] for f in features])

        # pixel_values, concatenate
        pixel_value_list = []
        image_grid_thw_list = []

        for feature in features:
            pixel_values = feature.get("pixel_values", None)
            if pixel_values is not None:
                pixel_value_list.append(pixel_values)

            image_grid_thw = feature.get("image_grid_thw", None)
            if image_grid_thw is not None:
                image_grid_thw_list.append(image_grid_thw)

        result["pixel_values"] = torch.concat(pixel_value_list, dim=0)
        result["image_grid_thw"] = torch.concat(image_grid_thw_list, dim=0)
        return result


def pad_inputs(inputs: Dict, tokenizer: Any, length: int) -> Dict:
    input_length = inputs["input_ids"].shape[-1]
    if input_length == length:
        return inputs

    assert input_length < length
    pad_length = length - input_length
    if tokenizer.padding_side == "right":
        inputs["input_ids"] = torch.concat(
            (
                inputs["input_ids"],
                torch.zeros(pad_length, dtype=torch.int) + tokenizer.pad_token_id,
            ),
            dim=-1,
        )
        inputs["attention_mask"] = torch.concat(
            (inputs["attention_mask"], torch.zeros(pad_length, dtype=torch.int)), dim=-1
        )
    else:
        inputs["input_ids"] = torch.concat(
            (
                torch.zeros(pad_length, dtype=torch.int) + tokenizer.pad_token_id,
                inputs["input_ids"],
            ),
            dim=-1,
        )
        inputs["attention_mask"] = torch.concat(
            (torch.zeros(pad_length, dtype=torch.int), inputs["attention_mask"]), dim=-1
        )
    return inputs


def find_image_token_indices(
    input_ids: List[int], start_token: int, end_token: int
) -> List[Tuple[int, int]]:
    """
    Find the start and end indices of image tokens in the input sequence.

    Args:
    input_ids (List[int]): The list of input token IDs.
    start_token (int): The token ID representing the start of an image.
    end_token (int): The token ID representing the end of an image.

    Returns:
    List[Tuple[int, int]]: A list of tuples, each containing the start and end indices of an image token pair.
    """
    indices = []
    stack = []

    for i, token in enumerate(input_ids):
        if token == start_token:
            stack.append(i)
        elif token == end_token and stack:
            start = stack.pop()
            indices.append((start, i))
    if len(stack) > 0:  # there exists image that is started but not ended
        start = stack.pop()
        indices.append((start, -1))
    return indices


def truncate_images_in_pixel_values(inputs: Dict, image_num: int) -> Dict:
    if image_num == 0:
        del inputs["pixel_values"]
        del inputs["image_grid_thw"]
        return inputs

    image_grid_thw = inputs["image_grid_thw"]
    img_lengths = []
    for index in range(image_grid_thw.shape[0]):
        img_length = image_grid_thw[index].prod()
        img_lengths.append(img_length)

    assert sum(img_lengths) == inputs["pixel_values"].shape[0]
    remaining_size = sum(img_lengths[:image_num])
    inputs["pixel_values"] = inputs["pixel_values"][:remaining_size]
    inputs["image_grid_thw"] = inputs["image_grid_thw"][:image_num]
    return inputs


def truncate_inputs_for_training(
    inputs: Dict,
    max_length: int,
    start_img_token: int,
    end_img_token: int,
    tokenizer: Any,
    img_num: int,
) -> Dict:
    """
    This function is used to truncate inputs in case the input_token ids are truncated, and some image tokens are truncated
    """
    input_ids = inputs["input_ids"].tolist()
    img_indices = find_image_token_indices(input_ids, start_img_token, end_img_token)
    # print("img_indices: ", img_indices, "img_num: ", img_num)
    if len(img_indices) == 0:  # all image tokens were truncated
        # print("all images are truncated ....")
        inputs["pixel_values"] = None
        inputs["image_grid_thw"] = None
        return inputs

    last_img_indices = img_indices[-1]
    if last_img_indices[1] == -1:  # the last image was partially truncated
        # print("an image was partially truncated")
        last_start_img_index = last_img_indices[0]
        inputs["input_ids"] = inputs["input_ids"][:last_start_img_index]
        inputs["attention_mask"] = inputs["attention_mask"][:last_start_img_index]
        if "labels" in inputs:
            inputs["labels"] = inputs["labels"][:last_start_img_index]
        inputs = pad_inputs(inputs, tokenizer, max_length)
        # pixel_values; image_grid_thw must be truncated
        inputs = truncate_images_in_pixel_values(inputs, len(img_indices) - 1)

        return inputs

    # pixel_values; image_grid_thw must be truncated
    if len(img_indices) < img_num:  # if there exists images that were not used
        # print("some images was truncated")
        inputs = truncate_images_in_pixel_values(inputs, len(img_indices))
    # else:
    #   print("no images were truncated")
    return inputs


def concatenate_pad_inputs(
    inputs: Dict, pad_token_inputs: Dict, padding_side: str
) -> Dict:
    concat_tuple = (pad_token_inputs["input_ids"], inputs["input_ids"])
    if padding_side == "right":
        concat_tuple = (concat_tuple[1], concat_tuple[0])

    inputs["input_ids"] = torch.concat(concat_tuple, dim=-1)
    concat_tuple = (
        torch.zeros_like(pad_token_inputs["attention_mask"]),
        inputs["attention_mask"],
    )
    if padding_side == "right":
        concat_tuple = (concat_tuple[1], concat_tuple[0])

    inputs["attention_mask"] = torch.concat(
        concat_tuple,
        dim=-1,
    )

    for key in ["pixel_values", "image_grid_thw"]:
        input_value = inputs.get(key, None)
        if input_value is not None:
            concat_tuple = (pad_token_inputs[key], input_value)
            if padding_side == "right":
                concat_tuple = (concat_tuple[1], concat_tuple[0])
            input_value = torch.concat(concat_tuple, dim=0)
        else:
            input_value = pad_token_inputs[key]
        inputs[key] = input_value
    return inputs


def assert_model_inputs(inputs: Dict, max_length: int, vision_start_id: int):
    assert (
        inputs["input_ids"].shape[-1]
        == inputs["labels"].shape[-1]
        == inputs["attention_mask"].shape[-1]
        == max_length
    ), f'{inputs["input_ids"].shape[-1]}, {inputs["labels"].shape[-1]}, {inputs["attention_mask"].shape[-1]}, {max_length}'

    # Assert the number of images == number of start_image_token
    num_imgs = (inputs["input_ids"] == vision_start_id).sum()
    assert (
        num_imgs == inputs["image_grid_thw"].shape[0]
    ), f'num_imgs={num_imgs}; shape: {inputs["image_grid_thw"].shape}'
    # assert to make sure that number of image tokens in image_grid_thw == number of tokens in pixel_values
    assert (
        sum(
            [
                inputs["image_grid_thw"][i].prod()
                for i in range(inputs["image_grid_thw"].shape[0])
            ]
        )
        == inputs["pixel_values"].shape[0]
    )


class LazyQwen2VLDataset(VisionDataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        raw_data,
        tokenizer: transformers.PreTrainedTokenizer,
        pretrained_path: str,
        pad_img_path: str,
        max_length: int,
        use_img_pad_token: bool = True,
        **kwargs,
    ):
        super().__init__(
            raw_data,
            tokenizer,
            pretrained_path,
            pad_img_path,
            max_length,
            use_img_pad_token,
            kwargs=kwargs,
        )
        self.input_processor = AutoProcessor.from_pretrained(pretrained_path)
        self.vision_start_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
        self.vision_end_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")
        self.pad_token_inputs = self.process_inputs(
            "<|vision_start|><|image_pad|><|vision_end|>",
            images=[self.pad_img],
            padding="do_not_pad",
        )

    def process_inputs(
        self,
        prompt: Union[str, List[str]],
        images: List[Any],
        padding: str = "max_length",
        max_length: Optional[int] = None,
    ) -> Dict:
        is_str = True
        if type(prompt) is list:
            is_str = False

        inputs = self.input_processor(
            text=[prompt] if is_str else prompt,
            images=images,
            padding=padding,
            max_length=max_length,
            return_tensors="pt",
            truncation=True,
        )
        if is_str:
            for key in inputs:
                if key in ["input_ids", "attention_mask"]:
                    inputs[key] = inputs[key].squeeze(0)
        return inputs

    def get_labels_from_input_ids(self, inputs: Dict, example: Dict) -> List[int]:
        input_ids = inputs["input_ids"].tolist()
        assistant_stop_token_ids = custom_datasets.get_assistant_stop_token_ids(
            self.prompt_template, self.tokenizer
        )
        assistant_prefix_tokens = custom_datasets.get_prefix_assistant_token_ids(
            self.prompt_template, self.tokenizer
        )

        masked_assistant_indices = (
            custom_datasets.get_masked_indices_of_assistant_messages(
                example["messages"]
            )
        )

        labels = custom_datasets.get_masked_labels(
            input_token_ids=input_ids,
            tokenizer=self.tokenizer,
            assistant_prefix_tokens=assistant_prefix_tokens,
            assistant_stop_tokens=assistant_stop_token_ids,
            keep_assistant_prefix=False,
            masked_assistant_indices=masked_assistant_indices,
        )

        labels_tensor = torch.tensor(labels)
        # Make sure that labels == 0 for all padded tokens
        labels_tensor[inputs["attention_mask"] == 0] = -100
        return labels_tensor

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        example = self.raw_data[i]
        prompt = self.prompt_template.get_prompt_from_messages(
            example["messages"], example["tools"], add_generation_prompt=False
        )
        images = prompt_utils.extract_images_from_messages(example["messages"])
        if len(images) == 0:
            images = None

        pad_token_num = self.pad_token_inputs["input_ids"].shape[-1]
        # we always save pad_token_num tokens for padding to make sure that the execution path is the same
        # for all GPUs
        max_length = (
            self.max_length - pad_token_num
            if self.use_img_pad_token
            else self.max_length
        )
        # print(f"-----------prompt: {prompt}")
        # print(f"max_length: {max_length}")
        inputs = self.process_inputs(
            prompt,
            images,
            padding="max_length",
            max_length=max_length,
        )

        # print("inputs: ", inputs)

        assert (
            inputs["input_ids"].shape[-1] == max_length
        ), f'input length: {inputs["input_ids"].shape}'
        if images:
            inputs = truncate_inputs_for_training(
                inputs,
                max_length,
                self.vision_start_id,
                self.vision_end_id,
                self.tokenizer,
                len(images),
            )

        # make sure that ther is alwasy a pad_token at the end
        if self.use_img_pad_token:
            inputs = concatenate_pad_inputs(
                inputs, self.pad_token_inputs, self.tokenizer.padding_side
            )

        # this chunk of code if for computing the labels
        inputs["labels"] = self.get_labels_from_input_ids(inputs, example)

        assert_model_inputs(inputs, self.max_length, self.vision_start_id)
        return inputs


class PackedQwen2VLDataset(LazyQwen2VLDataset):
    def __init__(
        self,
        raw_data,
        tokenizer: transformers.PreTrainedTokenizer,
        pretrained_path: str,
        pad_img_path: str,
        max_length: int,
        use_img_pad_token: bool = True,
        **kwargs,
    ):
        super().__init__(
            raw_data,
            tokenizer,
            pretrained_path,
            pad_img_path,
            max_length,
            use_img_pad_token,
            kwargs=kwargs,
        )
        self.max_packed_size = kwargs.get("max_packed_size", -1)
        cached_path = kwargs.get("cached_path", "")
        text_only_data = []
        img_data = []

        id_2_raw_data = {}
        self.id_2_length = {}
        pad_token_num = self.pad_token_inputs["input_ids"].shape[-1]
        if cached_path:
            with open(cached_path, "r") as f:
                cached_data = json.loads(f.read())
                id_2_length = cached_data["id_2_length"]
                for key in id_2_length:
                    self.id_2_length[int(key)] = id_2_length[key]

                self.invalid_data_ids = cached_data["invalid_data_ids"]
                assert (
                    self.max_length == cached_data["max_length"]
                ), f'max_length in cached_data ({cached_data["max_length"]}) != {self.max_length}'
                assert (
                    self.pretrained_path == cached_data["pretrained_path"]
                ), f'pretrained_path ({cached_data["pretrained_path"]}) in cached data != {self.pretrained_path}'
                assert len(self.id_2_length) == len(raw_data)
        else:
            for i, example in enumerate(self.raw_data):
                id_2_raw_data[i] = example
                example["original_index"] = i

                images = prompt_utils.extract_images_from_messages(example["messages"])
                if len(images) == 0:
                    text_only_data.append(example)
                else:
                    img_data.append(example)

            print(
                f"number of text_only data points: {len(text_only_data)}; number of data points with images: {len(img_data)}"
            )
            print("start to tokenize text-only images")

            text_data_points = custom_datasets.map_raw_data_to_input_dic(
                raw_data=text_only_data,
                tokenizer=tokenizer,
                padding="do_not_pad",
                batch_size=5000,
                keep_assistant_prefix=False,
                remove_invalid_data=False,
                max_length=self.max_length - pad_token_num,
            )
            assert len(text_only_data) == len(text_data_points)

            self.invalid_data_ids = []
            for dt, item in zip(text_only_data, text_data_points):
                if not custom_datasets.is_valid_labels(item["labels"]):
                    self.invalid_data_ids.append(dt["original_index"])

            lengths = [len(item["input_ids"]) for item in text_data_points]
            for dt, length in zip(text_only_data, lengths):
                self.id_2_length[dt["original_index"]] = length

            print(f"start to tokenize {len(img_data)} data points containing images")
            t1 = datetime.datetime.now()
            for i, example in enumerate(img_data):
                images = prompt_utils.extract_images_from_messages(example["messages"])
                assert len(images) > 0
                prompt = self.prompt_template.get_prompt_from_messages(
                    example["messages"], example["tools"], add_generation_prompt=False
                )

                inputs = self.process_inputs(
                    prompt,
                    images,
                    padding="do_not_pad",
                    max_length=self.max_length - pad_token_num,
                )
                # consider
                self.id_2_length[example["original_index"]] = inputs["input_ids"].shape[
                    -1
                ]
                labels = self.get_labels_from_input_ids(inputs, example)
                if not custom_datasets.is_valid_labels(labels.tolist()):
                    self.invalid_data_ids.append(example["original_index"])

                if i % 50 == 1:
                    t2 = datetime.datetime.now()
                    exe_time = (t2 - t1).total_seconds()
                    avg_time = exe_time / (i + 1)
                    remaining_time = (len(img_data) - i - 1) * avg_time
                    print(
                        f"+ {i}/{len(img_data)} avg_time: {avg_time}s, remaining time: {remaining_time / 60} minutes "
                    )

            # shuffle combinations of text_data_points & self.img_data
            if len(self.invalid_data_ids) > 0:
                print(f"******** NUMBER OF INVALID DATA: {len(self.invalid_data_ids)}")

        # remove all invalid datapoints
        self.final_raw_data = []
        self.final_lengths = []
        for i in range(len(raw_data)):
            if i not in self.invalid_data_ids:
                self.final_raw_data.append(raw_data[i])
                self.final_lengths.append(self.id_2_length[i])

        assert len(self.final_lengths) == len(raw_data) - len(self.invalid_data_ids)

        self.groups = custom_datasets.pack_data_points_by_length(
            self.final_lengths, self.max_length - pad_token_num, self.max_packed_size
        )

        self.stat()
        # print("final_lengths: ", self.final_lengths)
        # print("groups: ", self.groups)

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        group = self.groups[i]
        lengths = [self.final_lengths[index] for index in group]
        examples = [self.final_raw_data[index] for index in group]
        pad_token_num = self.pad_token_inputs["input_ids"].shape[-1]
        data_points = []
        pixel_value_list = []
        image_grid_thw_list = []

        for example in examples:
            prompt = self.prompt_template.get_prompt_from_messages(
                example["messages"], example["tools"], add_generation_prompt=False
            )
            images = prompt_utils.extract_images_from_messages(example["messages"])
            if len(images) == 0:
                images = None
            inputs = self.process_inputs(
                prompt,
                images,
                padding="do_not_pad",
                max_length=self.max_length - pad_token_num,
            )
            inputs["labels"] = self.get_labels_from_input_ids(inputs, example)
            data_points.append(
                {key: inputs[key] for key in ["input_ids", "labels", "attention_mask"]}
            )

            if inputs.get("pixel_values", None) is not None:
                pixel_value_list.append(inputs["pixel_values"])
                image_grid_thw_list.append(inputs["image_grid_thw"])

        # if this group is only 1 data point, some images might be truncated so far
        # if group is > 2 data points, no images were truncated
        if (
            len(group) == 1
            and data_points[0]["input_ids"].shape[-1] == self.max_length - pad_token_num
        ):
            final_inputs = data_points[0]
            final_inputs["pixel_values"] = (
                None if len(pixel_value_list) == 0 else pixel_value_list[0]
            )
            final_inputs["image_grid_thw"] = (
                None if len(pixel_value_list) == 0 else image_grid_thw_list[0]
            )
            images = prompt_utils.extract_images_from_messages(examples[0]["messages"])
            img_num = len(images)
            final_inputs = truncate_inputs_for_training(
                final_inputs,
                self.max_length - pad_token_num,
                self.vision_start_id,
                self.vision_end_id,
                self.tokenizer,
                img_num,
            )
        else:
            final_inputs = custom_datasets.pack_data_points_FA(
                data_points, self.tokenizer, self.max_length - pad_token_num
            )
            final_inputs["pixel_values"] = (
                torch.concat(pixel_value_list, dim=0) if pixel_value_list else None
            )
            final_inputs["image_grid_thw"] = (
                torch.concat(image_grid_thw_list, dim=0)
                if image_grid_thw_list
                else None
            )

        result = concatenate_pad_inputs(
            final_inputs, self.pad_token_inputs, self.tokenizer.padding_side
        )
        # add labels = [-100] to
        label_pads = torch.tensor([-100 for _ in range(pad_token_num)])
        if self.tokenizer.padding_side == "right":
            result["labels"] = torch.concat((result["labels"], label_pads), dim=0)
        else:
            result["labels"] = torch.concat((label_pads, result["labels"]), dim=0)

        assert_model_inputs(result, self.max_length, self.vision_start_id)
        # count label tokens
        return result

    def stat(self):
        print(
            f"number of original data points:{len(self.raw_data)}; packed to: {len(self.groups)} data points"
        )
        original_avg_length = sum(self.final_lengths) / len(self.final_lengths)
        packed_lengths = []
        for group in self.groups:
            lengths = [self.final_lengths[index] for index in group]
            packed_lengths.append(sum(lengths))
        avg_packed_length = sum(packed_lengths) / len(packed_lengths)
        print(
            f"original avg length: {original_avg_length}; avg packed length: {avg_packed_length}"
        )

    def save_cached(self, save_path: str):
        data = {
            "max_length": self.max_length,
            "pretrained_path": self.pretrained_path,
            "id_2_length": self.id_2_length,
            "invalid_data_ids": self.invalid_data_ids,
        }

        with open(save_path, "w") as f:
            f.write(json.dumps(data, ensure_ascii=False))
