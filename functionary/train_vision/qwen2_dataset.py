from typing import Any, Dict, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler
from typing import List, Tuple

import transformers
from functionary.prompt_template import prompt_utils, get_prompt_template_from_tokenizer
from functionary.train.custom_datasets import prepare_training_inputs
from transformers import AutoProcessor, Qwen2VLImageProcessor, Qwen2VLProcessor
from functionary.prompt_template.base_template import PromptTemplate
from functionary.train import custom_datasets

IMAGE_TOKEN_INDEX = -200


def pad_inputs(inputs: Dict, tokenizer: Any, length: int) -> Dict:
    input_length = inputs["input_ids"].shape[-1]
    if input_length == length:
        return inputs

    assert input_length < length
    pad_length = length - input_length
    inputs["input_ids"] = torch.concat(
        (inputs["input_ids"], torch.zeros(pad_length) + tokenizer.pad_token_id), dim=-1
    )
    inputs["attention_mask"] = torch.concat(
        (inputs["attention_mask"], torch.zeros(pad_length)), dim=-1
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
    image_grid_thw = inputs["image_grid_thw"]
    img_lengths = []
    for index in range(image_grid_thw.shape[0]):
        img_length = image_grid_thw[index].prod()
        img_lengths.append(img_length)
    
    assert sum(img_lengths) == inputs["pixel_values"].shape[0]
    remaining_size = sum(img_lengths[: image_num])
    inputs["pixel_values"] = inputs["pixel_values"][: remaining_size]
    inputs["image_grid_thw"] = inputs["image_grid_thw"][: image_num]
    return inputs        


def truncate_inputs_for_training(
    inputs: Dict,
    max_length: int,
    start_img_token: int,
    end_img_token: int,
    tokenizer: Any,
) -> Dict:
    """
    This function is used to truncate inputs in case the input_token ids are truncated, and some image tokens are truncated
    """
    input_ids = inputs["input_ids"].tolist()
    img_num = inputs["image_grid_thw"].shape[0]
    img_indices = find_image_token_indices(input_ids, start_img_token, end_img_token)
    if len(img_indices) == 0:  # all image tokens were truncated
        inputs["pixel_values"] = None
        inputs["image_grid_thw"] = None
        return inputs

    last_img_indices = img_indices[-1]
    if last_img_indices[1] == -1:  # the last image was partially truncated
        last_start_img_index = last_img_indices[0]
        inputs["input_ids"] = inputs["input_ids"][:last_start_img_index]
        inputs["attention_mask"] = inputs["attention_mask"][:last_start_img_index]
        inputs = pad_inputs(inputs, tokenizer, max_length)
        # pixel_values; image_grid_thw must be truncated
        inputs = truncate_images_in_pixel_values(inputs, len(last_img_indices) - 1)
        return inputs
    
    # pixel_values; image_grid_thw must be truncated
    if len(last_img_indices) < img_num: # if there exists images that were not used
        inputs = truncate_images_in_pixel_values(inputs, len(last_img_indices))
    return inputs


def concatenate_pad_inputs(inputs: Dict, pad_token_inputs: Dict) -> Dict:
    inputs["input_ids"] = torch.concat(
        ( pad_token_inputs["input_ids"], inputs["input_ids"]), dim=-1
    )
    inputs["attention_mask"] = torch.concat(
        (
            torch.zeros_like(pad_token_inputs["attention_mask"]),
            inputs["attention_mask"]
        ),
        dim=-1,
    )

    for key in ["pixel_values", "image_grid_thw"]:
        input_value = inputs.get(key, None)
        if input_value is not None:
            input_value = torch.concat((pad_token_inputs[key], input_value), dim=0)
        else:
            input_value = pad_token_inputs[key]
        inputs[key] = input_value
    return inputs


class LazyVisionDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        raw_data,
        tokenizer: transformers.PreTrainedTokenizer,
        pretrained_path: str,
        pad_img_path: str,
        use_img_pad_token: bool = True
    ):
        super().__init__()
        self.tokenizer = tokenizer

        # sampler = UniformSampler(raw_data, batch_size)
        # self.loop_indices = sampler.loop_indices
        # print(f"batch_size: {batch_size}; loop_index: {self.loop_indices[: 20]}")
        # # make sure that raw_data is in the correct order of reading data
        # self.raw_data = [raw_data[index] for index in self.loop_indices]

        self.raw_data = raw_data
        self.cached_data_dict = {}

        self.pad_img = None
        self.pad_img_path = pad_img_path
        self.input_processor = AutoProcessor.from_pretrained(pretrained_path)

        self.prompt_template = get_prompt_template_from_tokenizer(tokenizer)
        self.max_length = tokenizer.model_max_length
        self.vision_start_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
        self.vision_end_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")

        # compute the number of tokens for this image
        self.pad_img = Image.open(open(pad_img_path, "rb"))
        self.pad_token_inputs = self.process_inputs(
            "<|vision_start|><|image_pad|><|vision_end|>",
            images=[self.pad_img],
            padding="do_not_pad",
        )
        print("--------pad_token_inputs")
        print(self.pad_token_inputs)
        self.use_img_pad_token = use_img_pad_token

    def process_inputs(
        self,
        prompt: str,
        images: List[Any],
        padding: str = "max_length",
        max_length: Optional[int] = None,
    ) -> Dict:
        inputs = self.input_processor(
            text=[prompt],
            images=images,
            padding=padding,
            max_length=max_length,
            return_tensors="pt",
        )
        for key in inputs:
            if key in ["input_ids", "attention_mask"]:
                inputs[key] = inputs[key].squeeze(0)
        return inputs

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
        max_length = self.max_length - pad_token_num if self.use_img_pad_token else self.max_length
        inputs = self.process_inputs(
            prompt,
            images,
            padding="max_length",
            max_length=self.max_length - pad_token_num,
        )
        if images:
            inputs = truncate_inputs_for_training(
                inputs,
                self.max_length - pad_token_num,
                self.vision_start_id,
                self.vision_end_id,
                self.tokenizer,
            )
            assert inputs["input_ids"].shape[-1] == max_length

        # make sure that ther is alwasy a pad_token at the end
        if self.use_img_pad_token:
            inputs = concatenate_pad_inputs(inputs, self.pad_token_inputs)

        # this chunk of code if for computing the labels
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

        labels = torch.tensor(labels)
        labels = labels * inputs["attention_mask"]  # padded tokens --> 0
        labels[labels == 0] = -100  # padded tokens --> -100
        # mask pad_input_tokens
        inputs["labels"] = labels

        assert (
            inputs["input_ids"].shape[-1]
            == inputs["labels"].shape[-1]
            == inputs["attention_mask"].shape[-1]
            == self.max_length
        )
        return inputs
