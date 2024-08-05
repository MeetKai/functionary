from typing import Any, Dict, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler

import transformers
from functionary.prompt_template import prompt_utils
from functionary.train.custom_datasets import prepare_training_inputs

IMAGE_TOKEN_INDEX = -200


class LazyVisionDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        raw_data,
        tokenizer: transformers.PreTrainedTokenizer,
        pad_img_path: Optional[str] = "",
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
        token_ids = tokenizer.encode(
            "<|reserved_special_token_250|>", add_special_tokens=False
        )
        assert len(token_ids) == 1
        self.rep_token_id = token_ids[0]

        self.pad_img = None
        if pad_img_path:
            self.pad_img = Image.open(open(pad_img_path, "rb"))

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = prepare_training_inputs(
            messages=self.raw_data[i],
            tokenizer=self.tokenizer,
            keep_assistant_prefix=False,
        )
        example = self.raw_data[i]
        images = prompt_utils.extract_images_from_messages(example["messages"])

        if (
            len(images) == 0 and self.pad_img
        ):  # add pad_img_token to make sure that the graph is fixed
            images.append(self.pad_img)
            ret["inputs"] = prompt_utils.inject_image_token(
                ret["inputs"], self.rep_token_id
            )

        input_ids = ret["inputs"]["input_ids"]
        input_ids[input_ids == self.rep_token_id] = IMAGE_TOKEN_INDEX

        assert (input_ids == IMAGE_TOKEN_INDEX).sum() == len(images)
        ret = {
            "input_ids": input_ids,
            "labels": ret["inputs"]["labels"],
            "attention_mask": ret["inputs"]["attention_mask"],
            "images": images,
        }

        self.cached_data_dict[i] = ret
        return ret
