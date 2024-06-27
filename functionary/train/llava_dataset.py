from torch.utils.data import Dataset
import transformers
from typing import Dict, Any
import torch
from PIL import Image
from functionary.train.custom_datasets import prepare_training_inputs

IMAGE_TOKEN_INDEX = -200


class LazyVisionDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        raw_data,
        tokenizer: transformers.PreTrainedTokenizer
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}
        self.rep_token_id = tokenizer.encode(
            "<|reserved_special_token_250|>", add_special_tokens=False
        )[0]

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
        img_paths = []
        images, image_sizes = [], []
        if "metainfo" in example and "img_path" in example["metainfo"]:
            img_path = example["metainfo"]["img_path"]
            img_paths.append(img_path)
            image = Image.open(open(img_path, "rb"))
            images.append(image)

        input_ids = ret["inputs"]["input_ids"]
        # replace unused token with image_token_index
        input_ids[input_ids == self.rep_token_id] = IMAGE_TOKEN_INDEX

        # assert number of images == number of image tokens
        assert (input_ids == IMAGE_TOKEN_INDEX).sum() == len(images)
        ret = {
            "input_ids": input_ids,
            "labels": ret["inputs"]["labels"],
            "attention_mask": ret["inputs"]["attention_mask"],
            # "images": image_tensor,
            # "image_sizes": image_sizes,
            "images": images,
        }
        self.cached_data_dict[i] = ret
        return ret
