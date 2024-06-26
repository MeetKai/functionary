from torch.utils.data import Dataset
import transformers
from typing import Dict, Any
import torch
from PIL import Image
from functionary.train.custom_datasets import prepare_training_inputs
from llava.mm_utils import process_images


class LazyVisionDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        raw_data,
        tokenizer: transformers.PreTrainedTokenizer,
        image_processor: Any,
        model_config: Any
    ):
        super().__init__()
        self.tokenizer = tokenizer

        self.raw_data = raw_data
        self.cached_data_dict = {}
        self.model_config = model_config
        self.image_processor = image_processor

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
        images, image_sizes = [], []
        if "metainfo" in example and "img_path" in example["metainfo"]:
            img_path = example["metainfo"]["img_path"]
            with open(img_path, "rb") as f:
                image = Image.open(f)
                images.append(image)
                image_sizes.append(image.size)
        
        image_tensor = process_images(images, self.image_processor, self.model_config)
        
        ret = {
            "input_ids": ret["inputs"]["input_ids"],
            "labels": ret["inputs"]["labels"],
            "attention_mask": ret["inputs"]["attention_mask"],
            "images": image_tensor, 
            "image_sizes": image_sizes
        }
        self.cached_data_dict[i] = ret
        return ret