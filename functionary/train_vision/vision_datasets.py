from functionary.train_vision.qwen2_vl_dataset import LazyQwen2VLDataset
from torch.utils.data import Dataset
from typing import Any
import torch 
import numpy as np 

class CustomCollator:
    def __init__(self, tokenizer: Any, model: Any) -> None:
        self.tokenizer = tokenizer
        self.model = model

    def __call__(self, features: Any) -> Any:
        raise NotImplemented


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


def get_vision_dataset_class(dataset_type: str) -> Dataset:
    if dataset_type.lower() == "LazyQwen2VLDataset".lower():
        return LazyQwen2VLDataset
    raise Exception(f"dataset_type: {dataset_type} not found")

def get_collate_fn(dataset_type: str, model: Any, tokenizer: Any):
    if "qwen2vl" in dataset_type.lower():
        return Qwen2VLCollator(tokenizer, model)
    
    raise Exception(f"cannot find collate function for dataset_type: {dataset_type} not found")