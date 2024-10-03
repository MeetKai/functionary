from functionary.train_vision.qwen2_vl_dataset import LazyQwen2VLDataset, Qwen2VLCollator
from torch.utils.data import Dataset
from typing import Any
import torch 
import numpy as np 


def get_vision_dataset_class(dataset_type: str) -> Dataset:
    if dataset_type.lower() == "LazyQwen2VLDataset".lower():
        return LazyQwen2VLDataset
    raise Exception(f"dataset_type: {dataset_type} not found")

def get_collate_fn(dataset_type: str, model: Any, tokenizer: Any):
    if "qwen2vl" in dataset_type.lower():
        return Qwen2VLCollator(tokenizer, model)
    
    raise Exception(f"cannot find collate function for dataset_type: {dataset_type} not found")