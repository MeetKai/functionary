from transformers import AutoTokenizer
from functionary.prompt_template import get_prompt_template_by_version
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from functionary.train_vision.vision_datasets import (
    get_collate_fn,
    get_vision_dataset_class,
)
from functionary.train_vision.qwen2_vl_dataset import (
    PackedQwen2VLDataset,
    LazyQwen2VLDataset,
    Qwen2VLCollator,
)
import random
from transformers import Qwen2VLForConditionalGeneration
import torch 
from functionary.train.packing.monkey_patch_packing import monkey_patch_packing_for_model
from typing import Any
import typer
import json 
import math 


def compute_loss_from_ds(model: Any, tokenizer: Any, ds: Dataset, batch_size: int = 5):
    data_loader = DataLoader(ds, collate_fn=Qwen2VLCollator(tokenizer, model), batch_size=batch_size, shuffle=False)
    total_loss = 0
    model.eval()
    total_num_loss_tokens = 0  # this is the total number of tokens for computing loss

    for index, batch in enumerate(data_loader):
        print(f"compute loss for batch: {index}")
        for key in batch:
            batch[key] = batch[key].to(model.device)

        with torch.no_grad():
            avg_loss = model.forward(**batch).loss.item()
            # compute number of tokens used for computing loss
            labels = batch["labels"]
            shift_labels = labels[..., 1:].contiguous()
            shift_labels = shift_labels.view(-1)
            ignore_count = (shift_labels == -100).sum()
            num_tokens = shift_labels.size(0) - ignore_count

            total_num_loss_tokens += num_tokens.item()
            total_loss += avg_loss * num_tokens.item()
    return total_loss / total_num_loss_tokens, total_num_loss_tokens
        

def main(
    data_path: str,
    pretrained_path: str = "Qwen/Qwen2-VL-7B-Instruct",
    data_size: int = 100,
    max_length: int = 8192,
    seed: int = 10,
    batch_size: int = 5
):
    random.seed(seed)
    prompt_template = get_prompt_template_by_version("qwen2-vl")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
    tokenizer.chat_template = prompt_template.get_chat_template_jinja()

    with open(data_path, "r") as f:
        raw_data = [json.loads(line) for line in f]

    print(f"number of data points in total: {len(raw_data)}")
    random.shuffle(raw_data)

    raw_data = raw_data[:data_size]

    pad_img_path = "functionary/train_vision/pad_img2.png"
    normal_ds = LazyQwen2VLDataset(
        raw_data,
        tokenizer,
        pretrained_path,
        pad_img_path=pad_img_path,
        max_length=max_length,
    )

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        pretrained_path,
        torch_dtype=torch.bfloat16,
        use_flash_attention_2=True,
        device_map="auto"
    )
    
    normal_loss, normal_label_tokens = compute_loss_from_ds(model, tokenizer, normal_ds, batch_size)
    print(f"Normal ds: avg_loss: {normal_loss}, lab_tokens: {normal_loss}")

    print("Start packing: ")
    monkey_patch_packing_for_model(pretrained_path)
    packed_ds = PackedQwen2VLDataset(
        raw_data,
        tokenizer,
        pretrained_path,
        pad_img_path=pad_img_path,
        max_length=max_length,
        use_img_pad_token=True,
        store_img_data_in_memory=False,
        max_packed_size=-1,
    )
    packed_ds.stat()
    
    packed_loss, packed_label_tokens = compute_loss_from_ds(model, tokenizer, packed_ds, batch_size)
    print(f"Packed ds: avg_loss: {packed_loss}, lab_tokens: {packed_label_tokens}")
    print(f"Normal ds: avg_loss: {normal_loss}, lab_tokens: {normal_label_tokens}")
    difference_percentage = math.fabs(packed_loss - normal_loss) * 100 / normal_loss
    print(f"difference_percentage: {difference_percentage} %")

if __name__ == "__main__":
    typer.run(main)