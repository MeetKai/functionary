import typer
from transformers import AutoTokenizer
from functionary.prompt_template import get_prompt_template_by_version
from functionary.train_vision.qwen2_vl_dataset import PackedQwen2VLDataset
import json 


def main(pretrained_path: str, data_path: str, save_path: str, max_length: int = 8192):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
    prompt_template = get_prompt_template_by_version("qwen2-vl")
    tokenizer.chat_template = prompt_template.get_chat_template_jinja()
    
    with open(data_path, "r") as f:
        raw_data = [json.loads(line) for line in f]
    
    print("number of data points: ", len(raw_data))
    packed_ds = PackedQwen2VLDataset(raw_data, tokenizer, pretrained_path, pad_img_path="functionary/train_vision/pad_img2.png", max_length=max_length)
    packed_ds.save_cached(save_path)
    packed_ds.stat()
    # Try reloading again 
    packed_ds = PackedQwen2VLDataset(raw_data, tokenizer, pretrained_path, pad_img_path="functionary/train_vision/pad_img2.png", max_length=max_length, cached_path=save_path)
    print("successfully loaded the dataset from cached")


if __name__ == "__main__":
    typer.run(main)