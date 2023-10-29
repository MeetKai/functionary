from functionary.train import custom_datasets
from functionary.prompt import get_additional_tokens
from transformers import LlamaTokenizerFast
import json 
import typer
import random


def main(pretrained_path: str, data_path: str, save_path: str, max_length: int):
    tokenizer = LlamaTokenizerFast.from_pretrained(
        pretrained_path,
        model_max_length=max_length,
        legacy=True,
    )

    # Add special tokens
    tokenizer.pad_token = tokenizer.unk_token
    special_tokens = {"additional_special_tokens": get_additional_tokens()}
    num_new_tokens = tokenizer.add_special_tokens(special_tokens)
    print(f"add {num_new_tokens} tokens, max_length: {tokenizer.model_max_length}")
    raw_data = []
    with open(data_path, "r") as f:
        for line in f:
            if line.strip():
                raw_data.append(json.loads(line))
    print("number of data points: ", len(raw_data))
    ds = custom_datasets.DirectPackedDataset(raw_data, tokenizer, cached_path=save_path, batch_size=2000)
    ds.stat()
    ds_size = len(ds)
    index = random.randint(0, ds_size - 1)
    item = ds[index]
    for key in item:
        print(f"{key}, shape={item[key].shape}")
    

if __name__ == "__main__":
    typer.run(main)
