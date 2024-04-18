# This script is used to tokenize the dataset ahead for packing
# It is not necessary to run this script if you are using the PackedDataset class.
# Beware: This uses Functionary prompting template to tokenize.
import json
import os

import typer
from transformers import AutoTokenizer

from functionary.prompt_template import get_prompt_template_by_version
from functionary.train.custom_datasets import PackedDataset


def main(
    pretrained_path: str,
    data_path: str,
    save_folder: str,
    data_type: str,  # train/validation
    template_version: str = typer.Option(default="v2"),
    max_length: int = typer.Option(4096),
    pack_length: int = typer.Option(-1),
    max_packed_size: int = typer.Option(-1),
):
    """Tokenize the dataset ahead for packing

    Args:
        pretrained_path (str): pretrained model to use
        data_path (str): path to .jsonl file
        save_folder (str): where to save (the output_dir in training)
        data_type (str): one of: "train" or "validation"
        template_version: v1 or v2
        max_length (int, optional): max_length for tokenizer
    """
    assert data_type in ["train", "validation"]
    assert pack_length < max_length

    prompt_template = get_prompt_template_by_version(template_version)
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_path,
        model_max_length=max_length,
        legacy=True,
    )

    if pack_length == -1:
        pack_length = max_length

    tokenizer.pad_token = tokenizer.eos_token
    added_tokens = prompt_template.get_additional_tokens()
    special_tokens = {"additional_special_tokens": added_tokens}
    num_new_tokens = tokenizer.add_special_tokens(special_tokens)
    print("number of added tokens: ", num_new_tokens)

    with open(data_path, "r") as f:
        raw_data = [json.loads(line) for line in f]

    keep_assistant_prefix = True if data_type == "train" else False
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    cached_folder = f"{save_folder}/{data_type}_cached"
    if not os.path.exists(cached_folder):
        os.mkdir(cached_folder)

    print("number of items: ", len(raw_data))
    ds = PackedDataset(
        raw_data,
        tokenizer,
        cached_folder=cached_folder,
        ignore_cached=False,
        keep_assistant_prefix=keep_assistant_prefix,
        use_flash_attention=True,
        pack_length=pack_length,
        max_packed_size=max_packed_size,
    )
    ds.stat()


if __name__ == "__main__":
    typer.run(main)
