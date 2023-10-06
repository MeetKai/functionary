import json
import math
import pathlib
import random
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.distributed
from torch.nn import CrossEntropyLoss

from functionary.prompt import EndToken
from functionary.train.datasets import CustomDataset, split_data
from functionary.train.llama_flash_attn_monkey_patch import (
    replace_llama_attn_with_flash_attn,
)

replace_llama_attn_with_flash_attn()

import transformers
from transformers import LlamaTokenizer, Trainer


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="meta-llama/Llama-2-7b-hf")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    train_valid_split: float = field(
        default=0.9,
        metadata={
            "help": "Ratio to split overall data into train-validation. Must be between 0.0 and 1.0."
        },
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=4096,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    
    
def initialize_tokenizer(
    model: transformers.LlamaForCausalLM,
    model_name_or_path: str,
    model_max_length: int,
    cache_dir: str,
):
    """Initialize tokenizer and add special tokens, resizing vocab and embedding"""
    # note that must set legacy=True, read more: https://github.com/huggingface/transformers/issues/25176
    tokenizer = LlamaTokenizer.from_pretrained(
        model_name_or_path,
        cache_dir=cache_dir,
        model_max_length=model_max_length,
        padding_side="right",
        legacy=True,
    )

    # Add special tokens
    tokenizer.pad_token = tokenizer.unk_token
    special_tokens = {"additional_special_tokens": [e.value for e in EndToken]}
    num_new_tokens = tokenizer.add_special_tokens(special_tokens)
    
    # Resize embedding
    model.resize_token_embeddings(len(tokenizer))
    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

    return tokenizer


if __name__ == "__main__":
    argument_parser = transformers.HfArgumentParser(
            (ModelArguments, DataArguments, TrainingArguments)
        )
    model_args, data_args, training_args = argument_parser.parse_args_into_dataclasses()
    
    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
        scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    config.use_cache = False

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.config.use_cache = False

    tokenizer = initialize_tokenizer(
        model,
        model_args.model_name_or_path,
        training_args.model_max_length,
        training_args.cache_dir,
    )
    
    with open(data_args.data_path, "r") as file:
        raw_data = [json.loads(line) for line in file]

    if torch.distributed.get_rank() == 0:
        print(f"Data Loaded: #{len(raw_data)}")
        
    # Do train-validation split
    assert (
        0.0 < data_args.train_valid_split <= 1.0
    ), f"The `train_valid_split` argument of `{data_args.train_valid_split}` is not between 0.0 and 1.0."

    raw_train_data, raw_eval_data = split_data(
        raw_data, data_args.data_path, data_args.train_valid_split
    )
    eval_dataset = CustomDataset(raw_eval_data, tokenizer)
    
    datapoint = eval_dataset[0]
    input_ids = datapoint["input_ids"].tolist()
    labels = datapoint["labels"].tolist()
    
    # output = model.forward()
    
    breakpoint()