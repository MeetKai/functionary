import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from typing import Dict, Optional
import requests
import json
import random

# from torch.utils.data import Dataset
from datasets import Dataset
from transformers import AutoTokenizer, BitsAndBytesConfig
from transformers.modeling_utils import is_deepspeed_zero3_enabled

import transformers
import torch
from transformers.trainer_utils import is_main_process
from dataclasses import dataclass, field
from trl import DPOTrainer, DPOConfig, ModelConfig
from trl import get_kbit_device_map, get_peft_config, get_quantization_config
from transformers import TrainerCallback
import argparse
import os
from transformers import (
    Trainer,
    TrainingArguments,
    TrainerCallback,
    TrainerState,
    TrainerControl,
)

import os
from huggingface_hub import HfApi
from typing import Callable, Optional
import bitsandbytes as bnb
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from functionary.prompt_template import get_prompt_template_by_version

LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))


def print_rank0(*arg):
    if LOCAL_RANK == 0:
        print(*arg)


# class FunctionaryDPODataset(Dataset):
#     def __init__(self, data_path: str, prompt_template_version: str):
#         with open(data_path, "r") as f:
#             self.data = [json.loads(line) for line in f]
#         # assume that data with the fields: tools; messages; chosen; rejected
#         # chosen and rejected are assistant message; we will convert them to string
#         self.list_prompts = []
#         self.list_chosen = []
#         self.list_rejected = []
#         self.prompt_template = get_prompt_template_by_version(prompt_template_version)

#         for item in self.data:
#             messages = item["messages"]
#             tools = item.get("tools", []) or []
#             chosen = item["chosen"]
#             rejected = item["rejected"]
#             input_prompt = self.prompt_template.get_prompt_from_messages(
#                 messages, tools_or_functions=tools, add_generation_prompt=True
#             )
#             # compute the output prompt for chosen
#             full_prompt_chosen = self.prompt_template.get_prompt_from_messages(
#                 messages + [chosen], tools_or_functions=tools
#             )
#             chosen_output = full_prompt_chosen[len(input_prompt) :]

#             full_prompt_rejected = self.prompt_template.get_prompt_from_messages(
#                 messages + [rejected], tools_or_functions=tools
#             )
#             rejected_output = full_prompt_rejected[len(input_prompt) :]

#             self.list_prompts.append(input_prompt)
#             self.list_chosen.append(chosen_output)
#             self.list_rejected.append(rejected_output)

#     def __len__(self):
#         return len(self.list_prompts)

#     def __getitem__(self, idx):
#         return {
#             "prompt": self.list_prompts[idx],
#             "chosen": self.list_chosen[idx],
#             "rejected": self.list_rejected[idx],
#         }


def get_dataset_from_jsonl(data_path: str, prompt_template_version: str):
    with open(data_path, "r") as f:
        data = [json.loads(line) for line in f]
    list_prompts = []
    list_chosen = []
    list_rejected = []
    prompt_template = get_prompt_template_by_version(prompt_template_version)

    for item in data:
        messages = item["messages"]
        tools = item.get("tools", []) or []
        chosen = item["selected_answer"]
        rejected = item["rejected_answer"]
        input_prompt = prompt_template.get_prompt_from_messages(
            messages, tools_or_functions=tools, add_generation_prompt=True
        )
        # compute the output prompt for chosen
        full_prompt_chosen = prompt_template.get_prompt_from_messages(
            messages + [chosen], tools_or_functions=tools
        )
        chosen_output = full_prompt_chosen[len(input_prompt) :]

        full_prompt_rejected = prompt_template.get_prompt_from_messages(
            messages + [rejected], tools_or_functions=tools
        )
        rejected_output = full_prompt_rejected[len(input_prompt) :]

        list_prompts.append(input_prompt)
        list_chosen.append(chosen_output)
        list_rejected.append(rejected_output)

    return Dataset.from_dict(
        {"prompt": list_prompts, "chosen": list_chosen, "rejected": list_rejected}
    )


@dataclass
class ModelArguments(ModelConfig):
    model_name_or_path: Optional[str] = field(default="meta-llama/Llama-2-7b-hf")


@dataclass
class TrainingArguments(DPOConfig):
    use_liger: Optional[bool] = field(default=False)
    prompt_template_version: str = field(
        default="v2", metadata={"help": "choose prompt template to use for training"}
    )


@dataclass
class DataArguments:
    train_data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the eval data."}
    )


def trainer_save_model_safe(trainer: transformers.Trainer):
    """Saves the model in fsdp.FULL_STATE_DICT mode to have the model weights
    in .bin file format which is loadable by HF Transformers"""
    if trainer.accelerator.state.fsdp_plugin.state_dict_type.name != "FULL_STATE_DICT":
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model()


def main():
    argument_parser = transformers.HfArgumentParser(
        (DataArguments, TrainingArguments, ModelArguments)
    )
    data_args, training_args, model_args = argument_parser.parse_args_into_dataclasses()
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if LOCAL_RANK == 0:
        if not os.path.exists(training_args.output_dir):
            os.mkdir(training_args.output_dir)

        tokenizer.save_pretrained(training_args.output_dir)

    train_ds = get_dataset_from_jsonl(
        data_args.train_data_path, training_args.prompt_template_version
    )
    dev_ds = get_dataset_from_jsonl(
        data_args.eval_data_path, training_args.prompt_template_version
    )

    print_rank0(f"train_ds: {len(train_ds)}")
    print_rank0(f"dev_ds: {len(dev_ds)}")

    quantization_config = get_quantization_config(model_args)
    device_string = "cuda:" + str(LOCAL_RANK)
    device_map = (
        get_kbit_device_map()
        if quantization_config is not None
        else {"": device_string}
    )
    if len(training_args.fsdp) > 0 or is_deepspeed_zero3_enabled():
        device_map = None

    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=device_map,
        quantization_config=quantization_config,
    )

    if training_args.use_liger:
        from liger_kernel.transformers import AutoLigerKernelForCausalLM

        model_class = AutoLigerKernelForCausalLM
    else:
        model_class = transformers.AutoModelForCausalLM

    model = model_class.from_pretrained(model_args.model_name_or_path, **model_kwargs)
    model.resize_token_embeddings(len(tokenizer))
    peft_config = get_peft_config(model_args)
    ref_model = None
    if is_deepspeed_zero3_enabled():
        if peft_config is None:
            ref_model = model_class.from_pretrained(
                model_args.model_name_or_path, **model_kwargs
            )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()

    # FSDP requires state_dict_type=FULL_STATE_DICT in order to save the model weights in .bin format
    if trainer.is_fsdp_enabled:
        trainer_save_model_safe(trainer=trainer)
    else:
        trainer.save_model()


if __name__ == "__main__":
    main()
