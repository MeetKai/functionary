import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from typing import Dict, Optional
import json
from datasets import Dataset
from datetime import timezone
from transformers import AutoTokenizer, BitsAndBytesConfig
import transformers
import torch
from transformers.trainer_utils import is_main_process
from dataclasses import dataclass, field
from transformers import Trainer
from trl import GRPOConfig, GRPOTrainer, ModelConfig
from trl import get_kbit_device_map, get_peft_config, get_quantization_config

from transformers.modeling_utils import is_deepspeed_zero3_enabled
import os
from functionary.prompt_template import PromptTemplate, get_prompt_template_by_version

from transformers import (
    Trainer,
    TrainingArguments,
)
import os
import datetime
import shutil
from huggingface_hub import HfApi


import importlib.util
import os
from typing import Callable, Any

LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))
BETA_GRPO = 0.04


@dataclass
class TrainingArguments(GRPOConfig):
    use_liger: Optional[bool] = field(default=False)
    use_attn_implementation: Optional[str] = field(default="")
    prompt_template_version: str = field(
        default="qwen2.5-text-only",
        metadata={"help": "choose prompt template to use for training"},
    )
    reward_functions_path: str = field(
        default="functionary/train/example_rewards.py",
        metadata={"help": "Path to the reward functions."},
    )
    reward_function_names: str = field(
        default="reward_func",
        metadata={"help": "Name of the reward functions separated by comma."},
    )


@dataclass
class ModelArguments(ModelConfig):
    model_name_or_path: Optional[str] = field(default="meta-llama/Llama-2-7b-hf")


@dataclass
class DataArguments:
    train_data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    training_ratio: float = field(
        default=1.0, metadata={"help": "percentage of data used for training"}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the eval data."}
    )


def get_dataset(data_path: str, prompt_template: PromptTemplate):
    with open(data_path, "r") as f:
        data = [json.loads(line) for line in f]

    prompts = []
    extra_data_list = []
    for item in data:
        messages = item.get("messages", [])
        tools = item.get("tools", [])
        prompt = prompt_template.get_prompt_from_messages(
            messages, tools, add_generation_prompt=True
        )
        prompts.append(prompt)
        if "extra_data" in item:
            extra_data_list.append(item["extra_data"])
        else:
            extra_data_list.append(None)
    return Dataset.from_dict({"prompt": prompts, "extra_data": extra_data_list})


def load_function(file_path: str, function_name: str) -> Callable[..., Any]:
    """Dynamically load a function from a given Python file path."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No such file: {file_path}")

    # Create a module name based on file name
    module_name = os.path.splitext(os.path.basename(file_path))[0]

    # Load module spec
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        raise ImportError(f"Could not load spec from {file_path}")

    module = importlib.util.module_from_spec(spec)
    loader = spec.loader
    if loader is None:
        raise ImportError(f"No loader for {file_path}")

    loader.exec_module(module)

    # Get function from module
    func = getattr(module, function_name, None)
    if func is None:
        raise AttributeError(f"Function '{function_name}' not found in {file_path}")

    return func


def main():
    """Format of training requests"""
    argument_parser = transformers.HfArgumentParser(
        (TrainingArguments, ModelArguments, DataArguments)
    )
    training_args, model_args, data_args = argument_parser.parse_args_into_dataclasses()
    training_args.beta = BETA_GRPO

    assert training_args.reward_functions_path, "reward_functions_path is required"
    assert training_args.reward_function_names, "reward_function_names is required"
    reward_function_names = training_args.reward_function_names.split(",")
    reward_function_names = [
        name.strip() for name in reward_function_names if name.strip()
    ]
    reward_functions = [
        load_function(training_args.reward_functions_path, name)
        for name in reward_function_names
    ]

    output_dir = training_args.output_dir
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # max_length = get_max_length_config()
    # if "max_length" in train_request:
    #     max_length = train_request["max_length"]
    # default implementation, max_length=1024 (prompt + completion), max_prompt_length=512
    prompt_template = get_prompt_template_by_version(
        training_args.prompt_template_version
    )
    train_ds = get_dataset(data_args.train_data_path, prompt_template)
    dev_ds = get_dataset(data_args.eval_data_path, prompt_template)

    original_batch_size = training_args.per_device_train_batch_size

    quantization_config = get_quantization_config(model_args)
    device_string = "cuda:" + str(LOCAL_RANK)
    device_map = (
        get_kbit_device_map()
        if quantization_config is not None
        else {"": device_string}
    )
    if len(training_args.fsdp) > 0 or is_deepspeed_zero3_enabled():
        device_map = None

    attn_implementation = "flash_attention_2"
    if training_args.use_attn_implementation:
        attn_implementation = training_args.use_attn_implementation
        print(f"Using {attn_implementation} as the attention implementation")

    model_kwargs = dict(
        attn_implementation=attn_implementation,
        torch_dtype=torch.bfloat16,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=device_map,
    )

    # Only add quantization_config if it's not None
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config

    print(f"final training_args: {training_args}")

    if training_args.use_liger:
        from liger_kernel.transformers import AutoLigerKernelForCausalLM

        model_class = AutoLigerKernelForCausalLM
    else:
        model_class = transformers.AutoModelForCausalLM

    model = model_class.from_pretrained(model_args.model_name_or_path, **model_kwargs)

    peft_config = get_peft_config(model_args)

    # Check if this is the main process and create the output directory
    if is_main_process(LOCAL_RANK):  # Only create directory on main process
        os.makedirs(training_args.output_dir, exist_ok=True)
        print(f"Created output directory: {training_args.output_dir}")

    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_functions,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()


if __name__ == "__main__":
    main()
