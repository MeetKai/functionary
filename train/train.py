import random
from dataclasses import dataclass, field
import json
import pathlib
from typing import Dict, Optional

import torch
from torch.utils.data import Dataset
import torch.distributed

from llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn

replace_llama_attn_with_flash_attn()

import transformers
from transformers import Trainer, LlamaTokenizer


def create_target_tensors(input_ids, ignore_from=None, ignore_to=None):
    """Creates target tensors based on the ignoring range."""
    targets = input_ids.clone()
    if ignore_from is not None:
        targets[ignore_from:] = -100  # OR LabelSmoother.ignore_index
    if ignore_to is not None:
        targets[:ignore_to] = -100  # OR LabelSmoother.ignore_index
    return targets




'''
for llama 2, use this formatting function.

def prepare_message_for_model(messages, tokenizer):
    """Prepares given messages for the model by tokenizing the content and determining target tokens."""

    system_content = ""
    conversation_content = ""
    inst_flag = False  # Flag to check if a new turn has started

    for message in messages:
        if message["role"] == 'system':
            content = message.get("content", "")
            system_content += "<<SYS>>\n{content}\n<</SYS>>\n".format(content=content)

        elif message["role"] == "user" and message.get("content") is not None:
            if inst_flag:  # Check if a new turn should start
                conversation_content += "</s><s>[INST]"
            else:
                conversation_content += "[INST]"
            content = message.get("content", "")
            conversation_content += "{content}[/INST] ".format(content=content)
        ## this condition must be before 'elif message["role"] == 'assistant' and message.get("content") is not None:'
        elif message["role"] == "assistant" and message.get("to"):
            fn_call = "to={to}:\n{content}".format(to=message.get("to", ""), content=message.get("content", ""))
            print(fn_call)
            conversation_content += "{content}".format(content=fn_call)
            inst_flag = True

        elif message["role"] == 'assistant' and message.get("content") is not None:
            content = message.get("content", "")
            conversation_content += "{content}".format(content=content)
            inst_flag = True

        elif message["role"] == "function":
            text = "function name={name}:\n{content}\n".format(name=message.get("name", ""),
                                                           content=message.get("content", ""))
            if inst_flag:  # Check if a new turn should start
                conversation_content += "</s><s>[INST]".format(content=text)
            else:
                conversation_content += "[INST]"
            conversation_content += "{content}[/INST] ".format(content=text)

    # Check if the last turn was not closed
    if inst_flag:
        conversation_content += "</s>"

    text = "<s>[INST]{system_content}{conversation_content}".format(system_content=system_content, conversation_content=conversation_content)

    return text

'''

def prepare_message_for_model(message, tokenizer):
    """Prepares a given message for the model by tokenizing the content and determining target tokens."""

    if message["role"] == "system":
        text = "system:\n{content}\n".format(content=message.get("content", ""))
        input_ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids
        targets = create_target_tensors(input_ids, ignore_from=0, ignore_to=len(input_ids[0]))

    elif message["role"] == "function":
        text = "function name={name}:\n{content}\n".format(name=message.get("name", ""),
                                                           content=message.get("content", ""))
        input_ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids
        targets = create_target_tensors(input_ids, ignore_from=0, ignore_to=len(input_ids[0]))

    elif message["role"] == "user" and message.get("content") is None:
        text = "user:\n</s>"
        input_ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids
        targets = create_target_tensors(input_ids)

    elif message["role"] == "user":
        text = "user:\n</s>{content}\n".format(content=message.get("content", ""))
        input_ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids
        targets = create_target_tensors(input_ids, ignore_from=4)

    elif message["role"] == "assistant" and message.get("to") is not None:
        text = "assistant to={to}:\n{content}</s>".format(to=message.get("to", ""), content=message.get("content", ""))
        input_ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids
        targets = create_target_tensors(input_ids)

    elif message["role"] == "assistant" and message.get("content") is None:
        text = "assistant"
        input_ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids
        targets = create_target_tensors(input_ids)

    elif message["role"] == "assistant":
        text = "assistant:\n{content}\n".format(content=message.get("content", ""))
        input_ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids
        targets = create_target_tensors(input_ids)

    return text, input_ids, targets


def prepare_messages_for_model(messages, tokenizer):
    """Prepares a list of messages for the model by calling `prepare_message_for_model` function on each of them and
    concatenating the returned input_ids and targets. Also, the function merges the text of the messages."""
    all_texts = []
    all_input_ids = []
    all_targets = []

    for message in messages:
        text, input_ids, targets = prepare_message_for_model(message, tokenizer)
        all_texts.append(text)
        all_input_ids.append(input_ids.squeeze(0))
        all_targets.append(targets.squeeze(0))

    input_ids_tensor = torch.cat(all_input_ids, dim=-1)
    targets_tensor = torch.cat(all_targets, dim=-1)
    merged_text = "".join(all_texts)

    prepared_input = tokenizer.prepare_for_model(
        input_ids_tensor.tolist(),
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
    )

    prepared_targets = tokenizer.prepare_for_model(
        targets_tensor.tolist(),
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
    )

    return dict(
        input_ids=prepared_input["input_ids"],
        labels=prepared_targets["input_ids"],
        attention_mask=prepared_input["attention_mask"],
    )


class CustomDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(CustomDataset, self).__init__()
        self.tokenizer = tokenizer

        self.raw_data = raw_data
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = prepare_messages_for_model(self.raw_data[i], self.tokenizer)
        ret = {
            "input_ids": ret["input_ids"],
            "labels": ret["labels"],
            "attention_mask": ret["attention_mask"],
        }
        self.cached_data_dict[i] = ret
        return ret


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="meta-llama/Llama-2-7b-hf")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=4096,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

def train():
    argument_parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = argument_parser.parse_args_into_dataclasses()
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    model.config.use_cache = False
    tokenizer = LlamaTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        # legacy=False,  # See: https://github.com/huggingface/transformers/pull/24565
    )
    # See: https://github.com/facebookresearch/llama-recipes/blob/83fde7b94bd47e402731bfcf491beaf9950d2929/llama_finetuning.py#L111
    tokenizer.add_special_tokens(
        {

            "pad_token": "<PAD>",
        }
    )

    with open(data_args.data_path, "r") as file:
        raw_data = [json.loads(line) for line in file]

    if torch.distributed.get_rank() == 0:
        print(f"Data Loaded: #{len(raw_data)}")

    random.shuffle(raw_data)

    dataset = CustomDataset(raw_data, tokenizer)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
