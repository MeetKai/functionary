import json
import math
import os
import pathlib
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
import random
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import torch.distributed
from aenum import extend_enum
from PIL import Image
from torch.optim.lr_scheduler import LambdaLR
from functionary.train_vision.vision_datasets import (
    get_vision_dataset_class,
    get_collate_fn,
)
import re
from typing import Any

import transformers
from functionary.train.metrics import (
    extract_indices_of_first_tokens_of_param_values_in_assistant_response,
    extract_unmasked_chunks,
)
from functionary.train.packing.monkey_patch_packing import (
    monkey_patch_packing_for_model,
)

# set this so we can reproduce
random.seed(100)


extend_enum(
    transformers.trainer_utils.SchedulerType,
    "CUSTOMIZED_SCHEDULER",
    "customized_scheduler",
)


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    min_lr_ratio: float = 0.75,
    last_epoch: int = -1,
) -> LambdaLR:

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return current_step / max(1, num_warmup_steps)
        progress = (current_step - num_warmup_steps) / max(
            1, num_training_steps - num_warmup_steps
        )
        cosine_lr_multiple = (1.0 - min_lr_ratio) * 0.5 * (
            1.0 + math.cos(math.pi * num_cycles * 2.0 * progress)
        ) + min_lr_ratio
        return max(0.0, cosine_lr_multiple)

    return LambdaLR(optimizer, lr_lambda, last_epoch)


transformers.optimization.TYPE_TO_SCHEDULER_FUNCTION["customized_scheduler"] = (
    get_scheduler
)

from typing import Union

from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from functionary.prompt_template import PromptTemplate, get_prompt_template_by_version
from functionary.train_vision.llava_dataset import LazyVisionDataset
from transformers import AutoConfig, AutoTokenizer, Trainer

LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))


def print_rank0(*arg):
    if LOCAL_RANK == 0:
        print(*arg)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="meta-llama/Llama-2-7b-hf")
    model_class: str = field(
        default="", metadata={"help": "the model_class to load model"}
    )
    frozen_pattern: str = field(
        default="",
        metadata={
            "help": "regular expression of parameter names to be frozen during training"
        },
    )


@dataclass
class DataArguments:
    pad_img_path: str = field(
        default="functionary/train_vision/pad_img.png",
        metadata={"help": "pad image in case the data is text-only"},
    )
    train_data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    training_ratio: float = field(
        default=1.0, metadata={"help": "percentage of data used for training"}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the eval data."}
    )
    eval_ratio: float = field(
        default=1.0, metadata={"help": "percentage of data used for evluation"}
    )
    packing: bool = field(
        default=False, metadata={"help": "Whether use packing or not"}
    )
    max_packed_size: int = field(
        default=-1,
        metadata={
            "help": "maximum number of data points can be merged. For example, max_packed_size=3, we can only merge 2 or 3 data points into a new one"
        },
    )
    dataset_type: str = field(
        default="LazyQwen2VLDataset", metadata={"help": "The type of dataset to use"}
    )
    train_data_cached: str = field(
        default="",
        metadata={"help": "the path to the cached data for loading training data"},
    )
    validation_data_cached: str = field(
        default="",
        metadata={"help": "the path to the cached data for loading validation data"},
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_8bit")
    model_max_length: int = field(
        default=8192,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    keep_assistant_prefix: bool = field(
        default=False,
        metadata={
            "help": "Whether to mask the assistant prefix `<|from|>assistant\n<|recipient|>` during training"
        },
    )
    prompt_template_version: str = field(
        default="v2", metadata={"help": "choose prompt template to use for training"}
    )
    log_train_metrics: bool = field(
        default=False,
        metadata={"help": "set this true to log training metrics during training"},
    )
    
    use_liger: bool = field(
        default=False,
        metadata={
            "help": "Whether use liger or not. Refer to this link for more details: https://github.com/triton-lang/triton?tab=readme-ov-file#compatibility"
        },
    )


def trainer_save_model_safe(trainer: transformers.Trainer):
    """Saves the model in fsdp.FULL_STATE_DICT mode to have the model weights
    in .bin file format which is loadable by HF Transformers"""
    if trainer.accelerator.state.fsdp_plugin.state_dict_type.name != "FULL_STATE_DICT":
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model()


def freeze_model_parameters(model: Any, frozen_pattern: str):
    trainable_params = []
    for name, param in model.named_parameters():
        if re.search(frozen_pattern, name):
            print_rank0(f"-----Freeze parameter in the training: {name}")
            param.requires_grad = False
        else:
            trainable_params.append(name)
    print_rank0("------------TRAINABLE PARAMETERS--------")
    print_rank0("\n".join(trainable_params))
    print_rank0("------------------")


def initialize_tokenizer(
    *,
    model: transformers.AutoModelForCausalLM,
    model_name_or_path: str,
    prompt_template: PromptTemplate,
    model_max_length: int,
    cache_dir: str,
):
    """Initialize tokenizer and add special tokens, resizing vocab and embedding"""
    # Mistral requires left padding due to the Sliding Window Attention mechanism
    if "mistral" in type(model).__name__.lower():
        print("model is mistral so padding_side=left")
        padding_side = "left"
    else:
        padding_side = "right"

    # note that must set legacy=True, read more: https://github.com/huggingface/transformers/issues/25176
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        cache_dir=cache_dir,
        model_max_length=model_max_length,
        padding_side=padding_side,
        legacy=True,
    )

    # Add special tokens
    tokenizer.pad_token = tokenizer.eos_token
    added_tokens = prompt_template.get_additional_tokens()
    special_tokens = {"additional_special_tokens": added_tokens}
    num_new_tokens = tokenizer.add_special_tokens(special_tokens)

    # add chat_template for tokenizer
    tokenizer.chat_template = prompt_template.get_chat_template_jinja()
    # print("tokenizer: ", tokenizer)

    # Resize embedding if in the prompt template we add new tokens
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


def get_cached_path(data_args, training_args, model_args, file_name):
    current_folder = os.path.dirname(os.path.abspath(__file__))
    cache_folder = os.path.join(current_folder, "cached_data")
    if not os.path.exists(cache_folder):
        os.makedirs(cache_folder)
    
    model_name = model_args.model_name_or_path.replace("/", "_")
    data_name = file_name.replace("/", "_")
    length = training_args.model_max_length
    
    cached_path = os.path.join(cache_folder, f"{model_name}_{data_name}_{length}.json")
    return cached_path


def get_model_class(model_args):
    if model_args.model_class.lower() == "Qwen2VLForConditionalGeneration".lower():
        from transformers import Qwen2VLForConditionalGeneration
        from liger_kernel.transformers import apply_liger_kernel_to_qwen2_vl
        print("-------USE LIGER KERNEL-------")
        apply_liger_kernel_to_qwen2_vl()
        return Qwen2VLForConditionalGeneration
    if model_args.model_class.lower() == "Qwen2_5_VLForConditionalGeneration".lower():
        from transformers import Qwen2_5_VLForConditionalGeneration
        from liger_kernel.transformers import apply_liger_kernel_to_qwen2_vl
        print("-------USE LIGER KERNEL-------")
        apply_liger_kernel_to_qwen2_vl()
        return Qwen2_5_VLForConditionalGeneration
    return transformers.AutoModelForCausalLM


def train():
    argument_parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = argument_parser.parse_args_into_dataclasses()
    if data_args.packing:
        if not data_args.train_data_cached:
            data_args.train_data_cached = get_cached_path(data_args, training_args, model_args, data_args.train_data_path)
            
        if not data_args.validation_data_cached:
            data_args.validation_data_cached = get_cached_path(data_args, training_args, model_args, data_args.eval_data_path)
    # this is a must
    training_args.remove_unused_columns = False
    # this is a must
    print(
        "---------------------training_args.remove_unused_columns: ",
        training_args.remove_unused_columns,
    )

    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )


    if training_args.use_liger:
        from liger_kernel.transformers import AutoLigerKernelForCausalLM

        print_rank0("---------------using LIGER------------")
        model_class = AutoLigerKernelForCausalLM
    else:
        model_class = transformers.AutoModelForCausalLM
        
    model_class = get_model_class(model_args)
    model = model_class.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=compute_dtype,
        cache_dir=training_args.cache_dir,
        attn_implementation="flash_attention_2"
    )

    if model_args.frozen_pattern:
        freeze_model_parameters(model, model_args.frozen_pattern)

    if hasattr(model.config, "tokenizer_model_max_length"):
        model.config.tokenizer_model_max_length = training_args.model_max_length
    model.config.use_cache = False

    print_rank0("Prompt template to use: ", training_args.prompt_template_version)
    prompt_template = get_prompt_template_by_version(
        training_args.prompt_template_version
    )

    tokenizer = initialize_tokenizer(
        model=model,
        model_name_or_path=model_args.model_name_or_path,
        prompt_template=prompt_template,
        model_max_length=training_args.model_max_length,
        cache_dir=training_args.cache_dir,
    )

    if LOCAL_RANK == 0:
        if not os.path.exists(training_args.output_dir):
            os.mkdir(training_args.output_dir)

        tokenizer_folder = os.path.join(training_args.output_dir, "tokenizer")
        if not os.path.exists(tokenizer_folder):
            os.mkdir(tokenizer_folder)
        # Save tokenizer
        tokenizer.save_pretrained(tokenizer_folder)

    # get id of added tokens to compute the accuracy of predicing the token
    id2token = {
        tokenizer.encode(token)[-1]: token
        for token in prompt_template.get_additional_tokens()
    }
    print_rank0("id to tokens: ", id2token)

    assert data_args.train_data_path is not None, "Please provide a training data file."

    with open(data_args.train_data_path, "r") as f:
        raw_train_ds = [json.loads(line) for line in f]

    dataset_class = get_vision_dataset_class(
        data_args.dataset_type, packing=data_args.packing
    )
    print("dataset_class: ", dataset_class)
    add_params = {}
    if data_args.packing:
        # monkey-patch the model to support packing
        monkey_patch_packing_for_model(model_args.model_name_or_path)
        add_params = {"max_packed_size": data_args.max_packed_size}
        if data_args.train_data_cached:
            add_params["cached_path"] = data_args.train_data_cached
            print("Use cached to load training dataset")

    train_dataset = dataset_class(
        raw_train_ds,
        tokenizer,
        model_args.model_name_or_path,
        data_args.pad_img_path,
        training_args.model_max_length,
        use_img_pad_token=True,
        **add_params,
    )

    if torch.distributed.get_rank() == 0:
        print(f"Training Data Loaded: #{len(train_dataset)}")

    if training_args.do_eval:
        with open(data_args.eval_data_path, "r") as f:
            raw_eval_ds = [json.loads(line) for line in f]

        if data_args.validation_data_cached:
            add_params["cached_path"] = data_args.validation_data_cached
            print("use cached to load validation dataset")

        eval_dataset = dataset_class(
            raw_eval_ds,
            tokenizer,
            model_args.model_name_or_path,
            data_args.pad_img_path,
            training_args.model_max_length,
            use_img_pad_token=True,
            **add_params,
        )

        if torch.distributed.get_rank() == 0:
            print(f"Eval Data Loaded: #{len(eval_dataset)}")

    def preprocess_logits_for_metrics(logits, labels):
        """Preprocesses the logits during evaluation by computing the greedy token predictions for
        accuracy calculation and loss values for perplexity calculation. Both pred_ids and loss are
        of shape (batch_size x seq_len)"""
        correct_logits = logits
        added_labels = None
        if (
            type(logits) is tuple
        ):  # in mixtral logits is a tuple, correct logits is at the second index
            correct_logits = logits[1]
        elif type(logits) is dict:
            correct_logits = logits["logits"]
            added_labels = logits["labels"]
            labels = added_labels

        pred_ids = torch.argmax(correct_logits, dim=-1)

        loss_fn = CrossEntropyLoss(reduction="none")
        shift_logits = correct_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_logits = shift_logits.view(-1, len(tokenizer))
        shift_labels = shift_labels.view(-1)
        loss = loss_fn(shift_logits, shift_labels)
        loss = torch.mean(loss.view(correct_logits.shape[0], -1), dim=-1)

        return pred_ids, loss, added_labels

    def compute_accuracy_metrics(prediction_list, label_list):
        """Computes next-token accuracy and perplexity metrics for evaluation"""
        acc_count = 0
        total_num = 0
        dic = {token_id: {"acc": 0, "total": 0} for token_id in id2token}

        first_token_total_count, first_token_correct_count = 0, 0
        first_token_label_dic = {}

        for i in range(len(prediction_list)):
            pred, label = prediction_list[i], label_list[i]
            if i > 0 and label_list[i - 1] == -100 and label != -100:  # first token
                first_token_total_count += 1
                if label not in first_token_label_dic:
                    first_token_label_dic[label] = {"correct": 0, "total": 0}

                first_token_label_dic[label]["total"] += 1

                if label == pred:
                    first_token_correct_count += 1
                    first_token_label_dic[label]["correct"] += 1

            if label != -100:
                if label == pred:
                    acc_count += 1
                total_num += 1
            if label in dic:
                dic[label]["total"] += 1
                if label == pred:
                    dic[label]["acc"] += 1

        metrics = {"accuracy": acc_count / total_num}
        metrics = {
            "accuracy_recipient_token": first_token_correct_count
            / first_token_total_count,
            "total_number_recipient_token": first_token_total_count,
        }

        for token_id, stat in sorted(
            first_token_label_dic.items(), key=lambda x: -x[1]["total"]
        )[:5]:
            token = tokenizer.decode([token_id])
            metrics[f"accuracy_recipient_token_{token}"] = (
                stat["correct"] / stat["total"]
            )
            metrics[f"accuracy_recipient_token_{token}_total"] = stat["total"]

        # add accuracy for token: "all" if it is out of top-5
        if f"accuracy_recipient_token_all" not in metrics:
            all_token_id = tokenizer.encode("all", add_special_tokens=False)[0]
            if all_token_id in first_token_label_dic:
                stat = first_token_label_dic[all_token_id]
                metrics[f"accuracy_recipient_token_all"] = (
                    stat["correct"] / stat["total"]
                )
                metrics[f"accuracy_recipient_token_all_total"] = stat["total"]

        for token_id in dic:
            token = id2token[token_id]
            total_num = dic[token_id]["total"]
            acc = -1
            if total_num > 0:
                acc = dic[token_id]["acc"] / total_num
            metrics[f"accuracy_{token}"] = acc
            metrics[f"accuracy_total_num_{token}"] = total_num

        # Add metrics for accuracy of first tokens in arguments
        # only implemented for v3
        if "v3" in prompt_template.version:
            first_arguments_token_correct = 0
            first_arguments_token_total = 0

            unmasked_chunk_pairs = extract_unmasked_chunks(label_list, prediction_list)
            for label_chunk, pred_chunk in unmasked_chunk_pairs:
                # label_chunk_text = tokenizer.decode(label_chunk)
                # print_rank0(f"handle label_chunk:{label_chunk_text}")
                indices = extract_indices_of_first_tokens_of_param_values_in_assistant_response(
                    tokenizer, label_chunk
                )
                # if len(indices) > 0:
                #     included_tokens_str = ";".join(
                #         [tokenizer.decode([label_chunk[index]]) for index in indices]
                #     )
                #     print_rank0(
                #         f"label_chunk: {label_chunk_text}\nlabel: {included_tokens_str}"
                #     )
                for index in indices:
                    if label_chunk[index] == pred_chunk[index]:
                        first_arguments_token_correct += 1
                    first_arguments_token_total += 1

            if first_arguments_token_total > 0:
                metrics["accuracy_first_token_arguments"] = (
                    first_arguments_token_correct / first_arguments_token_total
                )
                metrics["accuracy_first_token_arguments_total"] = (
                    first_arguments_token_total
                )

        return metrics

    def compute_metrics(eval_preds):
        """Computes next-token accuracy and perplexity metrics for evaluation"""
        predictions = eval_preds.predictions[0][:, :-1]
        added_labels = eval_preds.predictions[2]
        labels = added_labels[:, 1:]

        prediction_list, label_list = (
            predictions.flatten().tolist(),
            labels.flatten().tolist(),
        )
        # Calculate perplexity
        loss = eval_preds.predictions[1].tolist()
        loss = sum(loss) / len(loss)
        perplexity = math.exp(loss)

        metrics = {"perplexity": perplexity}
        metrics.update(compute_accuracy_metrics(prediction_list, label_list))
        return metrics

    collate_fn = get_collate_fn(data_args.dataset_type, model, tokenizer)

    if training_args.do_eval:
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=None,
            preprocess_logits_for_metrics=None,
            data_collator=collate_fn,
        )
    else:
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=collate_fn,
            compute_metrics=None,
            preprocess_logits_for_metrics=None,
        )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()

    # FSDP requires state_dict_type=FULL_STATE_DICT in order to save the model weights in .bin format
    if trainer.is_fsdp_enabled:
        trainer_save_model_safe(trainer=trainer)
    else:
        trainer.save_model()


if __name__ == "__main__":
    train()
