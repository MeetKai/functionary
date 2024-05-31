import json
import math
import os
import pathlib
import sys
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import torch.distributed
import transformers
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer, Trainer

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from functionary.prompt_template import PromptTemplate, get_prompt_template_by_version
from functionary.train.custom_datasets import read_dataset

LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))


def print_rank0(*arg):
    if LOCAL_RANK == 0:
        print(*arg)


@dataclass
class ModelArguments:
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
    eval_ratio: float = field(
        default=1.0, metadata={"help": "percentage of data used for evluation"}
    )
    packing: bool = field(
        default=False, metadata={"help": "Whether use packing or not"}
    )
    pack_length: int = field(
        default=0,
        metadata={
            "help": "pack_length used to pack data points, default = 0 --> = model_max_length"
        },
    )
    max_packed_size: int = field(
        default=-1,
        metadata={
            "help": "maximum number of data points can be merged. For example, max_packed_size=3, we can only merge 2 or 3 data points into a new one"
        },
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_8bit")
    model_max_length: int = field(
        default=4096,
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


def trainer_save_model_safe(trainer: transformers.Trainer):
    """Saves the model in fsdp.FULL_STATE_DICT mode to have the model weights
    in .bin file format which is loadable by HF Transformers"""
    if trainer.accelerator.state.fsdp_plugin.state_dict_type.name != "FULL_STATE_DICT":
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model()


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
    print("tokenizer: ", tokenizer)

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


def extract_unmasked_chunks(labels: List[int], masked_value) -> List[List[int]]:
    """This function is used to extract unmasked chunks of integer
    For example, labels = [-100, -100, 1, 2, 3, -100, -100, 4, 5] --> chunks = [[1,2,3], [4,5]]
    Args:
        labels (List[int]): list of integer containing token_id and -100

    Returns:
        List[List[int]]: list of chunk, for example: [[1,2,3], [4,5]]
    """
    chunks = []
    chunk = []
    for token_id in labels:
        if token_id != masked_value:
            chunk.append(token_id)
        else:
            if len(chunk) > 0:
                chunks.append(chunk)
                chunk = []
    if len(chunk) > 0:
        chunks.append(chunk)
    return chunks


def print_some_examples(ds, tokenizer):
    data_loader = DataLoader(ds, batch_size=3)
    count = 0
    for batch in data_loader:
        if count == 0:
            print_rank0("keys in batch: ", batch.keys())
        print_rank0("--------------****Example data point****---------------")
        print_rank0("device: ", batch["input_ids"].device)
        print_rank0("shape of input_ids: ", batch["input_ids"].shape)  # B x L
        print_rank0("shape of labels: ", batch["labels"].shape)
        print_rank0("shape of attention_mask: ", batch["attention_mask"].shape)
        # print_rank0('input_ids: ', batch["input_ids"].tolist())
        # print_rank0('labels: ', batch["labels"].tolist())
        print_rank0("attention mask: ", batch["attention_mask"])
        input_ids = batch["input_ids"][0].tolist()
        input_chunk = extract_unmasked_chunks(input_ids, tokenizer.pad_token_id)
        assert len(input_chunk) == 1
        print_rank0("+ inputs: ")
        print_rank0(tokenizer.decode(input_chunk[0]))
        labels = batch["labels"][0].tolist()
        label_chunks = extract_unmasked_chunks(labels, -100)
        print_rank0("----------")
        for chunk in label_chunks:
            print_rank0("+ chunk: ")
            print_rank0(tokenizer.decode(chunk))
        count += 1
        if count == 5:
            break


def train():
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
        print_rank0("Rope scaling enabled")
    config.use_cache = False
    config.sliding_window = training_args.model_max_length

    if data_args.packing:
        print("Packing=True, using monkey-patched")
        config = AutoConfig.from_pretrained(model_args.model_name_or_path)
        config_type = type(config).__name__.lower()
        if "mistral" in config_type:
            print_rank0("using Monkey-patched MistralForCausalLM")
            from functionary.train.packing.monkey_patch_packing import (
                monkey_patch_packing_mistral,
            )

            monkey_patch_packing_mistral()

        elif "llama" in config_type:  # llama
            print_rank0("using Monkey-patched LlamaForCausalLM")
            from functionary.train.packing.monkey_patch_packing import (
                monkey_patch_packing_llama,
            )

            monkey_patch_packing_llama()

        elif "mixtral" in config_type:
            print_rank0("using Monkey-patched Mixtral")
            from functionary.train.packing.monkey_patch_packing import (
                monkey_patch_packing_mixtral,
            )

            monkey_patch_packing_mixtral()
        else:
            print("packing only supports models: Mistral, Llama, Mixtral")
            sys.exit(1)

    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=compute_dtype,
        config=config,
        cache_dir=training_args.cache_dir,
        use_flash_attention_2=True,
    )
    model.config.use_cache = False
    # Activate computing load balancing loss iin MixtralForCausalLM
    if hasattr(model.config, "output_router_logits"):
        setattr(model.config, "output_router_logits", True)
        print_rank0("Activate computing load balancing loss")

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

    train_dataset = read_dataset(
        model_args.model_name_or_path, data_args, training_args, tokenizer, "train"
    )

    if torch.distributed.get_rank() == 0:
        print(f"Training Data Loaded: #{len(train_dataset)}")

    if training_args.do_eval:
        eval_dataset = read_dataset(
            model_args.model_name_or_path,
            data_args,
            training_args,
            tokenizer,
            "validation",
        )

        if torch.distributed.get_rank() == 0:
            print(f"Eval Data Loaded: #{len(eval_dataset)}")

    print_rank0("***** HERE ARE SOME EXAMPLES FROM TRAINING ****")
    print_some_examples(train_dataset, tokenizer)

    print_rank0("***** HERE ARE SOME EXAMPLES FROM EVALUATION ***")
    print_some_examples(eval_dataset, tokenizer)

    def preprocess_logits_for_metrics(logits, labels):
        """Preprocesses the logits during evaluation by computing the greedy token predictions for
        accuracy calculation and loss values for perplexity calculation. Both pred_ids and loss are
        of shape (batch_size x seq_len)"""

        correct_logits = logits
        if (
            type(logits) is tuple
        ):  # in mixtral logits is a tuple, correct logits is at the second index
            correct_logits = logits[1]

        pred_ids = torch.argmax(correct_logits, dim=-1)

        loss_fn = CrossEntropyLoss(reduction="none")
        shift_logits = correct_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_logits = shift_logits.view(-1, len(tokenizer))
        shift_labels = shift_labels.view(-1)
        loss = loss_fn(shift_logits, shift_labels)
        loss = torch.mean(loss.view(correct_logits.shape[0], -1), dim=-1)

        return pred_ids, loss

    def compute_metrics(eval_preds):
        """Computes next-token accuracy and perplexity metrics for evaluation"""
        predictions = eval_preds.predictions[0][:, :-1]
        labels = eval_preds.label_ids[:, 1:]

        acc_count = 0
        total_num = 0
        dic = {token_id: {"acc": 0, "total": 0} for token_id in id2token}

        first_token_total_count, first_token_correct_count = 0, 0
        prediction_list, label_list = (
            predictions.flatten().tolist(),
            labels.flatten().tolist(),
        )
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

        # Calculate perplexity
        loss = eval_preds.predictions[1].tolist()
        loss = sum(loss) / len(loss)
        perplexity = math.exp(loss)

        metrics = {"accuracy": acc_count / total_num, "perplexity": perplexity}
        metrics = {
            "accuracy_first_token": first_token_correct_count / first_token_total_count,
            "total_number_first_token": first_token_total_count,
        }

        for token_id, stat in sorted(
            first_token_label_dic.items(), key=lambda x: -x[1]["total"]
        )[:5]:
            token = tokenizer.decode([token_id])
            metrics[f"accuracy_first_token_{token}"] = stat["correct"] / stat["total"]
            metrics[f"accuracy_first_token_{token}_total"] = stat["total"]

        for token_id in dic:
            token = id2token[token_id]
            total_num = dic[token_id]["total"]
            acc = -1
            if total_num > 0:
                acc = dic[token_id]["acc"] / total_num
            metrics[f"accuracy_{token}"] = acc
            metrics[f"accuracy_total_num_{token}"] = total_num

        return metrics

    if training_args.do_eval:
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
    else:
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
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
