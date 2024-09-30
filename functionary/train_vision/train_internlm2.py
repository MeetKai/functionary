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

import transformers
from functionary.train_vision.internlm_dataset import LazyVisionDataset
from functionary.train_vision.models.modeling_internvl.modeling_internvl_chat import InternVLChatModel

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
from transformers import AutoConfig, AutoTokenizer, Trainer

LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))


def print_rank0(*arg):
    if LOCAL_RANK == 0:
        print(*arg)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="meta-llama/Llama-2-7b-hf")


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


def trainer_save_model_safe(trainer: transformers.Trainer):
    """Saves the model in fsdp.FULL_STATE_DICT mode to have the model weights
    in .bin file format which is loadable by HF Transformers"""
    if trainer.accelerator.state.fsdp_plugin.state_dict_type.name != "FULL_STATE_DICT":
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model()


def print_parameters_info(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print_rank0(f"trainable: {name}")
        else:
            print_rank0(f"freezed: {name}")


def freeze_vision_model(model):
    for name, param in model.named_parameters():
        if name.startswith("vision_model"):
            param.requires_grad = False


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
        trust_remote_code=True
    )

    # Add special tokens
    tokenizer.pad_token = tokenizer.eos_token
    added_tokens = prompt_template.get_additional_tokens()
    special_tokens = {"additional_special_tokens": added_tokens}
    num_new_tokens = tokenizer.add_special_tokens(special_tokens)

    # add chat_template for tokenizer
    tokenizer.chat_template = prompt_template.get_chat_template_jinja()
    print("tokenizer: ", tokenizer)
    
    if num_new_tokens > 0:
        # Resize embedding if in the prompt template we add new tokens
        model.resize_token_embeddings(len(tokenizer))
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

    # store special tokens to model
    start_img_token_id = tokenizer.convert_tokens_to_ids(prompt_template.start_img_token)
    end_img_token_id = tokenizer.convert_tokens_to_ids(prompt_template.end_img_token)
    img_context_token_id = tokenizer.convert_tokens_to_ids(prompt_template.img_context)
    img_place_holder_token = tokenizer.convert_tokens_to_ids(prompt_template.start_img_token)
    
    model.img_start_token = start_img_token_id
    model.img_end_token = end_img_token_id
    model.img_context_token = img_context_token_id
    model.img_place_holder_token = img_place_holder_token
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


def train():
    argument_parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = argument_parser.parse_args_into_dataclasses()
    # this is a must
    training_args.remove_unused_columns = False
    # this is a must
    training_args.prompt_template_version = "internlm2-chat"
    print(
        "---------------------training_args.remove_unused_columns: ",
        training_args.remove_unused_columns,
    )

    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    model = InternVLChatModel.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=compute_dtype, use_flash_attention_2=True, trust_remote_code=True)
    
    model.config.use_cache = False
    
    setattr(model.config, "hidden_size", model.language_model.config.hidden_size)
    model.supports_gradient_checkpointing = True
    
    freeze_vision_model(model)
    
    print_parameters_info(model)

    print_rank0("Prompt template to use: ", training_args.prompt_template_version)
    prompt_template = get_prompt_template_by_version(
        training_args.prompt_template_version
    )

    tokenizer = initialize_tokenizer(
        model=model,
        model_name_or_path=model_args.model_name_or_path,
        prompt_template=prompt_template,
        model_max_length=training_args.model_max_length,
        cache_dir=training_args.cache_dir
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

    train_dataset = LazyVisionDataset(raw_train_ds, tokenizer, data_args.pad_img_path)
    if torch.distributed.get_rank() == 0:
        print(f"Training Data Loaded: #{len(train_dataset)}")

    if training_args.do_eval:
        with open(data_args.eval_data_path, "r") as f:
            raw_eval_ds = [json.loads(line) for line in f]

        eval_dataset = LazyVisionDataset(raw_eval_ds, tokenizer, data_args.pad_img_path)

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
            logits_dic_result = logits[0]
            if type(logits_dic_result) is dict:
                correct_logits = logits_dic_result["logits"]
                added_labels = logits_dic_result["labels"]
                labels = added_labels
            else:
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

    def collate_examples(features):
        result = {}
        first = features[0]
        for k, v in first.items():
            if k in ["input_ids", "attention_mask", "labels"]:
                if isinstance(v, torch.Tensor):
                    result[k] = torch.stack([f[k] for f in features])
                elif isinstance(v, np.ndarray):
                    result[k] = torch.tensor(np.stack([f[k] for f in features]))
                else:
                    result[k] = torch.tensor([f[k] for f in features])
        # aggregate images
        images = []
        for feature in features:
            images.extend(feature["images"])

        result["images"] = images
        return result

    class FunctionAccuracyTrackingTrainer(Trainer):
        """This trainer will also log the metrics for each training step

        Args:
            Trainer (_type_): _description_
        """

        def compute_loss(self, model, inputs, return_outputs=False):
            """
            How the loss is computed by Trainer. By default, all models return the loss in the first element.

            Subclass and override for custom behavior.
            Besides return the loss, this function also compute the metrics for each training step
            """
            outputs = model(**inputs)
            loss = outputs.loss
            logits = outputs.logits["logits"]
            labels = outputs.logits["labels"]

            pred_ids = torch.argmax(logits, dim=-1)

            # compute the metrics
            predictions = pred_ids[:, :-1]
            predictions = self.accelerator.pad_across_processes(
                predictions, dim=1, pad_index=-100
            )
            predictions = self.accelerator.gather_for_metrics((predictions))

            labels = labels[:, 1:]
            labels = self.accelerator.pad_across_processes(
                labels, dim=1, pad_index=-100
            )
            labels = self.accelerator.gather_for_metrics((labels))

            prediction_list, label_list = (
                predictions.flatten().tolist(),
                labels.flatten().tolist(),
            )
            metrics = compute_accuracy_metrics(prediction_list, label_list)
            prefix_metrics = {}
            for key in metrics:
                prefix_metrics[f"train_{key}"] = metrics[key]
            # Log to wandb
            self.log(prefix_metrics)
            return (loss, outputs) if return_outputs else loss

    # if set log_train_metrics=true will compute the metrics after each training step
    trainer_class = (
        FunctionAccuracyTrackingTrainer if training_args.log_train_metrics else Trainer
    )

    if training_args.do_eval:
        trainer = trainer_class(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            data_collator=collate_examples,
        )
    else:
        trainer = trainer_class(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=collate_examples,
            compute_metrics=compute_metrics,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
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
