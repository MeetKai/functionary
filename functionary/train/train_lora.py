import json
import math
import os
import pathlib
import random
import sys
from dataclasses import dataclass, field
from typing import Dict, Optional, List

import torch
import torch.distributed
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss

os.environ["WANDB_LOG_MODEL"] = "all"

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
import bitsandbytes as bnb
import transformers
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    BitsAndBytesConfig,
    LlamaTokenizer,
    LlamaTokenizerFast,
    Trainer,
    deepspeed,
)

from functionary.prompt import get_additional_tokens
from functionary.train.custom_datasets import CustomDataset, DirectPackedDataset
from functionary.train.llama_attention_mask_monkey_patch import LlamaForCausalLM

LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))


def print_rank0(*arg):
    if LOCAL_RANK == 0:
        print(*arg)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="meta-llama/Llama-2-7b-hf")


@dataclass
class DataArguments:
    train_data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    eval_data_path: str = field(default=None, metadata={"help": "Path to the eval data."})
    packing: bool = field(default=False, metadata={"help": "Whether use packing or not"})
    max_packing_length: int = field(default=4096, metadata={"help": "maximum sequence length we can pack"})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    model_max_length: int = field(
        default=4096,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )


@dataclass
class LoraArguments:
    lora_r: int = 16
    lora_alpha: int = 64
    lora_dropout: float = 0.1
    lora_target_modules: str = "all"  # all for all linear; "q_proj v_proj"
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False


def find_all_linear_names(model):
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, bnb.nn.Linear4bit) or isinstance(module, torch.nn.Linear):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    lora_param_count = 0
    all_param = 0
    embedding_lm_head_param_count = 0
    for name, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            print_rank0(f"trainable: {name}, num_params: {num_params}")
            if "lm_head" in name or "embed_tokens" in name:
                embedding_lm_head_param_count += num_params
            else:
                lora_param_count += num_params
    trainable_params = embedding_lm_head_param_count + lora_param_count
    print_rank0(
        f"all params: {all_param:,d} || trainable params: {trainable_params:,d} || trainable%: {100 * trainable_params / all_param}"
    )
    print_rank0(
        f"embedding_lm_head_param_count: {embedding_lm_head_param_count} = {embedding_lm_head_param_count * 100 / all_param} %"
    )
    print_rank0(f"loara_param: {lora_param_count} = {lora_param_count * 100 / all_param} %")


def get_device_map(training_args: TrainingArguments, lora_args: LoraArguments) -> Optional[Dict]:
    device_map = None
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if lora_args.q_lora:
        if ddp and training_args.fsdp:
            print("FSDP is incompatible with QLORA")
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else None
        if len(training_args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled():
            print("FSDP and ZeRO3 are both currently incompatible with QLoRA.")
    return device_map


def load_model_with_rope_scaling(
    model_args: ModelArguments, training_args: TrainingArguments, lora_args: LoraArguments, data_args: DataArguments
) -> transformers.AutoModelForCausalLM:
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
        scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    config.use_cache = False

    compute_dtype = torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)
    use_flash_attention_2 = True
    if data_args.packing:
        print_rank0("do not use flash attention because of using packing")
        use_flash_attention_2 = False # currently packing is only possible when use_flash_attention_2=False
    model = LlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        device_map=get_device_map(training_args, lora_args),
        use_flash_attention_2=use_flash_attention_2,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            use_flash_attention_2=use_flash_attention_2,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        if lora_args.q_lora
        else None,
    )
    return model


def prepare_model_for_training(
    model: transformers.AutoModelForCausalLM, training_args: TrainingArguments, lora_args: LoraArguments
):
    if lora_args.lora_target_modules == "all":
        target_modules = find_all_linear_names(model)
    else:
        modules = lora_args.lora_target_modules.split(" ")
        target_modules = [mod.strip() for mod in modules if len(mod.strip()) > 0]

    print_rank0("target_modules: ", target_modules)
    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_args.lora_dropout,
        bias=lora_args.lora_bias,
        task_type="CAUSAL_LM",
        modules_to_save=["lm_head", "embed_tokens"],  # because we retrain the embedding
    )

    if lora_args.q_lora:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    model = get_peft_model(model, lora_config)

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    model.config.use_cache = False
    print_trainable_parameters(model)
    return model


# Borrowed from: https://github.com/lm-sys/FastChat/blob/main/fastchat/train/train_lora.py#L68
def maybe_zero_3(param):
    if hasattr(param, "ds_id"):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
# Borrowed from: https://github.com/lm-sys/FastChat/blob/main/fastchat/train/train_lora.py#L68
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return


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


def initialize_tokenizer(
    model: transformers.AutoModelForCausalLM,
    model_name_or_path: str,
    model_max_length: int,
    cache_dir: str,
):
    """Initialize tokenizer and add special tokens, resizing vocab and embedding"""
    # note that must set legacy=True, read more: https://github.com/huggingface/transformers/issues/25176
    tokenizer = LlamaTokenizerFast.from_pretrained(
        model_name_or_path,
        cache_dir=cache_dir,
        model_max_length=model_max_length,
        legacy=True,
    )

    # Add special tokens
    tokenizer.pad_token = tokenizer.unk_token
    special_tokens = {"additional_special_tokens": get_additional_tokens()}
    num_new_tokens = tokenizer.add_special_tokens(special_tokens)

    # Resize embedding
    model.resize_token_embeddings(len(tokenizer))
    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

    return tokenizer


def get_dataset(data_args, training_args, tokenizer, dtype):
    ds_class = CustomDataset
    if data_args.packing:
        ds_class = DirectPackedDataset
    cached_path = os.path.join(training_args.output_dir, f"{dtype}_cached.jsonl")
    raw_train_data = None
    if training_args.local_rank > 0:
        print(f"process: {LOCAL_RANK} wait for main process to prepare the training data")
        torch.distributed.barrier()
    else:    
        if not os.path.exists(training_args.output_dir):
            os.mkdir(training_args.output_dir)
        data_path = data_args.train_data_path if dtype == "train" else data_args.eval_data_path
        with open(data_path, "r") as file:
            raw_train_data = [json.loads(line) for line in file]
        print_rank0(f"train_size: {len(raw_train_data)}")
        # TODO set ignore_cached=True to use the cached at the first time we run
        ds = ds_class(raw_train_data, tokenizer, cached_path=cached_path, ignore_cached=False)
        print(f"process: {LOCAL_RANK} finish processing data")
        torch.distributed.barrier()
    ds = ds_class(raw_train_data, tokenizer, cached_path=cached_path, ignore_cached=False)
    ds.stat()
    return ds

def train():
    argument_parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, LoraArguments))
    model_args, data_args, training_args, lora_args = argument_parser.parse_args_into_dataclasses()

    # Set RoPE scaling factor

    print_rank0("lora args: ", lora_args)

    print_rank0("training args: ", training_args)

    model = load_model_with_rope_scaling(model_args, training_args, lora_args, data_args)
    print_rank0(model)

    tokenizer = initialize_tokenizer(
        model,
        model_args.model_name_or_path,
        training_args.model_max_length,
        training_args.cache_dir,
    )
    
    assert data_args.train_data_path is not None, "Please provide a training data file."
    
    train_dataset = get_dataset(data_args, training_args, tokenizer, "train")
    print_some_examples(train_dataset, tokenizer)
    print_rank0("final train size: ", len(train_dataset))
    
    if training_args.do_eval:
        eval_dataset = get_dataset(data_args, training_args, tokenizer, "eval")
        print_rank0("final train size: ", len(eval_dataset))
    
    print_rank0("tokenizer.model_max_length: ", tokenizer.model_max_length)

    model = prepare_model_for_training(model, training_args, lora_args)
    
    def preprocess_logits_for_metrics(logits, labels):
        """Preprocesses the logits during evaluation by computing the greedy token predictions for
        accuracy calculation and loss values for perplexity calculation. Both pred_ids and loss are
        of shape (batch_size x seq_len)"""
        pred_ids = torch.argmax(logits, dim=-1)

        loss_fn = CrossEntropyLoss(reduction="none")
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_logits = shift_logits.view(-1, len(tokenizer))
        shift_labels = shift_labels.view(-1)
        loss = loss_fn(shift_logits, shift_labels)
        loss = torch.mean(loss.view(logits.shape[0], -1), dim=-1)

        return pred_ids, loss

    def compute_metrics(eval_preds):
        """Computes next-token accuracy and perplexity metrics for evaluation"""
        predictions = eval_preds.predictions[0][:, :-1]
        labels = eval_preds.label_ids[:, 1:]

        # Calculate accuracy
        acc_count = 0
        total_num = 0
        for pred, label in zip(predictions.flatten().tolist(), labels.flatten().tolist()):
            if label != -100:
                if label == pred:
                    acc_count += 1
                total_num += 1

        # Calculate perplexity
        loss = eval_preds.predictions[1].tolist()
        loss = sum(loss) / len(loss)
        perplexity = math.exp(loss)

        return {"accuracy": acc_count / total_num, "perplexity": perplexity}

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

    # check if zero3 mode enabled
    if deepspeed.is_deepspeed_zero3_enabled():
        # use deepspeed engine internal function to gather state dict
        # state_dict_zero3 contains whole parameters of base and lora adapters
        # we will not extract lora parameters since peft save_pretrained will do that
        # https://github.com/huggingface/peft/blob/3714aa2fff158fdfa637b2b65952580801d890b2/src/peft/peft_model.py#L125
        # https://github.com/huggingface/peft/blob/3714aa2fff158fdfa637b2b65952580801d890b2/src/peft/utils/save_and_load.py#L19
        state_dict_zero3 = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
        if training_args.local_rank == 0:
            state_dict = state_dict_zero3
    else:
        # in other mode we use original code from fastchat team, to make sure our change is minimum
        state_dict = get_peft_state_maybe_zero_3(model.named_parameters(), lora_args.lora_bias)

    if training_args.local_rank == 0:
        model.save_pretrained(training_args.output_dir, state_dict=state_dict)


if __name__ == "__main__":
    train()
