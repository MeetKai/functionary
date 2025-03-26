import math, os, pathlib, sys, torch, torch.distributed, transformers

from dataclasses import dataclass, field
from typing import Dict, Optional

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
import bitsandbytes as bnb
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    BitsAndBytesConfig,
    Trainer,
)
from transformers.modeling_utils import is_deepspeed_zero3_enabled

from functionary.prompt_template import get_prompt_template_by_version
from functionary.train.custom_datasets import read_dataset
from functionary.train import training_utils
from functionary.train.training_utils import print_rank0

LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))


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
    use_lazy_loading: bool = field(
        default=False,
        metadata={"help": "Whether to use lazy loading for the dataset or not"},
    )
    ignore_cached: bool = field(
        default=False,
        metadata={"help": "Whether to ignore cached tokenized data or not"},
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    model_max_length: int = field(
        default=4096,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    report_to: str = field(
        default="wandb", metadata={"help": "Report logging to wandb"}
    )

    keep_assistant_prefix: bool = field(
        default=True,
        metadata={
            "help": "Whether to mask the assistant prefix `<|from|>assistant\n<|recipient|>` during training"
        },
    )

    prompt_template_version: str = field(
        default="v2", metadata={"help": "choose prompt template to use for training"}
    )

    use_liger: bool = field(
        default=False,
        metadata={
            "help": "Whether use liger or not. Refer to this link for more details: https://github.com/triton-lang/triton?tab=readme-ov-file#compatibility"
        },
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
    print_rank0(
        f"loara_param: {lora_param_count} = {lora_param_count * 100 / all_param} %"
    )


def get_device_map(
    training_args: TrainingArguments, lora_args: LoraArguments
) -> Optional[Dict]:
    device_map = None
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if lora_args.q_lora:
        if ddp and training_args.fsdp:
            print("FSDP is incompatible with QLORA")
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else None
        if len(training_args.fsdp) > 0 or is_deepspeed_zero3_enabled():
            print("FSDP and ZeRO3 are both currently incompatible with QLoRA.")
    return device_map


def load_model_with_rope_scaling(
    model_args: ModelArguments,
    training_args: TrainingArguments,
    lora_args: LoraArguments,
    data_args: DataArguments,
) -> transformers.AutoModelForCausalLM:
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
        print(
            f"have to use rope-scaling, original max_leng={orig_ctx_len}, scaled to: {training_args.model_max_length}",
        )
        scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    config.use_cache = False

    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    if data_args.packing:
        from functionary.train.packing.monkey_patch_packing import (
            monkey_patch_packing_for_model,
        )

        monkey_patch_packing_for_model(model_args.model_name_or_path)

    if training_args.use_liger:
        from liger_kernel.transformers import AutoLigerKernelForCausalLM

        print_rank0("---------------using LIGER------------")
        model_class = AutoLigerKernelForCausalLM
    else:
        model_class = transformers.AutoModelForCausalLM

    model = model_class.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        device_map=get_device_map(training_args, lora_args),
        attn_implementation="flash_attention_2",  # use_flash_attention_2 is replaced by this from version: 4.36.0
        torch_dtype=compute_dtype,
        quantization_config=(
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                attn_implementation="flash_attention_2",
                bnb_4bit_compute_dtype=compute_dtype,
            )
            if lora_args.q_lora
            else None
        ),
    )
    return model


def prepare_model_for_training(
    model: transformers.AutoModelForCausalLM,
    training_args: TrainingArguments,
    lora_args: LoraArguments,
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
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=training_args.gradient_checkpointing
        )

    model = get_peft_model(model, lora_config)

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    model.config.use_cache = False
    # Activate computing load balancing loss iin MixtralForCausalLM
    if hasattr(model.config, "output_router_logits"):
        setattr(model.config, "output_router_logits", True)
        print_rank0("Activate computing load balancing loss")

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


"""
Below is the updated train() function from LEVENT OZBEK.
Most of it are just adaptions from changes in training_utils.py
I commented out the original train() function

- training_utils.tokenize_and_cache() to preprocess and cache tokenized datasets.
- directly using of DataLoader with training_utils.create_data_loader() / training_utils.create_distributed_data_loader() depending on the distributed training setup.
- integrated the updated preprocess_logits_for_metrics() and compute_metrics functions
- automatically switches between standard and distributed data loaders based on training_args.local_rank.
"""


def train():
    argument_parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = argument_parser.parse_args_into_dataclasses()

    print_rank0("lora args: ", lora_args)
    print_rank0("training args: ", training_args)

    # loading the model with RoPE scaling if required
    model = load_model_with_rope_scaling(
        model_args, training_args, lora_args, data_args
    )
    print_rank0(model)

    # loading and initializing the tokenizer
    prompt_template = get_prompt_template_by_version(
        training_args.prompt_template_version
    )
    tokenizer = training_utils.initialize_tokenizer(
        model=model,
        model_name_or_path=model_args.model_name_or_path,
        prompt_template=prompt_template,
        model_max_length=training_args.model_max_length,
        cache_dir=training_args.cache_dir,
    )

    id2token = {
        tokenizer.encode(token)[-1]: token
        for token in prompt_template.get_additional_tokens()
    }
    print_rank0("id to tokens: ", id2token)

    assert data_args.train_data_path is not None, "Please provide a training data file."

    # caching tokenized training data
    raw_train_dataset = read_dataset(
        model_args.model_name_or_path, data_args, training_args, tokenizer, "train"
    )
    train_dataset = training_utils.tokenize_and_cache(
        raw_train_dataset, tokenizer, training_args.cache_dir
    )
    print_rank0("****** Examples from train_dataset *****")
    training_utils.print_some_examples(train_dataset, tokenizer)
    print_rank0("final train size: ", len(train_dataset))

    if training_args.do_eval:
        # Caching already tokenized eval data
        raw_eval_dataset = read_dataset(
            model_args.model_name_or_path, data_args, training_args, tokenizer, "eval"
        )
        eval_dataset = training_utils.tokenize_and_cache(
            raw_eval_dataset, tokenizer, training_args.cache_dir
        )
        print_rank0("****** Examples from eval_dataset *****")
        training_utils.print_some_examples(eval_dataset, tokenizer)
        print_rank0("final eval size: ", len(eval_dataset))

    print_rank0("tokenizer.model_max_length: ", tokenizer.model_max_length)

    # prepairing model for training
    model = prepare_model_for_training(model, training_args, lora_args)

    # preprocessing logits and computing metrics using updated functions
    def preprocess_logits_for_metrics(logits, labels):
        return training_utils.preprocess_logits_for_metrics(
            logits, labels, len(tokenizer)
        )

    def compute_metrics(eval_preds):
        return training_utils.compute_metrics(eval_preds, id2token, tokenizer)

    # using the optimized data loaders in case of single or multi gpu setups
    train_loader = (
        training_utils.create_distributed_data_loader(
            train_dataset, batch_size=training_args.per_device_train_batch_size
        )
        if training_args.local_rank != -1
        else training_utils.create_data_loader(
            train_dataset, batch_size=training_args.per_device_train_batch_size
        )
    )

    if training_args.do_eval:
        eval_loader = (
            training_utils.create_distributed_data_loader(
                eval_dataset, batch_size=training_args.per_device_eval_batch_size
            )
            if training_args.local_rank != -1
            else training_utils.create_data_loader(
                eval_dataset, batch_size=training_args.per_device_eval_batch_size
            )
        )

    # initializing the trainer with the updated data loaders and metrics
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_loader.dataset if not training_args.do_eval else None,
        eval_dataset=eval_loader.dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics if training_args.do_eval else None,
        preprocess_logits_for_metrics=(
            preprocess_logits_for_metrics if training_args.do_eval else None
        ),
    )

    # resuming training if a checkpoint exists
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()

    # handling saving model and tokenizer in a distributed setting
    if training_args.local_rank == 0 or training_args.local_rank == -1:
        model.save_pretrained(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    train()
