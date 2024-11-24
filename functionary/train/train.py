import math, os, pathlib, sys, torch, transformers, torch.distributed

from dataclasses import dataclass, field
from typing import Optional
from aenum import extend_enum
from torch.optim.lr_scheduler import LambdaLR
from transformers import Trainer

from functionary.prompt_template import get_prompt_template_by_version
from functionary.train.custom_datasets import read_dataset
from functionary.train import training_utils
from training_utils import print_rank0

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


sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

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


"""
Below is the updated train() function from LEVENT OZBEK.
Most of the changes are identical to those in train_lora.py. I simply applied the changes to the utility code in training_utils.py
I commented out the original train() function

- training_utils.tokenize_and_cache() is used for both training and evaluation datasets to avoid repetition.
- dynamic_batch_size() function auto adjusts batch sizes based on token counts. I did not implement this in train_lora.py since loras are trained on a smaller data so I felt that it wasn't too necessary there.
- DataLoaders are constructed using BatchSampler to dynamically adjust the batch size per epoch.
- distributed DataLoader is used if local_rank != -1.
- updated to use the optimized preprocess_logits_for_metrics dynamically compute_metrics from training_utils.py.

Advantages of These Changes:
- handles datasets with varying sequence lengths dynamically
- supports both single-GPU and distributed setups.
"""


def train():
    """Training loop"""

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
        from functionary.train.packing.monkey_patch_packing import (
            monkey_patch_packing_for_model,
        )

        monkey_patch_packing_for_model(model_args.model_name_or_path)

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

    model = model_class.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=compute_dtype,
        config=config,
        cache_dir=training_args.cache_dir,
        attn_implementation="flash_attention_2",
    )
    model.config.use_cache = False
    if hasattr(model.config, "output_router_logits"):
        setattr(model.config, "output_router_logits", True)
        print_rank0("Activate computing load balancing loss")

    print_rank0("Prompt template to use: ", training_args.prompt_template_version)
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

    if LOCAL_RANK == 0:
        if not os.path.exists(training_args.output_dir):
            os.mkdir(training_args.output_dir)

        tokenizer.save_pretrained(training_args.output_dir)

    id2token = {
        tokenizer.encode(token)[-1]: token
        for token in prompt_template.get_additional_tokens()
    }
    print_rank0("id to tokens: ", id2token)

    assert data_args.train_data_path is not None, "Please provide a training data file."

    # Cache and tokenize training data
    raw_train_dataset = read_dataset(
        model_args.model_name_or_path, data_args, training_args, tokenizer, "train"
    )
    train_dataset = training_utils.tokenize_and_cache(
        raw_train_dataset, tokenizer, training_args.cache_dir
    )

    if torch.distributed.get_rank() == 0:
        print(f"Training Data Loaded: #{len(train_dataset)}")

    if training_args.do_eval:
        # Cache and tokenize evaluation data
        raw_eval_dataset = read_dataset(
            model_args.model_name_or_path,
            data_args,
            training_args,
            tokenizer,
            "validation",
        )
        eval_dataset = training_utils.tokenize_and_cache(
            raw_eval_dataset, tokenizer, training_args.cache_dir
        )
        if torch.distributed.get_rank() == 0:
            print(f"Eval Data Loaded: #{len(eval_dataset)}")

    print_rank0("***** HERE ARE SOME EXAMPLES FROM TRAINING ****")
    training_utils.print_some_examples(train_dataset, tokenizer)

    if training_args.do_eval:
        print_rank0("***** HERE ARE SOME EXAMPLES FROM EVALUATION ***")
        training_utils.print_some_examples(eval_dataset, tokenizer)

    # Dynamic batch size based on max tokens per batch
    max_tokens_per_batch = 2048  # You can adjust this as needed
    train_batch_sizes = training_utils.dynamic_batch_size(
        train_dataset, max_tokens_per_batch, tokenizer
    )
    print_rank0(f"Dynamic train batch sizes: {train_batch_sizes}")

    if training_args.do_eval:
        eval_batch_sizes = training_utils.dynamic_batch_size(
            eval_dataset, max_tokens_per_batch, tokenizer
        )
        print_rank0(f"Dynamic eval batch sizes: {eval_batch_sizes}")

    # DataLoaders with dynamic batch sizes
    train_loader = (
        DataLoader(
            train_dataset,
            batch_sampler=torch.utils.data.BatchSampler(
                sampler=torch.utils.data.SequentialSampler(train_dataset),
                batch_size=max(train_batch_sizes),  # Adjust batch size dynamically
                drop_last=False,
            ),
            num_workers=4,
            pin_memory=True,
        )
        if training_args.local_rank == -1
        else training_utils.create_distributed_data_loader(
            train_dataset, batch_size=max(train_batch_sizes)
        )
    )

    if training_args.do_eval:
        eval_loader = (
            DataLoader(
                eval_dataset,
                batch_sampler=torch.utils.data.BatchSampler(
                    sampler=torch.utils.data.SequentialSampler(eval_dataset),
                    batch_size=max(eval_batch_sizes),  # Adjust batch size dynamically
                    drop_last=False,
                ),
                num_workers=4,
                pin_memory=True,
            )
            if training_args.local_rank == -1
            else training_utils.create_distributed_data_loader(
                eval_dataset, batch_size=max(eval_batch_sizes)
            )
        )

    def preprocess_logits_for_metrics(logits, labels):
        return training_utils.preprocess_logits_for_metrics(
            logits, labels, len(tokenizer)
        )

    def compute_metrics(eval_preds):
        return training_utils.compute_metrics(eval_preds, id2token, tokenizer)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_loader.dataset,
        eval_dataset=eval_loader.dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics if training_args.do_eval else None,
        preprocess_logits_for_metrics=(
            preprocess_logits_for_metrics if training_args.do_eval else None
        ),
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()

    if trainer.is_fsdp_enabled:
        trainer_save_model_safe(trainer=trainer)
    else:
        trainer.save_model()


# def train():
#     """Training loop"""

#     argument_parser = transformers.HfArgumentParser(
#         (ModelArguments, DataArguments, TrainingArguments)
#     )
#     model_args, data_args, training_args = argument_parser.parse_args_into_dataclasses()

#     # Set RoPE scaling factor
#     config = transformers.AutoConfig.from_pretrained(
#         model_args.model_name_or_path,
#         cache_dir=training_args.cache_dir,
#     )
#     orig_ctx_len = getattr(config, "max_position_embeddings", None)
#     if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
#         scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
#         config.rope_scaling = {"type": "linear", "factor": scaling_factor}
#         print_rank0("Rope scaling enabled")
#     config.use_cache = False
#     config.sliding_window = training_args.model_max_length

#     if data_args.packing:
#         from functionary.train.packing.monkey_patch_packing import (
#             monkey_patch_packing_for_model,
#         )

#         monkey_patch_packing_for_model(model_args.model_name_or_path)

#     compute_dtype = (
#         torch.float16
#         if training_args.fp16
#         else (torch.bfloat16 if training_args.bf16 else torch.float32)
#     )

#     if training_args.use_liger:
#         from liger_kernel.transformers import AutoLigerKernelForCausalLM

#         print_rank0("---------------using LIGER------------")
#         model_class = AutoLigerKernelForCausalLM
#     else:
#         model_class = transformers.AutoModelForCausalLM

#     model = model_class.from_pretrained(
#         model_args.model_name_or_path,
#         torch_dtype=compute_dtype,
#         config=config,
#         cache_dir=training_args.cache_dir,
#         attn_implementation="flash_attention_2",  # use_flash_attention_2 is replaced by this from version: 4.36.0
#     )
#     model.config.use_cache = False
#     # Activate computing load balancing loss iin MixtralForCausalLM
#     if hasattr(model.config, "output_router_logits"):
#         setattr(model.config, "output_router_logits", True)
#         print_rank0("Activate computing load balancing loss")

#     print_rank0("Prompt template to use: ", training_args.prompt_template_version)
#     prompt_template = get_prompt_template_by_version(
#         training_args.prompt_template_version
#     )

#     tokenizer = training_utils.initialize_tokenizer(
#         model=model,
#         model_name_or_path=model_args.model_name_or_path,
#         prompt_template=prompt_template,
#         model_max_length=training_args.model_max_length,
#         cache_dir=training_args.cache_dir,
#     )

#     if LOCAL_RANK == 0:
#         if not os.path.exists(training_args.output_dir):
#             os.mkdir(training_args.output_dir)

#         tokenizer.save_pretrained(training_args.output_dir)

#     # get id of added tokens to compute the accuracy of predicing the token
#     id2token = {
#         tokenizer.encode(token)[-1]: token
#         for token in prompt_template.get_additional_tokens()
#     }
#     print_rank0("id to tokens: ", id2token)

#     assert data_args.train_data_path is not None, "Please provide a training data file."

#     train_dataset = read_dataset(
#         model_args.model_name_or_path, data_args, training_args, tokenizer, "train"
#     )

#     if torch.distributed.get_rank() == 0:
#         print(f"Training Data Loaded: #{len(train_dataset)}")

#     if training_args.do_eval:
#         eval_dataset = read_dataset(
#             model_args.model_name_or_path,
#             data_args,
#             training_args,
#             tokenizer,
#             "validation",
#         )

#         if torch.distributed.get_rank() == 0:
#             print(f"Eval Data Loaded: #{len(eval_dataset)}")

#     print_rank0("***** HERE ARE SOME EXAMPLES FROM TRAINING ****")
#     training_utils.print_some_examples(train_dataset, tokenizer)

#     if training_args.do_eval:
#         print_rank0("***** HERE ARE SOME EXAMPLES FROM EVALUATION ***")
#         training_utils.print_some_examples(eval_dataset, tokenizer)

#     def preprocess_logits_for_metrics(logits, labels):
#         return training_utils.preprocess_logits_for_metrics(
#             logits, labels, len(tokenizer)
#         )

#     def compute_metrics(eval_preds):
#         return training_utils.compute_metrics(eval_preds, id2token, tokenizer)

#     if training_args.do_eval:
#         trainer = Trainer(
#             model=model,
#             tokenizer=tokenizer,
#             args=training_args,
#             train_dataset=train_dataset,
#             eval_dataset=eval_dataset,
#             compute_metrics=compute_metrics,
#             preprocess_logits_for_metrics=preprocess_logits_for_metrics,
#         )
#     else:
#         trainer = Trainer(
#             model=model,
#             tokenizer=tokenizer,
#             args=training_args,
#             train_dataset=train_dataset,
#         )

#     if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
#         trainer.train(resume_from_checkpoint=True)
#     else:
#         trainer.train()

#     trainer.save_state()

#     # FSDP requires state_dict_type=FULL_STATE_DICT in order to save the model weights in .bin format
#     if trainer.is_fsdp_enabled:
#         trainer_save_model_safe(trainer=trainer)
#     else:
#         trainer.save_model()


if __name__ == "__main__":
    train()
