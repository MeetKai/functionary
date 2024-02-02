## Full Finetuning
```shell
# Create new virtual environment
python3 -m venv venv && source venv/bin/activate

# Install Torch 2.0.1
pip3 install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# Install Dependencies
pip install accelerate==0.23.0 transformers==4.37.2 bitsandbytes==0.41.1 scipy==1.11.3 sentencepiece==0.1.99 packaging==23.1 ninja==1.11.1 einops==0.7.0 wandb==0.15.11 jsonref==1.1.0 deepspeed==0.11.1 typer==0.9.0 tensorboard==2.15.1 wheel==0.42.0

# Install Flash Attention 2
pip install flash-attn==2.3.3 --no-build-isolation
```

### Llama-2 models

<details>
    We have produced full-parameter finetuning scripts compatible with FSDP and DDP respectively.

```shell
# FSDP with accelerate launcher
# 2xA100 80GB, from the root directory of the repository
export WANDB_ENTITY=NAME_OF_ENTITY
export WANDB_PROJECT=NAME_OF_PROJECT
accelerate launch --config_file "functionary/train/accelerate_configs/fsdp_config.yaml" -m functionary.train.train \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --train_data_path train_dataset.jsonl \
    --eval_data_path eval_dataset.jsonl \
    --bf16 True \
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 64 \
    --per_device_eval_batch_size 4 \
    --eval_accumulation_steps 16 \
    --evaluation_strategy "steps" \
    --eval_steps 200 \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 5 \
    --learning_rate 2e-5 \
    --weight_decay 0.3 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 4096 \
    --optim "paged_adamw_32bit" \
    --gradient_checkpointing True \
    --output_dir functionary-v1


# DDP
# 2xA100 80GB, from the root directory of the repository
export WANDB_ENTITY=NAME_OF_ENTITY
export WANDB_PROJECT=NAME_OF_PROJECT
torchrun --nproc_per_node=2 --master_port=20001 -m functionary.train.train \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --train_data_path train_dataset.jsonl \
    --eval_data_path eval_dataset.jsonl \
    --bf16 True \
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 64 \
    --per_device_eval_batch_size 4 \
    --eval_accumulation_steps 12 \
    --evaluation_strategy "steps" \
    --eval_steps 400 \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 5 \
    --learning_rate 2e-5 \
    --weight_decay 0.3 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True \
    --model_max_length 4096 \
    --report_to all \
    --gradient_checkpointing True \
    --output_dir functionary-v1
```

*If the training is stuck before training, run the following command before starting the training*:

```shell
export NCCL_P2P_DISABLE=1
```
</details>

### Mistral models

DeepSpeed is the recommended multi-GPU option for Mistral finetuning currently because FSDP may experience [loss instability](https://github.com/huggingface/transformers/issues/26498). Therefore, we have produced training scripts for Mistral based on DeepSpeed ZeRO Stage 3.

```shell
# DeepSpeed ZeRO3 with accelerate launcher
# 4xA100 80GB, from the root directory of the repository
export WANDB_ENTITY=NAME_OF_ENTITY
export WANDB_PROJECT=functionary
accelerate launch --config_file "functionary/train/accelerate_configs/ds3_config.yaml" -m functionary.train.train \
    --model_name_or_path mistralai/Mistral-7B-v0.1 \
    --train_data_path 2023-12-20_train.jsonl \
    --eval_data_path 2023-12-20_val.jsonl \
    --bf16 True \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --per_device_eval_batch_size 8 \
    --eval_accumulation_steps 1 \
    --evaluation_strategy "steps" \
    --eval_steps 75 \
    --save_strategy "no" \
    --save_steps 200 \
    --save_total_limit 5 \
    --learning_rate 9e-6 \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1.0 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 8192 \
    --optim "paged_adamw_32bit" \
    --gradient_checkpointing True \
    --output_dir functionary-v2.1
```

### Compute requirements

| Model    | Minimum Number of GPUs (A100-80GB) |
| :--------: | :-------: |
| Llama-2-7b  | 2 |
| Llama-2-13b | 4 |

### Logging model to WandB

As WandB is [not compatible](https://github.com/huggingface/accelerate/issues/1845) with DeepSpeed, we have provided a script that uploads the model as an artifact to a specified WandB training run. This script should be run after the training is completed.

**Cache management**

If you are running the training on container-based GPU instances which do not provide large enough volumes of disk space in the root directory (like RunPod), you should set the following WandB environment variables to a disk with large enough disk space. Else, you will encounter disk out of space error when trying to upload the artifacts.

```shell
export WANDB_DATA_DIR=/workspace/wandb_data
```

```shell
# From the root directory of the repository
python3 -m functionary.train.log_final_model_to_wandb [WANDB_ENTITY] [WANDB_PROJECT] [WANDB_RUN_ID] [MODEL_DIR]
```

Arguments:

- [WANDB_ENTITY] is the entity used in wandb for the training run.
- [WANDB_PROJECT] is the name of the project used to host the training run.
- [WANDB_RUN_ID] can be found in the `Overview` page of the training run. Find `Run path` and it is in the form `{entity}/{project}/{run_id}` (E.g.: meetkai/functionary/phwsc2oo).
- [MODEL_DIR] is the path of the directory where the final model weights are stored at.

## Lora Finetuning
### Finetuning
For Lora fintuning, you need to install additional requirements:

```
peft==0.5.0
datasets==2.8.0
```
Run script:

```shell
export WANDB_PROJECT=NAME_OF_PROJECT
export WANDB_LOG_MODEL=all
deepspeed functionary/train/train_lora.py \
    --model_name_or_path PRETRAINED_PATH \
    --train_data_path train_dataset.jsonl \
    --eval_data_path train_dataset.jsonl \
    --q_lora True \
    --bf16 True \
    --output_dir OUTPUT_FOLDER \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --eval_accumulation_steps 1 \
    --evaluation_strategy "steps" \
    --eval_steps 5 \
    --save_strategy "steps" \
    --save_steps 5 \
    --save_total_limit 3 \
    --logging_steps 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --report_to wandb \
    --packing \
    --deepspeed ds_config/zero2.json
```

Using **--q_lora True** to use q_lora instead of *lora*
Using **--packing** to speed up training by packing short data points, currently only works for Llama.

### Merging Lora weights
After finish training, you can merge the Lora weights with the pretrained weights by the following commmand:
```shell
python functionary/train/merge_lora_weight.py save_folder pretrained_path checkpoint
```
