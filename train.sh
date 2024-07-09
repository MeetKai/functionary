# DeepSpeed ZeRO3 with accelerate launcher
# 4xA100 80GB, from the root directory of the repository
export WANDB_ENTITY=jeffreymeetkai
export WANDB_PROJECT=functionary-llama-3-yarn
accelerate launch --config_file "functionary/train/accelerate_configs/ds3_config.yaml" --num_processes 4 -m functionary.train.train \
    --model_name_or_path /workspace/Meta-Llama-3-8B-Instruct \
    --train_data_path 2024-06-12_train_extended.jsonl \
    --eval_data_path 2024-06-12_val_extended.jsonl \
    --bf16 True \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --per_device_eval_batch_size 1 \
    --eval_accumulation_steps 8 \
    --eval_strategy "steps" \
    --eval_steps 65 \
    --save_strategy "no" \
    --save_steps 200 \
    --save_total_limit 5 \
    --learning_rate 1e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1.0 \
    --lr_scheduler_type "customized_scheduler" \
    --logging_steps 1 \
    --model_max_length 32768 \
    --optim "paged_adamw_8bit" \
    --gradient_checkpointing True \
    --prompt_template_version v3.llama3 \
    --output_dir functionary-small-v3.0-32K