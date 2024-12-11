export WANDB_ENTITY=jeffreymeetkai
export WANDB_PROJECT=functionary-lora
deepspeed functionary/train/train_lora.py \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --bf16 True \
    --output_dir functionary-small-test-lora-2024-11-04 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --eval_accumulation_steps 8 \
    --max_steps 2 \
    --evaluation_strategy "no" \
    --eval_steps 5 \
    --save_strategy "no" \
    --save_steps 5 \
    --save_total_limit 3 \
    --logging_steps 1 \
    --learning_rate 8e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --tf32 True \
    --model_max_length 2048 \
    --use_liger True \
    --gradient_checkpointing True \
    --prompt_template_version v3.llama3 \
    --deepspeed functionary/train/ds_config/zero2.json