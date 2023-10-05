export WANDB_ENTITY=jeffreymeetkai
export WANDB_PROJECT=functionary-llama2-13b

torchrun --nproc_per_node=2 --master_port=20001 -m functionary.train.train \
    --model_name_or_path /workspace/Llama-2-13B-fp16  \
    --data_path llama_training_dataset.jsonl \
    --train_valid_split 0.9 \
    --bf16 True \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 12 \
    --per_device_eval_batch_size 4 \
    --eval_accumulation_steps 16 \
    --evaluation_strategy "steps" \
    --eval_steps 30 \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 5 \
    --learning_rate 2e-5 \
    --weight_decay 0.3 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 5 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --output_dir functionary-13b