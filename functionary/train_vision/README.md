# Install Dependencies
```shell
pip install accelerate==0.27.2 bitsandbytes==0.41.1 scipy==1.11.3 sentencepiece==0.1.99 packaging==23.1 ninja==1.11.1 einops==0.7.0 wandb==0.15.11 jsonref==1.1.0 deepspeed==0.14.2 typer==0.9.0 tensorboard==2.15.1 wheel==0.42.0 aenum==3.1.15 transformers==4.42.3 flash-attn==v2.5.9.post1 git+https://github.com/LLaVA-VL/LLaVA-NeXT.git timm==1.0.8
```

Example script:

```shell
deepspeed functionary/train_vision/train.py \
    --model_name_or_path lmms-lab/llama3-llava-next-8b \
    --train_data_path TRAIN_PATH \
    --eval_data_path VALIDATION_PATH \
    --bf16 True \
    --output_dir functionary-with-metrics \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2\
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --eval_accumulation_steps 1 \
    --evaluation_strategy "steps" \
    --eval_steps 100 \
    --save_strategy "no" \
    --save_steps 100 \
    --save_total_limit 3 \
    --logging_steps 1 \
    --learning_rate 1e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --tf32 True \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --packing False\
    --optim "paged_adamw_32bit" \
    --deepspeed functionary/train/ds_config/zero3_wo_offload.json \
    --pad_img_path pad_img.png \
    --log_train_metrics True

```