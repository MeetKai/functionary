```shell
# Create new virtual environment
python3 -m venv venv && source venv/bin/activate

# Install Torch 2.0.1
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install Dependencies, (Latest main version of huggingface is critical as its giving OOM without it, We need to use this until 4.32 is out)
pip install accelerate==0.21.0 git+https://github.com/huggingface/transformers sentencepiece packaging ninja einops wandb

# Install Flash Attention 2
git clone https://github.com/Dao-AILab/flash-attention && cd flash-attention && python setup.py install && cd ..

# 2xA100 80GB
torchrun --nproc_per_node=2 --master_port=20001 train.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf  \
    --data_path llama_training_dataset.jsonl \
    --bf16 True \
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
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
    --gradient_checkpointing True \
    --output_dir functionary-v1
```

*Note: This training process might not work exactly as is. We've changed a couple of things in the training code and not tested yet.*  