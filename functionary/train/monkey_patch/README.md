# Packing Inputs Without Cross-Contamination Attention

To speed up training, we have implemented packing without **cross-contamination attention** by monkey-patching ``MistralForCausalLM`` and ``LlamaForCausalLM``. The idea of packing is to merge/pack short inputs into a single input so the number of training data points would be reduced.

Concretely, when we pack inputs, the attention should be only within individual sequences. For example, assume that we are packing 2 inputs: packed input = [input 1] [input 2]. Tokens from **input 1** only attend to tokens from **input 1** and tokens from **input 2** only attend to tokens from **input 2**

<p align="center">
  <img src="assets/cross_contamination.png", width="300", height="300">
  <img src="assets/correct_packing_attention.png", width="300", height="300">
</p>
<p align="center">
Examples of packing 2 input sequences: "good morning my name is John" and "This is a dog". The left is the attention matrix of packing with cross-contamination, the right is the correct attention matrix of packing</p>

## Reducing Training Time
The obvious benefit of packing is reducing the training time. This reduction depends on the **pack_length** and the **lengths** of original data points. 

For example, in the training of our model (functionary), we found that the training time was **reduced from 15 hours to 5 hours, almost 1/3** of that without packing. For short data like [tatsu-lab/alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) if we choose pack_length=4096, from original 100 data points, we can packed into only **4** data points. So the training time is only about 4% of original training data.

Some notes:
+ We ususally set pack_length = max_length but this is not a must
+ When using packing, especially for short dataset, the number of data points are drastically reduced, so the number of training steps would be also dratically reduced, if you set high value for batch_size or gradient_accumulation_steps the learning rate (assumed to be be decreased in accordance with trained steps) might drop much faster and might degrade the trained model.

## How to use

**Note that our implementation is only correct if using [Flash Attenion](https://github.com/Dao-AILab/flash-attention)**

We recommend using ``transformers==4.35.0``

So the additional requirement is only Flash Attention:

```
pip install flash-attn --no-build-isolation
```

To use, you just need to replace the ``MistralForCausalLM`` or ``LlamaForCausalLM`` with our monkey-patched versions:
```python 
#from transformers import LlamaForCausalLM
from llama_monkey_patch import LlamaForCausalLM

#from transformers import MistralForCausalLM
from mistral_monkey_patch import MistralForCausalLM
model = MistralForCausalLM(model_path, ...,use_flash_attention_2=True)
```

### Convert to Packed Dataset
The input format to monkey-patched models:

+ input_ids = [input_ids of sequence 1] + [input_ids of sequence 2] + ...  + [input_ids of sequence n] + [padding tokens]

+ attention_mask = [1, ..., 1, 2, ..., 2, ... n, ..., n, 0, 0, ..0]
where: 
  + number of 1s = len(input_ids of sequence 1)
  + ...
  + number of n = len(input_ids of sequence n)
  + number of 0s = len(padding tokens)

+ labels = [labels of sequence 1] + [labels of sequence 2] + ...  + [labels of sequence n] + [padding tokens]

Actually, we had already implemented converting Original Dataset (function ``__item__ return {"input_ids": xxx, "attention_mask": xxx, "labels": xxxx})`` to Packed Dataset, you can use with just one line of code:
```python
from packed_dataset import PackedDataset
packed_ds = PackedDataset(original_ds, tokenizer, pack_length)
```

## Assert Implementation
To make sure that the implementation is correct, we implemented a script for:
+ Ensuring the average loss over tokens from **(original dataset, original model)** == **(packed dataset, monkey-patched model)**
+ Ensuring the number of tokens used for **computing loss** from **original dataset** == The number of tokens used for **computing loss** from **packed dataset**

The script will randomly select 100 items from: "tatsu-lab/alpaca" for comparing the loss from **(original dataset, original model)** and **(packed dataset, monkey-patched model)**

To run the script you need to install:
```shell
# Install Dependencies
pip install accelerate==0.23.0 transformers==4.35.0 bitsandbytes==0.41.1 scipy==1.11.3 sentencepiece==0.1.99 packaging==23.1 ninja==1.11.1 einops==0.7.0 wandb==0.15.11 jsonref==1.1.0 deepspeed==0.11.1 typer==0.9.0

pip install flash-attn==2.3.2 --no-build-isolation
```

You can run the script to verify that the implementation of monkey-patch is correct:

```shell
python assert_monkey_patch.py mistralai/Mistral-7B-v0.1 mistral
```

The output would show that the difference between loss from **(original dataset, original model)** and **(packed dataset, monkey-patched model)** is almost 0%.