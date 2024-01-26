# Packing Inputs Without Cross-Contamination Attention

To speed up training, we have implemented packing without **cross-contamination attention** by monkey-patching ``MistralForCausalLM``, ``LlamaForCausalLM`` and ``MixtralForCausalLM``. The idea of packing is to merge/pack short inputs into a single input so the number of training data points would be reduced.

Concretely, when we pack inputs, the attention should be only within individual sequences. For example, assume that we are packing 2 inputs: packed input = [input 1] [input 2]. Tokens from **input 1** only attend to tokens from **input 1** and tokens from **input 2** only attend to tokens from **input 2**

<p align="center">
  <img src="assets/cross_contamination.png", width="300", height="300">
  <img src="assets/correct_packing_attention.png", width="300", height="300">
</p>
<p align="center">
Examples of packing 2 input sequences: "good morning my name is John" and "This is a dog". The left is the attention matrix of packing with cross-contamination, the right is the correct attention matrix of packing</p>

## Reduce The Training Time
The obvious benefit of packing is reducing the training time. This reduction depends on the **pack_length** and the **lengths** of original data points. 

For example, in the training of our model (functionary), we found that the training time was **reduced from 15 hours to 5 hours, almost 1/3** of that without packing. For short data like [tatsu-lab/alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) if we choose pack_length=4096, from original **52k data points**, we can packed into only **1567** data points. So the training time is only about **3%** of original training data.

Some notes:
+ pack_length >= max_length (used to tokenize data). If pack_length is big, the number of data points after packing will be small.
+ Pay attention to the **number of data points after packing**. For example, your original datset contains 5000 data points, packed into: 96 data points. If you set batch_size=32 in the training --> model's weights are updated only **3 times** --> **would be poor**. So When you use packing, if you should take a look at the number of data points after packing to tune the hyperparameters for training such as: pack_length, gradient_accumulation_steps, ... To make sure that the number of training steps would be big enough

## How To use

To use Packing in the training we just need to:
+ Convert original dataset to packed datasets
+ Use monkey-patched implementation of: ``MistralForCausalLM``, ``LlamaForCausalLM`` or ``MixtralForCausalLM``

### Convert to Packed Dataset
The format of packed input (assume that packing n inputs into one)
+ input_ids = 
  + [input_ids of sequence 1] + [input_ids of sequence 2] + ...  + [input_ids of sequence n] + [padding tokens] **if padding_side=right**
  + [padding tokens] + [input_ids of sequence 1] + [input_ids of sequence 2] + ...  + [input_ids of sequence n] **if padding_side=left**

+ attention_mask = 
  + [1, ..., 1, 2, ..., 2, ... n, ..., n, 0, 0, ..0] if **padding_side=right** 
  + [0, 0, ..0, 1, ..., 1, 2, ..., 2, ... n, ..., n] if **padding_side=left**
  
  where: 
  + number of 1s = len(input_ids of sequence 1)
  + ...
  + number of n = len(input_ids of sequence n)
  + number of 0s = len(padding tokens)

+ labels = 
  + [labels of sequence 1] + [labels of sequence 2] + ...  + [labels of sequence n] + [-100, ... -100] **if padding_side = right** where -100: means masking **pad_token** (excluding from computing loss)
  + [-100, ... -100] + [labels of sequence 1] + [labels of sequence 2] + ...  + [labels of sequence n] **if padding_side = left** 


Actually, we had already implemented converting Original Dataset (the function ``__item__ return {"input_ids": xxx, "attention_mask": xxx, "labels": xxxx})``) to Packed Dataset, you can use this with just one line of code:
```python
from packed_dataset import PackedDataset
# Note that pack_length must be >= max_leng used tokenizing the dataset
# original_ds[index] --> {"input_ids": xxx, "attention_mask": xxx, "labels": xxx}, labels is not necessarily required
packed_ds = PackedDataset(original_ds, tokenizer, pack_length)
```
### Use monkey-patched implementation

**Note that our implementation is only correct if using [Flash Attenion](https://github.com/Dao-AILab/flash-attention)**

We recommend using ``transformers==4.36.2`` for **finetuning LLama or Mistral**. **For Mixtral**, should use the latest implementation: ``pip install git+https://github.com/huggingface/transformers.git``

So the additional requirement is only **Flash Attention**:

```
pip install flash-attn --no-build-isolation
```

Based on your model: ``MistralForCausalLM``,  ``LlamaForCausalLM`` or ``MixtralForCausalLM`` that you will call the function for monkey-patching from: ``monkey_patch_packing.py`` accordingly

**For LlamaForCausalLM**
```python 
from monkey_patch_packing import monkey_patch_packing_llama
monkey_patch_packing_llama() # Monkey-patch LlamaForCausalLM
...
# Load the model
model = transformers.AutoModelForCausalLM.from_pretrained(model_path, ...)
model.config.use_cache = False # In the training, we don't need to use cache, note: must add this or can encounter assertion error
```
**For MistralForCausalLM**
```python
from monkey_patch_packing import monkey_patch_packing_mistral
monkey_patch_packing_mistral()
...
# Load the model
model = transformers.AutoModelForCausalLM.from_pretrained(model_path, ...)
model.config.use_cache = False # In the training, we don't need to use cache, note: must add this
```

**For MixtralForCausalLM**
```python
from monkey_patch_packing import monkey_patch_packing_mixtral
monkey_patch_packing_mixtral()
...
# Load the model
model = transformers.AutoModelForCausalLM.from_pretrained(model_path, ...)
model.config.use_cache = False # In the training, we don't need to use cache, note: must add this
```

The implementation is based on the idea of overwriting the function: ``_get_unpad_data`` of ``MistralForCausalLM``, ``LlamaForCausalLM`` and ``MixtralForCausalLM`` with a monkey-patched function that can handle ``attentions_mask`` of packed inputs. You can take a look at the file: **monkey_patch_packing.py**

## Assert Implementation
To make sure that the implementation is correct, we implemented a script for:
+ Ensuring the average loss over tokens from **(original dataset, original model)** == **(packed dataset, monkey-patched model)**
+ Ensuring the number of tokens used for **computing loss** from **original dataset** == The number of tokens used for **computing loss** from **packed dataset**

The script will randomly select 50 items from: "tatsu-lab/alpaca" for comparing the loss from **(original dataset, original model)** and **(packed dataset, monkey-patched model)**

To run the script you need to install:
```shell
# Install Dependencies
pip install git+https://github.com/huggingface/transformers.git
pip install accelerate==0.23.0 bitsandbytes==0.41.1 scipy==1.11.3 sentencepiece==0.1.99 packaging==23.1 ninja==1.11.1 einops==0.7.0 wandb==0.15.11 jsonref==1.1.0 deepspeed==0.11.1 typer==0.9.0

pip install flash-attn==2.3.2 --no-build-isolation
```

You can run the script to verify that the implementation of monkey-patch is correct:

```shell
python assert_monkey_patch.py mistralai/Mixtral-8x7B-v0.1
```

The output would show:
+ 50 random data points are packed into only 2 data points 
+ Time for computing loss from original model + original dataset is significantly greater than that monkey-patched model + packed dataset (In my run: 30.38 seconds vs 1.14 seconds)
+ The difference between loss from **(original dataset, original model)** and **(packed dataset, monkey-patched model)** is almost 0%.