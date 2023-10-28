from transformers import LlamaTokenizerFast, LlamaTokenizer
from functionary.prompt import get_additional_tokens, get_prompt_from_messages
import datetime 
import json 
import random 

model_path = "models/Llama-2-7b-hf"

def get_tokenizer(use_fast: bool):
    if use_fast:
        tokenizer = LlamaTokenizerFast.from_pretrained(model_path, legacy=True)
    else:
        tokenizer = LlamaTokenizer.from_pretrained(model_path, legacy=True)
    tokenizer.pad_token = tokenizer.unk_token
    special_tokens = {"additional_special_tokens": get_additional_tokens()}
    tokenizer.add_special_tokens(special_tokens)
    return tokenizer 

tokenizer = get_tokenizer(False)
fast_tokenizer = get_tokenizer(True)

total_data = []
with open("2023-10-20_train.jsonl", "r") as file:
    raw_train_data = [json.loads(line) for line in file]
    total_data.extend(raw_train_data)


prompts = []
for _, item in enumerate(total_data):
    prompts.append(get_prompt_from_messages(
        item["messages"], item["functions"]
    ))
random.shuffle(prompts)

print("start testing now")
for prompt in prompts[: 500]:
    t1 = tokenizer(prompt, padding="max_length", max_length=4096, truncation=True)["input_ids"]
    t2 = fast_tokenizer(prompt, padding="max_length", max_length=4096, truncation=True)["input_ids"]
    if t1 != t2:
        for i in range(len(t1)):
            if t1[i] != t2[i]:
                print(f"differ at: {i}: {t1[i]} and {t2[i]}")
        break