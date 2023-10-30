from functionary.train import custom_datasets
from transformers import LlamaTokenizerFast
from functionary.prompt import get_additional_tokens
import json 


model_path = "models/Llama-2-7b-hf"
tokenizer = LlamaTokenizerFast.from_pretrained(model_path, legacy=True, model_max_length=7000)
tokenizer.pad_token = tokenizer.unk_token
special_tokens = {"additional_special_tokens": get_additional_tokens()}
tokenizer.add_special_tokens(special_tokens)

train_path = "2023-10-27_train.jsonl"
raw_data_points = []
with open(train_path) as f:
    for line in f:
        raw_data_points.append(json.loads(line))

print("number of dp: ", len(raw_data_points))
ds = custom_datasets.DirectPackedDataset(raw_data_points, tokenizer, cached_path="cached_ds/7000_train", ignore_cached=True)

#ds = custom_datasets.PackedDataset(tokenizer, cached_path="models/llama-2-7b-functionary-2023-10-27/train_cached")
# wrong_item = ds[13]
# input_ids = wrong_item["input_ids"].tolist()
# for chunk in [(200, 250), (1000, 1050), (2000, 2050)]:
#     start, end = chunk
#     #print(f"{chunk}: {input_ids[start: end]}")

# input_ids = wrong_item["input_ids"].tolist()
# labels_ids = wrong_item["labels"].tolist()

# for i in range(len(ds)):
#     print("-------------i=", i)
#     ds[i]

# print("why: ")
# labels_ids = [label for label in labels_ids if label != -100]
# print("final label_ids: ", labels_ids)
# input_text = tokenizer.decode(input_ids)
# print(input_text)