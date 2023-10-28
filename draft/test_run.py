from functionary.train import custom_datasets
from transformers import LlamaTokenizerFast
from functionary.prompt import get_additional_tokens
import json 
import datetime 
import torch
import random 

from functionary.train.llama_attention_mask_monkey_patch import LlamaForCausalLM


data_path = "2023-10-27_val.jsonl"
model_path = "models/Llama-2-7b-hf"
tokenizer = LlamaTokenizerFast.from_pretrained(model_path, legacy=True, model_max_length=1024)
tokenizer.pad_token = tokenizer.unk_token
special_tokens = {"additional_special_tokens": get_additional_tokens()}
tokenizer.add_special_tokens(special_tokens)

with open(data_path, "r") as f:
    raw_data = [json.loads(line) for line in f]
raw_data = raw_data
random.shuffle(raw_data)

def assert_batch_processing():
    batch_size = 50
    return_tensor = True
    result = custom_datasets.prepare_training_inputs_batch(raw_data[: batch_size], tokenizer, padding="max_length", max_length=4096, return_tensor=return_tensor)
    for index, item in enumerate(raw_data[: batch_size]):
        res = custom_datasets.prepare_training_inputs_old(item, tokenizer, padding="max_length", max_length=4096, return_tensor=return_tensor)
        assert res["final_prompt"] == result["batch_prompts"][index]
        inputs = res["inputs"]
        b_inputs = result["batch_inputs"][index]
        for key in b_inputs:
            if not return_tensor:
                assert b_inputs[key] == inputs[key]
            else:
                assert b_inputs[key].tolist() == inputs[key].tolist()
        print("correct: ", index)
    print("Done !!!")


customized_dataset = custom_datasets.CustomDataset(raw_data, tokenizer)
print("number of data-points before: ", len(customized_dataset))
t1 = datetime.datetime.now()
packed_dataset = custom_datasets.PackedDataset(customized_dataset, 1024, tokenizer.pad_token_id)
t2 = datetime.datetime.now()
print("total time for handling: ", (t2 - t1).total_seconds())
print("number of data points after: ", len(packed_dataset))

# RUN_DEVICE = "cuda:1"

# def create_batch(ds):
#     input_dic = {}
#     for i in range(len(ds)):
#         print("i =", i)
#         item = ds[i]
#         if len(input_dic) == 0:
#             for key in item:
#                 input_dic[key] = []
#         for key in item:
#             input_dic[key].append(item[key])
        
#     for key in input_dic:
#         input_dic[key] = torch.stack(input_dic[key], 0)
#     return input_dic

# def prepare_inputs(input_dic):
#     for key in input_dic:
#         input_dic[key] = input_dic[key].to(RUN_DEVICE)
#     input_dic["return_dict"] = True


# def print_inputs_info(inputs):
#     for key in inputs:
#         if key != "return_dict":
#             print(f"{key}: {inputs[key].shape}")

# print("----------loss 1--------")
# inputs = create_batch(customized_dataset)
# prepare_inputs(inputs)

# print_inputs_info(inputs)

# model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map=RUN_DEVICE, use_flash_attention_2=False)
# model.resize_token_embeddings(len(tokenizer))
# model.train()


# print("inputs1: ", inputs)
# with torch.no_grad():
#     loss1 = model.forward(**inputs).loss.item()
#     print("loss1 = ", loss1)

# inputs2 = packed_dataset[0]
# for key in inputs2:
#     inputs2[key] = torch.unsqueeze(inputs2[key], 0)
# print_inputs_info(inputs2)
# print("inputs2: ", inputs2)
# prepare_inputs(inputs2)
# with torch.no_grad():
#     loss2 = model.forward(**inputs2).loss.item()
#     print("loss2 = ", loss2)