# reference: https://github.com/huggingface/trl/issues/805
from transformers import LlamaTokenizer
from functionary.train.llama_attention_mask_monkey_patch import LlamaForCausalLM
import torch 



model_path = "models/Llama-2-7b-hf"
device = "cuda:1"

tokenizer = LlamaTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token


def get_causal_mask(l, m_value):
    result = torch.zeros((l, l))
    result = result + m_value
    for i in range(l):
        for j in range(i + 1):
            result[i][j] = 0
    return result
    
def create_mask_from_lengths(lengths, max_length, m_value):
    result = torch.zeros((max_length, max_length))
    result += m_value
    acc_leng = 0
    print("lengths: ", lengths)
    for length in lengths:
        x = get_causal_mask(length, m_value)
        result[acc_leng: acc_leng + length, acc_leng: acc_leng + length] = x
        acc_leng += length
    pad_length = max_length - sum(lengths)
    result[-pad_length: , :] = 0
    result[:, -pad_length: ] = m_value
    return result

        
def pack_inputs(tokenizer, texts, max_length=4096, m_value=float("-inf")):
    input_ids = tokenizer(texts)["input_ids"]
    new_input = []
    lengths = []
    for input_id in input_ids:
        new_input += input_id
        lengths.append(len(input_id))
    masks = create_mask_from_lengths(lengths, max_length, m_value)
    labels = list(new_input)
    pad_leng = max_length - len(new_input)
    assert pad_leng >= 0
    print("pad_leng: ", pad_leng)
    new_input += [tokenizer.pad_token_id for _ in range(pad_leng)]
    labels += [-100 for _ in range(pad_leng)]
    assert len(new_input) == max_length == len(labels)
    return {"input_ids": new_input, "attention_mask": masks.tolist(), "labels": labels}


def create_batch_inputs(list_inputs):
    result = {}
    keys = list_inputs[0].keys()
    for key in list(keys):
        result[key] = []
    for inputs in list_inputs:
        for key in inputs:
            if key == "attention_mask":
                result[key].append([inputs[key]])   # attention_mask needs: Bx1xNxN
            else:
                result[key].append(inputs[key]) 
    for key in result:
        #print(f"key: {key}, result: {result[key]}")
        result[key] = torch.tensor(result[key])
    return result


def prepare_inputs(tokenizer, data_points, max_leng):
    list_inputs = []
    for dp in data_points:
        print("pack: ", dp)
        inputs = pack_inputs(tokenizer, dp, max_leng)
        list_inputs.append(inputs)
    return create_batch_inputs(list_inputs)

def test1():
    data_point1 = ["hello", "how old are"]
    data_point2 = ["this is my son"]
    # inputs = tokenizer(texts, padding=True, max_length=10, truncation=True, return_tensors="pt")
    # inputs["labels"] = inputs["input_ids"]
    # for key in ["input_ids", "attention_mask", "labels"]:
    #     inputs[key] = inputs[key].to(device)
    # inputs["return_dict"] = True
    # print(inputs)
    # output = model.forward(**inputs)
    # print("loss: ", output.loss)
    x = prepare_inputs(tokenizer, [data_point1, data_point2], 10)
    for key in x:
        print("---------------------------")
        print(f"key: {key}, shape: {x[key].shape}")
        print(x[key])
        x[key] = x[key].to(device)


    model = LlamaForCausalLM.from_pretrained(model_path, device_map=device, use_flash_attention_2=False)
    model.train()

    x["return_dict"] = True
    print(x)
    output = model.forward(**x)
    print("loss: ", output.loss)
    print("==============")
    texts= ["hello", "how old are", "this is my son"]
    inputs = tokenizer(texts, padding="max_length", max_length=10, truncation=True, return_tensors="pt")
    inputs["labels"] = inputs["input_ids"]

    for key in ["input_ids", "attention_mask", "labels"]:
        print("----------")
        print("key=", key)
        print(inputs[key])
        inputs[key] = inputs[key].to(device)

    inputs["return_dict"] = True
    output = model.forward(**inputs)
    print("Loss from normal attention_mask: ", output.loss)


def get_label_from_inputs(input_ids):
    all_labels = input_ids.tolist()
    for labels in all_labels:
        for i in range(len(labels)):
            if labels[i] == tokenizer.pad_token_id or labels[i] == 1:
                labels[i] = -100
    return torch.tensor(all_labels)


def get_loss(test_inputs, max_length, model):
    inputs = tokenizer(test_inputs, padding="max_length", max_length=max_length, truncation=True, return_tensors="pt")
    inputs["labels"] = get_label_from_inputs(inputs["input_ids"])
    
    for key in ["input_ids", "attention_mask", "labels"]:
        inputs[key] = inputs[key].to(device)
        
    print("inputs: ", inputs)
    inputs["return_dict"] = True
    output = model.forward(**inputs)
    loss2 = output.loss
    return loss2


def get_loss_packed(test_inputs, max_length, model):
    inputs = prepare_inputs(tokenizer, test_inputs, max_length)
    inputs["labels"] = get_label_from_inputs(inputs["input_ids"])
    
    for key in ["input_ids", "attention_mask", "labels"]:
        inputs[key] = inputs[key].to(device)
        
    print("inputs: ", inputs)
    inputs["return_dict"] = True
    output = model.forward(**inputs)
    loss2 = output.loss
    return loss2
    

model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map=device, use_flash_attention_2=False)
model.train()
# l1 = get_loss(["this is my son"], 8, model)
# print("l1: ", l1)
# l2 = get_loss(["this is my son"], 10, model)
# print("l2: ", l2)

# l3 = get_loss(["nice to meet you again here"], 10, model)
# print("l3: ", l3)

l4 = get_loss(["nice to meet you again here", "this is my son", "oh my god, I think I really love you boy"], 15, model)
print("l4: ", l4)

# print("----------------------")

# p1 = get_loss_packed(["nice to meet you again here"], 10, model)
# print("p1: ", p1)

# p2 = get_loss_packed([["this is my son"]], 20, model)
# print("p2: ", p2)

p3 = get_loss_packed([["nice to meet you again here", "this is my son", "oh my god, I think I really love you boy"]], 30, model)
print("p3: ", p3)

# p4 = get_loss_packed([["nice to meet you again here", "this is my son"]], 50, model)
# print("p4: ", p4)