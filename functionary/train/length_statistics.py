from transformers import LlamaTokenizerFast
from functionary.prompt import get_additional_tokens
from functionary.train.custom_datasets import get_prompt_from_messages
import json 
import typer
import datetime
import csv
import os


def save_json(data, path):
    with open(path, "w") as f:
        f.write(json.dumps(data, ensure_ascii=False, indent=4))


def save_csv(rows, path):
    with open(path, "w") as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)
            

def get_batch_indices(batch_size: int, total_size: int):
    iter_num = total_size // batch_size
    result = []
    for i in range(iter_num + 1):
        start = i * batch_size
        end = i * batch_size + batch_size
        if end > total_size:
            end = total_size
        if end > start: 
            result.append((start, end))
    return result


def compute_length_statistics(pretrained: str, train_path: str, valid_path: str, save_folder: str):
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    tokenizer = LlamaTokenizerFast.from_pretrained(pretrained, legacy=True)
    tokenizer.pad_token = tokenizer.unk_token
    special_tokens = {"additional_special_tokens": get_additional_tokens()}
    num_new_tokens = tokenizer.add_special_tokens(special_tokens)
    print("number of new tokens: ", num_new_tokens)
    total_data = []
    with open(train_path, "r") as file:
        raw_train_data = [json.loads(line) for line in file]
        total_data.extend(raw_train_data)
    
    with open(valid_path, "r") as file:
        raw_val_data = [json.loads(line) for line in file]
        total_data.extend(raw_val_data)
        
    prompts = []
    for _, item in enumerate(total_data):
        prompts.append(get_prompt_from_messages(
            item["messages"], item["functions"]
        ))
    print(f"start tokenizing {len(prompts)} now !!!")
    batch_size = 3000
    batches = get_batch_indices(batch_size, len(prompts))
    count_dic = {}
    t1 = datetime.datetime.now()
    for index, (start, end) in enumerate(batches):
        inputs = tokenizer(prompts[start: end])["input_ids"]
        for item in inputs:
            length = len(item)
            count_dic[length] = count_dic.get(length, 0) + 1
        t2 = datetime.datetime.now()
        acc_time = (t2 - t1).total_seconds()
        avg_time = acc_time / (index + 1)
        print(f"{index} / {len(batches)}; avg_time: {avg_time}; remaining time: {avg_time * (len(batches) - index -1)}")
            
    sorted_lengths = sorted(count_dic.items(), key=lambda x: x[0])
    acc_count = 0
    pairs = []
    rows = []
    for length, count in sorted_lengths:
        acc_count += count
        pairs.append((length, acc_count))
        rows.append((str(length), str(count)))
    rows.reverse()
    save_csv([["length", "count"]] + rows, f"{save_folder}/length_dic_count.csv")
    total_count = acc_count
    assert total_count == len(total_data)
    pairs.reverse()
    rows = [["length", "accumulated_count", "percentage"]]
    for i in range(len(pairs)):
        length, count = pairs[i]
        rows.append([str(length), str(count), str(count/total_count)])
    save_csv(rows, f"{save_folder}/accumulated.csv")
 

if __name__ == "__main__":
    typer.run(compute_length_statistics)