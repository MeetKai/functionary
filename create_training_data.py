import json 
import random 

def read_data(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def main():
    text_train_data = read_data("2024-03-27/2024-03-27_train.jsonl")
    text_dev_data = read_data("2024-03-27/2024-03-27_val.jsonl")
    
    img_train_data = read_data("2024-07-30_train.jsonl")
    img_dev_data = read_data("2024-07-30_val.jsonl")
    
    print(f"text_train_data:{len(text_train_data)}; text_dev_data: {len(text_dev_data)}; img_train_data: {len(img_train_data)}; img_dev_data:{len(img_dev_data)}")
    
    for item in [text_train_data, text_dev_data, img_train_data, img_dev_data]:
        random.shuffle(item)
    
    total_train = text_train_data[: 4000] + img_train_data
    total_dev = text_dev_data[: 100] + img_dev_data
    
    random.shuffle(total_train)
    random.shuffle(total_dev)
    
    print(f"number of total_train: {len(total_train)}; total_dev: {len(total_dev)}")
    
    with open("train_4k_text_4k_img.jsonl", "w") as f:
        for item in total_train:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    with open("dev_100_text_100_img.jsonl", "w") as f:
        for item in total_dev:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()