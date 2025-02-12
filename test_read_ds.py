from transformers import AutoTokenizer
from functionary.prompt_template import get_prompt_template_by_version
from torch.utils.data import DataLoader
from functionary.train_vision.vision_datasets import get_collate_fn, get_vision_dataset_class
import json 


pretrained_path = "Qwen/Qwen2-VL-7B-Instruct"
prompt_template = get_prompt_template_by_version("qwen2-vl")
tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
tokenizer.chat_template = prompt_template.get_chat_template_jinja()


# contain images all
with open("tests/test_case.json", "r") as f:
    raw_data = [json.loads(f.read())]


dataset_type = "LazyQwen2VLDataset"
ds = get_vision_dataset_class(dataset_type)(
    raw_data,
    tokenizer,
    pretrained_path=pretrained_path,
    pad_img_path="functionary/train_vision/pad_img2.png",
    max_length = 800,
    use_img_pad_token=True,
)


loader = DataLoader(ds, collate_fn=get_collate_fn(dataset_type, None, tokenizer), batch_size=3, shuffle=False)

for batch in loader:
    for key in batch:
        print(f"{key}, shape={batch[key].shape}")
    print(batch["image_grid_thw"])
    print("input_ids: ",batch["input_ids"])
    input_ids = batch["input_ids"][0].tolist()
    print("attention_mask: ",batch["attention_mask"])
    print("labels: ", batch["labels"])
    labels = batch["labels"][0].tolist()
    print("--------------INPUT----------")
    print(tokenizer.decode(input_ids))
    print("----------------LABEL--------")
    for i in range(len(labels)):
        if labels[i] == -100:
            labels[i] = 2
    print(tokenizer.decode(labels))