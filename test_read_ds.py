from functionary.train_vision.qwen2_dataset import LazyVisionDataset
from transformers import AutoTokenizer
from functionary.prompt_template import get_prompt_template_by_version
from torch.utils.data import DataLoader

pretrained_path = "Qwen/Qwen2-VL-7B-Instruct"
prompt_template = get_prompt_template_by_version("qwen2-vl")
tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
tokenizer.model_max_length = 512
tokenizer.chat_template = prompt_template.get_chat_template_jinja()

raw_data = [
    {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "file://functionary_logo.jpg"},
                    },
                    {"type": "text", "text": "can you describe this image"},
                ],
            },
            {
                "role": "assistant",
                "content": "This image is about a tree"
            }
        ],
        "tools": [],
    },
    {
        "messages": [
            {
                "role": "user",
                "content": "what do you do for a living?"
            },
            {
                "role": "assistant",
                "content": "I am a doctor"
            }
        ],
        "tools": [],
    }
]

ds = LazyVisionDataset(
    raw_data,
    tokenizer,
    pretrained_path=pretrained_path,
    pad_img_path="functionary/train_vision/pad_img2.png",
)

for i in range(len(ds)):
    print("---------------------------------")
    dt = ds[i]
    for key in dt:
        print(f"{key}; shape: {dt[key].shape}: {dt[key]}")
        print(f"key: ----")