from functionary.train_vision.qwen2_dataset import LazyVisionDataset
from transformers import AutoTokenizer
from functionary.prompt_template import get_prompt_template_by_version
from torch.utils.data import DataLoader

pretrained_path = "Qwen/Qwen2-VL-7B-Instruct"
prompt_template = get_prompt_template_by_version("qwen2-vl")
tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
tokenizer.chat_template = prompt_template.get_chat_template_jinja()


# contain images all
raw_data = [
    {
        "messages": [
            {"role": "user", "content": "what do you do for a living?"},
            {"role": "assistant", "content": "I am a doctor"},
        ],
        "tools": [],
    },
    {
        "messages": [
            {"role": "user", "content": "what do you do for a living?"},
            {"role": "assistant", "content": "I am a doctor"},
            {
                "role": "user",
                "content": f"can you count number of letters s in this string: "
                + " ".join(["this" for _ in range(100)]),
            },
            {"role": "assistant", "content": "The number is 100"},
        ],
        "tools": [],
    },
    {  # the last image as partially truncated
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "file://assets/s.png"}},
                    {"type": "text", "text": "can you describe this image"},
                ],
            },
            {"role": "assistant", "content": "the first image is letter s"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "how about this image ?"},
                    {"type": "image_url", "image_url": {"url": "file://assets/as.png"}},
                ],
            },
            {
                "role": "assistant",
                "content": "this image is the word: 'as' I can predict this",
            },
        ],
        "tools": [],
    },
]

# For index = 2
# 47: the first image is partially truncated
# 10: all img tokens are truncated
# 85 for partially truncated;
# 155: all images are included;
# 67: only the first image is included
tokenizer.model_max_length = 155

ds = LazyVisionDataset(
    raw_data,
    tokenizer,
    pretrained_path=pretrained_path,
    pad_img_path="functionary/train_vision/pad_img2.png",
    use_img_pad_token=False,
)


def display_data(index):
    dt = ds[index]
    for key in dt:
        print(f"{key}; shape: {dt[key].shape}: {dt[key]}")
        print(f"---------------------")
    input_ids = dt["input_ids"].tolist()
    print(f"text: {tokenizer.decode(input_ids)}")
    labels = dt["labels"]
    labels[labels == -100] = tokenizer.pad_token_id
    print(f"---------------------")
    print(f"labels: {tokenizer.decode(labels)}")


display_data(2)
