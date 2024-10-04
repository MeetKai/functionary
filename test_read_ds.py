from transformers import AutoTokenizer
from functionary.prompt_template import get_prompt_template_by_version
from torch.utils.data import DataLoader
from functionary.train_vision.vision_datasets import get_collate_fn, get_vision_dataset_class


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
                    {"type": "image_url", "image_url": {"url": "file://assets/Functionary_32.png"}},
                    {"type": "text", "text": "can you describe this image"},
                ],
            },
            {"role": "assistant", "content": "the first image is letter s"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "how about this image ?"},
                    {"type": "image_url", "image_url": {"url": "file://assets/SGD_acc.png"}},
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
dataset_type = "LazyQwen2VLDataset"
ds = get_vision_dataset_class(dataset_type)(
    raw_data,
    tokenizer,
    pretrained_path=pretrained_path,
    pad_img_path="functionary/train_vision/pad_img2.png",
    max_length = 155,
    use_img_pad_token=True,
)


loader = DataLoader(ds, collate_fn=get_collate_fn(dataset_type, None, tokenizer), batch_size=3, shuffle=False)

for batch in loader:
    for key in batch:
        print(f"{key}, shape={batch[key].shape}")
    print(batch["image_grid_thw"])