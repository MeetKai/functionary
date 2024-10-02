import typer
from transformers import AutoTokenizer
from functionary.prompt_template import get_prompt_template_by_version
from functionary.train_vision.qwen2_dataset import LazyVisionDataset
import torch
from transformers import Qwen2VLForConditionalGeneration
import math


def get_raw_data():
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
                        {
                            "type": "image_url",
                            "image_url": {"url": "file://assets/s.png"},
                        },
                        {"type": "text", "text": "can you describe this image"},
                    ],
                },
                {"role": "assistant", "content": "the first image is letter s"},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "how about this image ?"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "file://assets/as.png"},
                        },
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
    return raw_data


def get_loss_from_ds(ds, model):
    loss_list = []
    model.eval()
    with torch.no_grad():
        for i in range(len(ds)):
            print("------------------")
            data = ds[i]
            print("data: ", data)
            for key in ["input_ids", "labels", "attention_mask"]:
                data[key] = data[key][None, :]

            for key in data:
                data[key] = data[key].to(model.device)
                print(f"{key}: {data[key].shape}")

            labels = data["labels"]
            label_count = (labels != -100).sum().item()
            if label_count == 0:
                loss_list.append((label_count, -1))
            else:
                output = model.forward(**data)
                loss = output.loss.item()
                loss_list.append((label_count, loss))
    return loss_list


def main(max_length: int):
    pretrained_path = "Qwen/Qwen2-VL-7B-Instruct"
    prompt_template = get_prompt_template_by_version("qwen2-vl")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
    tokenizer.chat_template = prompt_template.get_chat_template_jinja()

    raw_data = get_raw_data()
    # raw_data = raw_data[-1: ]
    ds = LazyVisionDataset(
        raw_data,
        tokenizer,
        pretrained_path=pretrained_path,
        pad_img_path="functionary/train_vision/pad_img2.png",
        max_length=max_length,
        use_img_pad_token=False,
    )

    pad_ds = LazyVisionDataset(
        raw_data,
        tokenizer,
        pretrained_path=pretrained_path,
        pad_img_path="functionary/train_vision/pad_img2.png",
        max_length=max_length,
        use_img_pad_token=True,
    )

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        pretrained_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        use_flash_attention_2=True,
    )
    loss_list = get_loss_from_ds(ds, model)
    print("loss_list: ", loss_list)
    pad_loss_list = get_loss_from_ds(pad_ds, model)
    print("pad_loss_list: ", pad_loss_list)
    print("----------------------------")
    for loss, pad_loss in zip(loss_list, pad_loss_list):
        count1, loss1 = loss
        count2, loss2 = pad_loss
        percentage = math.fabs(loss2 - loss1) * 100 / loss1
        print(
            f"count1: {count1}; count2: {count2}, loss1: {loss1}, loss2: {loss2}; percentage={percentage} %"
        )


if __name__ == "__main__":
    typer.run(main)
