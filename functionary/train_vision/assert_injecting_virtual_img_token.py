from functionary.train_vision.models.modeling_llava import (
    FixedLlavaLlamaForCausalLM as LlavaLlamaForCausalLM,
)
from functionary.train_vision.llava_dataset import LazyVisionDataset
from functionary.prompt_template import get_prompt_template_by_version
from transformers import AutoTokenizer
import random
import json
import torch
import math
import numpy as np
from torch.utils.data import DataLoader
from llava.mm_utils import process_images

random.seed(100)


def get_loss_from_ds(data_loader: DataLoader, model):
    model.eval()
    result = []
    count = 0
    with torch.no_grad():
        for batch in data_loader:
            for key in batch:
                if type(batch[key]) is torch.Tensor:
                    batch[key] = batch[key].to(model.device)
            loss = model.forward(**batch).loss.item()
            result.append(loss)
            count += 1
            print("count=", count)
    return result


def main(text_only_path: str = "2024-05-15_val.jsonl"):
    pretrained_path = "lmms-lab/llama3-llava-next-8b"
    with open(text_only_path, "r") as f:
        examples = [json.loads(line) for line in f]
    random.shuffle(examples)

    model = LlavaLlamaForCausalLM.from_pretrained(
        pretrained_path,
        torch_dtype=torch.bfloat16,
        use_flash_attention_2=True,
        device_map="cuda",
    )

    tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
    tokenizer.pad_token = tokenizer.eos_token
    prompt_template = get_prompt_template_by_version("v3.llava_llama")
    tokenizer.chat_template = prompt_template.get_chat_template_jinja()

    model.config.use_cache = False
    ds = LazyVisionDataset(examples[:20], tokenizer)
    padded_ds = LazyVisionDataset(examples[:20], tokenizer, "pad_img.png")

    def collate_examples(features):
        result = {}
        first = features[0]
        for k, v in first.items():
            if k in ["input_ids", "attention_mask", "labels"]:
                if isinstance(v, torch.Tensor):
                    result[k] = torch.stack([f[k] for f in features])
                elif isinstance(v, np.ndarray):
                    result[k] = torch.tensor(np.stack([f[k] for f in features]))
                else:
                    result[k] = torch.tensor([f[k] for f in features])
        # aggregate images & image_sizes
        images = []
        for feature in features:
            images.extend(feature["images"])

        image_sizes = [image.size for image in images]

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model(device_map="auto")
        image_processor = vision_tower.image_processor

        if len(images) > 0:
            image_tensor = process_images(images, image_processor, model.config)
        else:
            image_tensor = None

        if image_tensor is not None and type(image_tensor) is not list:
            image_tensor = image_tensor.to(model.dtype)

        result["images"] = image_tensor
        result["image_sizes"] = image_sizes
        return result

    data_loader1 = DataLoader(
        ds, batch_size=2, shuffle=False, collate_fn=collate_examples
    )
    data_loader2 = DataLoader(
        padded_ds, batch_size=2, shuffle=False, collate_fn=collate_examples
    )

    loss_list2 = get_loss_from_ds(data_loader2, model)
    print("--------------------------------------")
    loss_list1 = get_loss_from_ds(data_loader1, model)

    assert len(loss_list1) == len(loss_list2)
    for loss1, loss2 in zip(loss_list1, loss_list2):
        diff = loss2 - loss1
        percent = math.fabs(diff) / math.fabs(loss1)
        print(f"loss1={loss1}; loss2={loss2}, percent={percent}")


if __name__ == "__main__":
    main()
