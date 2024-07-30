from PIL import Image
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms as T
import torch
from intermlm.modeling_internvl_chat import InternVLChatModel
from transformers import AutoTokenizer
import transformers
from typing import Optional, Dict
from torch.utils.data import Dataset
from functionary.prompt_template import get_prompt_template_from_tokenizer
from functionary.prompt_template import prompt_utils
from functionary.train.custom_datasets import prepare_training_inputs


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ]
    )
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image, min_num=1, max_num=12, image_size=448, use_thumbnail=False
):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert("RGB")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num
    )
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def fill_img_tokens_to_prompt(
    prompt,
    num_patches_list,
    num_image_token,
    img_start_token,
    img_context_token,
    img_end_token,
):
    for num_patches in num_patches_list:
        image_tokens = (
            img_start_token
            + img_context_token * num_image_token * num_patches
            + img_end_token
        )
        prompt = prompt.replace("<image>", image_tokens, 1)
    return prompt


class LazyVisionDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        raw_data,
        tokenizer: transformers.PreTrainedTokenizer,
        pad_img_path: Optional[str] = "",
    ):
        super().__init__()
        self.tokenizer = tokenizer

        # sampler = UniformSampler(raw_data, batch_size)
        # self.loop_indices = sampler.loop_indices
        # print(f"batch_size: {batch_size}; loop_index: {self.loop_indices[: 20]}")
        # # make sure that raw_data is in the correct order of reading data
        # self.raw_data = [raw_data[index] for index in self.loop_indices]

        self.raw_data = raw_data
        self.cached_data_dict = {}
        self.prompt_template = get_prompt_template_from_tokenizer(tokenizer)

        self.pad_img = None
        if pad_img_path:
            self.pad_img = Image.open(open(pad_img_path, "rb"))

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = prepare_training_inputs(
            messages=self.raw_data[i],
            tokenizer=self.tokenizer,
            keep_assistant_prefix=False,
        )
        example = self.raw_data[i]
        images = prompt_utils.extract_images_from_messages(example["messages"])
        # if (
        #     len(images) == 0 and self.pad_img
        # ):  # add pad_img_token to make sure that the graph is fixed
        #     images.append(self.pad_img)
        #     ret["inputs"] = inject_image_token(ret["inputs"], self.rep_token_id)

        # input_ids = ret["inputs"]["input_ids"]
        # input_ids[input_ids == self.rep_token_id] = IMAGE_TOKEN_INDEX

        # assert (input_ids == IMAGE_TOKEN_INDEX).sum() == len(images)
        ret = {
            "input_ids": ret["inputs"]["input_ids"],
            "labels": ret["inputs"]["labels"],
            "attention_mask": ret["inputs"]["attention_mask"],
            "images": images,
        }

        self.cached_data_dict[i] = ret
        return ret
