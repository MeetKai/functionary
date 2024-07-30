from PIL import Image
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms as T
import torch 
from intermlm.modeling_internvl_chat import InternVLChatModel
from transformers import AutoTokenizer 


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
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


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

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
            ((i // (target_width // image_size)) + 1) * image_size
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
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def fill_img_tokens_to_prompt(prompt, num_patches_list, num_image_token, img_start_token, img_context_token, img_end_token):
    for num_patches in num_patches_list:
        image_tokens = img_start_token + img_context_token * num_image_token * num_patches + img_end_token
        prompt = prompt.replace('<image>', image_tokens, 1)
    return prompt


def compute_loss():
    prompt = """<|im_start|>system
你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。<|im_end|><|im_start|>user
Image-1: <image>
Image-2: <image>
Describe the two images in detail.<|im_end|><|im_start|>assistant
The two images depict distinctly different subjects, each with its unique characteristics and context.

**Image 1:**

This image features a prominent architectural landmark, the Marina Bay Sands in Singapore. The iconic hotel complex is characterized by its three towers, which are connected at the top by a horizontal structure resembling a ship's prow. The towers are clad in glass and steel, reflecting the sky and surrounding environment. In the foreground, there is a large sculpture of a lion's head, which is a significant part of the Singapore cityscape. The sculpture is made of white material and is positioned near a body of water, likely a man-made lagoon. The sky in the background is a beautiful gradient of colors, transitioning from warm hues near the horizon to cooler tones higher up, indicating either sunrise or sunset. The water in the foreground is calm, adding to the serene atmosphere of the scene. The overall composition of the image highlights the blend of modern architecture and natural beauty.

**Image 2:**

This image focuses on a close-up of a flower, specifically a globe thistle (Echium sphaerocephalum). The flower is characterized by its spherical shape and numerous small, delicate petals that radiate outward from the center. The petals are a soft purple color, and the flower is set against a backdrop of other similar flowers and green foliage. The background is slightly blurred, which helps to emphasize the intricate details of the globe thistle. The image captures the natural beauty and complexity of the flower, highlighting its unique structure and color. The lighting in the image is natural, likely taken during the day, which enhances the vibrant colors of the petals and the greenery surrounding the flower.

In summary, Image 1 showcases a modern architectural marvel with a serene waterfront setting, while Image 2 presents a close-up of a delicate and intricate natural flower, emphasizing its unique form and color. Both images capture the beauty of their respective subjects, one through human-made structures and the other through natural elements.<|im_end|><|im_start|>user
What are the similarities and differences between these two images.<|im_end|><|im_start|>assistant""".strip()
    img_paths = ["singapore.jpeg", "flower.jpg"]
    imgs = [load_image(img_path, max_num=12).to(torch.bfloat16).cuda() for img_path in img_paths]
    
    pixel_values = torch.cat(imgs, dim=0)
    pixel_values = pixel_values.cuda()
    num_patches_list = [item.size(0) for item in imgs]
    print("num_patches_list: ", num_patches_list)
    path = 'OpenGVLab/InternVL2-8B'
    
    model = InternVLChatModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16, use_flash_attention_2=True).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    full_prompt = fill_img_tokens_to_prompt(prompt, num_patches_list, model.num_image_token, "<img>", "<IMG_CONTEXT>", "</img>")
    #print("------------FULL_PROMPT-----------")
    #print(full_prompt)
    
    img_start = tokenizer.convert_tokens_to_ids("<img>")
    img_end = tokenizer.convert_tokens_to_ids("</img>")
    img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    
    model.img_context_token_id = img_context_token_id
    
    model_inputs = tokenizer(full_prompt, return_tensors='pt')
    input_ids = model_inputs['input_ids'].cuda()
    print("input_ids: ", input_ids.tolist())
    labels = torch.clone(input_ids)
    labels[labels == img_start] = -100
    labels[labels == img_end] = -100
    labels[labels == img_context_token_id] = -100
    
    # count number of labels
    loss_count = (labels != -100).sum()
    print("loss_count: ", loss_count)
    
    attention_mask = model_inputs['attention_mask'].cuda()
    print("attention_mask: ", attention_mask.sum())
    input_dic = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
        "attention_mask": attention_mask,
        "labels": labels
    }
    print("part of pixel: ", pixel_values[0][0][0][: 10])
    model.eval()
    with torch.no_grad():
        result = model.forward(**input_dic)
        print("loss: ", result.loss)


if __name__ == "__main__":
    compute_loss()