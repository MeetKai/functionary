from functionary.openai_types import (
    ChatMessage,
    Function,
    FunctionCall,
    Tool,
    ChatCompletionRequest,
)
from functionary.prompt_template import get_prompt_template_from_tokenizer
from functionary.prompt_template.prompt_utils import (
    prepare_messages_for_inference,
    extract_images_from_messages,
)
from typing import Any, List, Optional
from llava.mm_utils import process_images
import torch
from functionary.inference_utils import analyze_tools_and_tool_choice


def generate(
    *, model: Any, tokenizer: Any, request: ChatCompletionRequest
) -> ChatMessage:
    tools_or_functions, tool_func_choice = analyze_tools_and_tool_choice(request)

    prompt_token_ids = prepare_messages_for_inference(
        tokenizer=tokenizer,
        messages=request.messages,
        tools_or_functions=tools_or_functions,
        tool_choice=tool_func_choice,
        device=model.device,
    )

    prompt_template = get_prompt_template_from_tokenizer(tokenizer)

    # replace unused token --> image_token_id
    img_token_id = tokenizer.encode(
        prompt_template.image_token, add_special_tokens=False
    )[0]
    prompt_token_ids[prompt_token_ids == img_token_id] = -200

    images = extract_images_from_messages(
        [message.dict() for message in request.messages]
    )
    image_tensor, image_sizes = None, None
    image_processor = model.get_vision_tower().image_processor
    if images:
        image_tensor = process_images(images, image_processor, model.config)
        image_tensor = [
            _image.to(dtype=torch.float16, device=model.device)
            for _image in image_tensor
        ]
        image_sizes = [image.size for image in images]

    generate_ids = model.generate(
        prompt_token_ids,
        images=image_tensor,
        image_sizes=image_sizes,
        do_sample=False,
        temperature=0,
        max_new_tokens=1024,
    )

    text_output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0]

    result = prompt_template.parse_assistant_response(text_output, tool_choice="auto")
    return ChatMessage(**result)
