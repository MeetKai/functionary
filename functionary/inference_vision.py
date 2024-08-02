from functionary.openai_types import (
    ChatMessage,
    Function,
    FunctionCall,
    Tool,
    ChatCompletionRequest,
    UsageInfo,
)
from functionary.prompt_template import get_prompt_template_from_tokenizer
from functionary.prompt_template.prompt_utils import (
    prepare_messages_for_inference,
    extract_images_from_messages,
)
from typing import Any, List, Optional, Tuple
import torch
from functionary.inference_utils import analyze_tools_and_tool_choice
from enum import Enum


class ModelType(str, Enum):
    llama_llava = "llama_llava"
    internvl_chat = "internvl_chat"


def generate(
    *, model_type: ModelType, model: Any, tokenizer: Any, request: ChatCompletionRequest
) -> Tuple[ChatMessage, UsageInfo]:
    generate_func = generate_internvl_chat
    if model_type == ModelType.llama_llava:
        generate_func = generate_llava
    return generate_func(model=model, tokenizer=tokenizer, request=request)


def generate_internvl_chat(
    *, model: Any, tokenizer: Any, request: ChatCompletionRequest
) -> Tuple[ChatMessage, UsageInfo]:
    tools_or_functions, tool_func_choice = analyze_tools_and_tool_choice(request)

    prompt_token_ids = prepare_messages_for_inference(
        tokenizer=tokenizer,
        messages=request.messages,
        tools_or_functions=tools_or_functions,
        tool_choice=tool_func_choice,
        device=model.device,
    )
    input_ids = prompt_token_ids.unsqueeze(0)
    attention_mask = torch.ones_like(input_ids).to(model.device)
    images = extract_images_from_messages(
        [message.dict() for message in request.messages]
    )
    input_ids, attention_mask, _, pixel_values, _ = model.expand_input_ids(
        input_ids, None, attention_mask, images, training=False
    )

    prompt_template = get_prompt_template_from_tokenizer(tokenizer)
    eos_token_ids = [
        tokenizer.convert_tokens_to_ids(tok)
        for tok in prompt_template.get_stop_tokens_for_generation()
    ]

    generation_config = dict(
        max_new_tokens=2048,
        do_sample=False,
        eos_token_id=eos_token_ids,
        temperature=request.temperature,
    )

    generation_output = model.generate(
        pixel_values=pixel_values,
        input_ids=input_ids,
        attention_mask=attention_mask,
        **generation_config
    )

    response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
    assistant_response = prompt_template.parse_assistant_response(response)
    usage = UsageInfo(
        prompt_tokens=input_ids.shape[-1],
        completion_tokens=generation_output.shape[-1],
        total_tokens=input_ids.shape[-1] + generation_output.shape[-1],
    )
    return ChatMessage(**assistant_response), usage


def generate_llava(
    *, model: Any, tokenizer: Any, request: ChatCompletionRequest
) -> Tuple[ChatMessage, UsageInfo]:
    from llava.mm_utils import process_images

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
        temperature=request.temperature,
        max_new_tokens=1024,
    )

    text_output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0]

    result = prompt_template.parse_assistant_response(text_output, tool_choice="auto")
    return ChatMessage(**result), UsageInfo()
