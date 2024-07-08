from functionary.openai_types import ChatMessage, Function, FunctionCall, Tool, ChatInput
from functionary.prompt_template import get_prompt_template_from_tokenizer
from functionary.prompt_template.prompt_utils import prepare_messages_for_inference, get_images_from_messages
from typing import Any, List, Optional
from transformers import StoppingCriteriaList, StoppingCriteria
from llava.mm_utils import process_images
import torch
from functionary.inference_utils import analyze_tools_and_tool_choice, StopWordsCriteria


def generate_message(
    *,
    model: Any,
    tokenizer: Any,
    request: ChatInput
) -> ChatMessage:
    tools_or_functions, tool_func_choice = analyze_tools_and_tool_choice(request)

    prompt_token_ids = prepare_messages_for_inference(
        tokenizer=tokenizer,
        messages=request.messages,
        tools_or_functions=tools_or_functions,
        tool_choice=tool_func_choice,
        device=model.device
    )
    
    stop_token_ids = []
    prompt_template = get_prompt_template_from_tokenizer(tokenizer)
    for stop_tok in prompt_template.get_stop_tokens_for_generation():
        tok_ids = tokenizer.encode(stop_tok, add_special_tokens=False)
        stop_token_ids.append(tok_ids)
    
    images = get_images_from_messages(request.messages)
    image_tensor, image_sizes = None, None
    image_processor = model.get_vision_tower().image_processor
    if images:
        image_tensor = process_images(images, image_processor, model.config)
        image_tensor = [_image.to(dtype=torch.float16, device=model.device) for _image in image_tensor]
        image_sizes = [image.size for image in images]

    stopping_criteria = StoppingCriteriaList([StopWordsCriteria(stops=stop_token_ids)])
    generate_ids = model.generate(
        prompt_token_ids,
        max_new_tokens=request.max_new_tokens,
        temperature=0.001 if request.temperature == 0 else request.temperature,
        stopping_criteria=stopping_criteria,
    )
    token_ids = generate_ids[:, prompt_token_ids.shape[-1] :][0].tolist()

    generated_content = tokenizer.decode(
        token_ids,
        images=image_tensor,
        image_sizes=image_sizes,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    ).strip()
    result = prompt_template.parse_assistant_response(generated_content)
    return ChatMessage(**result)
