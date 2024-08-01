import argparse
import json
import uuid
from typing import Union

import torch
import uvicorn
from fastapi import FastAPI
import sys

from transformers import AutoTokenizer, AutoConfig

# from functionary.train_vision.models.modeling_llava import FixedLlavaLlamaForCausalLM as LlavaLlamaForCausalLM
from functionary.inference_vision import generate, ModelType
from functionary.openai_types import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionRequest,
    Choice,
    StreamChoice,
    ChatCompletionResponse,
)
from functionary.prompt_template import get_prompt_template_from_tokenizer
from typing import Any, Dict
import math

app = FastAPI(title="Functionary API")


@app.post("/v1/chat/completions")
async def chat_endpoint(chat_input: ChatCompletionRequest):
    request_id = str(uuid.uuid4())
    if not chat_input.stream:
        response_message, usage = generate(
            model_type=model_type, model=model, tokenizer=tokenizer, request=chat_input  # type: ignore
        )
        finish_reason = "stop"
        if response_message.tool_calls is not None:
            finish_reason = "tool_calls"  # need to add this to follow the format of openAI function calling
        result = ChatCompletion(
            id=request_id,
            choices=[Choice.from_message(response_message, finish_reason)],
        )
        return result.dict(exclude_none=True)
    else:
        raise Exception("streaming it not implemented now")


def store_special_tokens_to_model(tokenizer: Any, model: Any) -> None:
    prompt_template = get_prompt_template_from_tokenizer(tokenizer)
    start_img_token_id = tokenizer.convert_tokens_to_ids(
        prompt_template.start_img_token
    )
    end_img_token_id = tokenizer.convert_tokens_to_ids(prompt_template.end_img_token)
    img_context_token_id = tokenizer.convert_tokens_to_ids(prompt_template.img_context)
    img_place_holder_token = tokenizer.convert_tokens_to_ids(
        prompt_template.start_img_token
    )

    model.img_start_token = start_img_token_id
    model.img_end_token = end_img_token_id
    model.img_context_token = img_context_token_id
    model.img_place_holder_token = img_place_holder_token


def get_model_type(pretrained_path: str) -> ModelType:
    try:
        config = AutoConfig.from_pretrained(pretrained_path, trust_remote_code=True)
        if config.model_type == "internvl_chat":
            return ModelType.internvl_chat
        else:
            print(
                "COULD NOT DETERMINE THE MODEL TYPE, CURRENTLY WE ONLY SUPPORT: llava_llama and internvl_chat"
            )
            sys.exit(1)
    except:  # will encounter exception if the model is llama_llava
        return ModelType.llama_llava


def split_model(pretrained_path: str) -> Dict:
    config = AutoConfig.from_pretrained(pretrained_path, trust_remote_code=True)
    num_layers = config.llm_config.num_hidden_layers
    device_map = {}
    world_size = torch.cuda.device_count()
    # num_layers = {
    #     'InternVL2-1B': 24, 'InternVL2-2B': 24, 'InternVL2-4B': 32, 'InternVL2-8B': 32,
    #     'InternVL2-26B': 48, 'InternVL2-40B': 60, 'InternVL2-Llama3-76B': 80}[f"InternVL2-{model_size}B"]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f"language_model.model.layers.{layer_cnt}"] = i
            layer_cnt += 1
    device_map["vision_model"] = 0
    device_map["mlp1"] = 0
    device_map["language_model.model.tok_embeddings"] = 0
    device_map["language_model.model.embed_tokens"] = 0
    device_map["language_model.output"] = 0
    device_map["language_model.model.norm"] = 0
    device_map["language_model.lm_head"] = 0
    device_map[f"language_model.model.layers.{num_layers - 1}"] = 0

    return device_map


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Functionary API Server")
    parser.add_argument(
        "--model",
        type=str,
        default="musabgultekin/functionary-7b-v1",
        help="Model name",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="choose which device to host the model: cpu, cuda, cuda:xxx, or auto",
    )
    parser.add_argument("--load_in_8bit", type=bool, default=False)
    args = parser.parse_args()
    model_type = get_model_type(args.model)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, legacy=True, trust_remote_code=True
    )
    if model_type == ModelType.llama_llava:
        from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM

        model = LlavaLlamaForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16 if args.device == "cpu" else torch.float16,
            use_flash_attention_2=True,
            device_map=args.device,
        )
    else:
        from functionary.train_vision.models.modeling_internvl.modeling_internvl_chat import (
            InternVLChatModel,
        )

        device_map = args.device
        if args.device == "auto":
            device_map = split_model(args.model)

        model = InternVLChatModel.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16 if args.device == "cpu" else torch.float16,
            trust_remote_code=True,
            device_map=device_map,
        )

        store_special_tokens_to_model(tokenizer, model)

    model.eval()

    uvicorn.run(app, host="0.0.0.0", port=8000)
