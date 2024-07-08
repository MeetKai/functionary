import argparse
import json
import uuid
from typing import Union

import torch
import uvicorn
from fastapi import FastAPI

from transformers import AutoTokenizer
from functionary.train_vision.models.modeling_llava import FixedLlavaLlamaForCausalLM as LlavaLlamaForCausalLM
from functionary.inference_vision import generate_message
from functionary.openai_types import (ChatCompletion, ChatCompletionChunk,
                                      ChatInput, Choice, StreamChoice)

app = FastAPI(title="Functionary API")


@app.post("/v1/chat/completions")
async def chat_endpoint(chat_input: ChatInput):
    request_id = str(uuid.uuid4())
    if not chat_input.stream:
        response_message = generate_message(
            model=model,  # type: ignore
            tokenizer=tokenizer,  
            request=chat_input      
        )
        finish_reason = "stop"
        if response_message.function_call is not None:
            finish_reason = "function_call"  # need to add this to follow the format of openAI function calling
        result = ChatCompletion(
            id=request_id,
            choices=[Choice.from_message(response_message, finish_reason)],
        )
        return result.dict(exclude_none=True)
    else:
        raise Exception("streaming it not implemented now")


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
    model = LlavaLlamaForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if args.device == "cpu" else torch.float16,
        use_flash_attention_2=True,
        device_map=args.device
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, legacy=True)
    print(tokenizer)

    uvicorn.run(app, host="0.0.0.0", port=8000)
