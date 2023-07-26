from fastapi import FastAPI
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import uuid
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from inference import generate
import uvicorn
import argparse
from functionary_utils import SchemaGen
import os

app = FastAPI()


class ChatInput(BaseModel):
    messages: List[Dict[str, Any]]
    functions: Optional[List[Dict[str, Any]]]
    plugin_urls : Optional[list[str]] ## cannot use openai client module for this param. coming soon.
    temperature: float = 0.7  # set a default value


@app.post("/v1/chat/completions")
async def chat_endpoint(chat_input: ChatInput):
    generated_message = generate(model, tokenizer, chat_input.messages, chat_input.functions, chat_input.temperature)

    return {
        'id': str(uuid.uuid4()),
        'object': 'chat.completion',
        'created': int(time.time()),
        'model': model_name,
        'choices': [
            {
                'message': generated_message,
                'finish_reason': 'stop',
                'index': 0
            }
        ]
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Functionary API Server")
    parser.add_argument('--model', type=str, default='musabgultekin/functionary-7b-v1', help='The model name to be used.')
    parser.add_argument('--preserve_cpu_mem', type=bool, default=False, help="If you have a system with low CPU memory (~16gb or under depending on the model being used), then you may want to set '--preserve_cpu_mem True'")
    args = parser.parse_args()

    model_name = args.model
    device = os.environ['INFERENCE_DEVICE'] =="cuda:0" if torch.cuda.is_available else "cpu"
    if device =="cpu":
        print("using large language models without a GPU is not recommended. if you have a gpu on your system, then there may be a compatibility issue with pytorch and your gpu drivers.")
    model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=args.preserve_cpu_mem, torch_dtype=torch.float16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    uvicorn.run(app, host="0.0.0.0", port=8000)
