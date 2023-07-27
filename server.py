from fastapi import FastAPI
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import uuid
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from inference import Model, default_SYSTEM_MESSAGE
import uvicorn
import argparse
import os

app = FastAPI()


class ChatInput(BaseModel):
    messages: List[Dict[str, Any]]
    functions: Optional[List[Dict[str, Any]]] = []
    plugin_urls : Optional[List[str]] = [] ## cannot use openai client module for this param. coming soon.
    temperature: float = 0.7  # set a default value


@app.post("/v1/chat/completions")
async def chat_endpoint(chat_input: ChatInput):
    generated_message = model.generate( messages=chat_input.messages, functions=chat_input.functions, plugins=chat_input.plugin_urls, temperature=chat_input.temperature)
    
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
    parser.add_argument('--model', type=str, default='musabgultekin/functionary-7b-v0.2', help='The model name to be used.')
    parser.add_argument('--preserve_cpu_mem', type=bool, default=False, help="If you have a system with low CPU memory (~16gb or under depending on the model being used), then you may want to set '--preserve_cpu_mem True'")
    parser.add_argument('--system_message', type=str, default=default_SYSTEM_MESSAGE, help="The system message to give to the model.")
    parser.add_argument('--use_bitsandbytes', type=bool, help="whether to quantize the model using bitsandbytes. this is particularly useful for systems with low gpu memory. to enable set '--use_bitsandbytes True'")
    args, _ = parser.parse_known_args()  # only parse known arguments

    kwargs = {}
    if args.model:
        kwargs['pretrained_model_name_or_path'] = args.model
    if args.preserve_cpu_mem is not None:
        kwargs['low_cpu_mem_usage'] = str(args.preserve_cpu_mem)
    if args.use_bitsandbytes:
        kwargs['load_in_8bit'] = str(args.use_bitsandbytes)

    model_name = kwargs['pretrained_model_name_or_path']

    model = Model(system_message=args.system_message, model_kwargs=kwargs)
    uvicorn.run(app, host="0.0.0.0", port=8000)

    # Use the model argument locally
    model_name = args.model
