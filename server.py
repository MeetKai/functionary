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
    functions: Optional[List[Dict[str, Any]]]
    plugin_urls : Optional[list[str]] ## cannot use openai client module for this param. coming soon.
    temperature: float = 0.7  # set a default value


@app.post("/v1/chat/completions")
async def chat_endpoint(chat_input: ChatInput):
    generated_message = model.generate( chat_input.messages, chat_input.functions, chat_input.temperature)

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
    parser.add_argument('--system_message', type=str, default=default_SYSTEM_MESSAGE, help="The system message to give to the model.")
    args = parser.parse_args()
    model_name = args.model ## since it is used in this script
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    os.environ['MODEL_NAME'] = model_name
    os.environ['INFERENCE_DEVICE'] = device
    os.environ['SMALL_MEM'] = str(args.preserve_cpu_mem)
    
    model = Model(model_name=args.model, preserve_mem=args.preserve_cpu_mem, system_message=args.system_message, device=device)
    uvicorn.run(app, host="0.0.0.0", port=8000)
