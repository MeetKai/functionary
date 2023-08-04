import uuid
from fastapi import FastAPI
from transformers import LlamaTokenizer, LlamaForCausalLM
import torch
from inference import generate_message
import uvicorn
import argparse

from openai_types import ChatCompletion, ChatInput, Choice

app = FastAPI(title="Functionary API")


@app.post("/v1/chat/completions", response_model=ChatCompletion)
async def chat_endpoint(chat_input: ChatInput):
    request_id = str(uuid.uuid4())
    response_message = generate_message(
        messages=chat_input.messages,
        functions=chat_input.functions,
        temperature=chat_input.temperature,
        model=model,  # type: ignore
        tokenizer=tokenizer,
    )

    return ChatCompletion(
        id=request_id, choices=[Choice.from_message(response_message)]
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Functionary API Server")
    parser.add_argument(
        "--model",
        type=str,
        default="musabgultekin/functionary-7b-v1",
        help="Model name",
    )
    parser.add_argument("--load_in_8bit", type=bool, default=False)
    args = parser.parse_args()

    model = LlamaForCausalLM.from_pretrained(
        args.model,
        low_cpu_mem_usage=True,
        device_map="auto",
        torch_dtype=torch.float16,
        load_in_8bit=args.load_in_8bit,
    )
    tokenizer = LlamaTokenizer.from_pretrained(args.model, use_fast=False)

    uvicorn.run(app, host="0.0.0.0", port=8000)
