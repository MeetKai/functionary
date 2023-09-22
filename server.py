from typing import Union
import argparse
import uuid

import torch
import uvicorn
from fastapi import FastAPI
from transformers import LlamaTokenizer, LlamaForCausalLM
import json

from functionary.openai_types import ChatCompletion, ChatInput, Choice, StreamChoice, ChatCompletionChunk
from functionary.inference import generate_message
from functionary.inference_stream import generate_stream
from fastapi.responses import StreamingResponse, JSONResponse

app = FastAPI(title="Functionary API")


@app.post("/v1/chat/completions")
async def chat_endpoint(chat_input: ChatInput):
    request_id = str(uuid.uuid4())
    if not chat_input.stream:
        response_message = generate_message(
            messages=chat_input.messages,
            functions=chat_input.functions,
            temperature=chat_input.temperature,
            model=model,  # type: ignore
            tokenizer=tokenizer,
        )
        finish_reason = "stop" 
        if response_message.function_call is not None:
            finish_reason = "function_call"  # need to add this to follow the format of openAI function calling
        result = ChatCompletion(id=request_id, choices=[Choice.from_message(response_message, finish_reason)])
        return result.model_dump(exclude_none=True)
    else:
        response_generator = generate_stream(
            messages=chat_input.messages,
            functions=chat_input.functions,
            temperature=chat_input.temperature,
            model=model,  # type: ignore
            tokenizer=tokenizer,
        )
        def get_response_stream():
            for response in response_generator:
                chunk = StreamChoice(**response)
                result = ChatCompletionChunk(id=request_id, choices=[chunk])
                yield f"data: {result.model_dump_json(exclude_unset=True)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(get_response_stream(), media_type="text/event-stream")


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
