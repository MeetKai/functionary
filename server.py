import argparse
import json
import uuid
from typing import Union

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from transformers import AutoModelForCausalLM, AutoTokenizer

from functionary.inference import generate_message
from functionary.inference_stream import generate_stream
from functionary.openai_types import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatInput,
    Choice,
    ModelCard,
    ModelList,
    ModelPermission,
    StreamChoice,
)

app = FastAPI(title="Functionary API")


@app.get("/v1/models")
async def show_available_models():
    """Show available models. Right now we only have one model."""
    model_cards = [
        ModelCard(
            id="/workspace/functionary-v1-ds3/",
            root="workspace/functionary-v1-ds3/",
            permission=[ModelPermission()],
        )
    ]
    return ModelList(data=model_cards)


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
            device=model.device,
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
                chunk_dic = result.dict(exclude_unset=True)
                chunk_data = json.dumps(chunk_dic, ensure_ascii=False)
                yield f"data: {chunk_data}\n\n"
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
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="choose which device to host the model: cpu, cuda, cuda:xxx, or auto",
    )
    parser.add_argument("--load_in_8bit", type=bool, default=False)
    args = parser.parse_args()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        low_cpu_mem_usage=True,
        device_map=args.device,
        torch_dtype=torch.bfloat16 if args.device == "cpu" else torch.float16,
        load_in_8bit=args.load_in_8bit,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    uvicorn.run(app, host="0.0.0.0", port=80)
