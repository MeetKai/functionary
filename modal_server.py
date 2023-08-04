from functools import lru_cache
from fastapi import FastAPI
from inference import generate_models
from openai_types import ChatCompletion, ChatInput, Choice

import modal

stub = modal.Stub("functionary")
app = FastAPI(title="Functionary API")

MODEL = "musabgultekin/functionary-7b-v1"
LOADIN8BIT = False


def get_model():
    # this is lazy should be using the modal model class
    import torch
    from transformers import LlamaTokenizer, LlamaForCausalLM

    model = LlamaForCausalLM.from_pretrained(
        MODEL,
        low_cpu_mem_usage=True,
        device_map="auto",
        torch_dtype=torch.float16,
        load_in_8bit=LOADIN8BIT,
    )
    tokenizer = LlamaTokenizer.from_pretrained(MODEL, use_fast=False)
    return model, tokenizer


def download_model():
    get_model()


image = (
    modal.Image.debian_slim()
    .pip_install(
        "fastapi",
        "pydantic",
        "transformers",
        "sentencepiece",
        "torch",
        "bitsandbytes>=0.39.0",
        "accelerate",
        "einops",
        "scipy",
        "numpy",
    )
    .run_function(download_model)
)


@app.post("/v1/chat/completions", response_model=ChatCompletion)
async def chat_endpoint(chat_input: ChatInput):
    model, tokenizer = get_model()

    response_message = generate_models(
        messages=chat_input.messages,
        functions=chat_input.functions,
        temperature=chat_input.temperature,
        model=model,  # type: ignore
        tokenizer=tokenizer,
    )

    return ChatCompletion(choices=[Choice.from_message(response_message)])


@stub.function(image=image, gpu="A100")
@modal.asgi_app()
def fastapi_app():
    return app
