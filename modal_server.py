import uuid
from typing import List

import modal
from fastapi import FastAPI

from functionary.inference import generate_message
from functionary.openai_types import (
    ChatCompletion,
    ChatInput,
    ChatMessage,
    Choice,
    Function,
)

stub = modal.Stub("functionary")
app = FastAPI(title="Functionary API")

MODEL = "musabgultekin/functionary-7b-v1"
LOADIN8BIT = False


def get_model():
    # this is lazy should be using the modal model class
    import torch
    from transformers import LlamaForCausalLM, LlamaTokenizer

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


@stub.cls(gpu="A100", image=image)
class Model:
    def __enter__(self):
        model, tokenizer = get_model()
        self.model = model
        self.tokenizer = tokenizer

    @modal.method()
    def generate(
        self,
        messages: List[ChatMessage],
        functions: List[Function],
        temperature: float,
    ):
        return generate_message(
            messages=messages,
            functions=functions,
            temperature=temperature,
            model=self.model,  # type: ignore
            tokenizer=self.tokenizer,
        )


@app.post("/v1/chat/completions", response_model=ChatCompletion)
async def chat_endpoint(chat_input: ChatInput):
    request_id = str(uuid.uuid4())
    model = Model()

    response_message = model.generate.call(
        messages=chat_input.messages,
        functions=chat_input.functions,
        temperature=chat_input.temperature,
    )

    return ChatCompletion(
        id=request_id, choices=[Choice.from_message(response_message)]
    )


@stub.function(image=image)
@modal.asgi_app()
def fastapi_app():
    return app
