from http import HTTPStatus
import os
import time
import uuid
from typing import Annotated, Dict, List, Literal, Optional, Union

from fastapi.responses import JSONResponse
import modal
from fastapi import FastAPI, Header, Request
from pydantic import BaseModel, Field, field_validator, validator

from functionary.openai_types import (
    ChatMessage,
    Function,
    Tool,
)



app = modal.App("functionary_vllm")
fast_api_app = FastAPI(title="Functionary API")

#MODEL = "meetkai/functionary-small-v2.4"
MODEL = "meetkai/functionary-medium-v2.4"
MAX_MODEL_LENGTH = 8196
LOADIN8BIT = False
#GPU_CONFIG = modal.gpu.L4(count=1)
GPU_CONFIG = modal.gpu.A100(memory=80, count=2)
MODEL_DIR = "/model2"
GRAMMAR_SAMPLING = False

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    functions: Optional[List[Function]] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Union[str, Tool]] = Field(default = None, validate_default=True)
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    max_tokens: Optional[int] = 512
    stop: Optional[Union[str, List[str]]] = Field(default_factory=list)
    stream: Optional[bool] = False
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    # Additional parameters supported by vLLM
    best_of: Optional[int] = None
    top_k: Optional[int] = -1
    ignore_eos: Optional[bool] = False
    use_beam_search: Optional[bool] = False

    @field_validator("tool_choice")
    def validate_tool_choice(cls, value, values):
        if value is None:
            if values["tools"] is None and values["functions"] is None:
                return "none"
            else:
                return "auto"
        return value


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[
        Literal["stop", "length", "function_call", "tool_calls"]
    ] = None

class UsageInfo(BaseModel):
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_tokens:Optional[int]

class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4()}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage:UsageInfo

class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length", "function_call"]] = None


class ChatCompletionStreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4()}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseStreamChoice]


def download_model_to_image(model_dir, model_name):
    from huggingface_hub import snapshot_download
    from transformers.utils import move_cache

    os.makedirs(model_dir, exist_ok=True)

    snapshot_download(
        model_name,
        local_dir=model_dir,
        ignore_patterns=["*.pt", "*.bin"],  # Using safetensors
    )
    move_cache()

def get_model():
    # this is lazy should be using the modal model class
    import torch   
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.transformers_utils.tokenizer import get_tokenizer
    import asyncio
    from vllm.engine.arg_utils import AsyncEngineArgs

    engine_args = AsyncEngineArgs(
            model=MODEL_DIR,
            tensor_parallel_size=GPU_CONFIG.count,
            gpu_memory_utilization=0.90,
            enforce_eager=False,  # capture the graph for faster inference, but slower cold starts
            disable_log_stats=True,  # disable logging so we can stream tokens
            disable_log_requests=True,
            max_model_len = MAX_MODEL_LENGTH
        )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    engine_model_config = asyncio.run(engine.get_model_config())

    # A separate tokenizer to map token IDs to strings.
    tokenizer = get_tokenizer(
        engine_args.tokenizer, tokenizer_mode=engine_args.tokenizer_mode
    )
    return engine, tokenizer, engine_model_config


def download_model():
    get_model()


image = (
    modal.Image.debian_slim()
    .pip_install(
        "transformers==4.37.2",
        "accelerate~=0.21.0",
        "sentencepiece~=0.1.99",
        "fastapi~=0.104.0",
        "uvicorn~=0.23.1",
        "pydantic~=2.6.0",
        "scipy~=1.11.1",
        "jsonref~=1.1.0",
        "requests~=2.31.0",
        "PyYAML~=6.0.1",
        "typer==0.9.0",
        "protobuf==3.20.0",
        "tokenizers==0.15.0",
        "vllm==0.3.0",
        "hf-transfer==0.1.6",
        "huggingface_hub==0.22.2",
    )
    .run_function(download_model_to_image,
        timeout=60 * 20,
        kwargs={
            "model_dir": MODEL_DIR,
            "model_name": MODEL
        },
        secrets=[modal.Secret.from_name("huggingface")])
)
def create_error_response(status_code: HTTPStatus, message: str) -> JSONResponse:
    from vllm.entrypoints.openai.protocol import (ErrorResponse)   
    return JSONResponse(
        ErrorResponse(message=message, type="invalid_request_error").dict(),
        status_code=status_code.value,
    )

async def check_length(request, input_ids, model_config):
    if hasattr(model_config.hf_config, "max_sequence_length"):
        context_len = model_config.hf_config.max_sequence_length
    elif hasattr(model_config.hf_config, "seq_length"):
        context_len = model_config.hf_config.seq_length
    elif hasattr(model_config.hf_config, "max_position_embeddings"):
        context_len = model_config.hf_config.max_position_embeddings
    elif hasattr(model_config.hf_config, "seq_length"):
        context_len = model_config.hf_config.seq_length
    else:
        context_len = 4096

    token_num = len(input_ids)

    if token_num + request.max_tokens > context_len:
        return create_error_response(
            HTTPStatus.BAD_REQUEST,
            f"This model's maximum context length is {context_len} tokens. "
            f"However, you requested {request.max_tokens + token_num} tokens "
            f"({token_num} in the messages, "
            f"{request.max_tokens} in the completion). "
            f"Please reduce the length of the messages or completion.",
        )
    else:
        return None

@app.cls(gpu=GPU_CONFIG,timeout=60 * 10,
    container_idle_timeout=60 * 10,
    allow_concurrent_inputs=10, image=image)
class Model:
    @modal.enter()
    def start_engine(self):        
        model, tokenizer,engine_model_config = get_model()
        self.model = model
        self.tokenizer = tokenizer
        self.engine_model_config = engine_model_config

    @modal.method()
    async def generate(self, request:ChatCompletionRequest):
        from vllm.sampling_params import SamplingParams
        from functionary.inference import enforce_tool_choice, prepare_messages_for_inference
        from functionary.inference_stream import generate_openai_format_from_stream_async
        from vllm.utils import random_uuid
        from functionary.prompt_template import (      
            PredefinedFuncTypes,      
            get_prompt_template_from_tokenizer,
        )
        from vllm.outputs import RequestOutput
        from functionary.prompt_template.prompt_template_v2 import get_random_tool_call_id

        

        if request.tools:
            tools = enforce_tool_choice(
                tool_choice=request.tool_choice, tools=request.tools
            )
            tools_or_functions = [item.dict() for item in tools]
        elif request.functions:
            tools = None
            tools_or_functions = [item.dict() for item in request.functions]
        else:
            tools = None
            tools_or_functions = []
        # Remove any code_interpreter tools remaining
        if tools:
            tools = [tool for tool in tools if tool.type != "code_interpreter"]
            tools_or_functions = [
                tool for tool in tools_or_functions if tool["type"] != "code_interpreter"
            ]
        prompt_token_ids = prepare_messages_for_inference(
                        tokenizer=self.tokenizer,
                        messages=request.messages,
                        functions=request.functions,
                        tools=tools,
                        tool_choice=request.tool_choice,
                    ).tolist()[0]
        error_check_ret = await check_length(request, prompt_token_ids, self.engine_model_config)
        if error_check_ret is not None:
            return error_check_ret

        model_name = request.model
        request_id = f"cmpl-{random_uuid()}"
        created_time = int(time.time())

        # compute stop_token_ids
        stop_token_ids = []
        prompt_template = get_prompt_template_from_tokenizer(self.tokenizer)
        for stop_tok in prompt_template.get_stop_tokens_for_generation():
            tok_ids = self.tokenizer.encode(stop_tok, add_special_tokens=False)
            stop_token_ids.append(tok_ids[-1])

        try:
            sampling_params = SamplingParams(
                n=request.n,
                presence_penalty=request.presence_penalty,
                frequency_penalty=request.frequency_penalty,
                temperature=request.temperature,
                top_p=request.top_p,
                stop=request.stop,
                stop_token_ids=stop_token_ids,
                max_tokens=request.max_tokens,
                best_of=request.best_of,
                top_k=request.top_k,
                ignore_eos=request.ignore_eos,
                use_beam_search=request.use_beam_search,
                skip_special_tokens=False,
                logprobs=len(self.tokenizer.vocab.keys()),
            )
        except ValueError as e:
            return create_error_response(HTTPStatus.BAD_REQUEST, str(e))
        

        if GRAMMAR_SAMPLING:
            result_generator = self.model.generate(
                prompt=None,
                sampling_params=sampling_params,
                request_id=request_id,
                prompt_token_ids=prompt_token_ids,
                tools_or_functions=tools_or_functions,
                prompt_template_cls=prompt_template,
                tool_choice=request.tool_choice,
            )
        else:
            result_generator = self.model.generate(
                prompt=None,
                sampling_params=sampling_params,
                request_id=request_id,
                prompt_token_ids=prompt_token_ids,
            )
        
        # Non-streaming response
        final_res: RequestOutput = None
        async for res in result_generator:            
            final_res = res

        assert final_res is not None
        choices = []
        for output in final_res.outputs:
            text_response = output.text.strip()
            chat_mess = prompt_template.parse_assistant_response(
                llm_output=text_response, tool_choice=request.tool_choice
            )  # parse_generated_content(text_response)

            # Postprocess finish reason
            if "function_call" in chat_mess and chat_mess["function_call"] is not None:
                output.finish_reason = "function_call"
            if "tool_calls" in chat_mess and chat_mess["tool_calls"] is not None:
                output.finish_reason = "tool_calls"

            # Convert v1 from function_call to tool_calls if tools are provided instead of functions
            if (
                prompt_template.version == "v1"
                and output.finish_reason == "function_call"
                and (request.functions is None or len(request.functions) == 0)
            ):
                chat_mess = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "function": {
                                "name": chat_mess["function_call"]["name"],
                                "arguments": chat_mess["function_call"]["arguments"],
                            },
                            "id": get_random_tool_call_id(),
                            "type": "function",
                        }
                    ],
                }
                output.finish_reason = "tool_calls"

            choice_data = ChatCompletionResponseChoice(
                index=output.index,
                message=ChatMessage(**chat_mess),
                finish_reason=output.finish_reason,
            )
            choices.append(choice_data)

        num_prompt_tokens = len(final_res.prompt_token_ids)
        num_generated_tokens = sum(len(output.token_ids) for output in final_res.outputs)
        usage = UsageInfo(
            prompt_tokens=num_prompt_tokens,
            completion_tokens=num_generated_tokens,
            total_tokens=num_prompt_tokens + num_generated_tokens,
        )
        response = ChatCompletionResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            choices=choices,
            usage=usage
        )       

        return response
    @modal.exit()
    def stop_engine(self):
        if GPU_CONFIG.count > 1:
            import ray

            ray.shutdown()


async def check_model(request) -> Optional[JSONResponse]:
    if request.model == MODEL:
        return
    ret = create_error_response(
        HTTPStatus.NOT_FOUND,
        f"The model `{request.model}` does not exist.",
    )
    return ret

@fast_api_app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_endpoint(raw_request: Request):
    """Completion API similar to OpenAI's API.
    See  https://platform.openai.com/docs/api-reference/chat/create
    for the API specification. This API mimics the OpenAI ChatCompletion API.
    NOTE: Currently we do not support the following features:
        - logit_bias (to be supported by vLLM engine)
    """

    request_json = await raw_request.json()
    request = ChatCompletionRequest(**request_json)

    error_check_ret = await check_model(request)
    if error_check_ret is not None:
        return error_check_ret

    #if request.logit_bias is not None:
        # TODO: support logit_bias in vLLM engine.
    #    return create_error_response(
    #       HTTPStatus.BAD_REQUEST, "logit_bias is not currently supported"
    #    )

    model = Model()

    return model.generate.remote(request)
    


@app.function(image=image)
@modal.asgi_app()
def fastapi_app():
    return fast_api_app
