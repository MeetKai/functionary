"""
The `Model` class is an interface between the ML model that you're packaging and the model
server that you're running it on.

The main methods to implement here are:
* `load`: runs exactly once when the model server is spun up or patched and loads the
   model onto the model server. Include any logic for initializing your model, such
   as downloading model weights and loading the model into memory.
* `predict`: runs every time the model server is called. Include any logic for model
  inference and return the model output.

See https://truss.baseten.co/quickstart for more.
"""


import time
import logging
import uuid
from typing import Any, Dict

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.protocol import ChatCompletionResponse, ChatCompletionResponseChoice, ChatMessage, UsageInfo
from vllm.inputs import TokensPrompt
from vllm.transformers_utils.tokenizer import get_tokenizer

from vllm import SamplingParams

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_assistant_response(llm_output: str, idx: str, tool_choice: Any = None) -> Dict:
    # first remove stop tokens if there exists
    for stop in ["<|eot_id|>", "<|end_of_text|>"]:
        if llm_output.endswith(stop):
            llm_output = llm_output[: -len(stop)]

    chunks = llm_output.split(">>>")
    chunks = [chunk.strip() for chunk in chunks if len(chunk.strip()) > 0]

    tool_calls = []
    text_content = ""

    for chunk in chunks:
        # format: function_name\narguments<end_of_functioncall>
        index = chunk.find("\n")
        func_name = chunk[:index].strip()
        arguments = chunk[index + 1 :].strip()
        if func_name == "all":
            text_content = arguments
        else:
            tool_calls.append(
                {
                    "function": {"name": func_name, "arguments": arguments},
                    "id": idx,
                    "type": "function",
                }
            )
    if len(tool_calls) == 0:
        tool_calls = None

    return {"role": "assistant", "content": text_content, "tool_calls": tool_calls}


class Model:
    def __init__(self, **kwargs) -> None:
        self.model_args = None
        self.llm_engine = None
        self.tokenizer = None

    def load(self) -> None:
        self.model_name = "meetkai/functionary-medium-v3.2"
        self.load_start_time = time.time()
        self.model_args = AsyncEngineArgs(model=self.model_name, max_model_len=8192, tensor_parallel_size=2)
        self.tokenizer = get_tokenizer(
            self.model_args.tokenizer, tokenizer_mode=self.model_args.tokenizer_mode
        )
        # Overwrite vLLM's default ModelConfig.max_logprobs of 5
        self.model_args.max_logprobs = len(self.tokenizer.vocab.keys())
        self.llm_engine = AsyncLLMEngine.from_engine_args(self.model_args)
        self.load_end_time = time.time()
        cold_start_time = self.load_end_time - self.load_start_time
        logger.info(f"Cold start time: {cold_start_time:.5f} seconds")

    async def predict(self, request: dict) -> Any:
        messages = request.pop("messages", [])
        tools = request.pop("tools", None)
        sampling_params = SamplingParams(**request)
        
        prompt = self.tokenizer.apply_chat_template(
            conversation=messages,
            tools=tools,
            bos_token="",
            add_generation_prompt=True,
            tokenize=False,
        )
        
        total_passes = 30
        avg_ttft = 0.0
        total_tokens = 0
        total_generation_time = 0.0

        for i in range(total_passes):
            idx = str(uuid.uuid4().hex)
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
            input_ids = input_ids.to("cuda").tolist()[0]
            
            generator = self.llm_engine.generate(
                inputs=TokensPrompt(prompt_token_ids=input_ids),
                sampling_params=sampling_params,
                request_id=idx,
            )
            first_token_time = None
            generation_start_time = time.time()
            final_res = None
            async for res in generator:
                if first_token_time is None:
                    first_token_time = time.time()
                final_res = res
            choices = []
            for output in final_res.outputs:
                text_response = output.text.strip()
                chat_mess = parse_assistant_response(text_response, idx)
                if "tool_calls" in chat_mess and chat_mess["tool_calls"]:
                    output.finish_reason = "tool_calls"
                choice_data = ChatCompletionResponseChoice(
                    index=output.index,
                    message=ChatMessage(**chat_mess),
                    finish_reason=output.finish_reason,
                )
                choices.append(choice_data)

            generation_end_time = time.time()
            total_generation_time += generation_end_time - generation_start_time
                
            num_prompt_tokens = len(final_res.prompt_token_ids)
            num_generated_tokens = sum(len(output.token_ids) for output in final_res.outputs)
            usage = UsageInfo(
                prompt_tokens=num_prompt_tokens,
                completion_tokens=num_generated_tokens,
                total_tokens=num_prompt_tokens + num_generated_tokens,
            )
            response = ChatCompletionResponse(
                id=idx,
                model=self.model_name,
                choices=choices,
                usage=usage,
            )
        
            avg_ttft += first_token_time - generation_start_time
            total_tokens += num_generated_tokens
            tokens_per_second = total_tokens / total_generation_time if total_generation_time > 0 else 0
        logger.info(f"Time to first token: {avg_ttft:.5f} seconds")
        logger.info(f"Tokens generated: {total_tokens}")
        logger.info(f"Tokens per second: {tokens_per_second:.5f}")
        
        return response