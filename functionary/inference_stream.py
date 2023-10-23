import gc
import re
from typing import Any, AsyncGenerator, Dict, Generator, List, Optional, Tuple

import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from functionary.inference import prepare_messages_for_inference
from functionary.openai_types import ChatMessage, Function
from functionary.prompt import EndToken, StartToken


def prepare_logits_processor(
    temperature: float, repetition_penalty: float, top_p: float, top_k: int
) -> LogitsProcessorList:
    processor_list = LogitsProcessorList()
    # TemperatureLogitsWarper doesn't accept 0.0, 1.0 makes it a no-op so we skip two cases.
    if temperature >= 1e-5 and temperature != 1.0:
        processor_list.append(TemperatureLogitsWarper(temperature))
    if repetition_penalty > 1.0:
        processor_list.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
    if 1e-8 <= top_p < 1.0:
        processor_list.append(TopPLogitsWarper(top_p))
    if top_k > 0:
        processor_list.append(TopKLogitsWarper(top_k))
    return processor_list


def generate_text_stream(
    model: LlamaForCausalLM,
    tokenizer: LlamaTokenizer,
    messages: List[ChatMessage],
    functions: Optional[List[Function]] = None,
    temperature: float = 0.7,
    max_new_tokens=256,
    stop_token_ids=[],
    **kwargs,
) -> Generator[Tuple[str, Optional[str]], Any, Any]:
    if hasattr(model, "device"):
        device = model.device
    else:
        device = "cuda:0"
    repetition_penalty = float(kwargs.get("repetition_penalty", 1.0))
    top_p = float(kwargs.get("top_p", 1.0))
    top_k = int(kwargs.get("top_k", -1))  # -1 means disable
    _stop_token_ids = list(stop_token_ids)
    if tokenizer.eos_token_id not in _stop_token_ids:
        _stop_token_ids.append(tokenizer.eos_token_id)

    logits_processor = prepare_logits_processor(temperature, repetition_penalty, top_p, top_k)
    input_ids = prepare_messages_for_inference(
        tokenizer=tokenizer, messages=messages, functions=functions, device=device
    )
    output_ids = input_ids.clone().detach()
    past_key_values = None  # KV cached
    token_ts = None  # next token
    finish_reason = None
    reach_stop_token = False
    words = ""
    for i in range(max_new_tokens):
        if i == 0:  # prefill
            out = model(input_ids, use_cache=True)
        else:  # decoding
            out = model(
                input_ids=token_ts,
                use_cache=True,
                past_key_values=past_key_values,
            )
        logits = out.logits
        past_key_values = out.past_key_values

        if logits_processor:
            if repetition_penalty > 1.0:
                tmp_output_ids = torch.as_tensor([output_ids], device=logits.device)
            else:
                tmp_output_ids = None
            last_token_logits = logits_processor(tmp_output_ids, logits[:, -1, :])[0]
        else:
            last_token_logits = logits[0, -1, :]

        if temperature < 1e-5 or top_p < 1e-8:  # greedy
            _, indices = torch.topk(last_token_logits, 2)
            tokens = [int(index) for index in indices.tolist()]
        else:
            probs = torch.softmax(last_token_logits, dim=-1)
            indices = torch.multinomial(probs, num_samples=2)
            tokens = [int(token) for token in indices.tolist()]
        token_int = tokens[0]
        token_ts = torch.as_tensor([[token_int]], device=device)
        current_output_text = tokenizer.decode(
            output_ids[0].tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        output_ids = torch.cat((output_ids, token_ts), 1)
        next_output_text = tokenizer.decode(
            output_ids[0].tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        output = next_output_text[len(current_output_text) :]
        words += output
        if token_int in _stop_token_ids:
            reach_stop_token = True
            break
        yield (output, finish_reason)

    # Finish stream event, which contains finish reason
    if reach_stop_token:
        finish_reason = "stop"
    else:
        finish_reason = "length"
    yield ("", finish_reason)

    # Clean
    del past_key_values, out
    gc.collect()
    torch.cuda.empty_cache()


def generate_with_check_stop(
    generator: Generator[Tuple[int, str, Optional[str]], Any, Any], stop_list: List[List[int]]
) -> Generator[Tuple[str, Optional[str]], Any, Any]:
    max_leng = max([len(stop) for stop in stop_list])
    temp_list: List[
        Tuple[int, str, Optional[str]]
    ] = []  # buffer of tokens; len(temp_list) <= max_leng, will yield a token if len(temp_list) == max_leng + 1

    def check_stop_criteria():
        for stop in stop_list:
            if len(temp_list) >= len(stop):
                token_ids = [
                    item[0] for item in temp_list[-len(stop) :]
                ]  # get sequence of token_ids to check; item[0] is token_id
                # print(f"check: {token_ids} vs {stop}")
                if token_ids == stop:
                    return True, stop
        return False, None

    for item in generator:
        print("gen item: ", item)
        temp_list.append(item)
        stop_now, stop = check_stop_criteria()
        if stop_now:
            temp_list = temp_list[: -len(stop)]
            # change finish_reason=stop if it is stopped if not the finish_reason is still None
            if len(temp_list) > 0:
                last_item = temp_list[-1]
                new_item = (last_item[0], last_item[1], "stop")
                temp_list[-1] = new_item
            break
        if len(temp_list) == max_leng + 1:
            return_item = temp_list.pop(0)
            yield return_item[1:]

    for return_item in temp_list:
        yield return_item[1:]


def update_response_state_from_delta_text(
    cur_text: str, func_name: Optional[str], response_type: Optional[str], delta_text: str, finish_reason: Optional[str]
) -> Tuple[Optional[str], Optional[str], Optional[Dict]]:
    cur_text += delta_text
    response: Optional[Dict[str, Any]] = None
    if response_type is None:
        if cur_text.strip().startswith(StartToken.function.value):  # if text_response
            if cur_text.endswith(":"):
                f_index = cur_text.find(StartToken.function.value)
                func_name = cur_text[f_index + len(StartToken.function.value): -1].strip()
                response = {
                    "delta": {
                        "role": "assistant",
                        "content": None,
                        "function_call": {"arguments": "", "name": func_name},
                    },
                    "finish_reason": None,
                }
                response_type = "function"
        else:  # if function_response
            response_type = "text"
            response = {"delta": {"content": "", "role": "assistant"}, "finish_reason": None}
            
    elif response_type == "function":
        if finish_reason is None:
            response = {
                "delta": {
                    "role": "assistant",
                    "function_call": {"arguments": delta_text},
                },  # format of openAI at the second return, don't need to add function_name
                "finish_reason": None,
            }
        else:
            response = {
                "delta": {},
                "finish_reason": "function_call",
            }  # format of openAI at the end, delta must be empty
            
    elif response_type == "text":
        if finish_reason is None:
            # need to check if call a function or not
            if cur_text.endswith(StartToken.function.value):  # if call another function
                print("call another function in the mean time")
                cur_text = StartToken.function.value
                response_type = None
            else:
                response = {"delta": {"content": delta_text, "role": "assistant"}, "finish_reason": None}
        else:  # finish generating
            response = {
                "delta": {},
                "finish_reason": finish_reason,
            }  # format of openAI at the end, delta must be empty
    return func_name, response_type, response, cur_text


def generate_openai_format_from_stream(
    generator: Generator[Tuple[str, Optional[str]], Any, Any]
) -> Generator[Dict, Any, Any]:
    cur_text = ""
    func_name = None
    response_type = None  # = function if it is function call; = text if it is chit-chat
    for delta_text, finish_reason in generator:
        #print(f"delta_text: {delta_text}, finish_reason: {finish_reason}")
        # print(f"item:{item}, finish_reason: {finish_reason}; response_type: {response_type}")
        func_name, response_type, response, cur_text = update_response_state_from_delta_text(
            cur_text, func_name, response_type, delta_text, finish_reason
        )
        if response is not None:
            yield response


def generate_stream(
    model: LlamaForCausalLM,
    tokenizer: LlamaTokenizer,
    messages: List[ChatMessage],
    functions: Optional[List[Function]] = None,
    temperature: float = 0.7,
    max_new_tokens=256,
    **kwargs,
) -> Generator[Dict, Any, Any]:
    stop_tokens = [EndToken.assistant, EndToken.function_call]
    stop_token_lists = []
    for stop in stop_tokens:
        token_ids = tokenizer.encode(stop)
        stop_token_lists.append(token_ids[-1])
    generator = generate_text_stream(
        model=model,
        tokenizer=tokenizer,
        messages=messages,
        functions=functions,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        stop_token_ids=stop_token_lists,
        **kwargs,
    )
    #checked_generator = generate_with_check_stop(generator, stop_tokens_list)
    for item in generate_openai_format_from_stream(generator):
        yield item


async def generate_openai_format_from_stream_async(
    generator: AsyncGenerator[Tuple[str, Optional[str]], None]
) -> AsyncGenerator[Dict, None]:
    cur_text = ""
    func_name = None
    response_type = None  # = function if it is function call; = text if it is chit-chat
    async for delta_text, finish_reason in generator:
        #""print(f"delta_text:{delta_text}, finish_reason: {finish_reason}; response_type:{response_type}")
        func_name, response_type, response, cur_text = update_response_state_from_delta_text(
            cur_text, func_name, response_type, delta_text, finish_reason
        )
        if response is not None:
            yield response
