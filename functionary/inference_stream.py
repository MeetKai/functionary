from transformers import LlamaForCausalLM, LlamaTokenizer
from typing import List, Optional, Tuple, Generator, Any, Dict
import gc
import torch
import re
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from functionary.openai_types import ChatMessage, Function
from functionary.inference import prepare_messages_for_inference


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
    **kwargs,
) -> Generator[Tuple[int, str, Optional[str]], Any, Any]:
    if hasattr(model, "device"):
        device = model.device
    else:
        device = "cuda:0"
    repetition_penalty = float(kwargs.get("repetition_penalty", 1.0))
    top_p = float(kwargs.get("top_p", 1.0))
    top_k = int(kwargs.get("top_k", -1))  # -1 means disable
    stop_token_ids = []
    if tokenizer.eos_token_id not in stop_token_ids:
        stop_token_ids.append(tokenizer.eos_token_id)

    logits_processor = prepare_logits_processor(temperature, repetition_penalty, top_p, top_k)
    input_ids = prepare_messages_for_inference(tokenizer=tokenizer, messages=messages, functions=functions)
    input_ids = input_ids.to(device)
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
        if token_int in stop_token_ids:
            reach_stop_token = True
            break
        yield (token_int, output, finish_reason)

    # Finish stream event, which contains finish reason
    if reach_stop_token:
        finish_reason = "stop"
    else:
        finish_reason = "lenghth"

    yield (token_int, "", finish_reason)

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


def generate_stream(
    model: LlamaForCausalLM,
    tokenizer: LlamaTokenizer,
    messages: List[ChatMessage],
    functions: Optional[List[Function]] = None,
    temperature: float = 0.7,
    max_new_tokens=256,
    **kwargs,
) -> Generator[Dict, Any, Any]:
    generator = generate_text_stream(
        model=model,
        tokenizer=tokenizer,
        messages=messages,
        functions=functions,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        **kwargs,
    )
    stop_list = kwargs.get("stops", [])
    stop_list.append("\n user:\n")
    stop_tokens_list = [tokenizer.encode(stop, add_special_tokens=False) for stop in stop_list]
    # We need to remove 29871 because sometimes Llamatokenizer automatically add: 29871
    # (take a loot at this: https://github.com/huggingface/transformers/issues/26273)
    stop_tokens_list = [item[1:] if len(item) > 1 and item[0] == 29871 else item for item in stop_tokens_list]
    checked_generator = generate_with_check_stop(generator, stop_tokens_list)

    cur_text = ""
    func_name = None
    response_type = None  # = function if it is function call; = text if it is chit-chat
    response: Dict[str, Any] = {}
    for item, finish_reason in checked_generator:
        # print(f"item:{item}, finish_reason: {finish_reason}; response_type: {response_type}")
        cur_text += item
        if response_type is None:
            if cur_text.lstrip() == ":\n":
                response_type = "text"
                response = {"delta": {"content": "", "role": "assistant"}, "finish_reason": None}
                yield response
            else:
                match = re.search(r"to=functions\.(?P<f>.+?):", cur_text.strip())
                if match is not None:
                    response_type = "function"
                    func_name = match.group("f").strip()
                    response = {
                        "delta": {
                            "role": "assistant",
                            "content": None,
                            "function_call": {"arguments": "", "name": func_name},
                        },
                        "finish_reason": None,
                    }
                    yield response
        elif response_type == "function":
            if finish_reason is None:
                response = {
                    "delta": {"role": "assistant", "function_call": {"arguments": item}},  # format of openAI at the second return, don't need to add function_name
                    "finish_reason": None,
                }
            else:
                response = {"delta": {}, "finish_reason": "function_call"}  # format of openAI at the end, delta must be empty
            yield response
        elif response_type == "text":
            if finish_reason is None:
                response = {"delta": {"content": item, "role": "assistant"}, "finish_reason": None}
            else:
                response = {"delta": {}, "finish_reason": finish_reason}  # format of openAI at the end, delta must be empty
            yield response
