from typing import List, Optional

import torch
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)

from functionary.openai_types import ChatMessage, Function, FunctionCall
from functionary.prompt import (
    SYSTEM_MESSAGE,
    EndToken,
    StartToken,
    get_prompt_from_messages,
)
from functionary.schema import generate_schema_from_functions


class StopWordsCriteria(StoppingCriteria):
    def __init__(self, stops=[]):
        StoppingCriteria.__init__(self)
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        inputs = input_ids[0].tolist()
        for stop in self.stops:
            if len(inputs) >= len(stop) and inputs[-len(stop) :] == stop:
                return True
        return False


def tokenize(message: ChatMessage, tokenizer: LlamaTokenizer, device="cuda:0"):
    text = str(message)
    return tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids.to(
        device
    )


def prepare_messages_for_inference(
    tokenizer: LlamaTokenizer,
    messages: List[ChatMessage],
    functions=None,
    device="cuda:0",
) -> torch.Tensor:
    dic_messages = [mess.dict() for mess in messages]
    dic_messages.append({"role": "assistant"})
    func_list = []
    if functions is not None:
        for item in functions:
            func_list.append(item.dict())
    final_prompt = get_prompt_from_messages(dic_messages, func_list)
    input_ids = tokenizer(final_prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(device)
    return input_ids


def remove_stop_tokens_from_end(
    token_ids: List[int], stop_sequences: List[List[int]]
) -> List[int]:
    """This function is used to remove the hitting stop-sequence of id at the end of generated token_ids

    Args:
        token_ids (List[int]): generated token_id from model
        stop_sequences (List[List[int]]): List of stop sequence of ids

    Returns:
        List[int]: the result after removing hitting stop sequence
    """
    # sort the stop_sequences by length in descending order
    sorted_sequences = sorted(stop_sequences, key=lambda x: -len(x))
    for seg in sorted_sequences:
        if len(token_ids) >= len(seg):
            if token_ids[-len(seg) :] == seg:
                return token_ids[: -len(seg)]
    return token_ids


def parse_generated_content(generated_content: str) -> ChatMessage:
    """Parse LLM output into ChatMessage

    Args:
        generated_content (str): llm output

    Returns:
        ChatMessage: _description_
    """
    # strip end_of_function_call and end_of_assistant
    generated_content = generated_content.strip()
    for endtoken in [EndToken.function_call, EndToken.assistant]:
        if generated_content.endswith(endtoken):
            generated_content = generated_content[: -len(endtoken)].strip()
    # First we need to check if llm_output contains start_token or not
    start_function_index = generated_content.find(StartToken.function.value)
    text_content = generated_content
    result = ChatMessage(role="assistant")
    if start_function_index >= 0:
        func_info = generated_content[
            start_function_index + len(StartToken.function.value) :
        ].strip()
        index = func_info.find(":")
        func_name = func_info[:index].strip()
        arguments = func_info[index + 1 :].strip()
        text_content = generated_content[:start_function_index].strip()
        result.function_call = FunctionCall(name=func_name, arguments=arguments)
    if len(text_content) > 0:
        result.content = text_content
    return result


def generate_message(
    model: LlamaForCausalLM,
    tokenizer: LlamaTokenizer,
    messages: List[ChatMessage],
    functions: Optional[List[Function]] = None,
    temperature: float = 0.7,
    max_new_tokens=256,
    device="cuda:0",
    **kwargs,
) -> ChatMessage:
    inputs = prepare_messages_for_inference(
        tokenizer=tokenizer, messages=messages, functions=functions, device=device
    )
    stop_words_ids = []
    # [EndToken.assistant, EndToken.function_call]
    for stop in kwargs.get("stops", []) + [EndToken.assistant, EndToken.function_call]:
        tok_ids = tokenizer.encode(stop, add_special_tokens=False)
        if (
            len(tok_ids) > 1 and tok_ids[0] == 29871
        ):  # this is the issue of Llamatokenizer, sometimes they add this token
            tok_ids = tok_ids[1:]
        stop_words_ids.append(tok_ids)

    stopping_criteria = StoppingCriteriaList([StopWordsCriteria(stops=stop_words_ids)])
    generate_ids = model.generate(
        inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.001 if temperature == 0 else temperature,
        stopping_criteria=stopping_criteria,
    )
    token_ids = generate_ids[:, inputs.shape[1] :][0].tolist()

    # token_ids = remove_stop_tokens_from_end(token_ids, stop_words_ids)

    generated_content = tokenizer.decode(
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
        max_new_tokens=max_new_tokens,
    ).strip()
    return parse_generated_content(generated_content)


if __name__ == "__main__":
    # First lets create an example messages list with all different types of roles and content.
    functions = [
        {
            "name": "get_current_weather",
            "description": "Get the current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "format": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The temperature unit to use. Infer this from the users location.",
                    },
                },
                "required": ["location", "format"],
            },
        }
    ]

    messages = [
        ChatMessage(role="assistant", content="Hi there!"),
        ChatMessage(role="user", content="How are you?"),
        ChatMessage(role="assistant", content="I'm good thanks!"),
        ChatMessage(
            role="user", content="What's the weather like today in san francisco?"
        ),
        ChatMessage(
            role="assistant",
            content="I can help you find out! Lets call the get_current_weather function.",
            function_call=FunctionCall(
                name="get_current_weather",
                arguments='{"location": "San Francisco, CA", "format": "celsius"}',
            ),
        ),
        ChatMessage(
            role="function", name="get_current_weather", content='{"value": 32}'
        ),
        ChatMessage(
            role="assistant", content="It's 32 degrees celsius in San Francisco today."
        ),
        ChatMessage(role="user", content="Thanks!"),
        ChatMessage(role="assistant", content="No problem!"),
    ]

    # Now Lets prepare the messages for inference
    tokenizer = LlamaTokenizer.from_pretrained("musabgultekin/functionary-7b-v1")
    inputs = prepare_messages_for_inference(
        tokenizer=tokenizer, messages=messages, functions=functions, device="cpu"
    )
    print(inputs.shape)
