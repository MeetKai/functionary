from typing import List, Optional

import torch
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)

from functionary.openai_types import ChatMessage, Function, FunctionCall, Tool
from functionary.prompt_template import get_prompt_template_from_tokenizer, PromptTemplate


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
    *,
    tokenizer: LlamaTokenizer,
    messages: List[ChatMessage],
    functions: Optional[List[Function]] = None,
    tools: Optional[List[Tool]] = None,
    device="cuda:0",
) -> torch.Tensor:
    
    prompt_template = get_prompt_template_from_tokenizer(tokenizer)
    
    dic_messages = [mess.dict() for mess in messages]
    dic_messages.append({"role": "assistant"})

    tools_or_functions = []
    if functions:
        tools_or_functions = [item.dict() for item in functions]
    elif tools:
        tools_or_functions = [item.dict() for item in tools]

    dic_messages = prompt_template.pre_process_messages_before_inference(dic_messages)
    final_prompt = prompt_template.get_prompt_from_messages(
        dic_messages, tools_or_functions=tools_or_functions
    )
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


def generate_message(
    *,
    model: LlamaForCausalLM,
    tokenizer: LlamaTokenizer,
    messages: List[ChatMessage],
    functions: Optional[List[Function]] = None,
    tools: Optional[List[Tool]] = None,
    temperature: float = 0.7,
    max_new_tokens=256,
    device="cuda:0",
    **kwargs,
) -> ChatMessage:
    prompt_template = get_prompt_template_from_tokenizer(tokenizer)
    inputs = prepare_messages_for_inference(
        tokenizer=tokenizer,
        messages=messages,
        functions=functions,
        tools=tools,
        device=device,
    )
    stop_words_ids = []
    # [EndToken.assistant, EndToken.function_call]
    for stop in (
        kwargs.get("stops", []) + prompt_template.get_stop_tokens_for_generation()
    ):
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

    generated_content = tokenizer.decode(
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
        max_new_tokens=max_new_tokens,
    ).strip()
    result = prompt_template.parse_assistant_response(generated_content)
    return ChatMessage(**result)


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
