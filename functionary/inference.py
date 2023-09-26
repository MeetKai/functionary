from typing import List, Optional

import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

from functionary.openai_types import ChatMessage, Function, FunctionCall
from functionary.schema import generate_schema_from_functions

SYSTEM_MESSAGE = """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. The assistant calls functions with appropriate input when necessary"""


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
    all_messages = []
    if functions is not None:
        all_messages.append(
            ChatMessage(
                role="system", content=generate_schema_from_functions(functions)
            )
        )

    all_messages.append(ChatMessage(role="system", content=SYSTEM_MESSAGE))

    for message in messages:
        # Function call responses
        if message.role == "function":
            message.name = f"functions.{message.name}"
        # Function call requests by assistant
        if message.function_call:
            message.function_call.name = f"functions.{message.function_call.name}"
        all_messages.append(message)

    all_messages.append(ChatMessage(role="assistant", content=None))

    # ! should this be done as concatting strings and then tokenizing?
    # ! >>> text = "".join([str(msg) for msg in all_messages]
    # ! >>> return tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids.to("cuda:0")
    all_input_ids = [
        tokenize(tokenizer=tokenizer, message=message, device=device)
        for message in all_messages
    ]
    # text = "".join([str(msg) for msg in all_messages])
    # print(text)
    return torch.cat(all_input_ids, dim=-1)


def generate_message(
    model: LlamaForCausalLM,
    tokenizer: LlamaTokenizer,
    messages: List[ChatMessage],
    functions: Optional[List[Function]] = None,
    temperature: float = 0.7,
    max_new_tokens=256,
    device="cuda:0",
) -> ChatMessage:
    inputs = prepare_messages_for_inference(
        tokenizer=tokenizer, messages=messages, functions=functions, device=device
    )
    generate_ids = model.generate(
        inputs, max_new_tokens=max_new_tokens, temperature=temperature
    )
    generated_content = tokenizer.batch_decode(
        generate_ids[:, inputs.shape[1] :],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    # If it's a function call:
    if generated_content.startswith("to=functions."):
        function_call_content = generated_content[len("to=functions.") :]
        function_name, arguments = function_call_content.split(":\n")
        return ChatMessage(
            role="assistant",
            function_call=FunctionCall(name=function_name, arguments=arguments),
        )
    return ChatMessage(
        role="assistant",
        content=generated_content.lstrip("assistant:\n").rstrip("\n user:\n"),
    )


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
