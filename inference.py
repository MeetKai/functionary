import torch
from typing import List, Optional
from transformers import LlamaTokenizer, LlamaForCausalLM

from openai_types import FunctionCall, Function, TurnMessage
from schema import generate_schema_from_functions

SYSTEM_MESSAGE = """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. The assistant calls functions with appropriate and correct input when necessary"""


def to_tokens(message: TurnMessage, tokenizer: LlamaTokenizer):
    text = str(message)
    return tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids.to(
        "cuda:0"
    )


def prepare_messages_for_inference(
    tokenizer: LlamaTokenizer, messages: List[TurnMessage], functions=None
):
    all_messages = []
    if functions is not None:
        all_messages.append(
            TurnMessage(
                role="system", content=generate_schema_from_functions(functions)
            )
        )
    all_messages.append(TurnMessage(role="system", content=SYSTEM_MESSAGE))

    for message in messages:
        if message.role == "assistant":
            if message:
                all_messages.append(
                    TurnMessage(role="assistant", content=message.content)
                )
            if message.function_call:
                fc = message.function_call
                all_messages.append(
                    TurnMessage(
                        role="assistant",
                        to=f"functions.{fc.name}",
                        content=fc.arguments,
                    )
                )
        elif message.role == "function":
            all_messages.append(
                TurnMessage(
                    role="function",
                    name=f"functions.{message.name}",
                    content=message.content,
                )
            )
        all_messages.append(message)

    all_messages.append(TurnMessage(role="assistant", content=None))

    # ! should this be done as concatting strings and then tokenizing?
    # ! >>> text = "".join([str(msg) for msg in all_messages]
    # ! >>> return tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids.to("cuda:0")
    all_input_ids = [
        to_tokens(tokenizer=tokenizer, message=message) for message in all_messages
    ]
    return torch.cat(all_input_ids, dim=-1)


def generate_models(
    model: LlamaForCausalLM,
    tokenizer: LlamaTokenizer,
    messages: List[TurnMessage],
    functions: Optional[List[Function]] = None,
    temperature: float = 0.7,
    max_new_tokens=256,
) -> TurnMessage:
    inputs = prepare_messages_for_inference(
        tokenizer=tokenizer, messages=messages, functions=functions
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
        return TurnMessage(
            role="assistant",
            function_call=FunctionCall(name=function_name, arguments=arguments),
        )
    return TurnMessage(
        role="assistant",
        content=generated_content.lstrip("assistant:\n").rstrip("\n user:\n"),
    )
