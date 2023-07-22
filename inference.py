import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from schema import generate_schema_for_functions

SYSTEM_MESSAGE = """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. The assistant calls functions with appropriate input when necessary"""


def prepare_message_for_inference(tokenizer, message):
    """Prepares a given message for the model by tokenizing the content."""

    if message["role"] == "system":
        text = "system:\n{content}\n".format(content=message.get("content", ""))

    elif message["role"] == "function":
        text = "function name={name}:\n{content}\n".format(name=message.get("name", ""), content= message.get("content", ""))

    elif message["role"] == "user" and message.get("content") is None:
        text = "user:\n</s>"

    elif message["role"] == "user":
        text = "user:\n</s>{content}\n".format(content=message.get("content", ""))

    elif message["role"] == "assistant" and message.get("to") is not None:
        text = "assistant to={to}:\n{content}</s>".format(to=message.get("to", ""), content=message.get("content", ""))

    elif message["role"] == "assistant" and message.get("content") is None:
        text = "assistant"

    elif message["role"] == "assistant":
        text = "assistant:\n{content}\n".format(content=message.get("content", ""))

    else:
        raise ValueError(f'Unsupported role: {message["role"]}')

    input_ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids.to("cuda:0")
    return input_ids


def prepare_messages_for_inference(tokenizer, messages, functions=None):
    all_messages = []
    if functions is not None:
        all_messages.append({"role": "system", "content": generate_schema_for_functions(functions)})
        #print(all_messages[0]["content"])
    all_messages.append({"role": "system", "content": SYSTEM_MESSAGE})
    all_messages.extend(messages)
    all_messages.append({"role": "assistant", "content": None})
    all_input_ids = [prepare_message_for_inference(tokenizer, msg) for msg in all_messages]
    return torch.cat(all_input_ids, dim=-1)


def generate(model, tokenizer, messages, functions=None, temperature=0.7, max_new_tokens=256):
    inputs = prepare_messages_for_inference(tokenizer, messages, functions)
    generate_ids = model.generate(inputs, max_new_tokens=max_new_tokens, temperature=temperature)
    generated_content = tokenizer.batch_decode(generate_ids[:, inputs.shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    #print(generated_content)

    # If its function call:
    if generated_content.startswith("to=functions."):
        function_call_content = generated_content.lstrip("to=functions.")
        function_name, arguments = function_call_content.split(":\n")
        return {
                "role": "assistant",
                "content": None,
                "function_call": {
                    "name": function_name,
                    "arguments": arguments,
                }
        }
    else:
        return {
                    'role': 'assistant',
                    'content': generated_content.lstrip("assistant:\n").rstrip("\n user:\n")
                }



if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained("musabgultekin/functionary-7b-v0.2", low_cpu_mem_usage=True, torch_dtype=torch.float16).to("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained("musabgultekin/functionary-7b-v0.2", use_fast=False)

    generate(
        tokenizer,
        messages=[
            {"role": "user", "content": "what is the weather for istanbul?"},
            {"role": "assistant", "to": "functions.get_current_weather", "content": '{\n  "location": "Istanbul",\n  "format": "celsius"\n}'},
            {"role": "function", "name": "functions.get_current_weather", "content": '{"value": 32}'},
            {"role": "assistant", "content": "The current weather in Istanbul is 32 degrees Celsius."},
            {"role": "user", "content": "what is the weather for san francisco?"},
        ], 
        functions=[
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
                },
        ]
    )
    
    
