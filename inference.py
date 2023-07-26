import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from functionary_utils import SchemaGen


default_SYSTEM_MESSAGE = """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. The assistant calls functions with appropriate input when necessary"""

class Model:

    def __init__(self, model_name, preserve_mem, device, system_message=default_SYSTEM_MESSAGE):
            self.SYSTEM_MESSAGE = system_message
            if preserve_mem == 'True':
                preserve_mem_real = True
            else:
                preserve_mem_real = False
            self.model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=preserve_mem_real, torch_dtype=torch.float16).to(device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    def prepare_message_for_inference(self, tokenizer, message):
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

        input_ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids.to(os.getenv('INFERENCE_DEVICE'))
        return input_ids


    def prepare_messages_for_inference(self, tokenizer, messages, functions=None, plugins=None):
        all_messages = []
        if functions is not None:
            all_messages.append({"role": "system", "content": SchemaGen.generate_schemas(functions=functions, plugins=plugins) })
        all_messages.append({"role": "system", "content": self.SYSTEM_MESSAGE})
        for message in messages:
            if message.get("role") == "assistant":
                if message.get("content"):
                    all_messages.append({"role": "assistant", "content": message.get("content")})
                if message.get("function_call"):
                    all_messages.append({"role": "assistant", 
                                        "to": "functions." + message.get("function_call", {}).get("name"), 
                                        "content": message.get("function_call", {}).get("arguments")})
            elif message.get("role") == "function":
                message["name"] = "functions." + message.get("name", "")
                all_messages.append(message)
            else:
                all_messages.append(message)
                
        all_messages.append({"role": "assistant", "content": None})
        #print(all_messages)
        all_input_ids = [self.prepare_message_for_inference(tokenizer, msg) for msg in all_messages]
        return torch.cat(all_input_ids, dim=-1)


    def generate(self, messages, functions=None, temperature=0.7, max_new_tokens=256):
        inputs = self.prepare_messages_for_inference(tokenizer, messages, functions)
        generate_ids = self.model.generate(inputs, max_new_tokens=max_new_tokens, temperature=temperature)
        generated_content = self.tokenizer.batch_decode(generate_ids[:, inputs.shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        # If it's a function call:
        if generated_content.startswith("to=functions."):
            function_call_content = generated_content[len("to=functions."):]  # Remove the prefix
            function_name, function_arguments = function_call_content.split(":\n")  # Split at the first ":\n"
        elif generated_content.startswith("to=plugins."):
            function_call_content = generated_content[len("to=plugins."):]  # Remove the prefix
            function_name, function_arguments = function_call_content.split(":\n")  # Split at the first ":\n"
        else:
            return {
                    'role': 'assistant',
                    'content': generated_content.lstrip("assistant:\n").rstrip("\n user:\n")
            }

        return {
                "role": "assistant",
                "content": None,
                "function_call": {
                    "name": function_name,
                    "arguments": function_arguments,
                }
        }
