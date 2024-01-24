# This example script showcases how to use llama_cpp to run inference
# as well as ChatLab's FunctionRegistry to run tools

import asyncio
import json
import random
import sys
from typing import List

from chatlab import FunctionRegistry, tool_result
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from pydantic import Field
from termcolor import colored
from transformers import AutoTokenizer

from functionary.prompt_template import get_prompt_template_from_tokenizer


class FunctionaryAPI:
    def __init__(self):
        # Model repository on the Hugging Face model hub
        model_repo = "meetkai/functionary-small-v2.2-GGUF"

        # File to download
        file_name = "functionary-small-v2.2.f16.gguf"

        # Download the file
        local_file_path = hf_hub_download(repo_id=model_repo, filename=file_name)

        # You can download gguf files from https://huggingface.co/meetkai/functionary-7b-v2-GGUF/tree/main
        self.llm = Llama(model_path=local_file_path, n_ctx=4096, n_gpu_layers=-1)

        # Create tokenizer from HF.
        # We found that the tokenizer from llama_cpp is not compatible with tokenizer from HF that we trained
        # The reason might be we added new tokens to the original tokenizer
        # So we will use tokenizer from HuggingFace
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_repo,
            legacy=True,
        )
        # prompt_template will be used for creating the prompt
        self.prompt_template = get_prompt_template_from_tokenizer(self.tokenizer)

    async def create(
        self,
        messages: List = Field(default_factory=list),
        tools: List = Field(default_factory=list),
        model="functionary-small-v2.2",  # ignore parameter
    ):
        """Creates a model response for the given chat conversation.

        Matches OpenAI's `chat.create()` function."""
        # Create the prompt to use for inference
        prompt_str = self.prompt_template.get_prompt_from_messages(
            messages + [{"role": "assistant"}], tools
        )
        token_ids = self.tokenizer.encode(prompt_str)

        gen_tokens = []
        # Get list of stop_tokens
        stop_token_ids = [
            self.tokenizer.encode(token)[-1]
            for token in self.prompt_template.get_stop_tokens_for_generation()
        ]

        # We use function generate (instead of __call__) so we can pass in list of token_ids
        for token_id in self.llm.generate(token_ids, temp=0):
            if token_id in stop_token_ids:
                break
            gen_tokens.append(token_id)

        llm_output = self.tokenizer.decode(gen_tokens)

        # parse the message from llm_output
        response = self.prompt_template.parse_assistant_response(llm_output)

        return response


async def main():
    functionary = FunctionaryAPI()

    # Provide some space after the llama_cpp logs
    print("\n\n")

    messages = []

    user_message = {
        "role": "user",
        "content": "what's the weather like in Santa Cruz, CA compared to Seattle, WA?",
    }
    print(colored(f"User: {user_message['content']}", "light_cyan", attrs=["bold"]))
    messages.append(user_message)

    def get_current_weather(
        location: str = Field(
            description="The city and state, e.g., San Francisco, CA"
        ),
    ):
        """Get the current weather"""

        return {
            "temperature": 75 + random.randint(-5, 5),
            "units": "F",
            "weather": random.choice(["sunny", "cloudy", "rainy", "windy"]),
        }

    fr = FunctionRegistry()
    fr.register(get_current_weather)

    print(colored("Tools: ", "dark_grey"))
    print(colored(json.dumps(fr.tools, indent=2), "dark_grey"))

    response = await functionary.create(messages=messages, tools=fr.tools)
    messages.append(response)

    if response.get("content") is not None:
        print(
            colored(
                f"Assistant: {response['content']}", "light_magenta", attrs=["bold"]
            )
        )

    if response.get("tool_calls") is not None:
        print()
        for tool in response["tool_calls"]:
            requested_function = tool["function"]
            result = await fr.call(
                requested_function["name"], requested_function["arguments"]
            )
            print(
                colored(
                    f"  𝑓  {requested_function['name']}({requested_function['arguments']})",
                    "green",
                ),
                " -> ",
                colored(str(result), "light_green"),
            )

            tool_call_response = tool_result(tool["id"], content=str(result))
            # OpenAI does not require the name field, but it is required for functionary's tool_result. See https://github.com/openai/openai-python/issues/1078
            tool_call_response["name"] = requested_function["name"]

            messages.append(tool_call_response)

        print()
        # Run inference again after running tools
        response = await functionary.create(messages=messages, tools=fr.tools)
        print(
            colored(
                f"Assistant: {response['content']}", "light_magenta", attrs=["bold"]
            )
        )
        messages.append(response)


if __name__ == "__main__":
    asyncio.run(main())
