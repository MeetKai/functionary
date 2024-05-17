from typing import Dict, List, Optional, Union

import torch
from lmformatenforcer import CharacterLevelParser, JsonSchemaParser
from lmformatenforcer.integrations.vllm import build_vllm_logits_processor
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)
from vllm.model_executor.guided_decoding.lm_format_enforcer_decoding import (
    _cached_build_vllm_token_enforcer_tokenizer_data,
    _normalize_json_schema_object,
)
from vllm.sampling_params import LogitsProcessor

from functionary.openai_types import ChatMessage, Function, FunctionCall, Tool
from functionary.prompt_template import (
    PromptTemplate,
    get_prompt_template_from_tokenizer,
)


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
    tools_or_functions: List[Dict],
    tool_choice: Optional[Union[str, Tool]] = None,
    device="cuda:0",
) -> torch.Tensor:
    prompt_template = get_prompt_template_from_tokenizer(tokenizer)

    dic_messages = [mess.dict() for mess in messages]
    dic_messages.append({"role": "assistant"})

    dic_messages = prompt_template.pre_process_messages_before_inference(dic_messages)

    # This also checks for code_interpreter and adds python default system message instead
    # default system message
    final_prompt = prompt_template.get_prompt_from_messages(
        dic_messages, tools_or_functions=tools_or_functions
    )

    if (
        prompt_template.version != "v1"
        and tool_choice is not None
        and tool_choice not in ["auto", "required"]
    ):
        if tool_choice == "none":
            final_prompt += prompt_template.get_force_text_generation_prefix()
        else:
            final_prompt += prompt_template.get_force_function_call_prefix(
                tool_choice.function.name
            )
    # some prompt template supports call a function directly such as: v2.llama3
    if tool_choice == "required":
        if hasattr(prompt_template, "function_separator"):
            final_prompt += getattr(prompt_template, "function_separator")

    input_ids = tokenizer(final_prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(device)
    return input_ids


def enforce_tool_choice(
    choice: Union[str, Tool, Function],
    tools_or_functions: Optional[List[Union[Tool, Function]]],
) -> Optional[List[Tool]]:
    """This function is used to enforce tool_choice in the list of tools if it is provided by the user

    Args:
        choice: (Union[str, Tool]): either "auto", "none" or Tool/Function object
        tools_or_functions (Optional[List[Tool, Function]]): the existing list of tools passed in from user

    Returns:
        List[Tool, Function]: the modified tools_or_functions based on tool_choice
    """
    if choice == "none":
        return []
    elif isinstance(choice, Tool):
        if choice.function.description == "" and choice.function.parameters is None:
            tools_or_functions = [
                tool
                for tool in tools_or_functions
                if tool.type == "function"
                and tool.function.name == choice.function.name
            ]
            assert (
                len(tools_or_functions) > 0
            ), f"Invalid value for 'tool_choice': no function named {choice.function.name} was specified in the 'tools' parameter"
        else:
            tools_or_functions = [choice]
    elif isinstance(choice, Function):
        tools_or_functions = [
            function for function in tools_or_functions if function.name == choice.name
        ]
        assert (
            len(tools_or_functions) > 0
        ), f"Invalid value for 'function_call': no function named {choice.name} was specified in the 'functions' parameter"

    return tools_or_functions


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


async def get_lm_format_enforcer_vllm_logits_processor_from_tool_name(
    tool_name, tools_or_functions, tokenizer
) -> LogitsProcessor:
    """
    Given a tool_name and list of tool definitions, find the json schema
    of the tool with tool_name name and get the necessary vLLM logits processor
    for the given tool schema."""

    tokenizer_data = _cached_build_vllm_token_enforcer_tokenizer_data(tokenizer)
    character_level_parser: CharacterLevelParser

    # Get the tool schema
    for tool_or_function in tools_or_functions:
        if tool_or_function["name"] == tool_name:
            raw_tool_schema = tool_or_function["parameters"]
            break
    schema = _normalize_json_schema_object(raw_tool_schema)
    character_level_parser = JsonSchemaParser(schema)
    logits_processor = build_vllm_logits_processor(
        tokenizer_data, character_level_parser
    )
    return logits_processor


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
