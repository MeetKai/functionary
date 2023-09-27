from transformers import LlamaTokenizer
from typing import List, Dict, Any, Optional, Tuple
import torch


ROLE_MAPPING = {
    "user": "<|END_OF_USER|>",
    "system": "<|END_OF_SYSTEM|>",
    "assistant": "<|END_OF_ASSISTANT|>",
    "function": "<|END_OF_FUNCTION|>",
}


def get_text_from_message(message: Dict) -> str:
    """Prepares a given message for the model by tokenizing the content and determining target tokens."""
    stop_token = ROLE_MAPPING[message["role"]]
    content = message.get("content", "")
    if content is not None:
        content = f"{content}{stop_token}"

    if message["role"] == "system":
        text = "system:\n{content}\n".format(content=content)

    elif message["role"] == "function":
        text = "function name={name}:\n{content}\n".format(name=message.get("name", ""), content=content)

    elif message["role"] == "user" and content is None:
        text = "user:\n"

    elif message["role"] == "user":
        text = "user:\n{content}\n".format(content=content)
    elif message["role"] == "assistant":
        function = None
        arguments = None
        if (
            "to" in message
        ):  # if "to" in message --> format of Musab: {"role": "assistant", "to": func_name, "content": arguments}
            function = message["to"]
            arguments = content
        elif (
            "function_call" in message
        ):  # format of openai: {"role": assistant, "function_call": {"name": xxx, "arguments": xxx}}
            assert content is None
            function = message["function_call"]["name"]
            arguments = message["function_call"]["arguments"] + stop_token

        if function is not None:  # function call
            text = f"assistant to={function}:\n{arguments}\n"
        elif content is not None:  # this is text content
            text = f"assistant:\n{content}\n"
        else:  # if no function call and content is None --> this is used at inference
            text = "assistant"

    # This chunk of code below is to assert that stop_token is added correctly
    no_stop_token = False
    if content is None:
        if message["role"] == "user":
            no_stop_token = True
            assert not text.endswith(stop_token)
        elif message["role"] == "assistant":  # used at inference
            if "function_call" not in message and "to" not in message:
                no_stop_token = True
                assert not text.endswith(stop_token)
    if not no_stop_token:
        assert text.endswith(stop_token + "\n")

    return text


def get_prompt_from_messages(messages: List[Dict]) -> str:
    result = ""
    for mess in messages:
        result += get_text_from_message(mess)
    return result


def get_role_token_id_map(tokenizer: Any) -> Dict:
    result = {}
    for role, role_stop_token in ROLE_MAPPING.items():
        tok_ids = tokenizer.encode(role_stop_token, add_special_tokens=False)
        assert len(tok_ids) <= 2
        if len(tok_ids) == 2:
            assert tok_ids[0] == 29871  # tokenizer add this token intentionally
        tok_id = tok_ids[0]
        if len(tok_ids) == 2:
            tok_id = tok_ids[1]
        result[tok_id] = role
    return result


def prepare_training_inputs(
    messages: List[Dict],
    tokenizer: Any,
    padding: str = "max_length",
    max_length: Optional[int] = None,
    return_tensor: bool = True,
    verbose=False,
) -> Tuple[str, Dict]:
    stop_id_to_role = get_role_token_id_map(tokenizer)
    prompt_str = get_prompt_from_messages(messages)
    max_length = max_length if max_length is not None else tokenizer.model_max_length
    input_dic = tokenizer(prompt_str, padding=padding, max_length=max_length, truncation=True)
    input_token_ids = input_dic["input_ids"]
    start = 0
    labels = [-100 for _ in range(len(input_token_ids))]
    for index, tok_id in enumerate(input_token_ids):
        if tok_id in stop_id_to_role:
            role = stop_id_to_role[tok_id]
            if role == "assistant":  # only compute loss from tokens of assistant
                for i in range(start + 2, index + 1):  # The reason for start + 2 is to ignore: "\nassistant" (2 tokens)
                    labels[i] = input_token_ids[i]
                if verbose:
                    chunk = input_token_ids[start + 2 : index + 1]
                    print("+++ chunk assistant to compute loss: ", tokenizer.decode(chunk))
                    print("chunk tokens: ", chunk)
            start = index + 1
    input_dic["labels"] = labels
    assert len(labels) == len(input_dic["input_ids"]) == len(input_dic["attention_mask"])
    if return_tensor:
        for key in input_dic:
            input_dic[key] = torch.tensor(input_dic[key])
    return prompt_str, input_dic


def get_test_messages_musab_format():
    return [
        {"role": "system", "content": "This is the conversation between AI and HUman"},
        {"role": "user", "content": "what is the definition of mammal?"},
        {"role": "assistant", "content": "mammal is a vertebrate animal of the class Mammalia"},
        {"role": "user", "content": "what is the weather in Hanoi now"},
        {"role": "assistant", "to": "call_weather_check", "content": '{"city": "Hanoi"}'},
        {
            "role": "function",
            "name": "call_weather_check",
            "content": '{"weather_info": "The weather is cool, temperature is about 20 Degree Celcius"}',
        },
        {"role": "assistant", "content": "The weather in Hanoi is cool, about 20 Degree Celcius"},
        {"role": "user", "content": "Thank you"},
        {"role": "assistant", "content": None},
    ]


def get_test_messages_openai_format():
    return [
        {"role": "user", "content": "is the car Song more expensive than car Tang?"},
        {
            "role": "assistant",
            "content": None,
            "function_call": {"name": "get_car_price", "arguments": '{\n  "car_name": "Song"\n}'},
        },
        {"role": "function", "content": "{'price': {'price': '$25000'}}", "name": "get_car_price"},
        {
            "role": "assistant",
            "content": None,
            "function_call": {"name": "get_car_price", "arguments": '{\n  "car_name": "Tang"\n}'},
        },
        {"role": "function", "content": "{'price': {'price': '$20000'}}", "name": "get_car_price"},
        {
            "role": "assistant",
            "content": "No, the car Tang is less expensive than the car Song. The car Song is priced at $25,000, while the car Tang is priced at $20,000.",
        },
    ]


def test_prompt():
    tokenizer = LlamaTokenizer.from_pretrained("musabgultekin/functionary-7b-v1")
    added_tokens = list(ROLE_MAPPING.values())
    print("token_size: ", len(tokenizer))
    print("added_tokens: ", added_tokens)
    tokenizer.add_tokens(added_tokens)
    print("token_size after: ", len(tokenizer))
    messages1 = get_test_messages_musab_format()
    messages2 = get_test_messages_openai_format()
    for messages in [messages1, messages2]:
        print("--------------------------------------")
        prompt_str, input_dic = prepare_training_inputs(
            messages, tokenizer, padding="longest", max_length=200, verbose=True
        )
        print(f"prompt_str\n{prompt_str}")
        print("input_dic: ", input_dic)


if __name__ == "__main__":
    test_prompt()
