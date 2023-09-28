import sys
from functionary.prompt_utils import prepare_training_inputs, EndToken
from transformers import LlamaTokenizer


def get_test_messages_openai_format():
    return [
        {"role": "system", "content": "This is the conversation between Human and AI"},
        {"role": "user", "content": "who is the president of US"},
        {"role": "assistant", "content": "Biden is the president of US"},
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
    # first we add stop_tokens to the tokenizer
    added_tokens = [e.value for e in EndToken]
    print("added_tokens: ", added_tokens)
    tokenizer.add_tokens(added_tokens)
    print("token_size after: ", len(tokenizer))

    messages = get_test_messages_openai_format()
    prompt_str, input_dic = prepare_training_inputs(
        messages, tokenizer, padding="longest", max_length=256, verbose=True
    )
    print(f"final prompt str: \n{prompt_str}")
    print("input_dic: ", input_dic)


if __name__ == "__main__":
    test_prompt()
