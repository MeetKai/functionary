import sys
from functionary.prompt_utils import prepare_training_inputs, EndToken
from functionary.prompt_utils import get_text_from_message
from transformers import LlamaTokenizer
from typing import List
import re
import unittest


def extract_unmasked_chunks(labels: List[int]) -> List[List[int]]:
    """This function is used to extract unmasked chunks of integer
    For example, labels = [-100, -100, 1, 2, 3, -100, -100, 4, 5] --> chunks = [[1,2,3], [4,5]]
    Args:
        labels (List[int]): list of integer containing token_id and -100

    Returns:
        List[List[int]]: list of chunk, for example: [[1,2,3], [4,5]]
    """
    chunks = []
    chunk = []
    for token_id in labels:
        if token_id != -100:
            chunk.append(token_id)
        else:
            if len(chunk) > 0:
                chunks.append(chunk)
                chunk = []
    if len(chunk) > 0:
        chunks.append(chunk)
    return chunks


class TestInsertingEndToken(unittest.TestCase):
    def test_correct_end_token(self):
        """this function is to test if endtoken is correctly inserted at the end"""
        # these cases requires end_token at the end
        cases = [
            ({"role": "system", "content": "This is a conversation between Human and AI"}, EndToken.system.value),
            ({"role": "user", "content": "hello"}, EndToken.user.value),
            ({"role": "assistant", "content": "nice to meet you"}, EndToken.assistant.value),
            (
                {
                    "role": "assistant",
                    "content": None,
                    "function_call": {"name": "get_weather", "arguments": '{"city": "Hanoi"}'},
                },
                EndToken.function_call.value,
            ),
            ({"role": "function", "content": '{"temperature": 30}', "name": "get_weather"}, EndToken.function.value),
        ]

        for message, stop_token in cases:
            prompt = get_text_from_message(message).strip()
            #  Check if prompt ends with stop_token
            self.assertTrue(prompt.endswith(stop_token), f"`{prompt}` doesn't end with: `{stop_token}`")

        # these cases don't require end_token at the end
        edge_cases = [
            ({"role": "user", "content": None}, EndToken.user.value),
            ({"role": "assistant", "content": None}, EndToken.assistant.value),
        ]

        for message, stop_token in edge_cases:
            prompt = get_text_from_message(message).strip()
            #  Check if prompt doesn't endswith stop_token
            self.assertFalse(prompt.endswith(stop_token), f"`{prompt}` ends with: `{stop_token}`")

    def test_prepare_training_inputs(self):
        """this function is used to test function: prepare_training_inputs"""
        tokenizer = LlamaTokenizer.from_pretrained("musabgultekin/functionary-7b-v1")
        # first we add stop_tokens to the tokenizer
        length_before = len(tokenizer)
        added_tokens = [e.value for e in EndToken]
        tokenizer.add_special_tokens({"additional_special_tokens": added_tokens})
        length_after = len(tokenizer)
        # check if tokenizer added new stop tokens successfully
        self.assertEqual(length_before + len(added_tokens), length_after)

        test_case = [
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
        _, inputs = prepare_training_inputs(
            test_case, tokenizer, padding="longest", max_length=256, return_tensor=False
        )
        input_ids = inputs["input_ids"]
        labels = inputs["labels"]
        self.assertEqual(len(input_ids), len(labels), "length of inputs and labels are different")

        # check if input_ids[i] == labels[i] if labels[i] != -100
        for input_token_id, label_token_id in zip(input_ids, labels):
            if label_token_id != -100:
                self.assertEqual(input_token_id, label_token_id, "input_token_id != label_token_id")

        # Check if only messages where role=assistant are remained, others will be masked as -100
        assistant_message = [item for item in test_case if item["role"] == "assistant"]
        # find unmasked chunks in labels (chunk[i] != -100), there chunks are associated with assistant messages
        # for example: labels=[-100, -100, 1, 2, 3, -100, -100, 4, 5] --> chunks = [[1,2,3], [4,5]]
        chunks = extract_unmasked_chunks(labels)

        self.assertEqual(
            len(chunks),
            len(assistant_message),
            "number of unmasked chunks in labels is different from number of messages where role=assistant",
        )
        for chunk, message in zip(chunks, assistant_message):
            decoded_content = "\nassistant" + tokenizer.decode(
                chunk
            )  # note that need to add: "\nassistant" because we mask this, see line 194 in prompt_utils.py
            prompt = get_text_from_message(message)
            # decoded_content and prompt should be the same
            # to avoid any mistakes of tokenizer like adding white space we will compare after removing space
            self.assertEqual(
                re.sub("\s", "", decoded_content),
                re.sub("\s", "", prompt),
                "decoded content is different from original content",
            )


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
    unittest.main()
