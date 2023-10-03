import os
import re
import unittest
from typing import List

from transformers import LlamaTokenizer

from functionary.prompt import EndToken, get_prompt_from_messages, get_text_from_message
from functionary.schema import generate_schema_from_functions
from functionary.train.custom_datasets import prepare_training_inputs


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
    def __init__(self, *args, **kwargs):
        super(TestInsertingEndToken, self).__init__(*args, **kwargs)
        self.test_case = {
            "functions": [
                {
                    "name": "get_car_price",
                    "description": "Get the price of a particular car model",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "car_name": {
                                "type": "string",
                                "description": "The name of the car model",
                            }
                        },
                        "required": ["get_car_price"],
                    },
                }
            ],
            "messages": [
                {"role": "user", "content": "who is the president of US"},
                {"role": "assistant", "content": "Biden is the president of US"},
                {
                    "role": "user",
                    "content": "is the car Song more expensive than car Tang?",
                },
                {
                    "role": "assistant",
                    "content": None,
                    "function_call": {
                        "name": "get_car_price",
                        "arguments": '{\n  "car_name": "Song"\n}',
                    },
                },
                {
                    "role": "function",
                    "content": "{'price': {'price': '$25000'}}",
                    "name": "get_car_price",
                },
                {
                    "role": "assistant",
                    "content": None,
                    "function_call": {
                        "name": "get_car_price",
                        "arguments": '{\n  "car_name": "Tang"\n}',
                    },
                },
                {
                    "role": "function",
                    "content": "{'price': {'price': '$20000'}}",
                    "name": "get_car_price",
                },
                {
                    "role": "assistant",
                    "content": "No, the car Tang is less expensive than the car Song. The car Song is priced at $25,000, while the car Tang is priced at $20,000.",
                },
            ],
        }
        current_folder = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(current_folder, "prompt_test.txt")) as f:
            self.final_prompt = f.read()

    def test_final_prompt_generation(self):
        final_prompt = (
            "system:\n"
            + generate_schema_from_functions(functions=self.test_case["functions"])
            + f"\nsystem:\nA chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. The assistant calls functions with appropriate input when necessary{EndToken.system.value}\n"
        )
        final_prompt += get_prompt_from_messages(self.test_case["messages"])

        self.assertEqual(
            final_prompt.strip(),
            self.final_prompt.strip(),
            "wrong final prompt from: get_prompt_from_messages",
        )

    def test_correct_end_token(self):
        """this function is to test if endtoken is correctly inserted at the end"""
        # these cases requires end_token at the end
        cases = [
            (
                {
                    "role": "system",
                    "content": "This is a conversation between Human and AI",
                },
                EndToken.system.value,
            ),
            ({"role": "user", "content": "hello"}, EndToken.user.value),
            (
                {"role": "assistant", "content": "nice to meet you"},
                EndToken.assistant.value,
            ),
            (
                {
                    "role": "assistant",
                    "content": None,
                    "function_call": {
                        "name": "get_weather",
                        "arguments": '{"city": "Hanoi"}',
                    },
                },
                EndToken.function_call.value,
            ),
            (
                {
                    "role": "function",
                    "content": '{"temperature": 30}',
                    "name": "get_weather",
                },
                EndToken.function.value,
            ),
        ]

        for message, stop_token in cases:
            prompt = get_text_from_message(message).strip()
            #  Check if prompt ends with stop_token
            self.assertTrue(
                prompt.endswith(stop_token),
                f"`{prompt}` doesn't end with: `{stop_token}`",
            )

        # these cases don't require end_token at the end
        edge_cases = [
            ({"role": "user", "content": None}, EndToken.user.value),
            ({"role": "assistant", "content": None}, EndToken.assistant.value),
        ]

        for message, stop_token in edge_cases:
            prompt = get_text_from_message(message).strip()
            #  Check if prompt doesn't endswith stop_token
            self.assertFalse(
                prompt.endswith(stop_token), f"`{prompt}` ends with: `{stop_token}`"
            )

    def test_prepare_training_inputs(self):
        """this function is used to test function: prepare_training_inputs"""
        # note that must set legacy=True, read more: https://github.com/huggingface/transformers/issues/25176
        tokenizer = LlamaTokenizer.from_pretrained(
            "musabgultekin/functionary-7b-v1", legacy=True
        )
        # first we add stop_tokens to the tokenizer
        length_before = len(tokenizer)
        added_tokens = [e.value for e in EndToken]
        tokenizer.add_special_tokens({"additional_special_tokens": added_tokens})
        length_after = len(tokenizer)
        # check if tokenizer added new stop tokens successfully
        self.assertEqual(length_before + len(added_tokens), length_after)

        _, inputs = prepare_training_inputs(
            self.test_case,
            tokenizer,
            padding="longest",
            max_length=512,
            return_tensor=False,
        )
        input_ids = inputs["input_ids"]
        labels = inputs["labels"]
        self.assertEqual(
            len(input_ids), len(labels), "length of inputs and labels are different"
        )

        # check if input_ids[i] == labels[i] if labels[i] != -100
        for input_token_id, label_token_id in zip(input_ids, labels):
            if label_token_id != -100:
                self.assertEqual(
                    input_token_id, label_token_id, "input_token_id != label_token_id"
                )

        # Check if only messages where role=assistant are remained, others will be masked as -100
        assistant_message = [
            item for item in self.test_case["messages"] if item["role"] == "assistant"
        ]
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


if __name__ == "__main__":
    unittest.main()
