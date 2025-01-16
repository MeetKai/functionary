import unittest
from transformers import AutoTokenizer
from functionary.train.metrics import (
    extract_indices_of_first_tokens_of_param_values_in_assistant_response,
)


class TestMetrics(unittest.TestCase):
    def test_first_argument_value_token_prediction(self):
        # Each test case includes assistant_response and target_tokens
        # the purpose of the test is to make sure that the function `extract_indices_of_first_tokens_of_param_values_in_assistant_response` can extract the first token of target token
        test_cases = [
            {
                "assistant_response": """all
I will get the price of 2 cars and compare>>>get_car_price
{"car_name": "Song", "year": 2019, "old": false}>>>get_car_price
{"car_name": "Tang", "year": 2020, "old": true}<|eot_id|>""",
                "target_tokens": ["Song", "2019", " false", "Tang", "2020", " true"],
            },
            {"assistant_response": "all\ngood morning<|eot_id|>", "target_tokens": []},
            {
                "assistant_response": '<function=get_weather>{"a": "a", "b": {"b1": "value1", "b2": 10, "b3": 1.4, "b4": false, "b5": ["abc", 10]}}</function><|eom_id|>',
                "target_tokens": [
                    "a",
                    ' {"',
                    "value1",
                    "10",
                    "1.4",
                    " false",
                    ' ["',
                    "abc",
                    "10",
                ],
            },
        ]

        tokenizer = AutoTokenizer.from_pretrained("lmms-lab/llama3-llava-next-8b")

        for case in test_cases:
            token_ids = tokenizer.encode(
                case["assistant_response"], add_special_tokens=False
            )
            target_tokens = [
                tokenizer.encode(target_token, add_special_tokens=False)[0]
                for target_token in case["target_tokens"]
            ]
            included_indices = (
                extract_indices_of_first_tokens_of_param_values_in_assistant_response(
                    tokenizer, token_ids, verbose=True
                )
            )
            included_tokens = [token_ids[index] for index in included_indices]
            print("target_tokens: ", target_tokens)
            print("included_tokens: ", included_tokens)
            assert target_tokens == included_tokens
