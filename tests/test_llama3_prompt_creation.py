import json
import os
import re
import unittest
from typing import List

from transformers import AutoTokenizer

from functionary.prompt_template import get_prompt_template_by_version
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


class TestLlama3Template(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestLlama3Template, self).__init__(*args, **kwargs)

        self.template_version = "v2.llama3"
        self.prompt_template = get_prompt_template_by_version(self.template_version)

        current_folder = os.path.dirname(os.path.abspath(__file__))
        with open(
            os.path.join(current_folder, f"test_case_{self.template_version}.json")
        ) as f:
            self.test_case = json.loads(f.read())

        with open(
            os.path.join(current_folder, f"prompt_test_{self.template_version}.txt")
        ) as f:
            self.final_prompt = f.read().strip()

    def test_final_prompt_generation(self):
        tools_or_functions = (
            self.test_case["tools"]
            if "tools" in self.test_case
            else self.test_case["functions"]
        )
        final_prompt = self.prompt_template.get_prompt_from_messages(
            self.test_case["messages"], tools_or_functions
        )
        # print("--------------PROMPT-------")
        # print(final_prompt)
        # print("-------------")
        self.assertEqual(
            final_prompt,
            self.final_prompt,
            "wrong final prompt from: get_prompt_from_messages",
        )

    def test_prepare_training_inputs_normal_tokenizer(self):
        """this function is used to test function: prepare_training_inputs"""
        tokenizer = AutoTokenizer.from_pretrained(
            "gradientai/Llama-3-8B-Instruct-Gradient-1048k", legacy=True
        )
        tokenizer.pad_token = tokenizer.eos_token
        # first we add stop_tokens to the tokenizer
        prompt_template = self.prompt_template
        added_tokens = prompt_template.get_additional_tokens()
        if len(added_tokens) > 0:
            tokenizer.add_special_tokens({"additional_special_tokens": added_tokens})

        inputs = prepare_training_inputs(
            messages=self.test_case,
            tokenizer=tokenizer,
            padding="longest",
            max_length=1024,
            return_tensor=False,
            verbose=False,
            keep_assistant_prefix=False,
        )
        input_ids = inputs["inputs"]["input_ids"]
        labels = inputs["inputs"]["labels"]
        self.assertEqual(
            len(input_ids), len(labels), "length of inputs and labels are different"
        )

        # check if input_ids[i] == labels[i] if labels[i] != -100
        for input_token_id, label_token_id in zip(input_ids, labels):
            if label_token_id != -100:
                self.assertEqual(
                    input_token_id, label_token_id, "input_token_id != label_token_id"
                )

        # Check if only messages where role=assistant and unmasked are remained, others will be masked as -100
        assistant_message = []
        for message in self.test_case["messages"]:
            if message["role"] == "assistant":
                masked = False

                if "metadata" in message and message["metadata"].get("masked", False):
                    masked = True

                if not masked:
                    assistant_message.append(message)
        # find unmasked chunks in labels (chunk[i] != -100), there chunks are associated with assistant messages
        # for example: labels=[-100, -100, 1, 2, 3, -100, -100, 4, 5] --> chunks = [[1,2,3], [4,5]]
        chunks = extract_unmasked_chunks(labels)

        self.assertEqual(
            len(chunks),
            len(assistant_message),
            "number of unmasked chunks in labels is different from number of messages where role=assistant",
        )

        print(f"number of unmasked chunks: {len(chunks)}")
        for chunk, message in zip(chunks, assistant_message):
            prefix = prompt_template.convert_message_to_prompt({"role": "assistant"})
            decoded_content = prefix + tokenizer.decode(chunk)

            prompt = prompt_template.convert_message_to_prompt(message)
            # decoded_content and prompt should be the same
            # to avoid any mistakes of tokenizer like adding white space we will compare after removing space
            self.assertEqual(
                re.sub("\s", "", decoded_content),
                re.sub("\s", "", prompt),
                "decoded content is different from original content",
            )

    def test_against_original_llama3_chat_template(self):
        tokenizer = AutoTokenizer.from_pretrained(
            "gradientai/Llama-3-8B-Instruct-Gradient-1048k", legacy=True
        )
        tokenizer.pad_token = tokenizer.eos_token
        messages = [
            {"role": "system", "content": "The current date is: 2024-04-23"},
            {"role": "user", "content": "Hello how are you?"},
            {"role": "assistant", "content": "Hi, I am good, and you?"},
            {"role": "user", "content": "Can you help me book a car?"},
            {"role": "assistant", "content": "Sure, I wil help you now"},
        ]

        inference_messages = messages[:-1] + [{"role": "assistant"}]

        for case in [messages]:
            created_prompt = self.prompt_template.get_prompt_from_messages(
                case, tools_or_functions=[]
            )
            correct_prompt = tokenizer.apply_chat_template(case, tokenize=False)

            # print("created_prompt: ", created_prompt)
            # print("correct: ", correct_prompt)

            if correct_prompt.startswith("<|begin_of_text|>"):
                correct_prompt = correct_prompt[len("<|begin_of_text|>") :]

            assert created_prompt.endswith(correct_prompt)


if __name__ == "__main__":
    unittest.main()
