import os
import re
import unittest
from typing import List

from transformers import LlamaTokenizer, LlamaTokenizerFast

from functionary.prompt import (
    get_default_prompt_template,
    PromptTemplateV1,
    PromptTemplateV2,
)
from functionary.schema import generate_schema_from_functions
from functionary.train.custom_datasets import prepare_training_inputs
import json


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

        self.prompt_template = get_default_prompt_template()
        version = "v1" if type(self.prompt_template) is PromptTemplateV1 else "v2"

        current_folder = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(current_folder, f"test_case_{version}.json")) as f:
            self.test_case = json.loads(f.read())

        with open(os.path.join(current_folder, f"prompt_test_{version}.txt")) as f:
            self.final_prompt = f.read()
            self.final_prompt = self.final_prompt.replace("\n\n<|from|>", "\n<|from|>")

    def test_final_prompt_generation(self):
        tools_or_functions = (
            self.test_case["tools"]
            if "tools" in self.test_case
            else self.test_case["functions"]
        )
        final_prompt = self.prompt_template.get_prompt_from_messages(
            self.test_case["messages"], tools_or_functions
        )
        # print("-----------------------------------------------")
        # print(final_prompt)
        # print("-------------------------------")
        self.assertEqual(
            final_prompt,
            self.final_prompt,
            "wrong final prompt from: get_prompt_from_messages",
        )

    def test_prepare_training_inputs_fast_tokenizer(self):
        print("start testing fast tokenizer")
        for keep_assistant_prefix in [True, False]:
            self.run_prepare_training_inputs(
                use_fast=True, 
                pretrained="mistralai/Mistral-7B-v0.1",
                keep_assistant_prefix=keep_assistant_prefix
            )

    def test_prepare_training_inputs_normal_tokenizer(self):
        print("start testing normal tokenizer")
        for keep_assistant_prefix in [True, False]:
            self.run_prepare_training_inputs(
                use_fast=False, 
                pretrained="mistralai/Mistral-7B-v0.1",
                keep_assistant_prefix=keep_assistant_prefix
            )

    def run_prepare_training_inputs(self, use_fast: bool, pretrained: str, keep_assistant_prefix: bool=False):
        """this function is used to test function: prepare_training_inputs"""
        # note that must set legacy=True, read more: https://github.com/huggingface/transformers/issues/25176
        tokenizer_class = LlamaTokenizer
        if use_fast:
            tokenizer_class = LlamaTokenizerFast
        tokenizer = tokenizer_class.from_pretrained(pretrained, legacy=True)
        tokenizer.pad_token = tokenizer.eos_token
        # first we add stop_tokens to the tokenizer
        prompt_template = self.prompt_template
        length_before = len(tokenizer)
        added_tokens = prompt_template.get_additional_tokens()
        tokenizer.add_special_tokens({"additional_special_tokens": added_tokens})
        length_after = len(tokenizer)
        # check if tokenizer added new stop tokens successfully
        self.assertEqual(length_before + len(added_tokens), length_after)

        inputs = prepare_training_inputs(
            messages=self.test_case,
            tokenizer=tokenizer,
            prompt_template=prompt_template,
            padding="longest",
            max_length=1024,
            return_tensor=False,
            verbose=True,
            keep_assistant_prefix=keep_assistant_prefix
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
            if keep_assistant_prefix:
                prefix = ""
            else:
                prefix = prompt_template.get_text_from_message({"role": "assistant"})
            decoded_content = prefix + tokenizer.decode(
                chunk
            )  # note that need to add: "\nassistant" because we mask this, see line 194 in prompt_utils.py
            prompt = prompt_template.get_text_from_message(message)
            # decoded_content and prompt should be the same
            # to avoid any mistakes of tokenizer like adding white space we will compare after removing space
            self.assertEqual(
                re.sub("\s", "", decoded_content),
                re.sub("\s", "", prompt),
                "decoded content is different from original content",
            )


if __name__ == "__main__":
    unittest.main()
