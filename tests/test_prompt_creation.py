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


class TestPromptTemplate(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestPromptTemplate, self).__init__(*args, **kwargs)

        self.template_versions = [
            "v2",
            "v2.llama3",
            "v3.llama3",
            "v3-llama3.1",
            "v3.json",
        ]
        self.pretrained_models = [
            "meetkai/functionary-small-v2.4",
            "meetkai/functionary-small-v2.5",
            "meetkai/functionary-medium-v3.0",
            "meetkai/functionary-small-v3.1",
            "meetkai/functionary-small-v3.1",
        ]

    def read_example_data(self, template_version: str):
        current_folder = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(current_folder, f"test_case.json")) as f:
            test_case = json.loads(f.read())

        with open(
            os.path.join(current_folder, f"prompt_test_{template_version}.txt")
        ) as f:
            final_prompt = f.read()
            if template_version == "v2":
                final_prompt = final_prompt.replace("\n\n<|from|>", "\n<|from|>")
        return test_case, final_prompt

    def test_final_prompt_generation(self):
        for template_version in self.template_versions:
            print("--------------test template_version: ", template_version)
            test_case, final_prompt = self.read_example_data(template_version)
            tools_or_functions = (
                test_case["tools"] if "tools" in test_case else test_case["functions"]
            )
            prompt_template = get_prompt_template_by_version(template_version)
            created_prompt = prompt_template.get_prompt_from_messages(
                test_case["messages"], tools_or_functions
            )
            self.assertEqual(
                final_prompt.strip(),
                created_prompt.strip(),
                f"wrong final prompt from: get_prompt_from_messages, for version={template_version}",
            )

    def test_prepare_training_inputs_normal_tokenizer(self):
        for template_version, pretrained_model in zip(
            self.template_versions, self.pretrained_models
        ):
            print(f"-------------_TEST: {template_version}, {pretrained_model}")
            self.run_prepare_training_inputs(
                template_version=template_version,
                pretrained=pretrained_model,
                verbose=False,
            )

    def run_prepare_training_inputs(
        self,
        template_version: str,
        pretrained: str,
        keep_assistant_prefix: bool = False,
        verbose: bool = False,
    ):
        """this function is used to test function: prepare_training_inputs"""
        # note that must set legacy=True, read more: https://github.com/huggingface/transformers/issues/25176
        tokenizer = AutoTokenizer.from_pretrained(pretrained, legacy=True)
        tokenizer.pad_token = tokenizer.eos_token
        # first we add stop_tokens to the tokenizer
        prompt_template = get_prompt_template_by_version(template_version)
        tokenizer.chat_template = prompt_template.get_chat_template_jinja()

        added_tokens = prompt_template.get_additional_tokens()
        special_tokens = {"additional_special_tokens": added_tokens}
        tokenizer.add_special_tokens(special_tokens)

        test_case, _ = self.read_example_data(template_version)

        inputs = prepare_training_inputs(
            messages=test_case,
            tokenizer=tokenizer,
            padding="longest",
            max_length=1024,
            return_tensor=False,
            verbose=True,
            keep_assistant_prefix=keep_assistant_prefix,
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
        for message in test_case["messages"]:
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
            if keep_assistant_prefix:
                prefix = ""
            else:
                prefix = prompt_template.convert_message_to_prompt(
                    {"role": "assistant"}
                )
            decoded_content = prefix + tokenizer.decode(
                chunk
            )  # note that need to add: "\nassistant" because we mask this, see line 194 in prompt_utils.py
            prompt = prompt_template.convert_message_to_prompt(message)
            # decoded_content and prompt should be the same
            # to avoid any mistakes of tokenizer like adding white space we will compare after removing space
            self.assertEqual(
                re.sub("\s", "", decoded_content),
                re.sub("\s", "", prompt),
                f"decoded content is different from original content:\ndecoded_content:{decoded_content}\nprompt:{prompt}",
            )


if __name__ == "__main__":
    unittest.main()
