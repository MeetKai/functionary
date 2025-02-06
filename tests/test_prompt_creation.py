import json
import os
import re
import unittest
from typing import List, Callable

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

        self.template_version_to_model_name = {
            "v2": "meetkai/functionary-small-v2.4",
            "v2.llama3": "meetkai/functionary-small-v2.5",
            "v3.llama3": "meetkai/functionary-medium-v3.0",
            "v3-llama3.1": "meetkai/functionary-small-v3.1",
        }
        self.image_template_version_to_model_name = {
            # "v3.llava_llama": "meetkai/functionary-vision-small-v0.1",
            # "qwen2-vl": "Qwen/Qwen2-VL-7B-Instruct",
            "qwen2.5": "Qwen/Qwen2.5-VL-7B-Instruct"
        }

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

    def read_image_example_data(self, template_version: str):
        current_folder = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(current_folder, f"test_case_vision.json")) as f:
            test_case = json.loads(f.read())

        with open(
            os.path.join(current_folder, f"prompt_test_{template_version}.txt")
        ) as f:
            final_prompt = f.read()
        return test_case, final_prompt

    def test_final_prompt_generation(self):
        for template_version in self.template_version_to_model_name.keys():
            print("--------------test template_version: ", template_version)
            test_case, final_prompt = self.read_example_data(template_version)
            tools_or_functions = (
                test_case["tools"] if "tools" in test_case else test_case["functions"]
            )
            prompt_template = get_prompt_template_by_version(template_version)
            created_prompt = prompt_template.get_prompt_from_messages(
                test_case["messages"], tools_or_functions
            )
            print(created_prompt)
            self.assertEqual(
                final_prompt.strip(),
                created_prompt.strip(),
                f"wrong final prompt from: get_prompt_from_messages, for version={template_version}",
            )

        for image_template_version in self.image_template_version_to_model_name.keys():
            print("--------------test image template_version: ", image_template_version)
            test_case, final_prompt = self.read_image_example_data(
                image_template_version
            )
            tools_or_functions = (
                test_case["tools"] if "tools" in test_case else test_case["functions"]
            )
            prompt_template = get_prompt_template_by_version(image_template_version)
            created_prompt = prompt_template.get_prompt_from_messages(
                test_case["messages"], tools_or_functions
            )
            print(created_prompt)

            self.assertEqual(
                final_prompt.strip(),
                created_prompt.strip(),
                f"wrong final prompt for vision from: get_prompt_from_messages, for version={image_template_version}",
            )

    def test_prepare_training_inputs_normal_tokenizer(self):
        for (
            template_version,
            pretrained_model,
        ) in self.template_version_to_model_name.items():
            print(f"-------------_TEST: {template_version}, {pretrained_model}")
            self.run_prepare_training_inputs(
                template_version=template_version,
                pretrained=pretrained_model,
                read_test_case_func=self.read_example_data,
                verbose=False,
            )

        print("test vision models")
        for (
            template_version,
            pretrained_model,
        ) in self.image_template_version_to_model_name.items():
            print(f"-------------_TEST: {template_version}, {pretrained_model}")
            self.run_prepare_training_inputs(
                template_version=template_version,
                pretrained=pretrained_model,
                read_test_case_func=self.read_image_example_data,
                verbose=False,
            )

    def run_prepare_training_inputs(
        self,
        template_version: str,
        pretrained: str,
        keep_assistant_prefix: bool = False,
        read_test_case_func: Callable = None,
        verbose: bool = False,
    ):
        """this function is used to test function: prepare_training_inputs"""
        # note that must set legacy=True, read more: https://github.com/huggingface/transformers/issues/25176
        tokenizer = AutoTokenizer.from_pretrained(pretrained)
        tokenizer.pad_token = tokenizer.eos_token
        # first we add stop_tokens to the tokenizer
        prompt_template = get_prompt_template_by_version(template_version)
        tokenizer.chat_template = prompt_template.get_chat_template_jinja()

        added_tokens = prompt_template.get_additional_tokens()
        special_tokens = {"additional_special_tokens": added_tokens}
        tokenizer.add_special_tokens(special_tokens)

        test_case, _ = read_test_case_func(template_version)

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
        assistant_indices = []
        for index, message in enumerate(test_case["messages"]):
            if message["role"] == "assistant":
                masked = False

                if "metadata" in message and message["metadata"].get("masked", False):
                    masked = True

                if not masked:
                    assistant_message.append(message)
                    assistant_indices.append(index)
        # find unmasked chunks in labels (chunk[i] != -100), there chunks are associated with assistant messages
        # for example: labels=[-100, -100, 1, 2, 3, -100, -100, 4, 5] --> chunks = [[1,2,3], [4,5]]
        chunks = extract_unmasked_chunks(labels)

        self.assertEqual(
            len(chunks),
            len(assistant_message),
            "number of unmasked chunks in labels is different from number of messages where role=assistant",
        )

        print(f"number of unmasked chunks: {len(chunks)}")
        for chunk, message, assistant_index in zip(
            chunks, assistant_message, assistant_indices
        ):
            prompt_wo_assistant = prompt_template.get_prompt_from_messages(
                test_case["messages"][:assistant_index],
                test_case["tools"],
                add_generation_prompt=True,
            )
            prompt_w_assistant = prompt_template.get_prompt_from_messages(
                test_case["messages"][: assistant_index + 1], test_case["tools"]
            )

            inference_text = prompt_w_assistant[len(prompt_wo_assistant) :]
            decoded_content = tokenizer.decode(
                chunk
            )  # note that need to add: "\nassistant" because we mask this, see line 194 in

            # decoded_content and prompt should be the same
            # to avoid any mistakes of tokenizer like adding white space we will compare after removing space
            self.assertEqual(
                re.sub("\s", "", decoded_content),
                re.sub("\s", "", inference_text),
                f"decoded content is different from original content:\ndecoded_content:{decoded_content}\nprompt:{inference_text}",
            )


if __name__ == "__main__":
    unittest.main()
