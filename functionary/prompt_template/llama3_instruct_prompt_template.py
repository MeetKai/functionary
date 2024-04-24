import json
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
import random, string
from functionary.prompt_template.base_template import PromptTemplate


def convert_to_llama3_messages(messages: List[Dict]) -> List[Dict]:
    result = []
    index = 0
    while index < len(messages):
        if messages[index]["role"] in ["user", "system"]:
            result.append(messages[index])
            index += 1
        else:
            if messages[index]["role"] == "assistant":
                tool_calls = messages[index].get("tool_calls", [])
                if len(tool_calls) == 0:
                    result.append(messages[index])
                else:
                    messages


def get_random_tool_call_id():
    return "call_" + "".join(
        [random.choice(string.ascii_letters + string.digits) for _ in range(24)]
    )


class Llam3InstructTemplate(PromptTemplate):
    function_separator = "<|reserved_special_token_249|>"
    version = "v2.llama3_instruct"

    def get_additional_tokens(self) -> List[str]:
        return []

    def get_assistant_prefixes(self) -> List[str]:
        # return [f"{self.from_token}assistant\n{self.recipient_token}"]
        return ["<|start_header_id|>assistant<|end_header_id|>\n\n"]

    def get_stop_tokens_for_generation(self) -> List[str]:
        return ["<|eot_id|>", "<|end_of_text|>"]

    def parse_assistant_response(
        self, llm_output: str, tool_choice: Any | None
    ) -> Dict:
        # first remove stop tokens if there exists
        for stop in self.get_stop_tokens_for_generation():
            if llm_output.endswith(stop):
                llm_output = llm_output[: -len(stop)]

        chunks = llm_output.split(self.function_separator)
        chunks = [chunk.strip() for chunk in chunks if len(chunk.strip()) > 0]

        tool_calls = []
        text_content = None
        tool_call_start_index = 0

        if not llm_output.startswith(
            self.function_separator
        ):  # if first token is not function_call
            text_content = chunks[0]
            tool_call_start_index = 1

        for chunk in chunks[tool_call_start_index:]:
            # format: function_name\narguments<end_of_functioncall>
            index = chunk.find("\n")
            func_name = chunk[:index].strip()
            arguments = chunk[index + 1 :].strip()
            tool_calls.append(
                {
                    "function": {"name": func_name, "arguments": arguments},
                    "id": get_random_tool_call_id(),
                    "type": "function",
                }
            )
        return {"role": "assistant", "content": text_content, "tool_calls": tool_calls}

    def convert_message_to_prompt(self, message: Dict) -> str:
        role = message["role"]
        content = message.get("content", None)
        if role == "tool":
            tool_name = message["name"]
            content = f"name={tool_name}\n{content}"

        prompt_template = (
            "<|start_header_id|>%s<|end_header_id|>\n\n{text}<|eot_id|>" % role
        )

        if role in ["user", "system", "tool"]:
            return prompt_template.format(text=content)

        assert role == "assistant"
        tool_calls = message.get("tool_calls", [])
        if tool_calls is None:
            tool_calls = []

        if content is None and len(tool_calls) == 0:
            return f"<|start_header_id|>{role}<|end_header_id|>\n\n"

        if content is not None and len(tool_calls) == 0:  # text-only
            return prompt_template.format(text=content)

        tool_call_prompts = []
        for tool_call in tool_calls:
            arguments = tool_call["function"]["arguments"]
            tool_name = tool_call["function"]["name"]
            tool_prompt = f"{tool_name}\n{arguments}"
            tool_call_prompts.append(tool_prompt)

        if (
            content is None and len(tool_calls) > 0
        ):  # function call only (<sep>fc1<sep>fc2<sep>)
            tool_call_content = self.function_separator + self.function_separator.join(
                tool_call_prompts
            )
            return prompt_template.format(text=tool_call_content)

        # Here is the case contains both text-response and tool_calls (content<sep>fc1<sep>fc2<sep>)
        total_content = (
            content
            + self.function_separator
            + self.function_separator.join(tool_call_prompts)
        )
        return prompt_template.format(text=total_content)

    def pre_process_messages_before_inference(self, messages: List[Dict]) -> List[Dict]:
        """re-order the messages where role = tool to match the order in tool_calls by tool_call_id
        Args:
            messages (List[Dict]): list of messages containing: tool_call_id

        Returns:
            List[Dict]: _description_
        """
        result = []
        index = 0
        while index < len(messages):
            message = messages[index]
            tool_calls = message.get("tool_calls", None)

            result.append(message)
            if message["role"] == "assistant" and tool_calls:
                num_calls = len(tool_calls)
                if (
                    tool_calls[0].get("id", None) is not None
                ):  # if tool_call contains "id" for mapping
                    tool_call_ids = [item["id"] for item in tool_calls]

                    tool_messages = [messages[index + 1 + j] for j in range(num_calls)]
                    id_2_tool_messages = {
                        item["tool_call_id"]: item for item in tool_messages
                    }
                    new_messages = [id_2_tool_messages[cid] for cid in tool_call_ids]

                    result.extend(new_messages)
                    index += num_calls + 1
                else:
                    index += 1
            else:
                index += 1
        return result

    def get_chat_template_jinja(self):
        return "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
