import json
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from functionary.prompt_template import prompt_utils
from functionary.prompt_template.llama3_prompt_template_v3 import Llama3TemplateV3


class LlavaLlama(Llama3TemplateV3):
    version = "v3.llava_llama"
    # This token will be replaced with image_token_id (-200) after we tokenize the text
    image_token = "<|reserved_special_token_250|>"

    def convert_message_to_prompt(self, message: Dict) -> str:
        role = message["role"]
        content = message.get("content", None)

        # comment this as currently the Llama-70b was trained using this
        # if role == "tool":
        #     tool_name = message["name"]
        #     content = f"name={tool_name}\n{content}"

        prompt_template = (
            f"{self.start_header}{role}{self.end_header}\n\n"
            + "{text}"
            + self.eos_token
        )

        if role == "user":  # Check if contain uploaded image or not
            if type(content) is list:
                content = prompt_utils.get_content_str_from_multi_modal_input(
                    message["content"], self.image_token
                )

        if role in ["user", "system", "tool"]:
            return prompt_template.format(text=content)

        assert role == "assistant", f"role must be assistant, but: {role}"
        tool_calls = message.get("tool_calls", [])
        if tool_calls is None:
            tool_calls = []

        if content is None and len(tool_calls) == 0:  # inference time
            return f"{self.start_header}{role}{self.end_header}\n\n{self.function_separator}"

        if content is not None:  # text-only
            tool_calls = [
                {"function": {"name": "all", "arguments": content}}
            ] + tool_calls

        tool_call_prompts = []
        for tool_call in tool_calls:
            arguments = tool_call["function"]["arguments"]
            tool_name = tool_call["function"]["name"]
            tool_prompt = f"{tool_name}\n{arguments}"
            tool_call_prompts.append(tool_prompt)

        total_content = self.function_separator + self.function_separator.join(
            tool_call_prompts
        )
        return prompt_template.format(text=total_content)

    def get_chat_template_jinja(self) -> str:
        chat_template = """{% for message in messages %}
        {% if message['role'] == 'user' or message['role'] == 'system' %}
            {% if 'metainfo' in message and 'img_path' in message['metainfo'] %}
                {{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n<|reserved_special_token_250|>\n' + message['content'] + '<|eot_id|>' }}<br>
            {% else %}
                {{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}<br>
        {% elif message['role'] == 'tool' %}
            {{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}<br>
        {% else %}
            {{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'}}<br>
            {% if message['content'] is not none %}
                {{ '>>>all\n' + message['content'] }}<br>
            {% endif %}
            {% if 'tool_calls' in message and message['tool_calls'] is not none %}
                {% for tool_call in message['tool_calls'] %}
                    {{ '>>>' + tool_call['function']['name'] + '\n' + tool_call['function']['arguments'] }}<br>
                {% endfor %}
            {% endif %}
            {{ '<|eot_id|>' }}<br>
        {% endif %}
        {% endfor %}
        {% if add_generation_prompt %}{{ '<|start_header_id|>{role}<|end_header_id|>\n\n' }}{% endif %}
        """
        chat_template = chat_template.replace("    ", "")
        chat_template = chat_template.replace("<br>\n", "")
        chat_template = chat_template.strip()
        return chat_template
