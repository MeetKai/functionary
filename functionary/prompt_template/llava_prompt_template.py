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

        # handle the case when user uploads images (content is a list)
        if role == "user" and type(content) is list:
            text_content = prompt_utils.stringify_content_with_images(
                message["content"], self.image_token
            )
            return f"{self.start_header}{role}{self.end_header}\n\n{text_content}{self.eos_token}"
        return super().convert_message_to_prompt(message)

    def get_chat_template_jinja(self) -> str:
        chat_template = """{% for message in messages %}
        {% if message['role'] == 'user'%}
            {{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' }}<br>
            {% if message['content'] is iterable and (message['content'] is not string and message['content'] is not mapping) %}
                {% for content_item in message['content'] %}
                    {% if content_item['type'] == 'image_url' %}
                        {{ '<|reserved_special_token_250|>' }}<br>
                    {% else %}
                        {{ content_item['text'] }}<br>
                    {% endif %}
                {% endfor %}
            {% else %}
                {{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}<br>
            {% endif %}
        {% elif message['role'] == 'tool' or message['role'] == 'system' %}
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
