import json
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from functionary.openai_types import Function, Tool
from functionary.prompt_template import prompt_utils
from functionary.prompt_template.llama3_prompt_template_v3 import Llama3TemplateV3
from functionary.schema import generate_schema_from_functions


class InternLMChat(Llama3TemplateV3):
    version = "internlm2-chat"
    start_img_token = "<img>"
    # "<img>", "<IMG_CONTEXT>", "</img>"
    end_img_token = "</img>"
    img_context = "<IMG_CONTEXT>"
    start_of_turn = "<|im_start|>"
    eos_token = "<|im_end|>"
    function_separator = ">>>"

    def inject_system_messages_based_on_tools(
        self, messages: List[Dict], tools_or_functions: Optional[List[Dict]] = None
    ) -> List[Dict]:
        """This will be used to add Default system message, code-interpreter system message if needed

        Args:
            messages (List[Dict]): List of messages
            tools_or_functions (Optional[List[Dict]], optional): List of tools, functions. Defaults to None.

        Returns:
            List[Dict]: _description_
        """
        result = super(Llama3TemplateV3, self).inject_system_messages_based_on_tools(
            messages, tools_or_functions
        )
        default_system_message = {
            "role": "system",
            "content": "你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。",
        }
        result = [default_system_message] + result
        return result

    def get_assistant_prefixes(self) -> List[str]:
        return [f"{self.start_of_turn}assistant\n{self.function_separator}"]

    def get_stop_tokens_for_generation(self) -> List[str]:
        return [self.eos_token]

    def get_force_function_call_prefix(self, function_name: str):
        return f"{function_name}\n"

    def get_start_of_function_call_token(self) -> str:
        return ""

    def pre_process_messages_before_inference(self, messages: List[Dict]) -> List[Dict]:
        """Order the tool results by the order of tool call ids

        Args:
            messages (List[Dict]): List of messages

        Returns:
            List[Dict]: List of messages
        """
        return prompt_utils.reorder_tool_messages_by_tool_call_ids(messages)

    def convert_message_to_prompt(self, message: Dict) -> str:
        role = message["role"]
        content = message.get("content", None)

        if role == "user" and type(content) is list:
            content = prompt_utils.stringify_content_with_images(
                message["content"], self.start_img_token
            )

        if role == "tool":
            tool_name = message["name"]
            content = f"name={tool_name}\n{content}"

        prompt_template = f"{self.start_of_turn}{role}\n" + "{text}" + self.eos_token

        if role in ["user", "system", "tool"]:
            return prompt_template.format(text=content)

        assert role == "assistant", f"role must be assistant, but: {role}"
        tool_calls = message.get("tool_calls", [])
        if tool_calls is None:
            tool_calls = []

        if content is None and len(tool_calls) == 0:  # inference time
            return f"{self.start_of_turn}{role}\n{self.function_separator}"

        if content is not None:  # text-only
            tool_calls = [
                {"function": {"name": "all", "arguments": content}}
            ] + tool_calls

        # list of text representing function calls: {function_name}\n{arguments}
        tool_call_prompts = []
        for tool_call in tool_calls:
            arguments = tool_call["function"]["arguments"]
            tool_name = tool_call["function"]["name"]
            tool_prompt = f"{tool_name}\n{arguments}"
            tool_call_prompts.append(tool_prompt)

        # join all function calls
        total_content = self.function_separator + self.function_separator.join(
            tool_call_prompts
        )
        return prompt_template.format(text=total_content)

    def get_force_text_generation_prefix(self):
        return f"all\n"

    def get_chat_template_jinja(self) -> str:
        chat_template = """{% for message in messages %}
        {% if message['role'] == 'user' or message['role'] == 'system' %}
            {{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' }}<br>
        {% elif message['role'] == 'tool' %}
            {{ '<|im_start|>' + message['role'] + '\nname=' + message['name'] + '\n' + message['content'] + '<|im_end|>' }}<br>
        {% else %}
            {{ '<|im_start|>' + message['role'] + '\n'}}<br>
            {% if message['content'] is not none %}
                {{ '>>>all\n' + message['content'] }}<br>
            {% endif %}
            {% if 'tool_calls' in message and message['tool_calls'] is not none %}
                {% for tool_call in message['tool_calls'] %}
                    {{ '>>>' + tool_call['function']['name'] + '\n' + tool_call['function']['arguments'] }}<br>
                {% endfor %}
            {% endif %}
            {{ '<|im_end|>' }}<br>
        {% endif %}
        {% endfor %}
        {% if add_generation_prompt %}{{ '<|im_start|>{role}\n' }}{% endif %}
        """
        chat_template = chat_template.replace("    ", "")
        chat_template = chat_template.replace("<br>\n", "")
        chat_template = chat_template.strip()
        return chat_template
