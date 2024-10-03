import datetime
import json
import re
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from functionary.openai_types import Function, Tool
from functionary.prompt_template import prompt_utils
from functionary.prompt_template.base_template import PromptTemplate
import copy

FN_NAME = "✿FUNCTION✿"
FN_ARGS = "✿ARGS✿"
FN_RESULT = "✿RESULT✿"
FN_EXIT = "✿RETURN✿"
FN_STOP_WORDS = [FN_RESULT, FN_EXIT]

FN_CALL_TEMPLATE_INFO_ZH = """# 工具

## 你拥有如下工具：

{tool_descs}"""

FN_CALL_TEMPLATE_INFO_EN = """# Tools

## You have access to the following tools:

{tool_descs}"""

FN_CALL_TEMPLATE_FMT_ZH = """## 你可以在回复中插入零次、一次或多次以下命令以调用工具：

%s: 工具名称，必须是[{tool_names}]之一。
%s: 工具输入
%s: 工具结果
%s: 根据工具结果进行回复，需将图片用![](url)渲染出来""" % (
    FN_NAME,
    FN_ARGS,
    FN_RESULT,
    FN_EXIT,
)

FN_CALL_TEMPLATE_FMT_EN = """## When you need to call a tool, please insert the following command in your reply, which can be called zero or multiple times according to your needs:

%s: The tool to use, should be one of [{tool_names}]
%s: The input of the tool
%s: Tool results
%s: Reply based on tool results. Images need to be rendered as ![](url)""" % (
    FN_NAME,
    FN_ARGS,
    FN_RESULT,
    FN_EXIT,
)

FN_CALL_TEMPLATE_FMT_PARA_ZH = """## 你可以在回复中插入以下命令以并行调用N个工具：

%s: 工具1的名称，必须是[{tool_names}]之一
%s: 工具1的输入
%s: 工具2的名称
%s: 工具2的输入
...
%s: 工具N的名称
%s: 工具N的输入
%s: 工具1的结果
%s: 工具2的结果
...
%s: 工具N的结果
%s: 根据工具结果进行回复，需将图片用![](url)渲染出来""" % (
    FN_NAME,
    FN_ARGS,
    FN_NAME,
    FN_ARGS,
    FN_NAME,
    FN_ARGS,
    FN_RESULT,
    FN_RESULT,
    FN_RESULT,
    FN_EXIT,
)

FN_CALL_TEMPLATE_FMT_PARA_EN = """## Insert the following command in your reply when you need to call N tools in parallel:

%s: The name of tool 1, should be one of [{tool_names}]
%s: The input of tool 1
%s: The name of tool 2
%s: The input of tool 2
...
%s: The name of tool N
%s: The input of tool N
%s: The result of tool 1
%s: The result of tool 2
...
%s: The result of tool N
%s: Reply based on tool results. Images need to be rendered as ![](url)""" % (
    FN_NAME,
    FN_ARGS,
    FN_NAME,
    FN_ARGS,
    FN_NAME,
    FN_ARGS,
    FN_RESULT,
    FN_RESULT,
    FN_RESULT,
    FN_EXIT,
)

FN_CALL_TEMPLATE = {
    "zh": FN_CALL_TEMPLATE_INFO_ZH + "\n\n" + FN_CALL_TEMPLATE_FMT_ZH,
    "en": FN_CALL_TEMPLATE_INFO_EN + "\n\n" + FN_CALL_TEMPLATE_FMT_EN,
    "zh_parallel": FN_CALL_TEMPLATE_INFO_ZH + "\n\n" + FN_CALL_TEMPLATE_FMT_PARA_ZH,
    "en_parallel": FN_CALL_TEMPLATE_INFO_EN + "\n\n" + FN_CALL_TEMPLATE_FMT_PARA_EN,
}


def get_function_description(function: Dict, lang: Literal["en", "zh"] = "en") -> str:
    """
    Text description of function
    """
    tool_desc_template = {
        "zh": "### {name_for_human}\n\n{name_for_model}: {description_for_model} 输入参数：{parameters} {args_format}",
        "en": "### {name_for_human}\n\n{name_for_model}: {description_for_model} Parameters: {parameters} {args_format}",
    }
    tool_desc = tool_desc_template[lang]
    name = function.get("name", None)
    name_for_human = function.get("name_for_human", name)
    name_for_model = function.get("name_for_model", name)
    assert name_for_human and name_for_model

    if name_for_model == "code_interpreter":
        args_format = {
            "zh": "此工具的输入应为Markdown代码块。",
            "en": "Enclose the code within triple backticks (`) at the beginning and end of the code.",
        }
    else:
        args_format = {
            "zh": "此工具的输入应为JSON对象。",
            "en": "Format the arguments as a JSON object.",
        }
    args_format = function.get("args_format", args_format[lang])

    return tool_desc.format(
        name_for_human=name_for_human,
        name_for_model=name_for_model,
        description_for_model=function["description"],
        parameters=json.dumps(function["parameters"], ensure_ascii=False),
        args_format=args_format,
    ).rstrip()


def convert_tools_to_system_message_content(functions: List[Dict], lang: str = "en"):
    tool_desc_template = FN_CALL_TEMPLATE[lang + ("_parallel")]
    tool_descs = "\n\n".join(
        get_function_description(function, lang=lang) for function in functions
    )
    tool_names = ",".join(
        function.get("name", function.get("name_for_model", ""))
        for function in functions
    )
    tool_system_content = tool_desc_template.format(
        tool_descs=tool_descs, tool_names=tool_names
    )
    return tool_system_content


def convert_openai_assistant_message_to_string(
    message: Dict, is_after_user_message: bool
):
    assistant_content = message.get("content", "")
    if assistant_content is None:
        assistant_content = ""

    tool_calls = message.get("tool_calls", [])
    if tool_calls:
        tool_call_info = []
        for tool_call in tool_calls:
            function_name = tool_call["function"]["name"]
            arguments = tool_call["function"]["arguments"]
            if function_name == "python":
                arguments = f"```\n{arguments}\n```"
            tool_call_str = f"✿FUNCTION✿: code_interpreter\n✿ARGS✿: {arguments}"
            tool_call_info.append(tool_call_str)

        if is_after_user_message:
            assistant_content += "\n".join(tool_call_info)
        else:  # previous message is tool
            if assistant_content:
                assistant_content = f"✿RETURN✿: {assistant_content}\n" + "\n".join(
                    tool_call_info
                )
    return assistant_content + "\n"


def convert_openai_tool_message_to_string(message):
    content = message["content"]
    return f"✿RESULT✿: {content}\n"


class Qwen2VLTemplate(PromptTemplate):
    version = "qwen2-vl"

    def get_additional_tokens(self) -> List[str]:
        return []

    def get_assistant_prefixes(self) -> List[str]:
        return ["<|im_start|>assistant\n"]

    def pre_process_messages_before_inference(self, messages: List[Dict]) -> List[Dict]:
        """Order the tool results by the order of tool call ids

        Args:
            messages (List[Dict]): List of messages

        Returns:
            List[Dict]: List of messages
        """
        return prompt_utils.reorder_tool_messages_by_tool_call_ids(messages)

    def get_stop_tokens_for_generation(self) -> List[str]:
        return ["<|im_end|>", "✿RESULT✿", "✿RETURN✿"]

    def get_chat_template_jinja(self):
        return super().get_chat_template_jinja()

    def get_prompt_from_messages(
        self,
        messages: List[Dict],
        tools_or_functions: Optional[List[Dict]] = None,
        bos_token: Optional[str] = "",
        add_generation_prompt: bool = False,
    ) -> str:
        """This function is used to get the complete prompt for list of messages

        Args:
            messages (List[Dict]): List of messages
            tools_or_functions (Optional[List[Dict]], optional): List of tools or functions. Defaults to None.

        Returns:
            str: the prompt for inference/training
        """
        if not tools_or_functions:  # if no functions are available
            return self._jinja_template.render(
                messages=messages,
                tools=tools_or_functions,
                bos_token=bos_token,
                add_generation_prompt=add_generation_prompt,
            )

        functions = []
        for func in tools_or_functions:
            if func["type"] == "function":
                functions.append(func["function"])
            elif func["type"] == "code_interpreter":
                functions.append({
                    "name": "code_interpreter",
                    "description": "",
                    "parameters": {}
                })
                # functions.append(
                #     {
                #         "name": "python",
                #         "description": 'When you send a message containing Python code to python, it will be executed in a stateful Jupyter notebook environment. python will respond with the output of the execution or time out after 60.0 seconds. The drive at "/mnt/data" can be used to save and persist user files.',
                #         "parameters": {
                #             "type": "object",
                #             "properties": {
                #                 "code": {"type": "string", "description": "Python code"}
                #             },
                #             "required": ["code"],
                #         },
                #     }
                # )

        tool_system_message_content = "You are a helpful assistant.\n\n"
        tool_system_message_content += convert_tools_to_system_message_content(
            functions
        )
        new_messages = [{"role": "system", "content": tool_system_message_content}]
        # merge tool messages into assistant message

        index = 0
        while index < len(messages):
            if messages[index]["role"] == "assistant":
                tool_calls = messages[index].get("tool_calls", [])
                if (
                    tool_calls
                ):  # at this time, assistant decided to use tools, need to find the final assistant message (no tool_calls)
                    final_assistant_index = -1
                    for j in range(index + 1, len(messages)):
                        if messages[j]["role"] == "assistant":
                            tool_calls = messages[j].get("tool_calls", [])
                            if not tool_calls:
                                final_assistant_index = j
                                break
                    if (
                        final_assistant_index == -1
                    ):  # if this request wasn't handled completely, set this as the last message
                        final_assistant_index = len(messages) - 1

                    total_assistant_content = ""
                    for j in range(index, final_assistant_index + 1):
                        assert messages[j]["role"] in ["assistant", "tool"]
                        if messages[j]["role"] == "assistant":
                            if not messages[j].get(
                                "tool_calls", []
                            ):  # no more tool_call --> this is the final response
                                final_response = messages[j]["content"]
                                total_assistant_content += f"✿RETURN✿: {final_response}"
                            else:
                                is_after_user_message = True if j == index else False
                                total_assistant_content += (
                                    convert_openai_assistant_message_to_string(
                                        messages[j], is_after_user_message
                                    )
                                )
                        else:  # role == tool
                            total_assistant_content += (
                                convert_openai_tool_message_to_string(messages[j])
                            )
                    new_messages.append(
                        {
                            "role": "assistant",
                            "content": total_assistant_content.strip(),
                        }
                    )
                    index = final_assistant_index + 1
                else:
                    new_messages.append(messages[index])
                    index += 1
            else:
                new_messages.append(messages[index])
                index += 1

        prompt = self._jinja_template.render(
            messages=new_messages,
            tools=tools_or_functions,
            bos_token=bos_token,
            add_generation_prompt=False,
        )
        # check last message to decide if the assistant turn is completed or not
        # last message is: system, user, assistant, tool 
        prompt = prompt.strip()
        if messages[-1]["role"] == "tool": # incomplete assistant's handling
            if prompt.endswith("<|im_end|>"):
                prompt = prompt[: -len("<|im_end|>")]
        else:
            prompt += "\n<|im_start|>assistant\n"
        return prompt
