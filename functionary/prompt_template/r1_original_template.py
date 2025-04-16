import datetime
import json
import re
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from functionary.openai_types import Function, Tool
from functionary.prompt_template import prompt_utils
from functionary.prompt_template.base_template import PromptTemplate


def parse_tool_call(tool_call_str: str) -> Dict:
    """parse the tool name and arguments from the tool call string. the format of the tool call string is:
    function<｜tool▁sep｜><function_name>
    ```type
    <argument_info>
    ```

    Args:
        tool_call_str (str): _description_

    Returns:
        Dict: {"name": <function_name>, "args": <argument_info>}
    """
    tool_sep_index = tool_call_str.find("<｜tool▁sep｜>")
    brkline_index = tool_call_str.find("\n", tool_sep_index)
    function_name = tool_call_str[
        tool_sep_index + len("<｜tool▁sep｜>") : brkline_index
    ].strip()
    # parse arguments
    arguments_content = tool_call_str[brkline_index:].strip()
    # strip ``` at the begining and the end
    arguments_content = arguments_content.replace("```", "")
    index = arguments_content.find("\n")  # ignore: json\n or python\n
    arguments_content = arguments_content[index:].strip()

    return {
        "id": prompt_utils.get_random_tool_call_id(),
        "type": "function",
        "function": {"name": function_name, "arguments": arguments_content},
    }


def extract_text_inside(start_prefix: str, end_prefix: str, text: str) -> List[str]:
    """extract all text inside the start_prefix and end_prefix, return a list of texts inside these two prefixes

    Args:
        start_prefix (str): the prefix before the text to extract
        end_prefix (str): the prefix after the text to extract
        text (str): the text to extract

    Returns:
        List[str]: a list of texts inside these two prefixes
    """
    result = []
    current_pos = 0

    while True:
        # Find next start position
        start_pos = text.find(start_prefix, current_pos)
        if start_pos == -1:
            break

        # Find matching end position
        end_pos = text.find(end_prefix, start_pos + len(start_prefix))
        if end_pos == -1:
            break

        # Extract text between prefixes
        extracted = text[start_pos + len(start_prefix) : end_pos]
        result.append(extracted)

        # Move current position past this match
        current_pos = end_pos + len(end_prefix)

    return result


class R1Template(PromptTemplate):
    version = "r1"

    def get_assistant_prefixes(self) -> List[str]:
        return [f"<｜Assistant｜>"]

    def pre_process_messages_before_inference(self, messages: List[Dict]) -> List[Dict]:
        """Order the tool results by the order of tool call ids

        Args:
            messages (List[Dict]): List of messages

        Returns:
            List[Dict]: List of messages
        """
        return prompt_utils.reorder_tool_messages_by_tool_call_ids(messages)

    def get_stop_tokens_for_generation(self) -> List[str]:
        return ["<｜end▁of▁sentence｜>"]

    def parse_assistant_response(
        self, llm_output: str, tool_choice: Optional[Any] = None
    ) -> Dict:
        for stop in self.get_stop_tokens_for_generation():
            if llm_output.endswith(stop):
                llm_output = llm_output[: -len(stop)]

        # add forced-function from tool_choice if exists
        llm_output = (
            self.get_generation_prefix_for_tool_choice(tool_choice) + llm_output
        )
        text_content = llm_output
        tool_calls = []
        # parse the tool calls
        # first extract the string about tool calls, which is inside: <｜tool▁calls▁begin｜><｜tool▁calls▁end｜>
        all_tool_calls_list = extract_text_inside(
            "<｜tool▁calls▁begin｜>", "<｜tool▁calls▁end｜>", llm_output
        )
        if len(all_tool_calls_list) > 0:
            all_tool_calls_str = all_tool_calls_list[0]
            index = llm_output.find("<｜tool▁calls▁begin｜")
            text_content = text_content[:index]
            # extract tool calls inside: <｜tool▁call▁begin｜> & <｜tool▁call▁end｜>
            tool_calls_strs = extract_text_inside(
                "<｜tool▁call▁begin｜>", "<｜tool▁call▁end｜>", all_tool_calls_str
            )
            for tool_call_str in tool_calls_strs:
                tool_calls.append(parse_tool_call(tool_call_str))
        return {
            "role": "assistant",
            "content": text_content if text_content else None,
            "tool_calls": tool_calls if tool_calls else None,
        }
