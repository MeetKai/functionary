from functionary.prompt_template.base_template import PromptTemplate
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from functionary.prompt_template import prompt_utils
from functionary.openai_types import Function, Tool
import json
import copy
import math
import re
from lxml import etree

# the following lines of code are copied from qwen_vl_utils
IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200

VIDEO_MIN_PIXELS = 128 * 28 * 28
VIDEO_MAX_PIXELS = 768 * 28 * 28
VIDEO_TOTAL_PIXELS = 24576 * 28 * 28
FRAME_FACTOR = 2
FPS = 2.0
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 768


def remove_emoji(text: str) -> str:
    """
    Remove emoji characters (such as: 📐, ⚗) from the text
    """
    # Unicode ranges for emojis
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"  # dingbats
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )

    return emoji_pattern.sub("", text)


def post_process_llm_output(llm_output: str) -> str:
    """
    Post-process the LLM output to remove emoji characters and other non-printable characters
    """
    llm_output = remove_emoji(llm_output)
    # fix invalid xml, inside llm_output, there can be invalid in complete xml tag, for example, it contains  <tool_call> but </tool_call> is missing
    # we need to check if there are missing </tool_call> and add it. Note thare there can be multiple tool_call tags in the llm_output
    # Count opening and closing tool_call tags
    open_tags = llm_output.count("<tool_call>")
    close_tags = llm_output.count("</tool_call>")

    # Add missing closing tags if needed
    if open_tags > close_tags:
        missing_tags = open_tags - close_tags
        llm_output = llm_output + "</tool_call>" * missing_tags
    return llm_output


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int = IMAGE_FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


class Qwen25PromptTemplate(PromptTemplate):
    version = "qwen2.5"
    start_of_turn = "<|im_start|>"
    end_of_turn = "<|im_end|>"

    def get_chat_template_jinja(self) -> str:
        path_prefix = "./functionary/prompt_template/jinja_templates/"
        with open(f"{path_prefix}{self.version}.txt", "r") as f:
            template = f.read()

        return template

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
        # handle code_interpreter
        _tools = []
        if tools_or_functions:
            for tool in tools_or_functions:
                if tool["type"] == "code_interpreter":
                    _tools.append(
                        {
                            "type": "function",
                            "function": {
                                "name": "python",
                                "description": "This tool is used to execute python code. Code will be executed in a stateful Jupyter notebook environment. Python will respond with the output of the execution or time out after 60.0 seconds. The drive at '/mnt/data' can be used to save and persist user files.",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "code": {
                                            "type": "string",
                                            "description": "The python code to run",
                                        }
                                    },
                                },
                            },
                        }
                    )
                else:
                    _tools.append(tool)

        # find the assistant message that tool_call is python
        _messages = []
        for message in messages:
            n_message = copy.deepcopy(message)
            tool_calls = n_message.get("tool_calls", []) or []
            if len(tool_calls) > 0:
                for tool_call in tool_calls:
                    if tool_call["function"]["name"] == "python":
                        arguments = tool_call["function"][
                            "arguments"
                        ]  # currently the code is in string format
                        # check if argument is a valid JSON string or python code
                        try:  # if this is a valid JSON string --> no need to change anything
                            json.loads(arguments)
                        except:
                            tool_call["function"]["arguments"] = json.dumps(
                                {"code": arguments}, ensure_ascii=False
                            )
            _messages.append(n_message)

        prompt = super().get_prompt_from_messages(
            messages=_messages,
            tools_or_functions=_tools,
            bos_token=bos_token,
            add_generation_prompt=add_generation_prompt,
        )
        return prompt

    def get_additional_tokens(self) -> List[str]:
        return []

    def get_assistant_prefixes(self) -> List[str]:
        return [f"{self.start_of_turn}assistant\n"]

    def get_stop_tokens_for_generation(self) -> List[str]:
        return [self.end_of_turn]

    def get_force_function_call_prefix(self, function_name: str):
        return (
            """<tool_call>
{"name": "%s}"""
            % function_name
        )

    def pre_process_messages_before_inference(self, messages: List[Dict]) -> List[Dict]:
        """Order the tool results by the order of tool call ids

        Args:
            messages (List[Dict]): List of messages

        Returns:
            List[Dict]: List of messages
        """
        return prompt_utils.reorder_tool_messages_by_tool_call_ids(messages)

    def parse_assistant_response(
        self, llm_output: str, tool_choice: Any = None
    ) -> Dict:
        # first remove stop tokens if there exists
        for stop in self.get_stop_tokens_for_generation():
            if llm_output.endswith(stop):
                llm_output = llm_output[: -len(stop)]

        # add forced-function from tool_choice if exists
        if type(tool_choice) is not str and tool_choice is not None:
            llm_output = (
                self.get_force_function_call_prefix(tool_choice.function.name)
                + llm_output
            )
        elif tool_choice == "required":
            llm_output = self.function_separator + llm_output

        print(f"+++LLM_OUTPUT: {llm_output}")
        llm_output = post_process_llm_output(llm_output)
        print(f"+++LLM_OUTPUT after post-processing: {llm_output}")
        text_content = ""
        tool_call_strs = []

        # Split on tool call tags
        parts = llm_output.split("<tool_call>")

        if len(parts) > 0:
            # First part is the text content
            text_content = parts[0].strip()

            # Process remaining parts as tool calls
            for part in parts[1:]:
                if "</tool_call>" in part:
                    tool_call = part.split("</tool_call>")[0].strip()
                    if tool_call:
                        tool_call_strs.append(tool_call)
        tool_calls = []
        for tool_call_str in tool_call_strs:
            tool_call_dic = json.loads(tool_call_str)
            tool_calls.append(
                {
                    "type": "function",
                    "id": prompt_utils.get_random_tool_call_id(),
                    "function": {
                        "name": tool_call_dic["name"],
                        "arguments": json.dumps(
                            tool_call_dic["arguments"], ensure_ascii=False
                        ),
                    },
                }
            )

        return {
            "role": "assistant",
            "content": text_content if len(text_content) > 0 else None,
            "tool_calls": tool_calls,
        }

    def preprocess_image_input(self, image: Any) -> Any:
        width, height = image.size
        min_pixels = MIN_PIXELS
        max_pixels = MAX_PIXELS
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=IMAGE_FACTOR,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        image = image.resize((resized_width, resized_height))
        return image

    def initialize_fsm_gen_state(
        self,
        tool_choice: Union[str, Tool],
        curr_text: str,
        curr_tokens: Optional[List[int]],
        add_code_interpreter: Optional[bool],
    ) -> Dict:
        """Initializes FSM state for both streaming and grammar sampling

        Args:
            tool_choice (str): tool_choice provided by user
            curr_text (str): Text to initialize in gen_state
            curr_tokens (List[int]): Corresponding tokens of curr_text
            add_code_interpreter (bool): Flag indicating whether to add "python" tool in options in "function" stage.
        Returns:
            Dict: generation state
        """
        result = {
            "stage": "start",
            "func_index": -1,
            "curr_text": curr_text,
            "curr_tokens": curr_tokens,
            "add_code_interpreter": add_code_interpreter,
        }
        return result

    def stream_delta_text(
        self,
        gen_state: Dict,
        delta_text: str,
        finish_reason: Optional[str],
        tools_or_functions: List[Dict],
        tool_choice: Any,
    ) -> Tuple[Dict, Optional[Union[Dict, List[Dict]]]]:
        if finish_reason is not None:  # handle if finish
            if gen_state["stage"] not in ["text_gen"]:
                finish_reason = "tool_calls"
            return gen_state, prompt_utils.get_text_delta_response(
                None, False, finish_reason
            )

        current_text = gen_state["curr_text"] + delta_text
        gen_state["curr_text"] = current_text
        if gen_state["stage"] == "start":
            if (
                gen_state.get("end_of_prev_function_call", False) and delta_text == "\n"
            ):  # ignore \n
                gen_state["end_of_prev_function_call"] = False
                gen_state["curr_text"] = ""

            elif delta_text == "<tool_call>":
                # print(f"delta text: {delta_text}; go to function_name")
                gen_state["stage"] = "function_name"
                gen_state["curr_text"] = ""
                gen_state["func_index"] += 1
                gen_state["call_id"] = prompt_utils.get_random_tool_call_id()
            else:
                # print(f"delta text: {delta_text}; go to text_gen")
                gen_state["stage"] = "text_gen"
                gen_state["curr_text"] = current_text
                gen_state["buffer"] = (
                    []
                )  # put to buffer before we return because we need to check the last item
                responses = [
                    prompt_utils.get_text_delta_response("", True, finish_reason)
                ]
                if len(delta_text) > 0:
                    gen_state["buffer"].append(delta_text)
                return gen_state, responses

        elif gen_state["stage"] == "function_name":
            # wait until we get '{"name": "func_name", "arguments": {'
            # print(f"current_text: {current_text}")
            pattern = (
                r'\s*{"name"\s*:\s*"(?P<function_name>.*)"\s*,\s*"arguments"\s*:\s*{'
            )
            match = re.search(pattern, current_text)
            if match:
                _, end_ind = match.start(), match.end()
                new_delta = current_text[end_ind - 1 :]
                gen_state["stage"] = "function_arguments"
                gen_state["curr_text"] = new_delta  # -1 to retain "{"
                gen_state["func_name"] = match.group("function_name")
                responses = [
                    prompt_utils.get_function_delta_response(
                        gen_state, "", True, True, finish_reason
                    )
                ]  # the chunk containing function_name only
                # generate another chunk for start of function_arguments
                gen_state["buffer"] = [
                    new_delta
                ]  # we only generate a chunk if len(buffer) >= 4
                # responses.append(prompt_utils.get_function_delta_response(
                #     gen_state, new_delta, False, False, finish_reason
                # ))
                return gen_state, responses
            else:
                return gen_state, None

        elif gen_state["stage"] == "text_gen":
            if delta_text == "<tool_call>":  # start a tool call
                # print("start a tool call after reasoning")
                gen_state["stage"] = "function_name"
                gen_state["curr_text"] = ""
                gen_state["func_index"] += 1
                gen_state["call_id"] = prompt_utils.get_random_tool_call_id()
                buffer_str = "".join(gen_state["buffer"]).strip()
                if len(buffer_str) > 0:
                    return gen_state, prompt_utils.get_text_delta_response(
                        buffer_str, False, finish_reason
                    )
            else:
                gen_state["buffer"].append(delta_text)
                if len(gen_state["buffer"]) >= 2:
                    delta_text_item = gen_state["buffer"].pop(0)
                    return gen_state, prompt_utils.get_text_delta_response(
                        delta_text_item, False, finish_reason
                    )

        elif gen_state["stage"] == "function_arguments":
            # generate until reach </tool_call>
            if delta_text == "</tool_call>":
                gen_state["stage"] = "start"
                gen_state["curr_text"] = ""
                gen_state["end_of_prev_function_call"] = True
                # return all in the buffer but need to strip and remove the last "}"
                buffer_str = "".join(gen_state["buffer"]).strip()
                if buffer_str.endswith("}}"):
                    buffer_str = buffer_str[:-1]  # remove the last "}"
                return gen_state, prompt_utils.get_function_delta_response(
                    gen_state, buffer_str, False, False, finish_reason
                )
            else:
                gen_state["buffer"].append(delta_text)
                if len(gen_state["buffer"]) >= 4:
                    delta_text_item = gen_state["buffer"].pop(0)
                    return gen_state, prompt_utils.get_function_delta_response(
                        gen_state, delta_text_item, False, False, finish_reason
                    )
        return gen_state, None


def match_pattern(pattern: str, text: str) -> Tuple[int, int]:
    match = re.search(pattern, text)
    if match:
        return match.start(), match.end()
    return -1, -1
