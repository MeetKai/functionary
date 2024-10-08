from __future__ import annotations

import json
import re
from abc import abstractmethod
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import jinja2

from functionary.openai_types import Function, Tool
from functionary.prompt_template import prompt_utils
from PIL import Image


def raise_exception(message):
    raise jinja2.exceptions.TemplateError(message)


def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    # We override the built-in tojson filter because Jinja's default filter escapes HTML characters
    # We also expose some options like custom indents and separators
    return json.dumps(
        x,
        ensure_ascii=ensure_ascii,
        indent=indent,
        separators=separators,
        sort_keys=sort_keys,
    )


class PromptTemplate:
    _jinja_env = jinja2.Environment()
    _jinja_env.filters["tojson"] = tojson
    _jinja_env.globals["raise_exception"] = raise_exception
    # Mapping from class --> instance to create singleton instance
    _instances = {}

    def __init__(self):
        self._jinja_template = self._jinja_env.from_string(
            self.get_chat_template_jinja()
        )

    @abstractmethod
    def get_start_of_function_call_token(self) -> str:
        """returns a token that indicates the start of a function call in the prompt template
        Returns:
            str: a string token
        """
        raise NotImplementedError

    @abstractmethod
    def initialize_grammar_sampling_gen_state(self, tool_choice: Optional[Any]) -> Dict:
        """initializes and returns a new generation state. Each template version may be initialized
        at different starting stage
        Args:
            tool_choice (Optional[Any]): the tool_choice provided by the user, if any
        Returns:
            dict: the gen_state. It contains the following:
            - stage: one of the following:
              - pre-function: the generation prior to function name generation
              - function: when the model is generating a function name
              - pre-parameter: when the model is generating the part between function name and parameter
              - parameter: when the model is generating parameters
              - no-tool-call: when the model is generating content
            - curr_tokens: all the tokens for the current stage being generated
            - curr_text: curr_tokens but in string text form
            - func_name: the function name, if any
        """
        raise NotImplementedError

    @abstractmethod
    def get_additional_tokens(self) -> List[str]:
        """return list of added tokens if using this template
        Returns:
            List[str]: list of tokens, each token is a string
        """
        raise NotImplementedError

    @abstractmethod
    def get_stop_tokens_for_generation(self) -> List[str]:
        """Function to get list of stop tokens in generation

        Returns:
            List[str]: list of stop tokens
        """
        raise NotImplementedError

    @abstractmethod
    def get_assistant_prefixes(self) -> List[str]:
        """Return the assistant prefixs in the final prompt, this is used for masking the labels
        in unmasking labels, the system will unmask chunks that start with assistant prefixs and end with stop tokens.
        For example, assistant_prefixes might be: "<|from|>assistant\n<|recipient|>"
        In this case unmasked chunks in labels would be tokens in ... of: <|from|>assistant\n<|recipient|> ... <|stop|>
        Returns:
            List[str]: list of possible assistant prefixs
        """
        raise NotImplementedError

    def pre_process_messages_before_inference(self, messages: List[Dict]) -> List[Dict]:
        """This function is used if we need to process messages before doing inference.
        This is used when the messages in training and inference are different.
        For example, in training we have no: tool_call_id, but in inference, we have tool_call_id to know the order of function calls.
        This function woule be called to convert inference messages to the format of training messages.
        Args:
            messages (List[Dict]): list of input messages

        Returns:
            List[Dict]: list of output messages
        """
        return messages

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

        prompt = self._jinja_template.render(
            messages=messages,
            tools=tools_or_functions,
            bos_token=bos_token,
            add_generation_prompt=add_generation_prompt,
        )

        return prompt

    def get_end_token_to_token_id(self, tokenizer: Any) -> Dict[str, int]:
        """return a dictionary mapping from end_token --> token_id
        Args:
            tokenizer (Any): tokenizer in transformers

        Returns:
            Dict[int, EndToken]: the mapping from token_id --> end_token
        """
        result = {}
        for item in self.get_stop_tokens_for_generation():
            tok_ids = tokenizer.encode(item, add_special_tokens=False)
            assert len(tok_ids) <= 2, ""
            if len(tok_ids) == 2:
                assert tok_ids[0] in [
                    29871,
                    28705,
                ]  # Llama tokenizer adds this token intentionally
            result[item] = tok_ids[-1]
        return result

    @abstractmethod
    def parse_assistant_response(
        self, llm_output: str, tool_choice: Optional[Any]
    ) -> Dict:
        """This function is used to parse llm_output to the Message of OpenAI ({"role": xxx, "content": xxx, ...})
        this is used in inference.
        Args:
            llm_output (str): The generated content from Model
            tool_choice (Optional[Any]): Any choice of tool provided by the user

        Returns:
            Dict: Dictionary of OpenAI message format
        """
        raise NotImplementedError

    @abstractmethod
    def update_response_state_from_delta_text(
        self,
        *,
        current_state: Dict[str, Any],
        delta_text: str,
        finish_reason: Optional[str],
        tool_choice: Any,
    ) -> Tuple[Dict[str, Any], Union[None, Dict, List[Dict]]]:
        """This function is used for streaming

        Args:
            current_state (Dict[str, Any]):  a dictionary containing the state of the streaming: such as current function_name,
            delta_text: new token generated
            finish_reason: if finished or not

        Returns:
            Tuple[Dict[str, Any], Optional[Dict]]: updated state, response: can be None, a dictionary: {} or a list of dictionary: [{}, ..., {}]
        """
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
    def stream_delta_text(
        self,
        gen_state: Dict,
        delta_text: str,
        finish_reason: Optional[str],
        tools_or_functions: List[Dict],
        tool_choice: Any,
    ) -> Tuple[Dict, Optional[Union[Dict, List[Dict]]]]:
        """This function is used for streaming

        Args:
            gen_state (Dict[str, Any]):  a dictionary containing the state of the streaming: such as current function_name,
            delta_text: new token generated
            finish_reason: if finished or not
            tools_or_functions: list of tools or functions
            tool_choice: tool_choice

        Returns:
            Tuple[Dict[str, Any], Optional[Dict]]: updated state, response: can be None, a dictionary: {} or a list of dictionary: [{}, ..., {}]
        """

    def _update_gen_state_for_fn_call(self, gen_state: Dict, func_name: str):
        """update the state when a function is going to be called

        Args:
            current_state (_type_): _description_
        """
        gen_state["func_name"] = func_name
        gen_state["func_index"] += 1
        gen_state["call_id"] = prompt_utils.get_random_tool_call_id()
        gen_state["first_time_func"] = True

        return gen_state

    def _reset_fsm_curr_text_and_tokens(self, gen_state: Dict):
        gen_state["curr_text"] = ""
        gen_state["curr_tokens"] = [] if gen_state["curr_tokens"] is not None else None

        return gen_state

    @abstractmethod
    def update_fsm_gen_state(
        self,
        gen_state: Dict,
        new_token: Optional[str],
        new_token_id: Optional[int],
        options: Optional[List],
        tokenizer: Any,
    ) -> Dict:
        """Receives a generation state, updates and returns it. This functions parses the generated
        tokens and identifies the stage of generation (pre-function, function, parameter, etc.)

        Args:
            gen_state (Dict): The current generation state. It contains the following:
            - stage: some of the common stages:
              - function: when the model is generating a function name
              - pre-parameter: when the model is generating the part between function name and parameter
              - parameter: when the model is generating parameters
              - text-gen: when the model is generating content
              - code-interpreter: when the model is generating code
            - curr_tokens: all the tokens for the current stage being generated
            - curr_text: curr_tokens but in string text form
            - func_name: the function name, if any
            new_token (str): The newly sampled token in string
            new_token_id (int): The token id of the newly sampled token
            options (List): All available function/param names depending on the stage of gen_state
            tokenizer (Any): The tokenizer class passed in from Transformers or vLLM

        Returns:
            dict: The updated gen_state
        """
        raise NotImplementedError

    @abstractmethod
    def get_options_from_gen_state(self, gen_state: Dict, tools_or_functions: List):
        """Gets a list of options for grammar sampling to generate tokens to form given the gen state

        Args:
            gen_state (Dict): _description_
            tools_or_functions (List): _description_

        Returns:
            _type_: _description_
        """
        return []

    def get_force_text_generation_prefix(self):
        """This function will be used for force-text generation. Returns empty string by default"""
        return ""

    @abstractmethod
    def get_force_function_call_prefix(self, function_name: str):
        """This function will be used for force-function call

        Args:
            function_name (str): _description_

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError

    def get_tool_choice_required_prefix(self):
        """This function will be used when tool_choice='required'. Returns empty string by default"""
        return ""

    def get_raw_response_from_assistant_message(
        self,
        message: Dict[str, Any],
        tool_func_choice: Union[str, Tool, Function],
        default_tool_call_name: str,
    ):
        """This function generates a mock raw response from a assistant message dict
        given the tool_func_choice. This function is used in test_request_handling.py
        to unittest the processing of raw response to OpenAI response message.

        Args:
            message (Dict[str, Any]): _description_
            tool_func_choice (Union[str, Tool, Function]): _description_
            default_tool_call_name (str): _description_

        Returns:
            str: The mock raw response in str format
        """
        # Form raw response from messages list
        sys_msg = self.get_prompt_from_messages(
            messages=[], tools_or_functions=[], add_generation_prompt=True
        )
        assistant_msg = self.get_prompt_from_messages(
            messages=[message], tools_or_functions=[]
        )
        raw_response = assistant_msg[len(sys_msg) :]

        # Remove stop tokens
        for stop_token in self.get_stop_tokens_for_generation():
            raw_response = raw_response.replace(stop_token, "")

        generation_prefix = self.get_generation_prefix_for_tool_choice(tool_func_choice)
        raw_response = raw_response[len(generation_prefix) :]

        return raw_response.rstrip()

    def get_chat_template_jinja(self) -> str:
        path_prefix = "./functionary/prompt_template/jinja_templates/"
        with open(f"{path_prefix}json_to_ts_schema.txt", "r") as f:
            json_to_ts_schema = f.read()
        with open(f"{path_prefix}{self.version}.txt", "r") as f:
            template = f.read()

        return (
            template[: template.index("{%")]
            + json_to_ts_schema
            + "\n"
            + template[template.index("{%") :]
        )

    def get_generation_prefix_for_tool_choice(self, tool_choice: Any):
        if tool_choice == "auto" or tool_choice is None:
            return ""
        if tool_choice == "required":
            return self.get_tool_choice_required_prefix()
        elif tool_choice == "none":
            return self.get_force_text_generation_prefix()
        elif isinstance(tool_choice, Tool):
            return self.get_force_function_call_prefix(tool_choice.function.name)
        elif isinstance(tool_choice, Function):
            return self.get_force_function_call_prefix(tool_choice.name)
        raise Exception(
            "tool-choice must be one of: None, none, auto, required, or a specific tool"
        )

    def preprocess_image_input(self, image: Image) -> Image:
        return image

    @classmethod
    def get_prompt_template(cls):
        if cls._instances.get(cls, None) is None:
            cls._instances[cls] = cls()
        return cls._instances[cls]
