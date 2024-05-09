from __future__ import annotations

import json
import re
from abc import abstractmethod
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from functionary.schema import generate_schema_from_functions

SYSTEM_MESSAGE = """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. The assistant calls functions with appropriate input when necessary"""
PYTHON_RUN_SYS_MSG = "When you send a message containing Python code to python, it will be executed in a stateful Jupyter notebook environment. python will respond with the output of the execution or time out after 60.0 seconds. The drive at '/mnt/data' can be used to save and persist user files."


class PredefinedFuncTypes(str, Enum):
    no_tool_call = "no-tool-call"
    code_interpreter = "code-interpreter"


class PromptTemplate:
    _instance = None

    @abstractmethod
    def get_start_of_function_call_token(self) -> str:
        """returns a token that indicates the start of a function call in the prompt template
        Returns:
            str: a string token
        """
        raise NotImplementedError

    def get_predefined_function_names(self, function_types: Any) -> List[str]:
        """returns a list of predefined function names. Some prompt template versions may
        require a default/predefined function name to indicate for example, no function called.
        E.g.: in v2, 'all' is generated to indicate normal model response. In this case, the v2
        subclass will overwrite this base method.
        Args:
            function_types (Any): Either "all" or one of the function type in PredefinedFuncTypes enum class
        Returns:
            List[str]: list of predefined function names (default to [])
        """
        return []

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
            - param_names: the parameters names, if any
        """
        raise NotImplementedError

    def update_grammar_sampling_gen_state(
        self,
        gen_state: Dict,
        new_token_id: int,
        options: Optional[List],
        tokenizer: Any,
    ) -> Dict:
        """Receives a generation state, updates and returns it. This is only used when
        grammar sampling is enabled in inference. This functions parses the generated
        tokens and identifies the stage of generation (pre-function, function, parameter,
        etc.)
        Args:
            gen_state (Dict): The current generation state. It contains the following:
            - stage: one of the following:
              - pre-function: the generation prior to function name generation
              - function: when the model is generating a function name
              - pre-parameter: when the model is generating the part between function name and parameter
              - parameter: when the model is generating parameters
              - no-tool-call: when the model is generating content
              - code-interpreter: when the model is generating code
            - curr_tokens: all the tokens for the current stage being generated
            - curr_text: curr_tokens but in string text form
            - func_name: the function name, if any
            - param_names: the parameters names, if any
            new_token_id (int): The token id of the newly sampled token
            options (List): All available function/param names depending on the stage of gen_state
            tokenizer (Any): The tokenizer class passed in from Transformers or vLLM
        Returns:
            dict: The updated gen_state
        """
        # Update curr_tokens and curr_text
        gen_state["curr_tokens"].append(new_token_id)
        gen_state["curr_text"] = tokenizer.decode(gen_state["curr_tokens"])

        # v1: "assistant:\n{content}\n{self.start_function}{function}:\n{arguments}\n"
        # v2: "{func_name}\n<content|>{param_names}\n<|from|> assistant\n<|recipient|>"
        if gen_state["stage"] == "pre-function":
            # Check if the new state is in "function" stage
            if gen_state["curr_text"].endswith(self.get_start_of_function_call_token()):
                gen_state = {
                    "stage": "function",
                    "curr_tokens": [],
                    "curr_text": "",
                    "func_name": "",
                    "param_names": [],
                    "add_predefined_fns": gen_state["add_predefined_fns"],
                }
                gen_state["stage"] = "function"

        elif gen_state["stage"] == "function":
            curr_text = gen_state["curr_text"]
            # Generate options_mask
            options_mask = [
                (
                    True
                    if option.startswith(curr_text.lstrip(" "))
                    or curr_text.lstrip(" ").startswith(option)
                    else False
                )
                for option in options
            ]
            # Transition to "pre-parameter" when only 1 element in options_mask is True
            if (
                sum(options_mask) == 1
                and curr_text == options[options_mask.index(True)]
            ):
                # Use the suffix from curr_text as the prefix in "pre-parameter"
                tool_name = options[options_mask.index(True)]
                suffix = curr_text[len(tool_name) :]
                gen_state["func_name"] = tool_name
                gen_state["curr_text"] = suffix
                gen_state["curr_tokens"] = [new_token_id] if suffix != "" else []
                gen_state["stage"] = "pre-parameter"

        elif gen_state["stage"] == "pre-parameter":
            if self.fn_param_sep_token in gen_state["curr_text"]:
                gen_state["curr_text"], gen_state["curr_tokens"] = "", []
                # Check if the new state is "no-tool-call" or "code-interpreter" or "parameter"
                if gen_state["func_name"] in self.get_predefined_function_names(
                    function_types=PredefinedFuncTypes.no_tool_call
                ):
                    gen_state["stage"] = "no-tool-call"
                elif gen_state["func_name"] in self.get_predefined_function_names(
                    function_types=PredefinedFuncTypes.code_interpreter
                ):
                    gen_state["stage"] = "code-interpreter"
                else:
                    gen_state["stage"] = "parameter"

        elif gen_state["stage"] == "parameter":
            # Get the latest param
            latest_param_str = gen_state["curr_text"]
            # Check if the new state is in "pre-function" stage
            try:
                _ = json.loads(latest_param_str)
                gen_state["stage"] = "pre-function"
            except:
                pass
        elif gen_state["stage"] in ["no-tool-call", "code-interpreter"]:
            # probability of stop token is not 100% at the end of no-tool-call
            # We still need to check if the stage will go to "function" by checking
            # for the presence of the start_of_function_call token
            if gen_state["curr_text"].endswith(self.get_start_of_function_call_token()):
                gen_state = {
                    "stage": "function",
                    "curr_tokens": [],
                    "curr_text": "",
                    "func_name": "",
                    "param_names": [],
                    "add_predefined_fns": gen_state["add_predefined_fns"],
                }

        return gen_state

    def grammar_sample(
        self,
        gen_state: Dict,
        tools_or_functions: List,
        delta_token_ids: List,
        model_sampled_token_id: int,
        tokenizer: Any,
    ) -> Tuple[int, str]:
        """Applies grammar-sampling to the token generation and returns a
        newly sampled token.

        For function name, the list of token ids sorted in descending order by
        log probabilities will be looped through until the token that fits the
        function names is reached. The grammar-sampled token replaces the
        output token if it is different from the model-sampled token.

        For parameter name, the lm-format-enforcer package is used to generate
        the parameters in JSON format, obeying the schema of the tool.

        Args:
            gen_state (Dict): The current generation state
            options (List): The list of available function/parameter names depending on gen_state["stage"]
            delta_token_ids (List): The list of delta token ids sorted in descending order by log probabilities
            model_sampled_token_id (int): The token id of the token sampled by model
            tokenizer (Any): The tokenizer object passed in from Transformers, vLLM, etc.
        Returns:
            Tuple[int, str]: Tuple of grammar-sampled token id and grammar-sampled token in str format
        """
        grammar_sampled_token_id, grammar_sampled_token = None, None

        # Form the functions options
        options = []
        if gen_state["stage"] == "function":
            options = [tool_or_func["name"] for tool_or_func in tools_or_functions]
            # Assume prompt template versions > 1 have "all" in function options
            # Subjected to changes in future versions
            # Concatenate the list of predefined function names in the respective prompt
            # template version. For e.g., v2 returns ["all"]
            if gen_state["add_predefined_fns"] is True:
                options += self.get_predefined_function_names(function_types="all")

        # No grammar sampling needed if gen_state not in "function" or
        # "pre-parameter" stages. Just return the model_sampled_token_id
        if gen_state["stage"] not in ["function", "pre-parameter"]:
            grammar_sampled_token_id = model_sampled_token_id
            grammar_sampled_token = tokenizer.decode([model_sampled_token_id])

        # Loop through the list of token ids sorted in descending order. For "function"
        # stage, form a mask made up of booleans where the index of the mask == index
        # of function name in function options. The element is True if the sampled_token
        # helps in forming the function. Else, False.
        if grammar_sampled_token_id is None:
            for i, sampled_token_ind in enumerate(delta_token_ids):
                sampled_token = tokenizer.decode(
                    [sampled_token_ind], add_special_tokens=False
                )
                # Form the function name with the current sampled token id
                new_curr_tokens_id = gen_state["curr_tokens"] + [sampled_token_ind]
                new_curr_tokens = tokenizer.decode(new_curr_tokens_id)

                if gen_state["stage"] == "function":
                    options_mask = [
                        (
                            True
                            if option.startswith(new_curr_tokens.lstrip(" "))
                            or new_curr_tokens.lstrip(" ").startswith(option)
                            else False
                        )
                        for option in options
                    ]

                    # - In case of two fns having common prefixes (e.g.: get_weather and
                    # get_weather_and_time), we need to iterate until parts of the
                    # fn_param_sep_token is present in new_curr_tokens to know if the
                    # shorter or longer function name is preferred by the model.
                    # - Reject the whitespace (" ") and empty ("") tokens
                    if any(options_mask) and sampled_token.strip(" ") != "":
                        grammar_sampled_token_id = sampled_token_ind
                        grammar_sampled_token = sampled_token
                        break
                # "pre-parameter" stage
                elif gen_state["stage"] == "pre-parameter":
                    # Check if new_curr_tokens is a prefix of fn_param_sep_token
                    if self.fn_param_sep_token.startswith(new_curr_tokens):
                        grammar_sampled_token_id = sampled_token_ind
                        grammar_sampled_token = sampled_token
                        break

        # Update gen_state
        return (
            grammar_sampled_token_id,
            grammar_sampled_token,
            self.update_grammar_sampling_gen_state(
                gen_state=gen_state,
                new_token_id=grammar_sampled_token_id,
                options=options,
                tokenizer=tokenizer,
            ),
        )

    @abstractmethod
    def get_additional_tokens(self) -> List[str]:
        """return list of added tokens if using this template
        Returns:
            List[str]: list of tokens, each token is a string
        """
        raise NotImplementedError

    @abstractmethod
    def convert_message_to_prompt(self, message: Dict) -> str:
        """Return the prompt of this message

        Args:
            message (Dict): Dictionary of openAI format

        Returns:
            str: prompt of this message
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
        messages_clone = messages.copy()  # To avoid modifying the original list

        functions = []
        is_code_interpreter = False
        if tools_or_functions is not None:
            for item in tools_or_functions:
                if (
                    "function" in item and item["function"] is not None
                ):  #  new data format: tools: [{"type": xx, "function": xxx}]
                    functions.append(item["function"])
                elif "type" in item and item["type"] == "code_interpreter":
                    is_code_interpreter = True
                else:
                    functions.append(item)  #  old format

        messages_clone.insert(
            0, {"role": "system", "content": generate_schema_from_functions(functions)}
        )
        if is_code_interpreter:
            messages_clone.insert(1, {"role": "system", "content": PYTHON_RUN_SYS_MSG})
        else:
            messages_clone.insert(1, {"role": "system", "content": SYSTEM_MESSAGE})

        return messages_clone

    def get_prompt_from_messages(
        self,
        messages: List[Dict],
        tools_or_functions: Optional[List[Dict]] = None,
    ) -> str:
        """This function is used to get the complete prompt for list of messages

        Args:
            messages (List[Dict]): List of messages
            tools_or_functions (Optional[List[Dict]], optional): List of tools or functions. Defaults to None.

        Returns:
            str: the prompt for inference/training
        """
        messages_clone = self.inject_system_messages_based_on_tools(
            messages, tools_or_functions
        )

        full_text = ""
        for message in messages_clone:
            full_text += self.convert_message_to_prompt(message)
        return (
            full_text  # Do not strip because llama3-instruct uses: \n\n before content
        )

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
    def get_force_function_call_prefix(self, function_name: str):
        """This function will be used for force-function call 

        Args:
            function_name (str): _description_

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError

    @abstractmethod
    def get_chat_template_jinja(self):
        """Return chat_template in jinja format"""
        raise NotImplementedError

    @classmethod
    def get_prompt_template(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
        return cls._instance
