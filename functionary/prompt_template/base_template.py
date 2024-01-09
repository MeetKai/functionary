from __future__ import annotations

import json
import re
from abc import abstractmethod
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from functionary.schema import generate_schema_from_functions

SYSTEM_MESSAGE = """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. The assistant calls functions with appropriate input when necessary"""


class PromptTemplate:
    _instance = None

    @abstractmethod
    def get_start_of_function_call_token(self) -> str:
        """returns a token that indicates the start of a function call in the prompt template
        Returns:
            str: a string token
        """
        raise NotImplementedError

    @abstractmethod
    def get_stop_token_for_function_parameter(
        self, stage: Literal["function", "parameter"]
    ) -> str:
        """returns a str token which stops function/parameter name generation
        e.g.: `"get_current_weather` with v1 prompt template -> returns id = 28747 (':' token)
        so the generation gets forced towards `"get_current_weather:\n{...`
        Args:
            stage (str): Whether to get function name or parameter name stopping token
        Returns:
            str: str token
        """
        raise NotImplementedError

    def get_predefined_function_names(self) -> List[str]:
        """returns a list of predefined function names. Some prompt template versions may
        require a default/predefined function name to indicate for example, no function called.
        E.g.: in v2, 'all' is generated to indicate normal model response. In this case, the v2
        subclass will overwrite this base method.
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
              - parameter-name: when the model is generating a parameter name
              - parameter-value: when the model is generating a parameter value
              - no-function-call: when the model is generating content
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
        tokens and identifies the stage of generation (pre-function, function, parameter-name,
        etc.)
        Args:
            gen_state (Dict): The current generation state. It contains the following:
            - stage: one of the following:
              - pre-function: the generation prior to function name generation
              - function: when the model is generating a function name
              - pre-parameter: when the model is generating the part between function name and parameter
              - parameter-name: when the model is generating a parameter name
              - parameter-value: when the model is generating a parameter value
              - no-function-call: when the model is generating content
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
            # Remove all unnecessary suffixes by checking whether stop token is in curr_text
            if (
                self.get_stop_token_for_function_parameter(stage="function")
                in gen_state["curr_text"]
            ):
                curr_text = gen_state["curr_text"].rstrip()
                while True:
                    if any([curr_text == option for option in options]):
                        break
                    curr_text = curr_text[:-1]
                gen_state["func_name"] = curr_text
            else:
                gen_state["func_name"] = gen_state["curr_text"].rstrip()

            # Check if the new state is in "pre-parameter" or "no-function-call" stage
            if (
                sum([gen_state["func_name"] == option for option in options]) == 1
                and sum(
                    [option.startswith(gen_state["func_name"]) for option in options]
                )
                == 1
            ):
                gen_state["stage"] = "pre-parameter"

                # Update curr_text and curr_tokens
                if (
                    self.get_stop_token_for_function_parameter(stage="function")
                    in gen_state["curr_text"]
                ):
                    gen_state["curr_text"] = tokenizer.decode([new_token_id])
                    gen_state["curr_tokens"] = [new_token_id]
                else:
                    gen_state["curr_text"], gen_state["curr_tokens"] = "", []
        elif gen_state["stage"] == "pre-parameter":
            # Check if the new state is in "parameter" or "no-function-call" stage
            if (
                self.fn_param_sep_token.rstrip("{").rstrip() in gen_state["curr_text"]
                and gen_state["func_name"] in self.get_predefined_function_names()
            ):
                gen_state["stage"] = "no-function-call"
            # Either '{' or '{"' or '{}'
            elif self.fn_param_sep_token in gen_state["curr_text"]:
                # Check if no arguments are called and go straight to "pre-function"
                if "}" in gen_state["curr_text"]:
                    gen_state["stage"] = "pre-function"
                elif '"' in gen_state["curr_text"]:
                    gen_state["stage"] = "parameter-name"
                    if gen_state["curr_text"].endswith('"'):
                        gen_state["curr_text"], gen_state["curr_tokens"] = "", []
                    else:
                        gen_state["curr_tokens"] = [new_token_id]
                        gen_state["curr_text"] = tokenizer.decode([new_token_id])
        elif gen_state["stage"] == "parameter-name":
            # Get the latest param
            latest_param_str = gen_state["curr_text"]

            # Remove unneccesary prefixes before the parameter-name part
            if len(gen_state["curr_tokens"]) > 0 and '"' in tokenizer.decode(
                [gen_state["curr_tokens"][0]]
            ):
                latest_param_str = latest_param_str[latest_param_str.find('"') + 1 :]

            # Check if the new state is in "parameter-value" stage
            stop_token = self.get_stop_token_for_function_parameter(stage="parameter")
            if stop_token in latest_param_str:
                pattern = stop_token + r".*$"
                match_res = re.search(pattern, latest_param_str, re.DOTALL)
                if bool(match_res):
                    gen_state["param_names"].append(
                        gen_state["curr_text"].removesuffix(match_res.group(0))
                    )
                    gen_state["stage"] = "parameter-value"
                    gen_state["curr_text"] = match_res.group(0)
                    new_tokens = []
                    for token in gen_state["curr_tokens"][::-1]:
                        new_tokens = [token] + new_tokens
                        next_text = tokenizer.decode(new_tokens)
                        if next_text.endswith(match_res.group(0)):
                            gen_state["curr_tokens"] = new_tokens
                            break
        elif gen_state["stage"] == "parameter-value":
            latest_param_val = gen_state["curr_text"]
            stop_token = self.get_stop_token_for_function_parameter(stage="parameter")

            # Remove unnecessary prefixes in latest_param_val
            if not latest_param_val.startswith(stop_token):
                latest_param_val = latest_param_val[latest_param_val.find(stop_token) :]

            # Check if the new state is in "pre-function" stage
            try:
                _ = json.loads('{"' + gen_state["param_names"][-1] + latest_param_val)
                gen_state["stage"] = "pre-function"
            except:
                pass

            # Check if the current state can be converted to json, it means the
            # new state is back to "parameter-name" stage
            pattern = r',[\s]*"'
            match_res = re.findall(pattern, latest_param_val, re.DOTALL)
            if '"' in tokenizer.decode(new_token_id) and len(match_res) > 0:
                latest_match = match_res[-1]
                try:
                    _ = json.loads(
                        '{"'
                        + gen_state["param_names"][-1]
                        + latest_param_val[: latest_param_val.rfind(latest_match)]
                        + "}"
                    )
                    gen_state["stage"] = "parameter-name"
                    if latest_param_val.endswith('"'):
                        gen_state["curr_text"], gen_state["curr_tokens"] = "", []
                    else:
                        gen_state["curr_tokens"] = [new_token_id]
                        gen_state["curr_text"] = tokenizer.decode([new_token_id])
                except:
                    pass
        elif gen_state["stage"] == "no-function-call":
            # probability of stop token is not 100% at the end of no-function-call
            # We still need to check if the stage will go to "function" by checking
            # for the presence of the start_of_function_call token
            if gen_state["curr_text"].endswith(self.get_start_of_function_call_token()):
                gen_state = {
                    "stage": "function",
                    "curr_tokens": [],
                    "curr_text": "",
                    "func_name": "",
                    "param_names": [],
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

        This function checks whether the model-sampled token helps towards
        forming one of the function names or parameter names. It loops through
        a list of token ids sorted in descending order by the log probabilities.
        It replaces the output token if the grammar-sampled token is different
        from the model-sampled token
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

        # Form the functions/parameters options
        options = []
        if gen_state["stage"] in ["pre-function", "function"]:
            options = [tool_or_func["name"] for tool_or_func in tools_or_functions]
        elif gen_state["stage"] == "pre-parameter":
            options = [self.fn_param_sep_token]
        else:
            func_name = gen_state["func_name"]
            for tool_or_func in tools_or_functions:
                if tool_or_func["name"] == func_name:
                    options = list(tool_or_func["parameters"]["properties"].keys())
                    break
        # Assume prompt template versions > 1 have "all" in function options
        # Subjected to changes in future versions
        # Concatenate the list of predefined function names in the respective prompt
        # template version. For e.g., v2 returns ["all"]
        if gen_state["stage"] == "function" and gen_state["add_predefined_fns"] is True:
            options += self.get_predefined_function_names()

        # No grammar sampling needed if gen_state not in "function" or "pre-parameter"
        # or "parameter-name" stages. Just return the model_sampled_token_id
        if gen_state["stage"] not in ["function", "pre-parameter", "parameter-name"]:
            grammar_sampled_token_id = model_sampled_token_id
            grammar_sampled_token = tokenizer.decode([model_sampled_token_id])

        # Loop through the list of token ids sorted in descending order. Form a mask made
        # up of booleans where the index of the mask == index of function/parameter name
        # in function/parameter options. The element is True if the sampled_token
        # helps in forming the function/parameter name. Else, False.
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
                        True
                        if option.startswith(new_curr_tokens.lstrip(" "))
                        or new_curr_tokens.lstrip(" ").startswith(option)
                        else False
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
                elif gen_state["stage"] == "pre-parameter":
                    # Get the suffix after fn_param_sep_token and check if crit_char is in it
                    if self.fn_param_sep_token in new_curr_tokens:
                        suffix = new_curr_tokens[
                            new_curr_tokens.index(self.fn_param_sep_token)
                            + len(self.fn_param_sep_token) :
                        ]
                    else:
                        suffix = new_curr_tokens
                    crit_bool = any([crit_char in suffix for crit_char in ['"', "}"]])

                    options_mask = []
                    for option in options:
                        if option.startswith(new_curr_tokens.lstrip(" ")) or crit_bool:
                            options_mask.append(True)
                        else:
                            options_mask.append(False)

                    # We just need to check if the option (fn_param_sep_token) is True
                    # or fn_param_sep_token + one of ['}', '"'] is present
                    if any(options_mask) and sampled_token.strip(" ") != "":
                        grammar_sampled_token_id = sampled_token_ind
                        grammar_sampled_token = sampled_token
                        break
                else:
                    # Mask away those wellformed parameter names while creating options_mask
                    wellformed_params = gen_state["param_names"]

                    # Remove unneccesary prefixes before the parameter-name part
                    if len(gen_state["curr_tokens"]) > 0 and '"' in tokenizer.decode(
                        [gen_state["curr_tokens"][0]]
                    ):
                        new_curr_tokens = new_curr_tokens[
                            new_curr_tokens.find('"') + 1 :
                        ]

                    options_mask = []
                    for option in options:
                        if option not in wellformed_params and option.startswith(
                            new_curr_tokens
                        ):
                            options_mask.append(True)
                        else:
                            options_mask.append(False)

                    # Same logic as function name, except that we check whether the token
                    # is a stopping token for parameter name generation.
                    if (
                        (
                            self.get_stop_token_for_function_parameter(
                                stage="parameter"
                            )
                            in new_curr_tokens
                        )
                        or any(options_mask)
                        and sampled_token.strip(" ") != ""
                    ):
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
        messages_clone = messages.copy()  # To avoid modifying the original list

        functions = []
        if tools_or_functions is not None:
            for item in tools_or_functions:
                if (
                    "function" in item
                ):  #  new data format: tools: [{"type": xx, "function": xxx}]
                    functions.append(item["function"])
                else:
                    functions.append(item)  #  old format

        messages_clone.insert(
            0, {"role": "system", "content": generate_schema_from_functions(functions)}
        )
        messages_clone.insert(1, {"role": "system", "content": SYSTEM_MESSAGE})

        full_text = ""
        for message in messages_clone:
            full_text += self.convert_message_to_prompt(message)
        return full_text.strip()

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
    def get_chat_template_jinja(self):
        """Return chat_template in jinja format"""
        raise NotImplementedError

    @classmethod
    def get_prompt_template(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
        return cls._instance
