import json
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from functionary.openai_types import Function, Tool
from functionary.prompt_template import prompt_utils
from functionary.prompt_template.base_template import PromptTemplate


class Llama3TemplateV3(PromptTemplate):
    version = "v3.llama3"
    function_separator = ">>>"
    start_header = "<|start_header_id|>"
    end_header = "<|end_header_id|>"
    eos_token = "<|eot_id|>"
    # This token splits between function name and parameters
    fn_param_sep_token = "\n"

    def get_additional_tokens(self) -> List[str]:
        return []

    def get_assistant_prefixes(self) -> List[str]:
        return [
            f"{self.start_header}assistant{self.end_header}\n\n{self.function_separator}"
        ]

    def get_stop_tokens_for_generation(self) -> List[str]:
        return [self.eos_token, "<|end_of_text|>"]

    def get_force_function_call_prefix(self, function_name: str):
        return f"{function_name}\n"

    def get_start_of_function_call_token(self) -> str:
        return self.function_separator

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

        # Form the options for the following stages
        options = self.get_options_from_gen_state(
            gen_state=gen_state, tools_or_functions=tools_or_functions
        )

        # No grammar sampling needed if gen_state not in the following stages. Return model_sampled_token_id
        if gen_state["stage"] not in ["pre-function", "function", "pre-parameter"]:
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
                elif gen_state["stage"] in ["pre-function", "pre-parameter"]:
                    # Check if new_curr_tokens is a prefix of any of options
                    if any([option.startswith(new_curr_tokens) for option in options]):
                        grammar_sampled_token_id = sampled_token_ind
                        grammar_sampled_token = sampled_token
                        break

        # Update gen_state
        return (
            grammar_sampled_token_id,
            grammar_sampled_token,
            self.update_fsm_gen_state(
                gen_state=gen_state,
                new_token=None,
                new_token_id=grammar_sampled_token_id,
                options=options,
                tokenizer=tokenizer,
            ),
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

        llm_output = (
            self.get_generation_prefix_for_tool_choice(tool_choice) + llm_output
        )

        chunks = llm_output.split(self.function_separator)
        chunks = [chunk.strip() for chunk in chunks if len(chunk.strip()) > 0]

        tool_calls = []
        text_content = None

        for chunk in chunks:
            # format: function_name\narguments<end_of_functioncall>
            index = chunk.find("\n")
            func_name = chunk[:index].strip()
            arguments = chunk[index + 1 :].strip()
            if func_name == "all":
                text_content = arguments
            else:
                tool_calls.append(
                    {
                        "function": {"name": func_name, "arguments": arguments},
                        "id": prompt_utils.get_random_tool_call_id(),
                        "type": "function",
                    }
                )
        if len(tool_calls) == 0:
            tool_calls = None

        return {"role": "assistant", "content": text_content, "tool_calls": tool_calls}

    def get_force_text_generation_prefix(self):
        return f"all\n"

    def initialize_fsm_gen_state(
        self,
        tool_choice: Union[str, Tool],
        curr_text: str,
        curr_tokens: Optional[List[int]],
        add_code_interpreter: Optional[bool],
    ) -> Dict:
        add_all_recipient = False
        func_name = None
        # To force a text response ("tool_choice"="none")
        if tool_choice == "none":
            stage = "text-gen"
        # Normal generation (function name first without "all") (tool_choice="returned")
        elif tool_choice == "required":
            stage = "function"
        # To force a function call (tool_choice={"type": "function", "function": {...}})
        elif not isinstance(tool_choice, str):
            stage = "parameter"
            func_name = (
                tool_choice.function.name
                if isinstance(tool_choice, Tool)
                else tool_choice.name
            )
        # Normal generation (function name first) (tool_choice="auto")
        else:
            add_all_recipient = True
            stage = "function"

        gen_state = {
            "stage": stage,
            "curr_tokens": curr_tokens,
            "curr_text": curr_text,
            "func_name": func_name,
            "func_index": -1,  # index of the tool in tool_calls
            "call_id": None,  # call_id of the current tool
            "first_chunk": True,
            "first_function_chunk": True,
            "add_all_recipient": add_all_recipient,
            "add_code_interpreter": add_code_interpreter,
        }

        return (
            self._update_gen_state_for_fn_call(gen_state, func_name)
            if func_name is not None
            else gen_state
        )

    def stream_delta_text(
        self,
        gen_state: Dict,
        delta_text: str,
        finish_reason: Optional[str],
        tools_or_functions: List[Dict],
        tool_choice: Any,
    ) -> Tuple[Dict, Optional[Union[Dict, List[Dict]]]]:
        if finish_reason is not None:  # handle if finish
            if gen_state["func_name"] is not None and gen_state["func_name"] != "all":
                finish_reason = "tool_calls"
            return gen_state, prompt_utils.get_text_delta_response(
                None, False, finish_reason
            )

        responses = []

        options = self.get_options_from_gen_state(
            gen_state=gen_state, tools_or_functions=tools_or_functions
        )

        if gen_state["stage"] == "text-gen":
            if delta_text != self.function_separator:
                if gen_state["first_chunk"]:
                    responses.append(
                        prompt_utils.get_text_delta_response("", True, finish_reason)
                    )
                    gen_state["first_chunk"] = False
                    if gen_state["curr_text"] != "":
                        responses.append(
                            prompt_utils.get_text_delta_response(
                                gen_state["curr_text"], False, finish_reason
                            )
                        )
                responses.append(
                    prompt_utils.get_text_delta_response(
                        delta_text, False, finish_reason
                    )
                )
        elif gen_state["stage"] in ["parameter", "code-interpreter"]:
            if delta_text != self.function_separator:
                if gen_state["first_function_chunk"]:
                    responses.append(
                        prompt_utils.get_function_delta_response(
                            gen_state, "", True, gen_state["first_chunk"], finish_reason
                        )
                    )
                    gen_state["first_chunk"] = False
                    gen_state["first_function_chunk"] = False

                    if gen_state["curr_text"] != "":
                        responses.append(
                            prompt_utils.get_function_delta_response(
                                gen_state,
                                gen_state["curr_text"],
                                False,
                                False,
                                finish_reason,
                            )
                        )
                responses.append(
                    prompt_utils.get_function_delta_response(
                        gen_state, delta_text, False, False, finish_reason
                    )
                )

        gen_state = self.update_fsm_gen_state(
            gen_state=gen_state,
            new_token=delta_text,
            new_token_id=None,
            options=options,
            tokenizer=None,
        )

        return gen_state, responses

    def update_fsm_gen_state(
        self,
        gen_state: Dict,
        new_token: Optional[str],
        new_token_id: Optional[str],
        options: Optional[List],
        tokenizer: Any,
    ) -> Dict:
        if gen_state["curr_tokens"] is not None:
            # Update curr_tokens and curr_text
            gen_state["curr_tokens"].append(new_token_id)
            gen_state["curr_text"] = tokenizer.decode(gen_state["curr_tokens"])
        else:
            gen_state["curr_text"] += new_token

        # v2: "{func_name}\n<content|>{param_names}\n<|from|> assistant\n<|recipient|>"
        if gen_state["stage"] == "pre-function":
            # Check if the new state is in "function" stage
            if self.get_start_of_function_call_token() in gen_state["curr_text"]:
                gen_state["stage"] = "function"
                curr_text = gen_state["curr_text"]
                gen_state = self._reset_fsm_curr_text_and_tokens(gen_state=gen_state)
                gen_state["curr_text"] = curr_text.removeprefix(
                    self.get_start_of_function_call_token()
                )
                gen_state["func_name"] = ""

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
                if tool_name == "all":
                    gen_state["func_name"] = tool_name
                else:
                    gen_state = self._update_gen_state_for_fn_call(
                        gen_state=gen_state, func_name=tool_name
                    )
                gen_state = self._reset_fsm_curr_text_and_tokens(gen_state=gen_state)
                # Jump to "parameter" stage if suffix is "\n"
                gen_state["stage"] = "pre-parameter" if suffix == "" else "parameter"

        elif gen_state["stage"] == "pre-parameter":
            if self.fn_param_sep_token in gen_state["curr_text"]:
                curr_text = gen_state["curr_text"]
                gen_state = self._reset_fsm_curr_text_and_tokens(gen_state=gen_state)
                gen_state["curr_text"] = curr_text.removeprefix(self.fn_param_sep_token)
                # Check if the new state is "text-gen" or "code-interpreter" or "parameter"
                if gen_state["func_name"] == "all":
                    gen_state["stage"] = "text-gen"
                elif gen_state["func_name"] == "python":
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
                gen_state = self._reset_fsm_curr_text_and_tokens(gen_state=gen_state)
            except:
                pass
        elif gen_state["stage"] in ["text-gen", "code-interpreter"]:
            # Check if the new state is in "function" stage
            # This happens when the text-gen is a COT or another fn is called after code-interpreter
            if gen_state["curr_text"].endswith(self.get_start_of_function_call_token()):
                gen_state["stage"] = "function"
                gen_state = self._reset_fsm_curr_text_and_tokens(gen_state=gen_state)
                gen_state["func_name"] = ""

        return gen_state

    def get_options_from_gen_state(self, gen_state: Dict, tools_or_functions: List):
        # Form the options for the following stages
        options = []
        if gen_state["stage"] == "pre-function":
            options = [self.function_separator, self.eos_token]
        elif gen_state["stage"] == "function":
            options = [(tool_or_func["name"]) for tool_or_func in tools_or_functions]
            if gen_state["add_all_recipient"]:
                options.append("all")
            if gen_state["add_code_interpreter"]:
                options.append("python")
        elif gen_state["stage"] == "pre-parameter":
            options = [self.fn_param_sep_token]

        return options
