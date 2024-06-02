import json
import random
import string
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from functionary.openai_types import Tool
from functionary.prompt_template import prompt_utils
from functionary.prompt_template.base_template import PromptTemplate


class PromptTemplateV2(PromptTemplate):
    from_token = "<|from|>"
    recipient_token = "<|recipient|>"
    content_token = "<|content|>"
    stop_token = "<|stop|>"
    version = "v2"
    # This token splits between function name and parameters
    fn_param_sep_token = "\n<|content|>"

    def get_start_of_function_call_token(self) -> str:
        return self.recipient_token

    def initialize_grammar_sampling_gen_state(
        self,
        tool_choice: str,
        curr_text: str,
        curr_tokens: List[int],
        add_code_interpreter: bool,
    ) -> Dict:
        """Initializes grammar-sampling state

        Args:
            tool_choice (str): tool_choice provided by user
            curr_text (str): Text to initialize in gen_state
            curr_tokens (List[int]): Corresponding tokens of curr_text
            add_code_interpreter (bool): Flag indicating whether to add "python" tool in options in "function" stage.
        Returns:
            Dict: generation state
        """
        add_all_recipient = False
        # To force a text response ("tool_choice"="none")
        if tool_choice == "none":
            stage = "text-gen"
        # Normal generation (function name first without "all") (tool_choice="returned")
        elif tool_choice == "required":
            stage = "function"
        # To force a function call (tool_choice={"type": "function", "function": {...}})
        elif tool_choice != "":
            stage = "parameter"
        # Normal generation (function name first) (tool_choice="auto")
        else:
            add_all_recipient = True
            stage = "function"

        return {
            "stage": stage,
            "curr_tokens": curr_tokens,
            "curr_text": curr_text,
            "func_name": tool_choice,
            "add_all_recipient": add_all_recipient,
            "add_code_interpreter": add_code_interpreter,
        }

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
        options = []
        if gen_state["stage"] == "pre-function":
            options = [
                f"\n{self.from_token}assistant\n{self.recipient_token}",
                self.stop_token,
            ]
        elif gen_state["stage"] == "function":
            options = [tool_or_func["name"] for tool_or_func in tools_or_functions]
            if gen_state["add_all_recipient"]:
                options.append("all")
            if gen_state["add_code_interpreter"]:
                options.append("python")
        elif gen_state["stage"] == "pre-parameter":
            options = [self.fn_param_sep_token]

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
            self.update_grammar_sampling_gen_state(
                gen_state=gen_state,
                new_token_id=grammar_sampled_token_id,
                options=options,
                tokenizer=tokenizer,
            ),
        )

    def get_additional_tokens(self) -> List[str]:
        return [
            self.from_token,
            self.recipient_token,
            self.content_token,
            self.stop_token,
        ]

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
              - text-gen: when the model is generating content
              - code-interpreter: when the model is generating code
            - curr_tokens: all the tokens for the current stage being generated
            - curr_text: curr_tokens but in string text form
            - func_name: the function name, if any
            new_token_id (int): The token id of the newly sampled token
            options (List): All available function/param names depending on the stage of gen_state
            tokenizer (Any): The tokenizer class passed in from Transformers or vLLM
        Returns:
            dict: The updated gen_state
        """
        # Update curr_tokens and curr_text
        gen_state["curr_tokens"].append(new_token_id)
        gen_state["curr_text"] = tokenizer.decode(gen_state["curr_tokens"])

        # v2: "{func_name}\n<content|>{param_names}\n<|from|> assistant\n<|recipient|>"
        if gen_state["stage"] == "pre-function":
            # Check if the new state is in "function" stage
            if gen_state["curr_text"].endswith(self.get_start_of_function_call_token()):
                gen_state["stage"] = "function"
                gen_state["curr_text"], gen_state["curr_tokens"] = "", []
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
                gen_state["func_name"] = tool_name
                gen_state["curr_text"] = suffix
                gen_state["curr_tokens"] = [new_token_id] if suffix != "" else []
                gen_state["stage"] = "pre-parameter"

        elif gen_state["stage"] == "pre-parameter":
            if self.fn_param_sep_token in gen_state["curr_text"]:
                gen_state["curr_text"], gen_state["curr_tokens"] = "", []
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
                gen_state["curr_text"], gen_state["curr_tokens"] = "", []
            except:
                pass
        elif gen_state["stage"] in ["text-gen", "code-interpreter"]:
            # Check if the new state is in "function" stage
            # This happens when the text-gen is a COT or another fn is called after code-interpreter
            if gen_state["curr_text"].endswith(self.get_start_of_function_call_token()):
                gen_state["stage"] = "function"
                gen_state["curr_text"], gen_state["curr_tokens"] = "", []
                gen_state["func_name"] = ""

        return gen_state

    def convert_message_to_prompt(self, message: Dict) -> str:
        role = message["role"]
        content = message.get("content", None)

        if role in [
            "system",
            "user",
        ]:  # <|from|>system\n<|recipient|>all\n<|content|>xxx
            return f"{self.from_token}{role}\n{self.recipient_token}all\n{self.content_token}{content}\n"

        if role == "tool":  # <|from|>tool_name\n<|recipient|>all\n<|content|>xxx
            tool_name = message["name"]
            return f"{self.from_token}{tool_name}\n{self.recipient_token}all\n{self.content_token}{content}\n"

        assert role == "assistant"
        tool_calls = message.get("tool_calls", [])
        if tool_calls is None:
            tool_calls = []
        if (
            len(tool_calls) == 0 and content is None
        ):  # for inference: <|from|> assistant\n<|recipient|>
            return f"{self.from_token}{role}\n{self.recipient_token}"

        if len(tool_calls) == 0:  # <|from|>assistant\n<|recipient|>all\n<|content|>xxx
            return f"{self.from_token}{role}\n{self.recipient_token}all\n{self.content_token}{content}{self.stop_token}\n"

        result = ""
        if content is not None:  # both text-response and function_call
            result += f"{self.from_token}{role}\n{self.recipient_token}all\n{self.content_token}{content}\n"

        for tool in tool_calls:
            func_name = tool["function"]["name"]
            arguments = tool["function"]["arguments"]
            #  <|from|>assistant\n<|recipient|>func_name\n<|content|>xxxx
            result += f"{self.from_token}{role}\n{self.recipient_token}{func_name}\n{self.content_token}{arguments}\n"

        result = result.strip() + f"{self.stop_token}\n"
        return result

    def get_stop_tokens_for_generation(self) -> List[str]:
        return [self.stop_token]

    def get_assistant_prefixes(self) -> List[str]:
        return [f"{self.from_token}assistant\n{self.recipient_token}"]

    def parse_assistant_response(
        self, llm_output: str, tool_choice: Optional[Any] = None
    ) -> Dict:
        for stop in self.get_stop_tokens_for_generation():
            if llm_output.endswith(stop):
                llm_output = llm_output[: -len(stop)]

        recipient_to_fill = ""
        if tool_choice is not None:
            if tool_choice == "none":
                recipient_to_fill = "all" + self.fn_param_sep_token
            elif isinstance(tool_choice, Tool):
                recipient_to_fill = tool_choice.function.name + self.fn_param_sep_token

        llm_output = (
            f"{self.from_token}assistant\n{self.recipient_token}"
            + recipient_to_fill
            + llm_output
        )
        responses = llm_output.split(self.from_token)
        responses = [response.strip() for response in responses]

        tool_calls = []
        text_response = None
        for response in responses:
            if len(response) == 0:
                continue
            # response = assistant<|recipient|>xxxx\n<|content|>yyy
            recipient_index = response.find(self.recipient_token)
            content_index = response.find(self.content_token)
            recipient = response[
                recipient_index + len(self.recipient_token) : content_index
            ].strip()
            content = response[content_index + len(self.content_token) :].strip()
            # print(f"recipient: {recipient}, content={content}")
            if recipient == "all":
                text_response = content
            else:
                tool_calls.append(
                    {
                        "function": {"name": recipient, "arguments": content},
                        "id": prompt_utils.get_random_tool_call_id(),
                        "type": "function",
                    }
                )

        return {"role": "assistant", "content": text_response, "tool_calls": tool_calls}

    def pre_process_messages_before_inference(self, messages: List[Dict]) -> List[Dict]:
        """re-order the messages where role = tool to match the order in tool_calls by tool_call_id
        Args:
            messages (List[Dict]): list of messages containing: tool_call_id

        Returns:
            List[Dict]: _description_
        """
        return prompt_utils.reorder_tool_messages_by_tool_call_ids(messages)

    def get_recipient(self, current_text: str) -> str:
        """Get recipient from the llm_output

        Args:
            current_text (str): _description_

        Returns:
            str: _description_
        """
        recipient_index = current_text.find(self.recipient_token)
        start_index = 0
        if recipient_index >= 0:
            start_index = recipient_index + len(self.recipient_token)

        end_index = current_text.find(f"\n{self.content_token}")
        return current_text[start_index:end_index].strip()

    def get_chat_template_jinja(self) -> str:
        chat_template = """{% for message in messages %}
        {% if message['role'] == 'user' or message['role'] == 'system' %}
            {{ '<|from|>' + message['role'] + '\n<|recipient|>all\n<|content|>' + message['content'] + '\n' }}<br>
        {% elif message['role'] == 'tool' %}
            {{ '<|from|>' + message['name'] + '\n<|recipient|>all\n<|content|>' + message['content'] + '\n' }}<br>
        {% else %}
            {% set contain_content='no'%}
            {% if message['content'] is not none %}
                {{ '<|from|>assistant\n<|recipient|>all\n<|content|>' + message['content'] }}<br>
                {% set contain_content='yes'%}
            {% endif %}
            {% if 'tool_calls' in message and message['tool_calls'] is not none %}
                {% for tool_call in message['tool_calls'] %}
                    {% set prompt='<|from|>assistant\n<|recipient|>' + tool_call['function']['name'] + '\n<|content|>' + tool_call['function']['arguments'] %}
                    {% if loop.index == 1 and contain_content == "no" %}
                        {{ prompt }}<br>
                    {% else %}
                        {{ '\n' + prompt}}<br>
                    {% endif %}
                {% endfor %}
            {% endif %}
            {{ '<|stop|>\n' }}<br>
        {% endif %}
        {% endfor %}
        {% if add_generation_prompt %}{{ '<|from|>assistant\n<|recipient|>' }}{% endif %}
        """
        chat_template = chat_template.replace("    ", "")
        chat_template = chat_template.replace("<br>\n", "")
        chat_template = chat_template.strip()
        return chat_template

    def update_response_state_from_delta_text(
        self,
        *,
        current_state: Dict[str, Any],
        delta_text: str,
        finish_reason: Optional[str],
        tool_choice: Any,
    ) -> Tuple[Dict[str, Any], Union[None, Dict, List[Dict]]]:
        func_name = None
        response_type = None
        skip_until_reach = ""

        if len(current_state) == 0:  # empty dict, at the first_time
            response_type, skip_until_reach, func_name = None, self.content_token, None

            if tool_choice == "none":
                response_type, skip_until_reach = "text", ""
                func_name = "all"
            elif (
                type(tool_choice) is not str and tool_choice is not None
            ):  # tool_choice is a specific tool
                response_type, skip_until_reach = "function", ""
                func_name = tool_choice.function.name

            current_state = {
                "current_text": "",  # the concatenation of all tokens so far
                "func_name": func_name,  # function_name of the current tool, if the response requires to use tool
                "response_type": response_type,  # response_type=text(text response)/function (using tool)
                "func_index": -1,  # index of the tool in tool_calls
                "call_id": None,  # call_id of the current tool
                # skip_until_reach we skip new tokens until we reach certain token. This is used when we hit special tokens
                "skip_until_reach": skip_until_reach,  # at first we don't need to skip as we are generating function
                "first_time": True,  # if first_time we return an tempty delta with role=assistant
            }
        # check if previous token is <|content|>, there might be a space between this token and next token (ex, <|content|> Hello)
        if current_state["current_text"].endswith(self.content_token):
            if delta_text[0] == " ":
                delta_text = delta_text[1:]

        current_state["current_text"] += delta_text
        if finish_reason is not None:
            if current_state["response_type"] == "function":
                finish_reason = "tool_calls"

            return current_state, prompt_utils.get_text_delta_response(
                None, False, finish_reason
            )

        skip_until_reach = current_state.get("skip_until_reach", "")
        if skip_until_reach:
            if delta_text != skip_until_reach:
                return current_state, None
            else:
                current_state["skip_until_reach"] = ""  # once hit, no need to skip
                recipient = self.get_recipient(current_state["current_text"])
                first_time = current_state["first_time"]
                current_state["first_time"] = False

                if recipient == "all":
                    current_state["response_type"] = "text"
                    return current_state, prompt_utils.get_text_delta_response(
                        "", True, finish_reason
                    )
                else:
                    current_state["response_type"] = "function"
                    current_state["func_name"] = recipient
                    current_state["call_id"] = prompt_utils.get_random_tool_call_id()
                    current_state["func_index"] += 1

                    return current_state, prompt_utils.get_function_delta_response(
                        current_state, "", True, False, finish_reason
                    )
        else:
            assert current_state["response_type"] is not None

            first_time = current_state["first_time"]
            if (
                delta_text == self.from_token
            ):  # skip until reach <content> to check type of response
                current_state["current_text"] = ""
                current_state["skip_until_reach"] = self.content_token
                current_state["response_type"] = None
                return current_state, None

            else:
                if current_state["response_type"] == "function":
                    if first_time:
                        current_state["call_id"] = (
                            prompt_utils.get_random_tool_call_id()
                        )
                        current_state["func_index"] += 1
                        responses = []
                        responses.append(
                            prompt_utils.get_function_delta_response(
                                current_state, "", first_time, False, finish_reason
                            )
                        )
                        current_state["first_time"] = False
                        responses.append(
                            prompt_utils.get_function_delta_response(
                                current_state, delta_text, False, False, finish_reason
                            )
                        )
                    else:
                        responses = prompt_utils.get_function_delta_response(
                            current_state, delta_text, False, False, finish_reason
                        )
                    return current_state, responses
                else:  # response_type=text
                    responses = []
                    if first_time:
                        current_state["first_time"] = False
                        responses.append(
                            prompt_utils.get_text_delta_response(
                                "", True, finish_reason
                            )
                        )
                    responses.append(
                        prompt_utils.get_text_delta_response(
                            delta_text, True, finish_reason
                        )
                    )

                    return current_state, responses

    def get_force_text_generation_prefix(self):
        return f"all{self.fn_param_sep_token}"

    def get_force_function_call_prefix(self, function_name: str):
        return f"{function_name}{self.fn_param_sep_token}"
