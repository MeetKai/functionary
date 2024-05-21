import json
import random
import string
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from functionary.openai_types import Tool
from functionary.prompt_template import prompt_utils
from functionary.prompt_template.base_template import PromptTemplate


def convert_to_llama3_messages(messages: List[Dict]) -> List[Dict]:
    result = []
    index = 0
    while index < len(messages):
        if messages[index]["role"] in ["user", "system"]:
            result.append(messages[index])
            index += 1
        else:
            if messages[index]["role"] == "assistant":
                tool_calls = messages[index].get("tool_calls", [])
                if len(tool_calls) == 0:
                    result.append(messages[index])
                else:
                    messages


class Llama3Template(PromptTemplate):
    function_separator = "<|reserved_special_token_249|>"
    version = "v2.llama3"
    fn_param_sep_token = "\n"

    def get_additional_tokens(self) -> List[str]:
        return []

    def get_assistant_prefixes(self) -> List[str]:
        return ["<|start_header_id|>assistant<|end_header_id|>\n\n"]

    def get_stop_tokens_for_generation(self) -> List[str]:
        return ["<|eot_id|>", "<|end_of_text|>"]

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
        # To force a text response ("tool_choice"="none")
        if tool_choice == "none":
            stage = "text-gen"
        # Normal generation (function name first) (tool_choice="required")
        elif tool_choice == "required":
            stage = "function"
        # To force a function call (tool_choice={"type": "function", "function": {...}})
        elif tool_choice != "":
            stage = "parameter"
        # Normal generation (either <|reserved_token_249|> or text) (tool_choice="auto")
        else:
            stage = "pre-function"

        return {
            "stage": stage,
            "curr_tokens": curr_tokens,
            "curr_text": curr_text,
            "func_name": tool_choice,
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

        # No grammar sampling needed for the following stages
        if gen_state["stage"] in [
            "pre-function",
            "text-gen",
            "code-interpreter",
            "parameter",
        ]:
            grammar_sampled_token_id = model_sampled_token_id
            grammar_sampled_token = tokenizer.decode([model_sampled_token_id])

        options = []
        # Form the pre-function options (<|reserved_token_249|> or <|eot_id|>) to update gen_state
        if gen_state["stage"] == "pre-function":
            options = [self.function_separator, "<|eot_id|>"]
        # Form the functions options for grammar sampling
        elif gen_state["stage"] == "function":
            options = [tool_or_func["name"] for tool_or_func in tools_or_functions]
            if gen_state["add_code_interpreter"]:
                options.append("python")

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

                    # Use the token as long as 1 option is True
                    # - Reject the whitespace (" ") and empty ("") tokens
                    if any(options_mask) and sampled_token.strip(" ") != "":
                        grammar_sampled_token_id = sampled_token_ind
                        grammar_sampled_token = sampled_token
                        break
                if gen_state["stage"] == "pre-parameter":
                    if new_curr_tokens == self.fn_param_sep_token:
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

        if gen_state["stage"] == "pre-function":
            # Check if the new state is in "function" stage
            if gen_state["curr_text"] == self.function_separator:
                gen_state = {
                    "stage": "function",
                    "curr_tokens": [],
                    "curr_text": "",
                    "func_name": "",
                    "add_code_interpreter": gen_state["add_code_interpreter"],
                }
            # Check if the new state is in "text-gen" stage
            elif gen_state["curr_text"] not in self.get_stop_tokens_for_generation():
                gen_state["stage"] = "text-gen"
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
            # - In case of two fns having common prefixes (e.g.: get_weather and get_weather_and_time),
            # we need to iterate until parts of the fn_param_sep_token is present in new_curr_tokens
            # to know if the shorter or longer function name is preferred by the model.
            if sum(options_mask) == 1 and curr_text.startswith(
                options[options_mask.index(True)]
            ):
                tool_name = options[options_mask.index(True)]
                gen_state["func_name"] = tool_name
                suffix = curr_text[len(tool_name) :]
                # If suffix == self.fn_param_sep_token ("\n"), jump straight to "parameter" stage
                if suffix == self.fn_param_sep_token:
                    gen_state["stage"] = "parameter"
                    gen_state["curr_text"] = ""
                else:
                    gen_state["stage"] = "pre-parameter"
                    gen_state["curr_text"] = suffix
                gen_state["curr_tokens"] = []
        elif gen_state["stage"] == "pre-parameter":
            # Check if the new state is in "parameter" or "code-interpreter" stage
            if gen_state["curr_text"] == self.fn_param_sep_token:
                if gen_state["func_name"] == "python":
                    gen_state["stage"] = "code-interpreter"
                else:
                    gen_state["stage"] = "parameter"
                gen_state["curr_tokens"] = []
                gen_state["curr_text"] = ""
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
            if gen_state["curr_text"].endswith(self.function_separator):
                gen_state["stage"] = "function"
                gen_state["curr_text"], gen_state["curr_tokens"] = "", []
        return gen_state

    def parse_assistant_response(
        self, llm_output: str, tool_choice: Any = None
    ) -> Dict:
        # first remove stop tokens if there exists
        for stop in self.get_stop_tokens_for_generation():
            if llm_output.endswith(stop):
                llm_output = llm_output[: -len(stop)]

        # add forced-function from tool_choice if exists
        if type(tool_choice) is not str and tool_choice is not None:
            tool_choice_name = (
                tool_choice.function.name
                if isinstance(tool_choice, Tool)
                else tool_choice.name
            )
            llm_output = (
                self.get_force_function_call_prefix(tool_choice_name) + llm_output
            )
        elif tool_choice == "required":
            llm_output = self.function_separator + llm_output

        chunks = llm_output.split(self.function_separator)
        chunks = [chunk.strip() for chunk in chunks if len(chunk.strip()) > 0]

        tool_calls = []
        text_content = None
        tool_call_start_index = 0

        if not llm_output.startswith(
            self.function_separator
        ):  # if first token is not function_call
            text_content = chunks[0]
            tool_call_start_index = 1

        for chunk in chunks[tool_call_start_index:]:
            # format: function_name\narguments<end_of_functioncall>
            index = chunk.find("\n")
            func_name = chunk[:index].strip()
            arguments = chunk[index + 1 :].strip()
            tool_calls.append(
                {
                    "function": {"name": func_name, "arguments": arguments},
                    "id": prompt_utils.get_random_tool_call_id(),
                    "type": "function",
                }
            )
        tool_calls = None if len(tool_calls) == 0 else tool_calls
        return {"role": "assistant", "content": text_content, "tool_calls": tool_calls}

    def convert_message_to_prompt(self, message: Dict) -> str:
        role = message["role"]
        content = message.get("content", None)
        if role == "tool":
            tool_name = message["name"]
            content = f"name={tool_name}\n{content}"

        prompt_template = (
            "<|start_header_id|>%s<|end_header_id|>\n\n{text}<|eot_id|>" % role
        )

        if role in ["user", "system", "tool"]:
            return prompt_template.format(text=content)

        assert role == "assistant"
        tool_calls = message.get("tool_calls", [])
        if tool_calls is None:
            tool_calls = []

        if content is None and len(tool_calls) == 0:
            return f"<|start_header_id|>{role}<|end_header_id|>\n\n"

        if content is not None and len(tool_calls) == 0:  # text-only
            return prompt_template.format(text=content)

        tool_call_prompts = []
        for tool_call in tool_calls:
            arguments = tool_call["function"]["arguments"]
            tool_name = tool_call["function"]["name"]
            tool_prompt = f"{tool_name}\n{arguments}"
            tool_call_prompts.append(tool_prompt)

        if (
            content is None and len(tool_calls) > 0
        ):  # function call only (<sep>fc1<sep>fc2<sep>)
            tool_call_content = self.function_separator + self.function_separator.join(
                tool_call_prompts
            )
            return prompt_template.format(text=tool_call_content)

        # Here is the case contains both text-response and tool_calls (content<sep>fc1<sep>fc2<sep>)
        total_content = (
            content
            + self.function_separator
            + self.function_separator.join(tool_call_prompts)
        )
        return prompt_template.format(text=total_content)

    def pre_process_messages_before_inference(self, messages: List[Dict]) -> List[Dict]:
        """Order the tool results by the order of tool call ids

        Args:
            messages (List[Dict]): List of messages

        Returns:
            List[Dict]: List of messages
        """
        return prompt_utils.reorder_tool_messages_by_tool_call_ids(messages)

    def update_state_for_function(self, current_state):
        """update the state when a function is going to be called

        Args:
            current_state (_type_): _description_
        """
        current_state["response_type"] = "function"
        current_state["skip_until_reach"] = "\n"
        current_state["current_text"] = ""
        current_state["func_index"] += 1
        current_state["call_id"] = prompt_utils.get_random_tool_call_id()

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
        if len(current_state) == 0:  # empty dict, at the first_time
            current_state = {
                "current_text": "",  # the concatenation of all tokens so far
                "func_name": None,  # function_name of the current tool, if the response requires to use tool
                "response_type": None,  # response_type=text(text response)/function (using tool)
                "func_index": -1,  # index of the tool in tool_calls
                "call_id": None,  # call_id of the current tool
                # skip_until_reach we skip new tokens until we reach certain token. This is used when we hit special tokens
                "skip_until_reach": None,  # at first we will skip until reach <|content|>
                "first_time": True,  # if first_time we return an tempty delta with role=assistant
            }

            if tool_choice == "none":
                current_state["response_type"] = "text"

            elif tool_choice == "required":
                self.update_state_for_function(current_state)

            elif type(tool_choice) is not str and tool_choice is not None:
                self.update_state_for_function(current_state)
                current_state["func_name"] = tool_choice.function.name
                current_state["skip_until_reach"] = (
                    None  # function is already defined, no need to wait for new tokens to define
                )
                current_state["current_text"] += delta_text

                # first return a delta with function_name only
                responses = [
                    prompt_utils.get_function_delta_response(
                        current_state, "", True, False, finish_reason
                    )
                ]
                # next return the first chunk of params
                responses.append(
                    prompt_utils.get_function_delta_response(
                        current_state, delta_text, False, False, finish_reason
                    )
                )
                return current_state, responses
        else:
            current_state["first_time"] = False

        current_state["current_text"] += delta_text

        if finish_reason is not None:  # handle if finish
            if current_state["response_type"] == "function":
                finish_reason = "tool_calls"
            return current_state, prompt_utils.get_text_delta_response(
                None, False, finish_reason
            )

        skip_until_reach = current_state["skip_until_reach"]
        if skip_until_reach:  # if have to wait
            if (
                delta_text != current_state["skip_until_reach"]
            ):  # wait for until reach: "\n" to get function_name
                return current_state, None
            else:  # finally we get the end of function
                skip_until_reach = None
                current_state["skip_until_reach"] = None
                current_state["func_name"] = current_state["current_text"].strip()
                return current_state, prompt_utils.get_function_delta_response(
                    current_state, "", True, False, finish_reason
                )

        if (
            current_state["response_type"] is None
        ):  # Check if first token is function_separator
            if delta_text == self.function_separator:  # if first we call a function
                self.update_state_for_function(current_state)
                # first chunk of function_call is a message where all fields are None, except role
                return current_state, prompt_utils.get_text_delta_response(
                    None, True, finish_reason
                )
            else:
                current_state["response_type"] = "text"
                # The first chunk is always empty delta
                empty_response = prompt_utils.get_text_delta_response(
                    "", True, finish_reason
                )
                first_chunk = prompt_utils.get_text_delta_response(
                    delta_text, True, finish_reason
                )
                return current_state, [empty_response, first_chunk]
        else:  # already knew the curent type
            if (
                delta_text == self.function_separator
            ):  # end of current text_response or function
                self.update_state_for_function(current_state)
                # only first call contains empty delta
                return current_state, None
            else:  # not starting to call a function
                if current_state["response_type"] == "text":
                    responses = []
                    if current_state[
                        "first_time"
                    ]:  # if tool_choice=none, we still need to send an empty delta first
                        empty_response = prompt_utils.get_text_delta_response(
                            "", True, finish_reason
                        )
                        responses.append(empty_response)

                    responses.append(
                        prompt_utils.get_text_delta_response(
                            delta_text, True, finish_reason
                        )
                    )
                    return current_state, responses
                else:  # response_type = function
                    return current_state, prompt_utils.get_function_delta_response(
                        current_state, delta_text, False, False, finish_reason
                    )

    def get_chat_template_jinja(self) -> str:
        chat_template = """{% for message in messages %}
        {% if message['role'] == 'user' or message['role'] == 'system' %}
            {{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}<br>
        {% elif message['role'] == 'tool' %}
            {{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + 'name=' + message['name'] + '\n' + message['content'] + '<|eot_id|>' }}<br>
        {% else %}
            {{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'}}<br>
            {% if message['content'] is not none %}
                {{ message['content'] }}<br>
            {% endif %}
            {% if 'tool_calls' in message and message['tool_calls'] is not none %}
                {% for tool_call in message['tool_calls'] %}
                    {{ '<|reserved_special_token_249|>' + tool_call['function']['name'] + '\n' + tool_call['function']['arguments'] }}<br>
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

    def get_force_function_call_prefix(self, function_name: str):
        return f"{self.function_separator}{function_name}\n"
