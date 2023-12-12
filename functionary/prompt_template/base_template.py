from __future__ import annotations

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
    def get_stopping_token(self, stage: Literal["function", "parameter"]) -> int:
        """returns a token id which stops function/parameter name generation
        e.g.: `"get_current_weather` with v1 prompt template -> returns id = 28747 (':' token)
        so the generation gets forced towards `"get_current_weather:\n{...`
        Args:
            stage (str): Whether to get function name or parameter name stopping token
        Returns:
            int: integer token id
        """
        raise NotImplementedError

    @abstractmethod
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
            gen_state (Dict): The current generation state
            new_token_id (int): The token id of the newly sampled token
            options (List): All available function/param names depending on the stage of gen_state
            tokenizer (Any): The tokenizer class passed in from Transformers or vLLM
        Returns:
            dict: The updated gen_state
        """

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
        self, messages: List[Dict], tools_or_functions: Optional[List[Dict]] = None
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
    def parse_assistant_response(self, llm_output: str) -> Dict:
        """This function is used to parse llm_output to the Message of OpenAI ({"role": xxx, "content": xxx, ...})
        this is used in inference.
        Args:
            llm_output (str): The generated content from Model

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
