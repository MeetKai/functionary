import itertools

import numpy as np


class Trie(object):
    def __init__(self, strings=None, values=None, parent=None):
        self.children = {}
        self.value = None
        self.match_version = -1
        self.match = False
        self.partial_match = False
        self.parent = parent
        self.flag = None  # a spot for user code to store state

        if strings is not None:
            for i, s in enumerate(strings):
                self.insert(s, None if values is None else values[i])

    def insert(self, s, value):
        if len(s) == 0:
            self.value = value
        else:
            first_char = s[0]
            if first_char not in self.children:
                self.children[first_char] = Trie(parent=self)
            self.children[first_char].insert(s[1:], value)

    def values(self, prefix):
        if prefix == "":
            sub_values = list(
                itertools.chain.from_iterable(
                    self.children[k].values(prefix) for k in self.children
                )
            )
            if self.value is not None:
                sub_values.append(self.value)
            return sub_values
        else:
            return self.children[prefix[0]].values(prefix[1:])

    def __setitem__(self, key, value):
        if len(key) == 0:
            self.value = value
        else:
            if key[0] not in self.children:
                self.children[key[0]] = Trie(parent=self)
            self.children[key[0]].__setitem__(key[1:], value)

    def __contains__(self, key):
        return self.__getitem__(key) is not None

    def __getitem__(self, key):
        if len(key) == 0:
            return self
        elif key[0] in self.children:
            return self.children[key[0]].__getitem__(key[1:])
        else:
            return None


class ByteTrie(object):
    def __init__(self, byte_strings=None, values=None, parent=None):
        self.children = {}
        self.value = None
        self.match_version = -1
        self.match = False
        self.partial_match = False
        self.parent = parent
        self.flag = None  # a spot for user code to store state
        self.log_prob = 0

        if byte_strings is not None:
            for i, s in enumerate(byte_strings):
                self.insert(s, None if values is None else values[i])

    def insert(self, s, value):
        if len(s) == 0:
            self.value = value
        else:
            first_byte = s[0:1]
            if first_byte not in self.children:
                self.children[first_byte] = ByteTrie(parent=self)
            self.children[first_byte].insert(s[1:], value)


class GrammarSampler:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        byte_tokens = []
        for i in range(len(self.tokenizer)):
            byte_tokens.append(
                self.tokenizer.convert_tokens_to_string(
                    ["a", self.tokenizer.convert_ids_to_tokens(i)]
                )[1:],
            )
        self._token_trie = ByteTrie(byte_tokens, np.arange(len(byte_tokens)))
        self._token_trie.match = True
        self._token_trie.match_version = 0
        self.tokens = byte_tokens
        self.curr_tokens = ""

    def check_to_sample(self, text, start_token, functions):
        suffix = text[text.rfind(start_token) + len(start_token) :].lstrip()

        # Check the following:
        # the suffix after start_token is empty => True
        # the suffix after start_token is in the midst of generating a function name => True
        # the suffix after start_token has just completed generating a function name => False
        # the suffix after start_token is generating a normal response => False
        functions[0].name = "everywhere"
        functions.append(functions[0].copy())
        functions[1].name = "everyone"

        return any(
            [func.name.startswith(suffix) and func.name != suffix for func in functions]
        )

    def sample(
        self, functions, tools, delta_logprobs, delta_token_ids, prompt_template_version
    ):
        if functions:
            tools_or_functions = [item.dict() for item in functions]
        elif tools:
            tools_or_functions = [item.dict() for item in tools]

        select_options = [option["name"] for option in tools_or_functions]
        if prompt_template_version == "v2":
            select_options.append("all")
            
        select_options = ["everywhere", "everyone"]

        for i, sampled_token_ind in enumerate(delta_token_ids):
            sampled_token = self.tokens[sampled_token_ind]
            new_curr_tokens = self.curr_tokens + sampled_token
            options_mask = [
                True if option.startswith(new_curr_tokens.lstrip(" ")) else False
                for option in select_options
            ]

            if any(options_mask) and sampled_token != " ":
                if sum(options_mask) == 1:
                    has_space = True if sampled_token[0] == " " else False
                    sampled_token = select_options[options_mask.index(True)][
                        len(self.curr_tokens.lstrip()) :
                    ]
                    if has_space:
                        sampled_token = f" {sampled_token}"
                    self.curr_tokens = ""
                else:
                    self.curr_tokens += sampled_token
                return sampled_token

        return None
