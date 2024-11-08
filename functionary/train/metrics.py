from json_source_map import calculate
from typing import Any, List, Dict, Tuple
import json
from transformers import AutoTokenizer


def find_first_token_value(index: int, token_indices: List[Tuple[int, int]]) -> int:
    """This function return the index of token that contains the index
    For example: token_indices=[(1, 4), ()]
    Args:
        start_index (int): _description_
        token_indices (List): List of tuple: (index of start, index of end, token_index, token_str)

    Returns:
        int: _description_
    """
    for i, (start, end) in enumerate(token_indices):
        if start <= index and index < end:
            return i
    return None


def extract_indices_of_first_token_in_argument_values(
    argument_token_indices: List[int], argument_text: str, verbose: bool = False
) -> List[int]:
    """this function return indices of first tokens in argument values
    for example, argument_text: {"a": 12, "b": {"c": "Hanoi"}}; argument_token_indices = [(0, 1), ... (10, 12)]
    --> return the indices of first token of: 12; indices of first token of Hanoi
    Args:
        argument_token_indices (List[int]): List of (start, end) of tokens in argument_token_indices
        argument_text (str): The text of arguments, a python dictionary

    Returns:
        List[int]: indices of first token of values in argument_text
    """
    try:
        # Calculate the positions of the values in the argument_text
        field_dic = calculate(argument_text)
    except Exception as e:
        if verbose:
            print(f"exception using calculate to find key from: {argument_text}")
        return []

    result = []
    for field in field_dic:
        if len(field) > 0:
            entry = field_dic[field]
            start, end = entry.value_start.position, entry.value_end.position
            if argument_text[start] == '"':  # if parameter is string
                start += 1
            token_index = find_first_token_value(start, argument_token_indices)
            if verbose:
                print(
                    f"key={field}; at: {start}, {end}; --> token_index: {token_index}"
                )
            if token_index is not None:
                result.append(token_index)
    return result


def get_indices_of_tokens_in_string(
    tokenizer: Any, token_ids: List[int], verbose: bool = False
):
    text = tokenizer.decode(token_ids)
    tokens = [tokenizer.decode(token_id) for token_id in token_ids]
    pos = 0
    token_indices = []

    for token_index, token in enumerate(tokens):
        start = text.find(token, pos)
        if start == -1:
            if verbose:
                print("cannot find start")
            raise Exception(f"cannot find token: '{token}' in {text[pos: ]}")
        end = start + len(token)
        token_indices.append((start, end))
        pos = end
    return token_indices, text


def locate_start_end_indices_of_token(char_start, char_end, token_indices):
    token_index_start, token_index_end = -1, -1
    for index, (start, end) in enumerate(token_indices):
        if char_start >= start and char_start < end:
            token_index_start = index
        if char_end <= end and char_end > start:
            token_index_end = index
    return token_index_start, token_index_end


def extract_indices_of_json_objects(text: str) -> List[Tuple[int, int]]:
    """
    Extract all indices of JSON objects from a given text.

    Parameters:
    text (str): The input text containing JSON objects.

    Returns:
    list: A list of indices ([(start, end), ...]) of extracted JSON objects in text.
    """
    json_indices = []
    stack = []
    start_idx = None

    for i, char in enumerate(text):
        if char == "{":
            if not stack:
                start_idx = i  # Potential start of JSON object
            stack.append(char)
        elif char == "}":
            if stack:
                stack.pop()
                if not stack and start_idx is not None:
                    json_str = text[start_idx : i + 1]
                    try:
                        print("load json: ", json_str)
                        parsed_json = json.loads(json_str)
                        json_indices.append((start_idx, i + 1))
                    except json.JSONDecodeError:
                        # Invalid JSON, ignore and continue
                        pass
                    start_idx = None
    return json_indices


def extract_indices_of_first_tokens_of_param_values_in_assistant_response(
    tokenizer: Any, token_ids: List[int], verbose: bool = False
) -> List[int]:
    """Extract the first tokens of values of parameters in tool call
    For example, token_ids of assistant response= [27, 1723, 29380, 70464, 89963, 2588, 794, 330, 39, 73803, 498, 330, 817, 669, 301, 5979, 355, 794, 837, 5474, 1723, 29, 128008]
    this is for assistant response text= '<function=get_weather>{"location": "Hanoi", "use_celcius": true}</function><|eom_id|>'
    this function will extract the indices of first tokens associated with: Hanoi & true which are tokens: 39 & 837
    Args:
        tokenizer (Any): _description_
        token_ids (List[int]): token_ids of the assistant
        verbose (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    # first we compute the indices of tokens and the response_text from token_indices
    # For example token_ids=[27, 1723, 29380, 70464, 89963, 2588, 794, 330, 39, 73803, 498, 330, 817, 669, 301, 5979, 355, 794, 837, 5474, 1723, 29, 128008]
    # this is tokens = ['<', 'function', '=get', '_weather', '>{"', 'location', '":', ' "', 'H', 'anoi', '",', ' "', 'use', '_c', 'el', 'ci', 'us', '":', ' true', '}</', 'function', '>', '<|eom_id|>']
    # --> respone_text=<function=get_weather>{"location": "Hanoi", "use_celcius": true}</function><|eom_id|>
    # token_indices=[(0, 1), (1, 9), (9, 13), (13, 21), (21, 24), (24, 32), (32, 34), (34, 36), (36, 37), (37, 41), (41, 43), (43, 45), (45, 48), (48, 50), (50, 52), (52, 54), (54, 56), (56, 58), (58, 63), (63, 66), (66, 74), (74, 75), (75, 85)]
    # token_indices is list of indices (start, end) of token in response_text
    token_indices, response_text = get_indices_of_tokens_in_string(
        tokenizer, token_ids, verbose
    )
    if verbose:
        print(f"response_text:", response_text)
        tokens = [response_text[s:e] for s, e in token_indices]
        print(f"tokens: ", tokens)
        print("token_indices: ", token_indices)
        print("---------------")

    # Extract indices of jsons in response_text, indices is a list: [(start, end), ...] where response_text[start: end] is a json
    json_indices = extract_indices_of_json_objects(response_text)
    result = []
    for start, end in json_indices:
        # first find the token_start_ind, token_end_ind associated with start, end, this is mapping from character index --> token_index
        token_start_ind, token_end_ind = locate_start_end_indices_of_token(
            start, end, token_indices
        )
        if verbose:
            print("------------------------------")
            print(
                f"extract json: start={start}; end={end}; content: {response_text[start: end]}"
            )
            print(
                f"convert to token_indices: token_start_ind={token_start_ind}({token_indices[token_start_ind]}); token_end_ind={token_end_ind}({token_indices[token_end_ind]})"
            )

        argument_text = response_text[start:end]
        # This is the token_indices inside argument_text
        # for example: argument_text={"location": "Hanoi", "use_celcius": true}
        # argument_token_indices = [(0, 2), (2, 10), (10, 12), (12, 14), (14, 15), (15, 19), (19, 21), (21, 23), (23, 26), (26, 28), (28, 30), (30, 32), (32, 34), (34, 36), (36, 41)]
        argument_token_indices = []
        # in the best case, this is = 0, for example, >{"a": 10} --> '>{"' is a token, while the start is only {, we need to temporarily consider this token as: {"

        for p in token_indices[token_start_ind : token_end_ind + 1]:
            # compute the relative indices of original token indices in argument_text
            # if p[0] != start, this is the case where token p here is: '>{"' while start is at: {, which is in the middle of the token, so we need to trim the token into: {"
            argument_token_indices.append(
                (p[0] - start if p[0] >= start else 0, p[1] - start)
            )
        # check if the last token is longer than end --> trim. For example, last token=}</ --> trim to }
        argument_token_indices[-1] = (argument_token_indices[-1][0], end - start)

        first_token_of_values_indices = (
            extract_indices_of_first_token_in_argument_values(
                argument_token_indices, argument_text
            )
        )
        if verbose:
            print(
                f"argument_token_indices: {argument_token_indices}; argument_text: {argument_text}"
            )
            print(
                f"argument_tokens: ",
                [argument_text[s:e] for s, e in argument_token_indices],
            )
            print(f"first_token_of_values_indices={first_token_of_values_indices}")

        for index in first_token_of_values_indices:
            result.append(index + token_start_ind)
            if verbose:
                start, end = token_indices[index + token_start_ind]
                content = response_text[start:end]
                print(
                    f"the detected token at index: {index + token_start_ind}, token_id={token_ids[index + token_start_ind]}; content={content}"
                )
    return result


def extract_unmasked_chunks(labels: List[int], preds: List[int]):
    """labels = [-100, -100, ... id1, id2, ...-100, -100, ...]
    this function extract unmasked chunks: [[id1, id2, ...], ...] in both labels and preds

    Args:
        labels (List[int]): _description_
        preds (List[int]): _description_

    Returns:
        _type_: _description_
    """
    current_label_chunk = []
    current_pred_chunk = []
    result = []
    for i in range(len(labels)):
        if labels[i] != -100:
            current_label_chunk.append(labels[i])
            current_pred_chunk.append(preds[i])
        else:
            if len(current_label_chunk) > 0:  # end of the assistant response chunk
                result.append((current_label_chunk, current_pred_chunk))
                current_label_chunk = []
                current_pred_chunk = []

    if len(current_label_chunk) > 0:
        result.append((current_label_chunk, current_pred_chunk))
    return result


def test():
    text = """<function=get_weather>{"location": "Hanoi", "use_celcius": true}</function><|eom_id|>"""
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    print("token_ids: ", token_ids)
    extract_indices_of_first_tokens_of_param_values_in_assistant_response(
        tokenizer, token_ids, verbose=True
    )


if __name__ == "__main__":
    test()
