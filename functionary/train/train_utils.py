from json_source_map import calculate
from typing import Any, List, Dict


def find_first_token_value(index: int, token_indices: List) -> int:
    """This function return the index of token that contains the index
    For example: token_indices=[(1, 4), ()]
    Args:
        start_index (int): _description_
        token_indices (List): List of tuple: (index of start, index of end, token_index, token_str)

    Returns:
        int: _description_
    """
    for start, end, token_index, _ in token_indices:
        if start <= index and index < end:
            return token_index
    return None


def find_index_of_element_in_list(element: Any, alist: List) -> int:
    if element not in alist:
        return -1
    return alist.index(element)


def extract_indices_of_first_tokens_of_param_values(
    arguments_token_ids: List[int], tokenizer: Any, verbose: bool = False
) -> List[int]:
    argument_text = tokenizer.decode(arguments_token_ids)
    token_strings = [tokenizer.decode(token_id) for token_id in arguments_token_ids]
    token_indices = []
    pos = 0

    for token_index, token_str in enumerate(token_strings):
        start = argument_text.find(token_str, pos)
        if start == -1:
            if verbose:
                print("cannot find start")
            continue
        end = start + len(token_str)
        token_indices.append((start, end, token_index, token_str))
        pos = end

    if verbose:
        print("token_indices: ", token_indices)
    # locate the key in the dictionary
    field_dic = calculate(argument_text)
    result = []
    for field in field_dic:
        if len(field) > 0:
            if verbose:
                print("find param: ", field)
            entry = field_dic[field]
            start, end = entry.value_start.position, entry.value_end.position
            if argument_text[start] == '"':
                start += 1
            token_index = find_first_token_value(start, token_indices)
            if token_index:
                result.append(token_index)
    return result


def extract_indices_of_first_tokens_of_param_values_in_assistant_response(
    tokenizer: Any, token_ids: List[int], verbose: bool = False
) -> List[int]:
    """Extract the first tokens of values of parameters in tool call
    For example, token_ids of assistant response=get_current_weather\n{"location": "Hanoi"}
    this function will extract the indices of tokens associated with: Hanoi & 3
    Args:
        tokenizer (Any): _description_
        token_ids (List[int]): token_ids of the assistant
        verbose (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    function_sep = ">>>"
    function_sep_id = tokenizer.encode(function_sep, add_special_tokens=False)[0]
    break_line = "\n"
    brk_line_token_id = tokenizer.encode(break_line, add_special_tokens=False)[0]
    # print(f"function_sep_id: {function_sep_id}; brk_line_token_id:{brk_line_token_id}")
    sep_indices = [-1]
    for i in range(len(token_ids)):
        if token_ids[i] == function_sep_id:
            sep_indices.append(i - 1)

    if verbose:
        print("sep_indices: ", sep_indices)
    result = []
    for i, sep_index in enumerate(sep_indices):
        brk_index = find_index_of_element_in_list(
            brk_line_token_id, token_ids[sep_index + 1 :]
        )
        if brk_index >= 0:
            brk_index += sep_index + 1
            func_name = tokenizer.decode(token_ids[sep_index + 1 : brk_index])
            # print(f"func_name:{token_ids[sep_index + 1: brk_index]};{func_name};sep_index={sep_index}, brk_index:{brk_index}")
            if func_name != "all":
                end_index = len(token_ids) - 2  # exclude eos_token_id for the last call
                if i != len(sep_indices) - 1:
                    end_index = sep_indices[i + 1]
                start_argument_index = brk_index + 1
                # = brk_index + 1, end_index
                # token_ids[brk_index + 1: ] --> {"car_name": "Tang"}
                token_indices = extract_indices_of_first_tokens_of_param_values(
                    token_ids[start_argument_index : end_index + 1],
                    tokenizer,
                    verbose=verbose,
                )
                result.extend([start_argument_index + ind for ind in token_indices])
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
