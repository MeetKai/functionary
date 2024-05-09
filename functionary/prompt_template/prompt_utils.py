from typing import Optional, Dict, List
import random
import string


def get_function_delta_response(
    current_state: Dict,
    delta_text: str,
    first_call: bool,
    return_role: bool,
    finish_reason: Optional[str],
) -> Dict:
    """Return delta for tool_call in streaming

    Args:
        current_state (Dict): _description_
        delta_text (str): _description_
        first_call (bool): _description_
        return_role (bool): _description_
        finish_reason (Optional[str]): _description_

    Returns:
        Dict: _description_
    """
    return {
        "delta": {
            "content": None,
            "function_call": None,
            "role": None if not return_role else "assistant",
            "tool_calls": [
                {
                    "index": current_state["func_index"],
                    "id": (
                        current_state["call_id"] if first_call else None
                    ),  # only return call_id at the first time
                    "function": {
                        "arguments": delta_text,
                        "name": current_state["func_name"] if first_call else None,
                    },
                    "type": "function" if first_call else None,
                }
            ],
        },
        "finish_reason": finish_reason,
        "index": 0,
    }


def get_text_delta_response(
    delta_text: Optional[str], return_role: bool, finish_reason: Optional[str]
) -> Dict:
    """Return delta for text_response in streaming

    Args:
        delta_text (Optional[str]): _description_
        return_role (bool): _description_
        finish_reason (Optional[str]): _description_

    Returns:
        Dict: _description_
    """
    return {
        "delta": {
            "content": delta_text,
            "function_call": None,
            "role": None if not return_role else "assistant",
            "tool_calls": None,
        },
        "finish_reason": finish_reason,
        "index": 0,
    }


def get_random_tool_call_id():
    return "call_" + "".join(
        [random.choice(string.ascii_letters + string.digits) for _ in range(24)]
    )


def reorder_tool_messages_by_tool_call_ids(messages: List[Dict]) -> List[Dict]:
    """re-order the messages where role = tool to match the order in tool_calls by tool_call_id
    Args:
        messages (List[Dict]): list of messages containing: tool_call_id

    Returns:
        List[Dict]: _description_
    """
    result = []
    index = 0
    while index < len(messages):
        message = messages[index]
        tool_calls = message.get("tool_calls", None)

        result.append(message)
        if message["role"] == "assistant" and tool_calls:
            num_calls = len(tool_calls)
            if (
                tool_calls[0].get("id", None) is not None
            ):  # if tool_call contains "id" for mapping
                tool_call_ids = [item["id"] for item in tool_calls]

                tool_messages = [messages[index + 1 + j] for j in range(num_calls)]
                id_2_tool_messages = {
                    item["tool_call_id"]: item for item in tool_messages
                }
                new_messages = [id_2_tool_messages[cid] for cid in tool_call_ids]

                result.extend(new_messages)
                index += num_calls + 1
            else:
                index += 1
        else:
            index += 1
    return result
