from copy import deepcopy
from http import HTTPStatus
from typing import Dict, List, Optional

import jsonref
import torch
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import StoppingCriteria, StoppingCriteriaList

from functionary.openai_types import Function
from functionary.prompt_template.prompt_utils import enforce_tool_choice


class ErrorResponse(BaseModel):
    object: str = "error"
    message: str
    type: str
    param: Optional[str] = None
    code: int


class StopWordsCriteria(StoppingCriteria):
    def __init__(self, stops=[]):
        StoppingCriteria.__init__(self)
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        inputs = input_ids[0].tolist()
        for stop in self.stops:
            if len(inputs) >= len(stop) and inputs[-len(stop) :] == stop:
                return True
        return False


def analyze_tools_and_tool_choice(request):
    if request.tools:
        tools = enforce_tool_choice(
            choice=request.tool_choice, tools_or_functions=request.tools
        )
        tools_or_functions = [item.dict() for item in tools]
        tool_func_choice = request.tool_choice if request.tool_choice else "auto"
    elif request.functions:
        functions = enforce_tool_choice(
            choice=request.function_call, tools_or_functions=request.functions
        )
        tools_or_functions = [item.dict() for item in functions]
        tool_func_choice = request.function_call if request.function_call else "auto"
    else:
        tools_or_functions = []
        tool_func_choice = "none"

    return tools_or_functions, tool_func_choice


def create_error_response(
    status_code: HTTPStatus, message: str, param: Optional[str]
) -> JSONResponse:
    return JSONResponse(
        ErrorResponse(
            message=message,
            type="invalid_request_error",
            param=param,
            code=status_code.value,
        ).dict(),
        status_code=status_code.value,
    )


async def check_all_errors(request, served_model) -> Optional[JSONResponse]:
    if request.model not in served_model:
        return create_error_response(
            status_code=HTTPStatus.NOT_FOUND,
            message=f"The model `{request.model}` does not exist.",
            param=None,
        )
    if request.tools and request.functions:
        return create_error_response(
            status_code=HTTPStatus.BAD_REQUEST,
            message="'functions' and 'tools' cannot both be provided. 'functions' are deprecated; use the 'tools' parameter instead.",
            param=None,
        )
    if isinstance(request.function_call, str) and request.function_call not in [
        "none",
        "auto",
    ]:
        return create_error_response(
            status_code=HTTPStatus.BAD_REQUEST,
            message=f"Invalid value: '{request.function_call}'. Supported values are: 'none' and 'auto'.",
            param="function_call",
        )
    if isinstance(request.tool_choice, str) and request.tool_choice not in [
        "none",
        "auto",
        "required",
    ]:
        return create_error_response(
            status_code=HTTPStatus.BAD_REQUEST,
            message=f"Invalid value: '{request.tool_choice}'. Supported values are: 'none', 'auto', and 'required'.",
            param="tool_choice",
        )
    if request.functions is None and request.function_call is not None:
        return create_error_response(
            status_code=HTTPStatus.BAD_REQUEST,
            message=f"Invalid value for 'function_call': 'function_call' is only allowed when 'functions' are specified.",
            param="function_call",
        )
    if request.tools is None and request.tool_choice is not None:
        return create_error_response(
            status_code=HTTPStatus.BAD_REQUEST,
            message=f"Invalid value for 'tool_choice': 'tool_choice' is only allowed when 'tools' are specified.",
            param="tool_choice",
        )
    return


def resolve_json_refs(tools_or_functions):
    tools = deepcopy(tools_or_functions)
    if tools:
        for i in range(len(tools)):
            if "type" in tools[i]:
                if tools[i]["type"] == "function":
                    tools[i]["function"]["parameters"] = deepcopy(
                        jsonref.JsonRef.replace_refs(tools[i]["function"]["parameters"])
                    )
            else:
                tools[i]["parameters"] = deepcopy(
                    jsonref.JsonRef.replace_refs(tools[i]["parameters"])
                )

    return tools


def convert_tool_calls_to_function_call(
    functions: Optional[List[Function]], chat_message: Dict
) -> Dict:
    if "delta" not in chat_message:  # Non-streaming
        if (
            functions
            and len(functions) > 0
            and "tool_calls" in chat_message
            and chat_message["tool_calls"] is not None
            and len(chat_message["tool_calls"]) > 0
        ):
            chat_message["function_call"] = {
                "name": chat_message["tool_calls"][0]["function"]["name"],
                "arguments": chat_message["tool_calls"][0]["function"]["arguments"],
            }
            chat_message["tool_calls"] = None
    else:  # Streaming
        if (
            functions
            and len(functions) > 0
            and "tool_calls" in chat_message["delta"]
            and chat_message["delta"]["tool_calls"]
            and len(chat_message["delta"]["tool_calls"]) > 0
        ):
            chat_message["delta"]["function_call"] = chat_message["delta"][
                "tool_calls"
            ][0]["function"]
            chat_message["delta"]["tool_calls"] = None

    return chat_message
