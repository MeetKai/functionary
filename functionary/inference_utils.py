from http import HTTPStatus
from typing import Optional

import torch
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import StoppingCriteria, StoppingCriteriaList

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
