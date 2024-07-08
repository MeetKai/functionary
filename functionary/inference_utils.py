from functionary.prompt_template.prompt_utils import enforce_tool_choice
from transformers import StoppingCriteriaList, StoppingCriteria
import torch


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