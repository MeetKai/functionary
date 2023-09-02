"""Chat language model that can interpret and execute functions/plugins"""

from .inference import generate_message
from .openai_types import ChatMessage, Function, FunctionCall
from .schema import generate_schema_from_functions

__all__ = [
    "ChatMessage",
    "Function",
    "FunctionCall",
    "generate_message",
    "generate_schema_from_functions",
]
