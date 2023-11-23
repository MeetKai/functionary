import time
from typing import List, Optional

from pydantic import BaseModel, Field


class FunctionCall(BaseModel):
    name: Optional[str] = None
    arguments: str
    

class ToolCall(BaseModel):
    index: Optional[int] = None
    id: Optional[str] = None
    function: FunctionCall
    type: Optional[str] = "function"


class Function(BaseModel):
    name: str
    description: Optional[str] = Field(default="")
    parameters: dict


class Tool(BaseModel):
    type: str = "function"
    function: Function


class ChatMessage(BaseModel):
    role: Optional[str] = None
    tool_call_id: Optional[str] = None
    content: Optional[str] = None
    name: Optional[str] = None
    function_call: Optional[FunctionCall] = None
    tool_calls: Optional[List[ToolCall]] = None

    def __str__(self) -> str:
        if self.role == "system":
            return f"system:\n{self.content}\n"

        elif self.role == "function":
            return f"function name={self.name}:\n{self.content}\n"

        elif self.role == "user":
            if self.content is None:
                return "user:\n</s>"
            else:
                return f"user:\n</s>{self.content}\n"

        elif self.role == "assistant":
            if self.content is not None and self.function_call is not None:
                return f"assistant:\n{self.content}\nassistant to={self.function_call.name}:\n{self.function_call.arguments}</s>"

            elif self.function_call is not None:
                return f"assistant to={self.function_call.name}:\n{self.function_call.arguments}</s>"

            elif self.content is None:
                return "assistant"

            else:
                return f"assistant:\n{self.content}\n"

        else:
            raise ValueError(f"Unsupported role: {self.role}")


class ChatInput(BaseModel):
    messages: List[ChatMessage]
    functions: Optional[List[Function]] = None
    tools: Optional[List[Tool]] = None
    temperature: float = 0.9
    stream: bool = False


class Choice(BaseModel):
    message: ChatMessage
    finish_reason: str = "stop"
    index: int = 0

    @classmethod
    def from_message(cls, message: ChatMessage, finish_reason: str):
        return cls(message=message, finish_reason=finish_reason)


class ChatCompletion(BaseModel):
    id: str
    object: str = "chat.completion"
    created: float = Field(default_factory=time.time)
    choices: List[Choice]


class StreamChoice(BaseModel):
    delta: ChatMessage
    finish_reason: Optional[str] = "stop"
    index: int = 0


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: float = Field(default_factory=time.time)
    choices: List[StreamChoice]
