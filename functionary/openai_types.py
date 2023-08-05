import time
from typing import List, Optional

from pydantic import BaseModel, Field


class FunctionCall(BaseModel):
    name: str
    arguments: str


class Function(BaseModel):
    name: str
    description: Optional[str] = Field(default="")
    parameters: dict


class ChatMessage(BaseModel):
    role: str
    content: Optional[str] = None
    name: Optional[str] = None
    _to: Optional[str] = None
    function_call: Optional[FunctionCall] = None

    class Config:
        underscore_attrs_are_private = True

    def __str__(self) -> str:
        if self.role == "system":
            return f"system:\n{self.content}\n"

        elif self.role == "function":
            return f"function name={self.name}:\n{self.content}\n"

        elif self.role == "user" and self.content is None:
            return "user:\n</s>"

        elif self.role == "user":
            return f"user:\n</s>{self.content}\n"

        elif self.role == "assistant" and self._to is not None:
            return f"assistant to={self._to}:\n{self.content}</s>"

        elif self.role == "assistant" and self.content is None:
            return "assistant"

        elif self.role == "assistant":
            return f"assistant:\n{self.content}\n"

        else:
            raise ValueError(f"Unsupported role: {self.role}")


class ChatInput(BaseModel):
    messages: List[ChatMessage]
    functions: Optional[List[Function]] = None
    temperature: float = 0.9


class Choice(BaseModel):
    message: ChatMessage
    finish_reason: str = "stop"
    index: int = 0

    @classmethod
    def from_message(cls, message: ChatMessage):
        return cls(message=message)


class ChatCompletion(BaseModel):
    id: str
    object: str = "chat.completion"
    created: float = Field(default_factory=time.time)
    choices: List[Choice]
