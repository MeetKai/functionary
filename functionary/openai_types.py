import time
import uuid
from typing import List, Optional

from pydantic import BaseModel, Field


class ModelPermission(BaseModel):
    id: str = Field(default_factory=lambda: f"modelperm-{str(uuid.uuid4())}")
    object: str = "model_permission"
    created: int = Field(default_factory=lambda: int(time.time()))
    allow_create_engine: bool = False
    allow_sampling: bool = True
    allow_logprobs: bool = True
    allow_search_indices: bool = False
    allow_view: bool = True
    allow_fine_tuning: bool = False
    organization: str = "*"
    group: Optional[str] = None
    is_blocking: str = False


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "vllm"
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: List[ModelPermission] = Field(default_factory=list)


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = Field(default_factory=list)


class FunctionCall(BaseModel):
    name: Optional[str] = None
    arguments: str


class Function(BaseModel):
    name: str
    description: Optional[str] = Field(default="")
    parameters: dict


class ChatMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None
    name: Optional[str] = None
    function_call: Optional[FunctionCall] = None

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
