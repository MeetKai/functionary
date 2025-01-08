import time
import uuid
from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, validator


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
    parameters: Optional[dict] = None


class Tool(BaseModel):
    type: Literal["function", "code_interpreter"] = "function"
    function: Optional[Function] = None


class ChatMessage(BaseModel):
    role: Optional[str] = None
    tool_call_id: Optional[str] = None
    content: Union[None, str, List[Dict]] = None
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
    max_tokens: int = 1024


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


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: float = Field(default_factory=time.time)
    choices: List[StreamChoice]
    model: str
    usage: Optional[UsageInfo] = Field(default=None)


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    functions: Optional[List[Function]] = None
    tools: Optional[List[Tool]] = None
    function_call: Optional[Union[str, Function]] = None
    tool_choice: Optional[Union[str, Tool]] = None
    temperature: Union[Optional[float], str] = 0.6
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    max_tokens: Optional[int] = 512
    stop: Optional[Union[str, List[str]]] = Field(default_factory=list)
    stream: Optional[bool] = False
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None

    # Disable logprobs and top_logprobs currently first
    logprobs: Optional[bool] = False
    top_logprobs: Optional[int] = None

    # Additional parameters supported by vLLM
    best_of: Optional[int] = None
    top_k: Optional[int] = -1
    ignore_eos: Optional[bool] = False

    # Extra parameters for SRT backend only and will be ignored by OpenAI models.
    regex: Optional[str] = None
    min_tokens: Optional[int] = 0
    repetition_penalty: Optional[float] = 1.0
    stop_token_ids: Optional[List[int]] = Field(default_factory=list)

    # @validator("tool_choice", always=True)
    # def validate_tool_choice(cls, value, values):
    #     if value is None:
    #         if values["tools"] is None and values["functions"] is None:
    #             return "none"
    #         else:
    #             return "auto"
    #     return value

    # @validator("function_call", always=True)
    # def validate_function_call(cls, value, values):
    #     if value is None:
    #         if values["tools"] is None and values["functions"] is None:
    #             return "none"
    #         else:
    #             return "auto"
    #     return value


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[
        Literal["stop", "length", "function_call", "tool_calls"]
    ] = None


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{str(uuid.uuid4().hex)}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: UsageInfo


class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length", "function_call"]] = None


class ChatCompletionStreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{str(uuid.uuid4().hex)}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseStreamChoice]
