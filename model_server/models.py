from pydantic import BaseModel, ConfigDict
from typing import List, Optional, Any

class Message(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: Optional[str] = "user"
    content: Any = ""

class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: Optional[str] = None
    messages: Optional[List[Message]] = None
    prompt: Optional[Any] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    stream: Optional[bool] = None

class CompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: Optional[str] = None
    prompt: Optional[Any] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    stream: Optional[bool] = None
