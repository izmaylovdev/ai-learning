from pydantic import BaseModel, ConfigDict
from typing import List, Optional, Any, Dict

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

class LinkedInPostRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    topic: str
    style: Optional[str] = "professional"
    tone: Optional[str] = "informative"
    length: Optional[str] = "medium"
    include_hashtags: Optional[bool] = True
    auto_select_agents: Optional[bool] = True
    enabled_agents: Optional[List[str]] = None
    disabled_agents: Optional[List[str]] = None
    agent_params: Optional[Dict[str, Dict]] = None
    verbose: Optional[bool] = False

