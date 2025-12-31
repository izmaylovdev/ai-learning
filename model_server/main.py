from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Any
import os
import time
import uuid

from model_server.models import CompletionRequest, ChatCompletionRequest, Message
from text_generation.get_generator import get_generator
import config

app = FastAPI(title="Local OpenAI-compatible model server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GENERATOR_TYPE = os.getenv("GENERATOR_TYPE", config.GENERATOR_BACKEND)


def messages_to_prompt(messages: List[Message]) -> str:
    # Concatenate messages into a single prompt. Support strings or richer content
    parts = []
    for m in messages:
        role = (m.role or "user").strip()
        content = extract_text_from_content(m.content)
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


def extract_text_from_content(content: Any) -> str:
    """Extract a human-readable text from message content. Handles:
    - plain strings
    - lists of content blocks (e.g. [{"type": "text", "text": "..."}, ...])
    - dicts with a 'text' or 'content' field
    - fall back to str(content)
    """
    if content is None:
        return ""
    # plain string
    if isinstance(content, str):
        return content
    # list of blocks
    if isinstance(content, list):
        pieces = []
        for item in content:
            if isinstance(item, str):
                pieces.append(item)
            elif isinstance(item, dict):
                # common fields
                if "text" in item:
                    pieces.append(str(item.get("text") or ""))
                elif "content" in item:
                    pieces.append(str(item.get("content") or ""))
                else:
                    pieces.append(str(item))
            else:
                pieces.append(str(item))
        return "\n".join([p for p in pieces if p])
    # dict-like content
    if isinstance(content, dict):
        if "text" in content:
            return str(content.get("text") or "")
        if "content" in content:
            return str(content.get("content") or "")
        # handle openai 'content' where the value may be a list
        for k in ("message", "body", "data"):
            if k in content:
                return extract_text_from_content(content[k])
        # fallback
        return str(content)
    # fallback
    return str(content)


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    gen = get_generator(GENERATOR_TYPE)

    print("Answering chat completion request")

    if req.messages:
        prompt = messages_to_prompt(req.messages)
    elif req.prompt:
        # prompt can be string or list
        if isinstance(req.prompt, list):
            prompt = "\n".join(map(str, req.prompt))
        else:
            prompt = str(req.prompt)
    else:
        prompt = ""

    # call generator
    text = gen.generate(prompt)

    response = {
        "id": str(uuid.uuid4()),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model or GENERATOR_TYPE,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
    }
    return response


@app.post("/v1/completions")
async def completions(req: CompletionRequest):
    print("Answering chat completion request")

    gen = get_generator(GENERATOR_TYPE)

    if req.prompt is None:
        prompt = ""
    elif isinstance(req.prompt, list):
        prompt = "\n".join(map(str, req.prompt))
    else:
        prompt = str(req.prompt)

    text = gen.generate(prompt)

    response = {
        "id": str(uuid.uuid4()),
        "object": "text_completion",
        "created": int(time.time()),
        "model": req.model or GENERATOR_TYPE,
        "choices": [{"text": text, "index": 0, "finish_reason": "stop"}],
    }
    return response


@app.get("/health")
async def health():
    return {"ok": True, "generator_type": GENERATOR_TYPE }


@app.get("/v1/models")
async def list_models():
    model_info = {
        "id": 'ciklum AI learning',
        "object": "model",
        "owned_by": "local",
        # include lightweight metadata that some clients may inspect
        "permission": [],
    }
    return {"object": "list", "data": [model_info]}


@app.get("/v1/models/{model_id}")
async def get_model(model_id: str):
    """Return a single model description or 404 if not found."""
    expected_id = GENERATOR_TYPE or "local-model"
    if model_id != expected_id:
        # Follow OpenAI style: 404-like response (FastAPI will return 404 if we raise HTTPException)
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail={"error": {"message": f"Model '{model_id}' not found"}})

    model_info = {
        "id": expected_id,
        "object": "model",
        "owned_by": "local",
        "permission": [],
    }
    return model_info
