from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import time
import uuid

from linkedin_post_generation.agent import agent as linkedin_agent
from agents.learning_program_rag.agent import get_agent as get_rag_agent
from model_server.schemas import CompletionRequest, ChatCompletionRequest
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

# Model IDs
LINKEDIN_MODEL_ID = "linkedin-post-agent"
RAG_MODEL_ID = "rag-agent"

# Add logging for submodel selection
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ModelServer")


def get_agent_for_model(model_id: str):
    """Get the appropriate agent based on model ID."""
    if model_id and model_id.lower() == RAG_MODEL_ID:
        return get_rag_agent(), "RAG"
    else:
        # Default to LinkedIn agent
        return linkedin_agent, "LinkedIn"


def extract_response_text(result):
    """Extract text from agent response."""
    if isinstance(result, dict) and "messages" in result:
        messages = result["messages"]
        if messages:
            last_message = messages[-1]
            if hasattr(last_message, 'content'):
                if isinstance(last_message.content, str):
                    return last_message.content
                elif isinstance(last_message.content, dict):
                    return last_message.content.get('text', str(last_message.content))
                else:
                    return str(last_message.content)
            else:
                return str(last_message)
        else:
            return "No response generated."
    elif hasattr(result, 'content'):
        return result.content if isinstance(result.content, str) else str(result.content)
    else:
        return str(result)


# Update the chat_completions endpoint to include logging
@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    agent, agent_name = get_agent_for_model(req.model)
    logger.info(f"Answering chat completion request using {agent_name} agent")

    # Extract the content from messages
    if req.messages:
        # Get the last user message
        user_messages = [m for m in req.messages if m.role == "user"]
        if user_messages:
            last_user_message = user_messages[-1]
            # Handle content that could be string or other types
            if isinstance(last_user_message.content, str):
                content = last_user_message.content
            else:
                content = str(last_user_message.content)
        else:
            content = ""
    elif req.prompt:
        # prompt can be string or list
        if isinstance(req.prompt, list):
            content = "\n".join(map(str, req.prompt))
        else:
            content = str(req.prompt)
    else:
        content = ""

    try:
        # Use the agent to generate a response
        from langchain_core.messages import HumanMessage
        result = agent.invoke({"messages": [HumanMessage(content=content)]})
        text = extract_response_text(result)
        logger.info(f"Generated response using {agent_name} agent")

    except Exception as e:
        logger.error(f"Error using {agent_name} agent: {e}")
        text = f"Error: {str(e)}"

    response = {
        "id": str(uuid.uuid4()),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model or LINKEDIN_MODEL_ID,
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
    agent, agent_name = get_agent_for_model(req.model)
    logger.info(f"Answering completion request using {agent_name} agent")

    # Extract prompt content
    if req.prompt is None:
        content = ""
    elif isinstance(req.prompt, list):
        content = "\n".join(map(str, req.prompt))
    else:
        content = str(req.prompt)

    try:
        # Use the agent to generate a response
        from langchain_core.messages import HumanMessage
        result = agent.invoke({"messages": [HumanMessage(content=content)]})
        text = extract_response_text(result)
        logger.info(f"Generated response using {agent_name} agent")

    except Exception as e:
        logger.error(f"Error using {agent_name} agent: {e}")
        text = f"Error: {str(e)}"

    response = {
        "id": str(uuid.uuid4()),
        "object": "text_completion",
        "created": int(time.time()),
        "model": req.model or LINKEDIN_MODEL_ID,
        "choices": [{"text": text, "index": 0, "finish_reason": "stop"}],
    }
    return response


@app.get("/health")
async def health():
    return {"ok": True, "generator_type": GENERATOR_TYPE }


@app.get("/v1/linkedin/agents")
async def list_linkedin_agents():
    """List all available LinkedIn post generation agents and their metadata."""
    try:
        agents_info = linkedin_agent.list_available_agents()

        return {
            "object": "list",
            "data": agents_info,
        }
    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail={"error": {"message": f"Error listing agents: {str(e)}"}})


@app.get("/v1/models")
async def list_models():
    models = [
        {
            "id": LINKEDIN_MODEL_ID,
            "object": "model",
            "owned_by": "local",
            "description": "LinkedIn post generation agent",
            "permission": [],
        },
        {
            "id": RAG_MODEL_ID,
            "object": "model",
            "owned_by": "local",
            "description": "RAG agent for answering questions using document retrieval",
            "permission": [],
        },
    ]
    return {"object": "list", "data": models}


@app.get("/v1/models/{model_id}")
async def get_model(model_id: str):
    """Return a single model description or 404 if not found."""
    available_models = {
        LINKEDIN_MODEL_ID: {
            "id": LINKEDIN_MODEL_ID,
            "object": "model",
            "owned_by": "local",
            "description": "LinkedIn post generation agent",
            "permission": [],
        },
        RAG_MODEL_ID: {
            "id": RAG_MODEL_ID,
            "object": "model",
            "owned_by": "local",
            "description": "RAG agent for answering questions using document retrieval",
            "permission": [],
        },
    }

    if model_id not in available_models:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail={"error": {"message": f"Model '{model_id}' not found"}})

    return available_models[model_id]
