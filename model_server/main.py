from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import time
import uuid

from linkedin_post_generation.agent import agent as linkedin_agent
from model_server.models import CompletionRequest, ChatCompletionRequest
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

# Initialize LinkedIn Post Agent
def get_linkedin_agent():
    """Get the LinkedIn agent instance."""
    return linkedin_agent
# Add logging for submodel selection
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ModelServer")

# Update the chat_completions endpoint to include logging
@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    logger.info("Answering chat completion request using LinkedIn agent")

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

    # Always use LinkedIn agent
    try:
        # Get LinkedIn agent
        agent = get_linkedin_agent()

        # Use the agent to generate a response
        from langchain_core.messages import HumanMessage
        result = agent.invoke({"messages": [HumanMessage(content=content)]})

        # Extract the text from the result
        if isinstance(result, dict) and "messages" in result:
            # Get the last message from the agent
            messages = result["messages"]
            if messages:
                last_message = messages[-1]
                if hasattr(last_message, 'content'):
                    text = last_message.content
                else:
                    text = str(last_message)
            else:
                text = "No response generated."
        else:
            text = str(result)

        logger.info(f"Generated response using LinkedIn agent")

    except Exception as e:
        logger.error(f"Error using LinkedIn agent: {e}")
        text = f"Error: {str(e)}"

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
    logger.info("Answering completion request using LinkedIn agent")

    # Extract prompt content
    if req.prompt is None:
        content = ""
    elif isinstance(req.prompt, list):
        content = "\n".join(map(str, req.prompt))
    else:
        content = str(req.prompt)

    # Always use LinkedIn agent
    try:
        # Get LinkedIn agent
        agent = get_linkedin_agent()

        # Use the agent to generate a response
        from langchain_core.messages import HumanMessage
        result = agent.invoke({"messages": [HumanMessage(content=content)]})

        # Extract the text from the result
        if isinstance(result, dict) and "messages" in result:
            # Get the last message from the agent
            messages = result["messages"]
            if messages:
                last_message = messages[-1]
                if hasattr(last_message, 'content'):
                    text = last_message.content
                else:
                    text = str(last_message)
            else:
                text = "No response generated."
        else:
            text = str(result)

        logger.info(f"Generated response using LinkedIn agent")

    except Exception as e:
        logger.error(f"Error using LinkedIn agent: {e}")
        text = f"Error: {str(e)}"

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


@app.get("/v1/linkedin/agents")
async def list_linkedin_agents():
    """List all available LinkedIn post generation agents and their metadata."""
    try:
        agent = get_linkedin_agent()
        agents_info = agent.list_available_agents()

        return {
            "object": "list",
            "data": agents_info,
        }
    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail={"error": {"message": f"Error listing agents: {str(e)}"}})


@app.get("/v1/models")
async def list_models():
    model_info = {
        "id": 'Linkedin-Post-Agent' ,
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
