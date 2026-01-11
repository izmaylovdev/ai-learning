from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Any
import os
import time
import uuid

from model_server.models import CompletionRequest, ChatCompletionRequest, Message, LinkedInPostRequest
from text_generation.get_generator import get_generator
from linkedin_post_generation.agent import LinkedInPostAgent
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
linkedin_agent = None

def get_linkedin_agent():
    """Lazy initialization of LinkedIn Post Agent."""
    global linkedin_agent
    if linkedin_agent is None:
        linkedin_agent = LinkedInPostAgent(
            generator_backend=GENERATOR_TYPE,
            repo_root=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
    return linkedin_agent


def messages_to_prompt(messages: List[Message]) -> str:
    # Concatenate messages into a single prompt. Support strings or richer content
    parts = []
    for m in messages:
        role = (m.role or "user").strip()
        content = extract_text_from_content(m.content)
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


def extract_topic_from_prompt(prompt: str) -> str:
    """Extract the topic from a user prompt asking for a LinkedIn post."""
    # Remove common phrases to get the core topic
    lower_prompt = prompt.lower()

    # Common patterns
    patterns = [
        "write a linkedin post about ",
        "create a linkedin post about ",
        "generate a linkedin post about ",
        "linkedin post about ",
        "write about ",
        "post about ",
        "topic: ",
        "topic is ",
    ]

    topic = prompt
    for pattern in patterns:
        if pattern in lower_prompt:
            idx = lower_prompt.index(pattern)
            topic = prompt[idx + len(pattern):].strip()
            break

    # Clean up the topic
    topic = topic.rstrip('.!?')
    return topic


def should_generate_linkedin_post(prompt: str, is_first_message: bool = True) -> bool:
    """Determine if the user is asking for a LinkedIn post.

    Args:
        prompt: The user's prompt
        is_first_message: Whether this is the first message in a conversation

    Returns:
        True if we should generate a LinkedIn post
    """
    lower_prompt = prompt.lower().strip()

    # Check for explicit LinkedIn post keywords
    linkedin_keywords = [
        "linkedin post",
        "linkedin article",
        "post for linkedin",
        "write a post about",
        "create a post about",
        "generate a post",
    ]

    if any(keyword in lower_prompt for keyword in linkedin_keywords):
        return True

    # If it's a first message and looks like a simple topic (not a question or conversation)
    # we treat it as a LinkedIn post request
    # This allows users to just type: "RAG systems" or "Python best practices"
    if is_first_message:
        # Short prompts (1-10 words) without question marks or conversation markers
        word_count = len(lower_prompt.split())
        has_question = '?' in prompt or lower_prompt.startswith(('what', 'why', 'how', 'when', 'where', 'who'))
        has_conversation = any(word in lower_prompt for word in ['hello', 'hi', 'hey', 'help', 'can you', 'could you', 'please'])

        # If it's a short topic-like input without questions or conversation
        if 1 <= word_count <= 10 and not has_question and not has_conversation:
            return True

    return False


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


# Add logging for submodel selection
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ModelServer")

# Update the chat_completions endpoint to include logging
@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    gen = get_generator(GENERATOR_TYPE)

    logger.info("Answering chat completion request")

    # Determine if this is the first message (only user messages, no prior assistant responses)
    is_first_message = True
    if req.messages:
        # Check if there are any assistant messages in the history
        is_first_message = not any(m.role == "assistant" for m in req.messages)
        prompt = messages_to_prompt(req.messages)
        # For first message detection, only use the last user message
        user_messages = [m for m in req.messages if m.role == "user"]
        if user_messages:
            last_user_content = extract_text_from_content(user_messages[-1].content)
        else:
            last_user_content = prompt
    elif req.prompt:
        # prompt can be string or list
        if isinstance(req.prompt, list):
            prompt = "\n".join(map(str, req.prompt))
        else:
            prompt = str(req.prompt)
        last_user_content = prompt
    else:
        prompt = ""
        last_user_content = ""

    # Check if this is a LinkedIn post request
    # For multi-turn conversations, only check the last user message
    if should_generate_linkedin_post(last_user_content, is_first_message):
        logger.info("Detected LinkedIn post request, using LinkedIn agent...")

        try:
            # Extract topic from the last user message
            topic = extract_topic_from_prompt(last_user_content)
            logger.info(f"Extracted topic: {topic}")

            # Get or create LinkedIn agent
            agent = get_linkedin_agent()

            # Generate the LinkedIn post
            result = agent.generate_post(
                topic=topic,
                style="professional",
                tone="informative",
                length="medium",
                include_hashtags=True,
                auto_select_agents=True,
                verbose=False,
            )

            # Log the agents used
            if result["agents_used"]:
                agents_list = [name for name, used in result["agents_used"].items() if used]
                logger.info(f"Triggered subagents: {', '.join(agents_list)}")

            # Format the response to be conversational
            text = result["post"]

            # Add metadata footer if agents were used
            if result["agents_used"]:
                agents_list = [name for name, used in result["agents_used"].items() if used]
                if agents_list:
                    text += f"\n\n---\n_Generated using: {', '.join(agents_list)}_"

        except Exception as e:
            logger.error(f"Error generating LinkedIn post: {e}")
            # Fallback to normal generation
            text = gen.generate(prompt)
    else:
        # Normal conversation - use standard generator
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


@app.post("/v1/linkedin/generate")
async def generate_linkedin_post(req: LinkedInPostRequest):
    """Generate a LinkedIn post using the LinkedIn Post Agent."""
    print(f"Generating LinkedIn post for topic: {req.topic}")

    try:
        agent = get_linkedin_agent()

        result = agent.generate_post(
            topic=req.topic,
            style=req.style,
            tone=req.tone,
            length=req.length,
            include_hashtags=req.include_hashtags,
            auto_select_agents=req.auto_select_agents,
            enabled_agents=req.enabled_agents,
            disabled_agents=req.disabled_agents,
            agent_params=req.agent_params,
            verbose=req.verbose,
        )

        return {
            "id": str(uuid.uuid4()),
            "object": "linkedin.post",
            "created": int(time.time()),
            "post": result["post"],
            "topic": result["topic"],
            "style": result["style"],
            "tone": result["tone"],
            "length": result["length"],
            "agents_used": result["agents_used"],
            "context_gathered": result["context_gathered"],
        }
    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail={"error": {"message": f"Error generating LinkedIn post: {str(e)}"}})


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
