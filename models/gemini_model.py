"""Google Gemini model for LangChain - importable module."""
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
# Find the project root (parent of models directory)
project_root = Path(__file__).parent.parent
env_path = project_root / ".env"
load_dotenv(dotenv_path=env_path)

# Initialize Gemini model
# Note: Set GOOGLE_API_KEY environment variable before using
gemini_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    max_output_tokens=2048,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)

# Alternative: Gemini Pro model (more powerful but slower)
gemini_pro_model = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0.7,
    max_output_tokens=2048,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)

