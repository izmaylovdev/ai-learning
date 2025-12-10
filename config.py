"""Configuration for the AI learning project."""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "output")
AUDIO_DIR = os.path.join(OUTPUT_DIR, "audio")
TRANSCRIPTS_DIR = os.path.join(OUTPUT_DIR, "transcripts")

# Ensure directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)

# Qdrant settings
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "ai_learning_docs"

# Embedding settings
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# Text splitting settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Whisper settings
WHISPER_MODEL = "base"  # Options: tiny, base, small, medium, large
WHISPER_DEVICE = "cpu"  # Options: cpu, cuda
WHISPER_COMPUTE_TYPE = "int8"  # Options: int8, float16, float32


# Google Gemini Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-pro")

# Generation parameters
GEMINI_TEMPERATURE = float(os.getenv("GEMINI_TEMPERATURE", "0.7"))
GEMINI_TOP_P = float(os.getenv("GEMINI_TOP_P", "0.95"))
GEMINI_TOP_K = int(os.getenv("GEMINI_TOP_K", "40"))
GEMINI_MAX_OUTPUT_TOKENS = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "2048"))

# HuggingFace Local Model Configuration
HUGGINGFACE_MODEL = os.getenv("HUGGINGFACE_MODEL", "openai/gpt-oss-20b")
HUGGINGFACE_DEVICE = os.getenv("HUGGINGFACE_DEVICE", "auto")  # "auto", "cpu", "cuda", or "mps"
HUGGINGFACE_TEMPERATURE = float(os.getenv("HUGGINGFACE_TEMPERATURE", "0.7"))
HUGGINGFACE_TOP_P = float(os.getenv("HUGGINGFACE_TOP_P", "0.95"))
HUGGINGFACE_TOP_K = int(os.getenv("HUGGINGFACE_TOP_K", "50"))
HUGGINGFACE_MAX_NEW_TOKENS = int(os.getenv("HUGGINGFACE_MAX_NEW_TOKENS", "256"))
HUGGINGFACE_DO_SAMPLE = os.getenv("HUGGINGFACE_DO_SAMPLE", "true").lower() == "true"

# RAG Configuration
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "3"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.5"))

