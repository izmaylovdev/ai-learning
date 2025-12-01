"""Configuration settings for AI Learning project."""

import os

# Paths
DATA_DIR = "data"
OUTPUT_DIR = "output"
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

