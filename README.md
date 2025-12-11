# AI Learning RAG System

A modular Retrieval-Augmented Generation (RAG) system designed for question-answering on AI learning materials. The system processes PDFs and video transcripts, stores them in a vector database, and uses various text generation models to answer questions based on the retrieved context.

## Features

- **Modular Architecture**: Pluggable components for embeddings, vector stores, and text generators
- **Multiple Data Sources**: Support for PDF documents and video transcripts
- **Flexible Text Generation**: Choose between Google Gemini, HuggingFace models, or custom generators
- **Vector Storage**: Qdrant-based semantic search for efficient retrieval
- **Configurable RAG Pipeline**: Customizable chunk size, overlap, and retrieval parameters

## Project Structure

```
ai-learning/
├── answer_question.py          # Main RAG question-answering system
├── config.py                   # Central configuration file
├── embeddings.py               # Embedding model implementations
├── interfaces.py               # Abstract interfaces for modular components
├── vector_stores.py            # Vector database implementations
├── docker-compose.yml          # Docker setup for Qdrant
├── requirements.txt            # Python dependencies
│── get_data_for_query.py       # Query-based data retrieval
│
├── data/                       # Source data directory
│   ├── *.pdf                   # PDF learning materials
│   ├── *.mp4                   # Video learning materials
│   └── output/                 # Processed data
│       ├── audio/              # Extracted audio from videos
│       └── transcripts/        # Transcribed text from videos
│
├── data_extraction/            # Data processing modules
│   ├── extract_pdf.py          # PDF text extraction and chunking
│   ├── extract_video.py        # Video audio extraction and transcription
│
├── text_generation/            # Text generation implementations
│   ├── generator.py            # Generator interface
│   ├── get_generator.py        # Generator factory
│   ├── google_gemini.py        # Google Gemini implementation
│   └── gpt_oss_20b.py          # HuggingFace GPT implementation
│
└── tests/                      # Test suite
    ├── test_questions.py       # Question-answering tests
    └── test_results.log        # Test results log
```

## Installation

### Prerequisites

- Python 3.8+
- Docker and Docker Compose (for Qdrant)
- Google API key (for Gemini model)

### Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd ai-learning
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   Create a `.env` file in the project root:
   ```env
   # Google Gemini Configuration
   GOOGLE_API_KEY=your_google_api_key_here
   GEMINI_MODEL=gemini-2.5-flash
   GEMINI_TEMPERATURE=0.7
   GEMINI_TOP_P=0.95
   GEMINI_TOP_K=40
   GEMINI_MAX_OUTPUT_TOKENS=2048

   # HuggingFace Model Configuration (optional)
   HUGGINGFACE_MODEL=openai/gpt-oss-20b

   # RAG Configuration
   TOP_K_RESULTS=3
   SIMILARITY_THRESHOLD=0.5
   ```

5. **Start Qdrant vector database:**
   ```bash
   docker-compose up -d
   ```

## Configuration

All configuration is centralized in `config.py`. Key settings include:

- **Paths**: Data directories for source materials and outputs
- **Qdrant**: Vector database connection settings
- **Embeddings**: Model selection and dimensions
- **Text Splitting**: Chunk size and overlap for document processing
- **Whisper**: Audio transcription settings
- **Text Generation**: Parameters for Gemini and HuggingFace models
- **RAG**: Retrieval parameters (top-k, similarity threshold)

## Usage

### 1. Processing Data

**Extract and process PDFs:**
```python
from data_extraction.extract_pdf import PDFProcessor

processor = PDFProcessor()
processor.process_pdf("path/to/document.pdf")
```

**Extract and transcribe videos:**
```python
from data_extraction.extract_video import VideoProcessor

processor = VideoProcessor()
processor.process_video("path/to/video.mp4")
```

### 2. Question Answering

```python
from answer_question import RAGQuestionAnswerer

# Initialize RAG system (uses default components from config)
rag = RAGQuestionAnswerer()

# Ask a question
question = "What is RAG?"
answer = rag.answer_question(question)
print(answer)
```

### 3. Custom Configuration

**Use specific components:**
```python
from embeddings import get_embedding_model
from vector_stores import get_vector_store
from text_generation.get_generator import get_generator

# Initialize custom components
embeddings = get_embedding_model("huggingface")
vector_store = get_vector_store("qdrant")
generator = get_generator("gemini")  # or "huggingface"

# Initialize RAG with custom components
rag = RAGQuestionAnswerer(
    embedding_model=embeddings,
    vector_store=vector_store,
    generator=generator
)
```

## Architecture

### Core Components

1. **Interfaces** (`interfaces.py`):
   - `EmbeddingInterface`: Abstract interface for embedding models
   - `VectorStoreInterface`: Abstract interface for vector databases
   - `GeneratorInterface`: Abstract interface for text generators

2. **Embeddings** (`embeddings.py`):
   - HuggingFace sentence-transformers for semantic embeddings
   - Default: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)

3. **Vector Stores** (`vector_stores.py`):
   - Qdrant for efficient semantic search
   - Support for metadata filtering and similarity thresholds

4. **Text Generators** (`text_generation/`):
   - Google Gemini (`google_gemini.py`)
   - HuggingFace GPT models (`gpt_oss_20b.py`)
   - Factory pattern for easy switching (`get_generator.py`)

5. **RAG System** (`answer_question.py`):
   - Retrieval of relevant document chunks
   - Context-aware answer generation
   - Modular component integration

### Data Processing Pipeline

1. **Input**: PDFs or videos in the `data/` directory
2. **Extraction**: Text from PDFs, audio from videos
3. **Transcription**: Whisper-based speech-to-text for videos
4. **Chunking**: Text splitting with configurable size and overlap
5. **Embedding**: Convert chunks to vector representations
6. **Storage**: Index vectors in Qdrant with metadata
7. **Retrieval**: Semantic search for relevant chunks
8. **Generation**: Answer synthesis using retrieved context

## Testing

Run the test suite:
```bash
python -m pytest tests/
```

Or run specific tests:
```bash
python tests/test_questions.py
```

Test results are logged to `tests/test_results.log`.

## Dependencies

### Core Libraries

- **pypdf**: PDF text extraction
- **langchain**: Text processing and splitting framework
- **qdrant-client**: Vector database client
- **sentence-transformers**: Embedding models
- **transformers**: HuggingFace models
- **torch**: PyTorch for model inference
- **google-genai**: Google Gemini API client

### Optional

- **bitsandbytes**: Model quantization (8-bit/4-bit loading)
- **accelerate**: Optimized model loading

See `requirements.txt` for complete list and versions.

## Docker Services

The `docker-compose.yml` file sets up:
- **Qdrant**: Vector database (accessible at `localhost:6333`)

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GOOGLE_API_KEY` | Google Gemini API key | Required |
| `GEMINI_MODEL` | Gemini model name | `gemini-2.5-flash` |
| `GEMINI_TEMPERATURE` | Generation temperature | `0.7` |
| `GEMINI_TOP_P` | Nucleus sampling | `0.95` |
| `GEMINI_TOP_K` | Top-k sampling | `40` |
| `GEMINI_MAX_OUTPUT_TOKENS` | Max response length | `2048` |
| `HUGGINGFACE_MODEL` | HF model name | `openai/gpt-oss-20b` |
| `HUGGINGFACE_DEVICE` | Device for inference | `auto` |
| `TOP_K_RESULTS` | Number of chunks to retrieve | `3` |
| `SIMILARITY_THRESHOLD` | Minimum similarity score | `0.5` |
## License
MIT