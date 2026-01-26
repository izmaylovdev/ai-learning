# AI Learning Project

A comprehensive AI learning platform featuring multiple specialized agents for code analysis, retrieval-augmented generation (RAG), and LinkedIn content creation. This project demonstrates practical applications of AI in software development, document processing, and content generation.

## 🚀 Features

### Specialized AI Agents
- **Code Analysis Agent** - Analyzes codebases, explains architecture, and answers technical questions
- **RAG Agent** - Retrieval-augmented generation for question answering from learning materials
- **LinkedIn Post Generator** - Creates professional LinkedIn content from various topics

### Data Processing Pipeline
- PDF text extraction and processing
- Video to audio extraction and transcription
- Intelligent document chunking and indexing
- Vector embeddings for semantic search

### Evaluation Framework
- Comprehensive agent performance metrics (Accuracy, Relevance, Clarity)
- Automated benchmarking and testing
- Detailed evaluation reports and analytics

## 🏗️ File Structure Map

```
ai-learning/
├── agents/                         # Specialized AI agents
│   ├── code_analysis/             # Code repository analysis agent
│   │   ├── agent.py               # Main agent implementation
│   │   ├── cli.py                 # Command line interface
│   │   ├── indexer.py             # Code indexing functionality
│   │   └── tools.py               # Agent tools and utilities
│   ├── learning_program_rag/      # RAG-based Q&A agent
│   │   ├── agent.py               # Main RAG agent
│   │   ├── tools.py               # RAG tools and utilities
│   │   ├── data_extraction/       # Data processing pipeline
│   │   │   ├── extract_pdf.py     # PDF text extraction
│   │   │   └── extract_video.py   # Video transcription
│   │   └── tests/                 # Agent testing suite
│   │       ├── test_agent_langchain.py
│   │       ├── test_agentignore.py
│   │       └── test_questions.py
│   └── linkedin_post_generation/  # LinkedIn content creation
├── data/                          # Learning materials and outputs
│   ├── *.mp4                     # Video learning materials
│   ├── *.pdf                     # PDF documents
│   └── output/                   # Processed outputs
│       ├── audio/                # Extracted audio files
│       └── transcripts/          # Generated transcripts
├── evaluation/                    # Comprehensive evaluation framework
│   ├── benchmarks.py             # Performance benchmarks
│   ├── config.py                 # Evaluation configuration
│   ├── demo.py                   # Evaluation demonstrations
│   ├── evaluate_agents.py        # Main evaluation runner
│   ├── evaluator.py              # Core evaluation logic
│   ├── metrics.py                # Evaluation metrics
│   ├── quick_eval.py             # Quick evaluation script
│   └── results/                  # Evaluation results storage
├── models/                        # LLM integrations
│   ├── gemini_model.py           # Google Gemini integration
│   └── llm_studio_model.py       # LLM Studio integration
├── model_server/                  # FastAPI model serving
│   ├── main.py                   # FastAPI application
│   ├── run_model_server.py       # Server runner
│   └── schemas.py                # API schemas
├── util/                         # Core utilities
│   ├── embeddings.py             # Embedding utilities
│   ├── interfaces.py             # Common interfaces
│   └── vector_stores.py          # Vector storage implementations
├── config.py                     # Global configuration
├── docker-compose.yml            # Docker services configuration
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

## 🛠️ Technology Stack

- **Language Models**: Google Gemini, Local LLM via LLM Studio
- **Vector Database**: Qdrant for semantic search
- **Embeddings**: Sentence Transformers, HuggingFace
- **Framework**: LangChain for agent orchestration
- **Web Interface**: OpenWebUI integration
- **API**: FastAPI for model serving
- **Media Processing**: Whisper for audio transcription

## 📋 Prerequisites

- Python 3.8+
- Docker and Docker Compose
- Google API key for Gemini (optional)

## 🚀 Quick Start

### 1. Clone and Setup
```bash
git clone <repository-url>
cd ai-learning
pip install -r requirements.txt
```

### 2. Environment Configuration
Create a `.env` file with your configuration:
```env
# Google Gemini (optional)
GOOGLE_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-2.5-flash

# Backend selection
EMBEDDING_BACKEND=huggingface
VECTOR_STORE_BACKEND=qdrant
GENERATOR_BACKEND=gemini
```

### 3. Start Services
```bash
# Start Qdrant vector database and OpenWebUI
docker-compose up -d

# Start the model server (optional)
python model_server/run_model_server.py
```

### 4. Process Learning Materials
```bash
# Extract text from PDFs
python agents/learning_program_rag/data_extraction/extract_pdf.py

# Extract and transcribe videos
python agents/learning_program_rag/data_extraction/extract_video.py
```

## 💻 Usage Examples

### Code Analysis Agent
```python
from agents.code_analysis.agent import get_agent

agent = get_agent()
response = agent.invoke({
    "messages": [{
        "role": "user", 
        "content": "Explain the architecture of this project"
    }]
})
```

### RAG Agent
```python
from agents.learning_program_rag.agent import get_agent

agent = get_agent()
response = agent.invoke({
    "messages": [{
        "role": "user",
        "content": "What are the key concepts of RAG?"
    }]
})
```

### LinkedIn Post Generation
```python
from linkedin_post_generation.agent import get_agent

agent = get_agent()
response = agent.invoke({
    "messages": [{
        "role": "user",
        "content": "Create a post about implementing RAG systems"
    }]
})
```

## 🔧 Configuration

### Model Backends
- **Gemini**: Google's latest language model
- **LLM Studio**: Local model deployment via LLM Studio
- **Custom**: Extend with your own model implementations

### Vector Store Options
- **Qdrant**: Primary vector database (recommended)
- **In-memory**: For testing and development

### Embedding Models
- **sentence-transformers/all-MiniLM-L6-v2**: Default lightweight model
- **Custom models**: Configure via `EMBEDDING_MODEL` environment variable

## 📊 Evaluation

Run comprehensive agent evaluations:

```bash
# Evaluate all agents
python evaluation/evaluate_agents.py

# Quick evaluation
python evaluation/quick_eval.py

# View results
ls evaluation/results/
```

### Evaluation Metrics
- **Accuracy** (0.0-1.0): Factual correctness and reliability
- **Relevance** (0.0-1.0): How well responses address questions
- **Clarity** (0.0-1.0): Response structure and readability

## 🧪 Testing

```bash
# Run basic tests
python -m pytest agents/learning_program_rag/tests/

# Test specific agents
python agents/learning_program_rag/tests/test_agent_langchain.py
```

## 📂 Data Management

### Supported Formats
- **PDFs**: Automatic text extraction and chunking
- **Videos**: Audio extraction and Whisper transcription
- **Text files**: Direct processing and indexing

### Processing Pipeline
1. Extract content from source materials
2. Chunk text with overlap for context preservation
3. Generate embeddings using sentence transformers
4. Index in Qdrant vector database
5. Enable semantic search and retrieval

## 🌐 Web Interface

Access the web interface via OpenWebUI:
- URL: `http://localhost:3000`
- Features: Chat interface, agent selection, document upload
- Integration: Direct connection to your local agents

## 🔍 Monitoring

### Vector Database
- Qdrant Web UI: `http://localhost:6333/dashboard`
- Collection monitoring and search testing

### Model Server
- FastAPI docs: `http://localhost:8000/docs`
- Health checks and API documentation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Run evaluations: `python evaluation/evaluate_agents.py`
5. Submit a pull request

## 📄 License

This project is part of Ciklum's AI learning program and is intended for educational purposes.

## 🆘 Troubleshooting

### Common Issues
1. **Qdrant connection errors**: Ensure Docker is running and port 6333 is available
2. **Memory issues**: Reduce chunk size or use smaller embedding models
3. **API rate limits**: Configure appropriate delays for external API calls

### Getting Help
- Check the evaluation results for agent performance insights
- Review logs in `agents/learning_program_rag/tests/test_results.log`
- Ensure all dependencies are installed correctly
