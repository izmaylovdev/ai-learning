# AI Learning Project

A RAG (Retrieval Augmented Generation) system with Google Gemini integration for processing and answering questions about educational content.

## Features

- **Google Gemini Integration**: Advanced text generation using Google's Gemini models
- **RAG Architecture**: Interface-based design for embeddings, vector stores, and generators
- **PDF & Video Processing**: Extract content from PDFs and video transcripts
- **Flexible Configuration**: Environment-based configuration system

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Copy the example environment file and add your API keys:

```bash
cp .env.example .env
```

Edit `.env` and set your Google API key:

```
GOOGLE_API_KEY=your_google_api_key_here
```

### 3. Get Google API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the key and paste it in your `.env` file

## Usage

### Using GeminiGenerator

```python
from generators import GeminiGenerator

# Initialize with default config from .env
generator = GeminiGenerator()

# Simple text generation
response = generator.generate("Explain RAG in simple terms.")
print(response)

# Question answering with context
context = "RAG combines retrieval with generation..."
question = "What is RAG?"
answer = generator.generate_answer(question, context)
print(answer)

# Chat-style interaction
messages = [
    {"role": "user", "content": "What is a vector database?"},
    {"role": "assistant", "content": "A vector database stores..."},
    {"role": "user", "content": "Can you give an example?"}
]
response = generator.generate_with_chat(messages)
print(response)
```

### Custom Configuration

```python
# Initialize with custom parameters
generator = GeminiGenerator(
    model_name="gemini-pro",
    temperature=0.5,
    max_output_tokens=1024
)

# Update configuration dynamically
generator.update_generation_config(
    temperature=0.9,
    top_p=0.95
)
```

### Run Example

```bash
python example_gemini.py
```

## Project Structure

```
.
├── interfaces.py           # Abstract interfaces for RAG components
├── generators.py          # Generator implementations (Gemini)
├── embeddings.py          # Embedding model implementations
├── vector_stores.py       # Vector database implementations
├── config.py              # Configuration management
├── example_gemini.py      # Example usage
├── requirements.txt       # Python dependencies
├── .env.example          # Example environment variables
└── data/                 # Educational content (PDFs, videos)
```

## Configuration Options

### Gemini Model Settings

- `GOOGLE_API_KEY`: Your Google API key (required)
- `GEMINI_MODEL`: Model name (default: "gemini-pro")
- `GEMINI_TEMPERATURE`: Randomness in generation (0.0-1.0, default: 0.7)
- `GEMINI_TOP_P`: Nucleus sampling parameter (default: 0.95)
- `GEMINI_TOP_K`: Top-k sampling parameter (default: 40)
- `GEMINI_MAX_OUTPUT_TOKENS`: Maximum response length (default: 2048)

## Architecture

### Interface-Based Design

The project uses abstract interfaces for flexibility:

- **GeneratorInterface**: Text generation models
- **EmbeddingInterface**: Document and query embeddings
- **VectorStoreInterface**: Vector database operations

This allows easy swapping of implementations (e.g., switching from Gemini to OpenAI).

## Available Models

### Google Gemini Models

- `gemini-pro`: Best for text-based tasks
- `gemini-pro-vision`: Supports image inputs
- `gemini-1.5-pro`: Latest model with extended context

## Troubleshooting

### "No module named 'generativeai'"

Install the required packages:
```bash
pip install google-generativeai
```

### "API key is required"

Make sure your `.env` file exists and contains:
```
GOOGLE_API_KEY=your_actual_api_key
```

### Rate Limiting

If you hit rate limits, consider:
- Adding delays between requests
- Reducing `max_output_tokens`
- Using a paid API tier

## License

MIT License

## Contributing

Contributions welcome! Please feel free to submit a Pull Request.

