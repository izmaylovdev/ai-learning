# RAG Tests

This folder contains test scripts for the RAG Question Answering system.

## Test Questions Script

The `test_questions.py` script runs a set of predefined questions through the RAG system and logs the results.

### Questions Tested

1. What are the production 'Do's' for RAG?
2. What is the difference between standard retrieval and the ColPali approach?
3. Why is hybrid search better than vector-only search?

### Usage

```bash
# Make sure you're in the virtual environment
source ../.venv/bin/activate  # or activate your venv

# Run the test script
python test_questions.py
```

### Output

The script will:
- Process each question through the RAG system
- Display results in the console
- Write detailed results to `test_results.log` in this folder

The log file includes:
- Timestamp
- All questions and their answers
- Source documents used
- Relevance scores
- Summary statistics

### Requirements

- The RAG system must be initialized (Qdrant running, data indexed)
- All dependencies from `requirements.txt` must be installed
- Environment variables must be configured (.env file)

