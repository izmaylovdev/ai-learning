# Agent Evaluation Framework

A comprehensive evaluation system for measuring the success of AI agents across three key dimensions: **Accuracy**, **Relevance**, and **Clarity**.

## Overview

This evaluation framework provides:
- **Standardized metrics** for measuring agent performance
- **Benchmark test suites** for different agent types
- **Automated evaluation** with detailed reporting
- **Configurable evaluation criteria** based on agent purpose

## Key Metrics

### 1. Accuracy (0.0 - 1.0)
Measures factual correctness and reliability of responses:
- **Similarity to expected responses** (if available)
- **Presence of factual content** and specific details
- **Appropriate uncertainty handling** ("I don't know" when appropriate)

### 2. Relevance (0.0 - 1.0) 
Measures how well the response addresses the question:
- **Keyword overlap** between question and response
- **Question type alignment** (what/how/why questions get appropriate answer types)
- **Focus and topic adherence** without going off-topic

### 3. Clarity (0.0 - 1.0)
Measures how understandable and well-structured responses are:
- **Readability** (appropriate sentence length and structure)
- **Organization** (logical flow, connectors, formatting)
- **Complexity level** (appropriate technical depth)

## Quick Start

### Basic Evaluation

```python
from evaluation import AgentEvaluator, AccuracyMetric, RelevanceMetric, ClarityMetric

# Create evaluator with metrics
evaluator = AgentEvaluator([
    AccuracyMetric(),
    RelevanceMetric(), 
    ClarityMetric()
])

# Your agent function
def my_agent(question: str) -> str:
    return "Your agent's response here"

# Create test cases
test_cases = [
    TestCase(
        id="test_001",
        question="What is machine learning?",
        expected_response="Machine learning is a subset of AI..."
    )
]

# Run evaluation
results = evaluator.evaluate_agent(my_agent, test_cases, "my_agent")
```

### Using Pre-built Benchmarks

```python
from evaluation import create_benchmark_suite

# Get benchmark for specific agent type
rag_tests = create_benchmark_suite("rag")
code_tests = create_benchmark_suite("code_analysis") 
linkedin_tests = create_benchmark_suite("linkedin_post")

# Evaluate with benchmark
results = evaluator.evaluate_agent(my_agent, rag_tests, "my_rag_agent")
```

### Command Line Evaluation

Evaluate all agents at once:
```bash
python evaluation/evaluate_agents.py --agent all
```

Evaluate specific agent:
```bash
python evaluation/evaluate_agents.py --agent rag
python evaluation/evaluate_agents.py --agent code_analysis
python evaluation/evaluate_agents.py --agent linkedin
```

## Agent-Specific Evaluations

### RAG Agent
- **Focus**: Factual accuracy and source-based responses
- **Key metrics**: Accuracy (1.0), Relevance (1.0), Clarity (0.8)
- **Test areas**: Production best practices, technical comparisons, methodology explanations

### Code Analysis Agent  
- **Focus**: Technical accuracy and architectural understanding
- **Key metrics**: Accuracy (1.0), Relevance (1.0), Clarity (0.9)
- **Test areas**: Code structure, design patterns, implementation details

### LinkedIn Post Agent
- **Focus**: Engagement and professional communication
- **Key metrics**: Clarity (1.0), Relevance (0.9), Accuracy (0.7)  
- **Test areas**: Professional tone, technical content adaptation, call-to-actions

## Evaluation Results

Results are saved as JSON files in `evaluation_results/` with:
- **Individual test scores** for each metric
- **Overall performance** statistics
- **Execution time** measurements
- **Detailed failure** analysis

### Sample Result Structure

```json
{
  "agent_name": "rag_agent",
  "timestamp": "20260126_143022",
  "total_tests": 4,
  "successful_tests": 3,
  "failed_tests": 1,
  "average_score": 0.847,
  "results": [
    {
      "test_case_id": "rag_001",
      "question": "What are the production Do's for RAG?",
      "accuracy_score": 0.92,
      "relevance_score": 0.88,
      "clarity_score": 0.76,
      "overall_score": 0.853,
      "execution_time_ms": 1247.3
    }
  ]
}
```

## Customizing Evaluation

### Custom Metrics

```python
from evaluation.metrics import EvaluationMetric, MetricResult

class CustomMetric(EvaluationMetric):
    def __init__(self):
        super().__init__("custom_metric", weight=1.0)
    
    def calculate(self, question, agent_response, expected_response=None, context=None):
        # Your scoring logic here
        score = your_calculation(agent_response)
        return MetricResult(score=score, details={"custom_detail": "value"})
```

### Custom Test Cases

```python
from evaluation.evaluator import TestCase

custom_tests = [
    TestCase(
        id="custom_001",
        question="Your custom question",
        expected_response="Expected answer",
        context={"expected_facts": ["fact1", "fact2"]},
        tags=["custom", "domain_specific"],
        difficulty="medium"
    )
]
```

## Configuration

Edit `evaluation/config.py` to customize:
- **Metric weights** for different agent types
- **Performance targets** for response time
- **Scoring thresholds** for quality levels
- **Output preferences** and reporting options

## Best Practices

1. **Regular Evaluation**: Run evaluations after code changes
2. **Diverse Test Cases**: Include easy, medium, and hard questions
3. **Domain-Specific Tests**: Create tests specific to your agent's purpose  
4. **Baseline Establishment**: Track performance over time
5. **Failure Analysis**: Review failed test cases to improve agents

## Integration with CI/CD

Add evaluation to your pipeline:

```yaml
# GitHub Actions example
- name: Evaluate Agents
  run: |
    python evaluation/evaluate_agents.py --agent all
    # Upload results or fail pipeline if scores below threshold
```

## Dependencies

The evaluation framework requires:
- `langchain` (for agent interfaces)
- `dataclasses` (built-in Python)
- `json` (built-in Python)
- `re` (built-in Python) 
- `difflib` (built-in Python)

## Output Files

Evaluation generates:
- `evaluation_results/{agent_name}_evaluation_{timestamp}.json` - Detailed results
- Console reports with summary statistics
- Optional charts (if matplotlib is installed and enabled)

## Extending the Framework

### Adding New Agent Types

1. Create benchmark in `benchmarks.py`
2. Add agent wrapper in `evaluate_agents.py`
3. Configure metrics weights in `config.py`
4. Update this README

### Adding New Metrics

1. Create metric class inheriting from `EvaluationMetric`
2. Implement `calculate()` method
3. Add to metric imports in `__init__.py`
4. Use in evaluator configuration

## Troubleshooting

**Common Issues:**
- **Import errors**: Ensure proper Python path configuration
- **Agent initialization**: Check agent dependencies are available
- **Empty responses**: Verify agent functions return strings
- **Slow evaluation**: Consider reducing test case count or optimizing agents

For more help, check the evaluation logs or create an issue.
