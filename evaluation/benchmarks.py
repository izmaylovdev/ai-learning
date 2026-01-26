"""Benchmark test suites for agent evaluation."""

from typing import List, Dict, Any
from .evaluator import TestCase


def create_benchmark_suite(agent_type: str = "general") -> List[TestCase]:
    """
    Create a benchmark test suite based on agent type.

    Args:
        agent_type: Type of agent ("rag", "code_analysis", "linkedin_post", "general")

    Returns:
        List of test cases for evaluation
    """
    if agent_type == "rag":
        return create_rag_benchmark()
    elif agent_type == "code_analysis":
        return create_code_analysis_benchmark()
    elif agent_type == "linkedin_post":
        return create_linkedin_post_benchmark()
    else:
        return create_general_benchmark()


def create_rag_benchmark() -> List[TestCase]:
    """Create benchmark for RAG agents."""
    return [
        TestCase(
            id="rag_001",
            question="What are the production 'Do's' for RAG?",
            expected_response="Production RAG systems should implement proper data ingestion pipelines, use appropriate chunking strategies, implement robust vector search, and ensure proper retrieval relevance scoring.",
            context={
                "expected_facts": [
                    "data ingestion",
                    "chunking strategies",
                    "vector search",
                    "retrieval relevance",
                    "production"
                ]
            },
            tags=["rag", "production", "best_practices"],
            difficulty="medium"
        ),
        TestCase(
            id="rag_002",
            question="What is the difference between standard retrieval and the ColPali approach?",
            expected_response="ColPali uses visual document understanding to retrieve information from documents as images, while standard retrieval processes text-based chunks from parsed documents.",
            context={
                "expected_facts": [
                    "visual document understanding",
                    "images",
                    "text-based chunks",
                    "parsed documents"
                ]
            },
            tags=["rag", "colpali", "retrieval_methods"],
            difficulty="hard"
        ),
        TestCase(
            id="rag_003",
            question="Why is hybrid search better than vector-only search?",
            expected_response="Hybrid search combines semantic vector search with keyword-based search, providing better recall for exact matches while maintaining semantic understanding for conceptual queries.",
            context={
                "expected_facts": [
                    "semantic vector search",
                    "keyword-based search",
                    "better recall",
                    "exact matches",
                    "semantic understanding"
                ]
            },
            tags=["rag", "search", "hybrid"],
            difficulty="medium"
        ),
        TestCase(
            id="rag_004",
            question="How do you evaluate RAG system performance?",
            context={
                "expected_facts": [
                    "retrieval accuracy",
                    "answer relevance",
                    "factual correctness",
                    "response time"
                ]
            },
            tags=["rag", "evaluation", "metrics"],
            difficulty="medium"
        )
    ]


def create_code_analysis_benchmark() -> List[TestCase]:
    """Create benchmark for code analysis agents."""
    return [
        TestCase(
            id="code_001",
            question="What is the structure of the agents directory?",
            context={
                "expected_facts": [
                    "code_analysis",
                    "learning_program_rag",
                    "linkedin_post_generation",
                    "agent.py",
                    "tools.py"
                ]
            },
            tags=["code", "structure", "directory"],
            difficulty="easy"
        ),
        TestCase(
            id="code_002",
            question="How does the RAG agent work?",
            context={
                "expected_facts": [
                    "search_knowledge_base",
                    "vector store",
                    "embeddings",
                    "similarity search"
                ]
            },
            tags=["code", "rag", "implementation"],
            difficulty="medium"
        ),
        TestCase(
            id="code_003",
            question="What design patterns are used in the codebase?",
            context={
                "expected_facts": [
                    "singleton pattern",
                    "factory pattern",
                    "agent pattern",
                    "tool composition"
                ]
            },
            tags=["code", "patterns", "architecture"],
            difficulty="hard"
        ),
        TestCase(
            id="code_004",
            question="How are dependencies managed in this project?",
            context={
                "expected_facts": [
                    "requirements.txt",
                    "imports",
                    "package structure"
                ]
            },
            tags=["code", "dependencies", "structure"],
            difficulty="easy"
        )
    ]


def create_linkedin_post_benchmark() -> List[TestCase]:
    """Create benchmark for LinkedIn post generation agents."""
    return [
        TestCase(
            id="linkedin_001",
            question="Create a LinkedIn post about RAG systems in production",
            context={
                "expected_facts": [
                    "professional tone",
                    "engaging hook",
                    "technical content",
                    "call to action"
                ]
            },
            tags=["linkedin", "rag", "production"],
            difficulty="medium"
        ),
        TestCase(
            id="linkedin_002",
            question="Write a post about the benefits of AI learning programs",
            context={
                "expected_facts": [
                    "learning benefits",
                    "skill development",
                    "career growth",
                    "practical application"
                ]
            },
            tags=["linkedin", "learning", "ai"],
            difficulty="easy"
        ),
        TestCase(
            id="linkedin_003",
            question="Create a technical post about vector databases",
            context={
                "expected_facts": [
                    "vector databases",
                    "technical accuracy",
                    "use cases",
                    "benefits"
                ]
            },
            tags=["linkedin", "technical", "databases"],
            difficulty="medium"
        )
    ]


def create_general_benchmark() -> List[TestCase]:
    """Create general benchmark for any agent."""
    return [
        TestCase(
            id="general_001",
            question="What is artificial intelligence?",
            expected_response="Artificial intelligence (AI) is the simulation of human intelligence in machines that are programmed to think and learn like humans.",
            context={
                "expected_facts": [
                    "simulation",
                    "human intelligence",
                    "machines",
                    "think and learn"
                ]
            },
            tags=["general", "ai", "definition"],
            difficulty="easy"
        ),
        TestCase(
            id="general_002",
            question="Explain machine learning in simple terms",
            context={
                "expected_facts": [
                    "algorithms",
                    "data",
                    "patterns",
                    "predictions"
                ]
            },
            tags=["general", "ml", "explanation"],
            difficulty="easy"
        ),
        TestCase(
            id="general_003",
            question="What are the ethical considerations in AI development?",
            context={
                "expected_facts": [
                    "bias",
                    "fairness",
                    "privacy",
                    "transparency",
                    "accountability"
                ]
            },
            tags=["general", "ethics", "ai"],
            difficulty="medium"
        )
    ]


def create_custom_benchmark(test_cases_data: List[Dict[str, Any]]) -> List[TestCase]:
    """
    Create a custom benchmark from provided test case data.

    Args:
        test_cases_data: List of dictionaries with test case information

    Returns:
        List of TestCase objects
    """
    test_cases = []

    for data in test_cases_data:
        test_case = TestCase(
            id=data.get("id", f"custom_{len(test_cases)+1:03d}"),
            question=data["question"],
            expected_response=data.get("expected_response"),
            context=data.get("context", {}),
            tags=data.get("tags", []),
            difficulty=data.get("difficulty", "medium")
        )
        test_cases.append(test_case)

    return test_cases
