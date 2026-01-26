"""Configuration for agent evaluation."""

import os

# Evaluation settings
EVALUATION_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "results")
DEFAULT_METRICS = ["accuracy", "relevance", "clarity"]

# Metric weights by agent type
METRIC_WEIGHTS = {
    "rag_agent": {
        "accuracy": 1.0,
        "relevance": 1.0,
        "clarity": 0.8
    },
    "code_analysis_agent": {
        "accuracy": 1.0,
        "relevance": 1.0,
        "clarity": 0.9
    },
    "linkedin_agent": {
        "accuracy": 0.7,
        "relevance": 0.9,
        "clarity": 1.0
    }
}

# Benchmark configurations
BENCHMARK_CONFIG = {
    "rag": {
        "focus_areas": ["retrieval_accuracy", "factual_correctness", "source_attribution"],
        "difficulty_distribution": {"easy": 0.2, "medium": 0.6, "hard": 0.2}
    },
    "code_analysis": {
        "focus_areas": ["code_understanding", "architecture_analysis", "pattern_recognition"],
        "difficulty_distribution": {"easy": 0.3, "medium": 0.5, "hard": 0.2}
    },
    "linkedin_post": {
        "focus_areas": ["engagement", "professionalism", "clarity", "call_to_action"],
        "difficulty_distribution": {"easy": 0.4, "medium": 0.5, "hard": 0.1}
    }
}

# Scoring thresholds
SCORING_THRESHOLDS = {
    "excellent": 0.9,
    "good": 0.7,
    "acceptable": 0.5,
    "poor": 0.3
}

# Reporting settings
INCLUDE_DETAILED_METRICS = True
INCLUDE_FAILURE_ANALYSIS = True
GENERATE_CHARTS = False  # Set to True if you want to generate charts (requires matplotlib)

# Performance targets (in milliseconds)
PERFORMANCE_TARGETS = {
    "rag_agent": 5000,
    "code_analysis_agent": 3000,
    "linkedin_agent": 2000
}
