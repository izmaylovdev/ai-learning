"""Demo script showing evaluation framework usage."""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation import AgentEvaluator, AccuracyMetric, RelevanceMetric, ClarityMetric
from evaluation.evaluator import TestCase


def demo_simple_agent(question: str) -> str:
    """A simple demo agent for testing evaluation."""

    # Simple rule-based responses for demo
    question_lower = question.lower()

    if "machine learning" in question_lower:
        return """Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed for every task. It involves algorithms that can identify patterns in data and use those patterns to make predictions or classifications on new, unseen data."""

    elif "ai" in question_lower or "artificial intelligence" in question_lower:
        return """Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think and learn like humans. AI systems can perform tasks that typically require human intelligence, such as visual perception, speech recognition, decision-making, and language translation."""

    elif "python" in question_lower:
        return """Python is a high-level, interpreted programming language known for its simplicity and readability. It's widely used in data science, machine learning, web development, and automation. Python's syntax is designed to be intuitive and its extensive library ecosystem makes it powerful for various applications."""

    elif "evaluation" in question_lower or "measure" in question_lower:
        return """Evaluation in AI systems typically involves measuring performance across multiple dimensions such as accuracy (factual correctness), relevance (addressing the question properly), and clarity (understandability of responses). These metrics help ensure AI systems meet quality standards."""

    else:
        return """I understand you're asking about this topic, but I don't have specific information to provide a detailed answer. Could you please rephrase your question or provide more context?"""


def run_demo():
    """Run a demonstration of the evaluation framework."""

    print("=" * 80)
    print("AGENT EVALUATION FRAMEWORK DEMO")
    print("=" * 80)
    print()

    # Create evaluator with all three metrics
    metrics = [
        AccuracyMetric(weight=1.0),
        RelevanceMetric(weight=1.0),
        ClarityMetric(weight=1.0)
    ]

    evaluator = AgentEvaluator(
        metrics=metrics,
        output_dir="evaluation_results"
    )

    # Create demo test cases
    test_cases = [
        TestCase(
            id="demo_001",
            question="What is machine learning?",
            expected_response="Machine learning is a subset of AI that enables computers to learn from data without explicit programming.",
            context={
                "expected_facts": [
                    "subset of AI",
                    "learn from data",
                    "without explicit programming",
                    "patterns",
                    "predictions"
                ]
            },
            tags=["ml", "definition"],
            difficulty="easy"
        ),
        TestCase(
            id="demo_002",
            question="How do you evaluate AI system performance?",
            context={
                "expected_facts": [
                    "accuracy",
                    "relevance",
                    "clarity",
                    "metrics",
                    "quality standards"
                ]
            },
            tags=["evaluation", "metrics"],
            difficulty="medium"
        ),
        TestCase(
            id="demo_003",
            question="What are the advantages of Python for data science?",
            context={
                "expected_facts": [
                    "simplicity",
                    "readability",
                    "libraries",
                    "data science",
                    "ecosystem"
                ]
            },
            tags=["python", "data_science"],
            difficulty="easy"
        ),
        TestCase(
            id="demo_004",
            question="Explain quantum computing applications in cryptography",
            expected_response="This is a complex topic requiring specific domain knowledge.",
            tags=["quantum", "cryptography", "advanced"],
            difficulty="hard"
        )
    ]

    print("Running evaluation on demo agent...")
    print(f"Test cases: {len(test_cases)}")
    print()

    # Run evaluation
    results = evaluator.evaluate_agent(
        agent_function=demo_simple_agent,
        test_cases=test_cases,
        agent_name="demo_agent"
    )

    # Display results
    print("\nEVALUATION RESULTS:")
    print("-" * 40)

    successful_results = [r for r in results if r.success]
    failed_results = [r for r in results if not r.success]

    print(f"Total tests: {len(results)}")
    print(f"Successful: {len(successful_results)}")
    print(f"Failed: {len(failed_results)}")
    print()

    if successful_results:
        # Calculate averages
        avg_accuracy = sum(r.accuracy_score or 0 for r in successful_results) / len(successful_results)
        avg_relevance = sum(r.relevance_score or 0 for r in successful_results) / len(successful_results)
        avg_clarity = sum(r.clarity_score or 0 for r in successful_results) / len(successful_results)
        avg_overall = sum(r.overall_score or 0 for r in successful_results) / len(successful_results)
        avg_time = sum(r.execution_time_ms or 0 for r in successful_results) / len(successful_results)

        print("AVERAGE SCORES:")
        print(f"  Accuracy:  {avg_accuracy:.3f}")
        print(f"  Relevance: {avg_relevance:.3f}")
        print(f"  Clarity:   {avg_clarity:.3f}")
        print(f"  Overall:   {avg_overall:.3f}")
        print(f"  Avg Time:  {avg_time:.1f} ms")
        print()

        # Show individual test results
        print("INDIVIDUAL TEST RESULTS:")
        for result in successful_results:
            print(f"\n  {result.test_case_id}: {result.question[:50]}...")
            print(f"    Accuracy: {result.accuracy_score:.3f}")
            print(f"    Relevance: {result.relevance_score:.3f}")
            print(f"    Clarity: {result.clarity_score:.3f}")
            print(f"    Overall: {result.overall_score:.3f}")
            print(f"    Time: {result.execution_time_ms:.1f} ms")

    if failed_results:
        print("\nFAILED TESTS:")
        for result in failed_results:
            print(f"  {result.test_case_id}: {result.error_message}")

    # Generate comprehensive report
    report = evaluator.generate_report(results, "Demo Agent")
    print(f"\n{report}")

    print("\n" + "=" * 80)
    print("Demo completed! Check evaluation_results/ for detailed JSON output.")
    print("=" * 80)


def demonstrate_metrics():
    """Demonstrate individual metrics on sample responses."""

    print("\n" + "=" * 80)
    print("INDIVIDUAL METRICS DEMONSTRATION")
    print("=" * 80)

    # Sample question and responses
    question = "What is machine learning?"
    good_response = "Machine learning is a subset of artificial intelligence that enables computers to learn patterns from data and make predictions without being explicitly programmed for every task."
    poor_response = "ML is good."

    # Test each metric
    accuracy_metric = AccuracyMetric()
    relevance_metric = RelevanceMetric()
    clarity_metric = ClarityMetric()

    print(f"\nQuestion: {question}")
    print(f"\nGood response: {good_response}")
    print(f"Poor response: {poor_response}")

    for response_type, response in [("Good", good_response), ("Poor", poor_response)]:
        print(f"\n{response_type.upper()} RESPONSE METRICS:")

        acc_result = accuracy_metric.calculate(question, response)
        rel_result = relevance_metric.calculate(question, response)
        cla_result = clarity_metric.calculate(question, response)

        print(f"  Accuracy:  {acc_result.score:.3f}")
        print(f"  Relevance: {rel_result.score:.3f}")
        print(f"  Clarity:   {cla_result.score:.3f}")

        # Show some metric details
        if acc_result.details:
            print(f"    Accuracy details: {acc_result.details}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Demonstrate evaluation framework")
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Only demonstrate individual metrics"
    )

    args = parser.parse_args()

    if args.metrics_only:
        demonstrate_metrics()
    else:
        run_demo()
        demonstrate_metrics()
