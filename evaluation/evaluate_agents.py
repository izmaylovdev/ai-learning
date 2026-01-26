"""Agent-specific evaluation scripts."""

import sys
import os
from datetime import datetime
import logging

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation import AgentEvaluator, AccuracyMetric, RelevanceMetric, ClarityMetric, create_benchmark_suite
from agents.code_analysis.agent import analyze_code
from agents.learning_program_rag.agent import ask_rag_agent
from linkedin_post_generation.agent import agent as linkedin_agent
from langchain_core.messages import HumanMessage


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def evaluate_rag_agent():
    """Evaluate the RAG agent."""
    logger.info("Starting RAG agent evaluation")

    # Create evaluator with metrics
    metrics = [
        AccuracyMetric(weight=1.0),
        RelevanceMetric(weight=1.0),
        ClarityMetric(weight=0.8)
    ]

    evaluator = AgentEvaluator(
        metrics=metrics,
        output_dir="evaluation_results"
    )

    # Create test cases
    test_cases = create_benchmark_suite("rag")

    # Wrapper function for the agent
    def rag_function(question: str) -> str:
        try:
            return ask_rag_agent.invoke({"question": question})
        except Exception as e:
            return f"Error: {str(e)}"

    # Run evaluation
    results = evaluator.evaluate_agent(
        agent_function=rag_function,
        test_cases=test_cases,
        agent_name="rag_agent"
    )

    # Generate and print report
    report = evaluator.generate_report(results, "RAG Agent")
    print("\n" + "="*80)
    print("RAG AGENT EVALUATION REPORT")
    print("="*80)
    print(report)

    return results


def evaluate_code_analysis_agent():
    """Evaluate the code analysis agent."""
    logger.info("Starting code analysis agent evaluation")

    # Create evaluator with metrics
    metrics = [
        AccuracyMetric(weight=1.0),
        RelevanceMetric(weight=1.0),
        ClarityMetric(weight=0.9)
    ]

    evaluator = AgentEvaluator(
        metrics=metrics,
        output_dir="evaluation_results"
    )

    # Create test cases
    test_cases = create_benchmark_suite("code_analysis")

    # Wrapper function for the agent
    def code_analysis_function(question: str) -> str:
        try:
            return analyze_code.invoke({"question": question})
        except Exception as e:
            return f"Error: {str(e)}"

    # Run evaluation
    results = evaluator.evaluate_agent(
        agent_function=code_analysis_function,
        test_cases=test_cases,
        agent_name="code_analysis_agent"
    )

    # Generate and print report
    report = evaluator.generate_report(results, "Code Analysis Agent")
    print("\n" + "="*80)
    print("CODE ANALYSIS AGENT EVALUATION REPORT")
    print("="*80)
    print(report)

    return results


def evaluate_linkedin_agent():
    """Evaluate the LinkedIn post generation agent."""
    logger.info("Starting LinkedIn agent evaluation")

    # Create evaluator with metrics (clarity is most important for LinkedIn posts)
    metrics = [
        ClarityMetric(weight=1.0),
        RelevanceMetric(weight=0.9),
        AccuracyMetric(weight=0.7)
    ]

    evaluator = AgentEvaluator(
        metrics=metrics,
        output_dir="evaluation_results"
    )

    # Create test cases
    test_cases = create_benchmark_suite("linkedin_post")

    # Wrapper function for the agent
    def linkedin_function(question: str) -> str:
        try:
            response = linkedin_agent.invoke({"messages": [HumanMessage(question)]})
            if isinstance(response, dict):
                return response.get("output", str(response))
            return str(response)
        except Exception as e:
            return f"Error: {str(e)}"

    # Run evaluation
    results = evaluator.evaluate_agent(
        agent_function=linkedin_function,
        test_cases=test_cases,
        agent_name="linkedin_agent"
    )

    # Generate and print report
    report = evaluator.generate_report(results, "LinkedIn Post Generation Agent")
    print("\n" + "="*80)
    print("LINKEDIN AGENT EVALUATION REPORT")
    print("="*80)
    print(report)

    return results


def evaluate_all_agents():
    """Evaluate all agents and generate a comprehensive report."""
    logger.info("Starting comprehensive agent evaluation")

    all_results = {}

    try:
        logger.info("Evaluating RAG agent...")
        all_results["rag"] = evaluate_rag_agent()
    except Exception as e:
        logger.error(f"Failed to evaluate RAG agent: {e}")
        all_results["rag"] = []

    try:
        logger.info("Evaluating code analysis agent...")
        all_results["code_analysis"] = evaluate_code_analysis_agent()
    except Exception as e:
        logger.error(f"Failed to evaluate code analysis agent: {e}")
        all_results["code_analysis"] = []

    try:
        logger.info("Evaluating LinkedIn agent...")
        all_results["linkedin"] = evaluate_linkedin_agent()
    except Exception as e:
        logger.error(f"Failed to evaluate LinkedIn agent: {e}")
        all_results["linkedin"] = []

    # Generate comprehensive report
    print("\n" + "="*80)
    print("COMPREHENSIVE AGENT EVALUATION SUMMARY")
    print("="*80)
    print(f"Evaluation completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    for agent_name, results in all_results.items():
        if results:
            successful_results = [r for r in results if r.success]
            if successful_results:
                avg_score = sum(r.overall_score or 0 for r in successful_results) / len(successful_results)
                avg_accuracy = sum(r.accuracy_score or 0 for r in successful_results) / len(successful_results)
                avg_relevance = sum(r.relevance_score or 0 for r in successful_results) / len(successful_results)
                avg_clarity = sum(r.clarity_score or 0 for r in successful_results) / len(successful_results)

                print(f"{agent_name.upper()} AGENT:")
                print(f"  Total tests: {len(results)}")
                print(f"  Successful: {len(successful_results)}")
                print(f"  Overall score: {avg_score:.3f}")
                print(f"  Accuracy: {avg_accuracy:.3f}")
                print(f"  Relevance: {avg_relevance:.3f}")
                print(f"  Clarity: {avg_clarity:.3f}")
            else:
                print(f"{agent_name.upper()} AGENT: All tests failed")
        else:
            print(f"{agent_name.upper()} AGENT: Evaluation failed")
        print()

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate AI agents")
    parser.add_argument(
        "--agent",
        choices=["rag", "code_analysis", "linkedin", "all"],
        default="all",
        help="Which agent to evaluate"
    )

    args = parser.parse_args()

    if args.agent == "rag":
        evaluate_rag_agent()
    elif args.agent == "code_analysis":
        evaluate_code_analysis_agent()
    elif args.agent == "linkedin":
        evaluate_linkedin_agent()
    else:
        evaluate_all_agents()
