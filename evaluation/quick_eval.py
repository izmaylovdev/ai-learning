"""Simple evaluation runner that safely evaluates available agents."""

import sys
import os
import logging
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def safe_import_and_evaluate():
    """Safely import evaluation components and test available agents."""

    try:
        from evaluation import AgentEvaluator, AccuracyMetric, RelevanceMetric, ClarityMetric, create_benchmark_suite
        from evaluation.evaluator import TestCase
        logger.info("✅ Evaluation framework imported successfully")
    except Exception as e:
        logger.error(f"❌ Failed to import evaluation framework: {e}")
        return

    # Test with a simple demo function first
    def simple_test_agent(question: str) -> str:
        """Simple test agent to verify evaluation works."""
        return f"This is a test response to: {question}"

    logger.info("Testing evaluation framework with simple agent...")

    try:
        # Create basic evaluator
        evaluator = AgentEvaluator([
            AccuracyMetric(),
            RelevanceMetric(),
            ClarityMetric()
        ], output_dir="evaluation/results")

        # Simple test case
        test_cases = [
            TestCase(
                id="simple_test",
                question="What is AI?",
                expected_response="AI is artificial intelligence",
                tags=["test"]
            )
        ]

        # Run evaluation
        results = evaluator.evaluate_agent(
            agent_function=simple_test_agent,
            test_cases=test_cases,
            agent_name="simple_test_agent"
        )

        if results and len(results) > 0:
            logger.info(f"✅ Evaluation framework test successful - Score: {results[0].overall_score:.3f}")
        else:
            logger.error("❌ Evaluation returned no results")

    except Exception as e:
        logger.error(f"❌ Evaluation framework test failed: {e}")
        return

    # Now try to evaluate real agents if available
    logger.info("Checking available agents for evaluation...")

    available_agents = {}

    # Check RAG agent
    try:
        from agents.learning_program_rag.agent import ask_rag_agent

        def rag_wrapper(question: str) -> str:
            try:
                return ask_rag_agent.invoke({"question": question})
            except Exception as e:
                return f"Error: {str(e)}"

        available_agents["rag_agent"] = rag_wrapper
        logger.info("✅ RAG agent available")

    except Exception as e:
        logger.warning(f"⚠️ RAG agent not available: {e}")

    # Check code analysis agent
    try:
        from agents.code_analysis.agent import analyze_code

        def code_wrapper(question: str) -> str:
            try:
                return analyze_code.invoke({"question": question})
            except Exception as e:
                return f"Error: {str(e)}"

        available_agents["code_analysis_agent"] = code_wrapper
        logger.info("✅ Code analysis agent available")

    except Exception as e:
        logger.warning(f"⚠️ Code analysis agent not available: {e}")

    # Check LinkedIn agent
    try:
        from linkedin_post_generation.agent import agent as linkedin_agent
        from langchain_core.messages import HumanMessage

        def linkedin_wrapper(question: str) -> str:
            try:
                response = linkedin_agent.invoke({"messages": [HumanMessage(question)]})
                if isinstance(response, dict):
                    return response.get("output", str(response))
                return str(response)
            except Exception as e:
                return f"Error: {str(e)}"

        available_agents["linkedin_agent"] = linkedin_wrapper
        logger.info("✅ LinkedIn agent available")

    except Exception as e:
        logger.warning(f"⚠️ LinkedIn agent not available: {e}")

    # Evaluate available agents
    if not available_agents:
        logger.warning("⚠️ No agents available for evaluation")
        return

    logger.info(f"Evaluating {len(available_agents)} available agents...")

    # Create limited test cases for quick evaluation
    quick_tests = [
        TestCase(
            id="quick_001",
            question="What is artificial intelligence?",
            tags=["general", "ai"],
            difficulty="easy"
        ),
        TestCase(
            id="quick_002",
            question="How does machine learning work?",
            tags=["general", "ml"],
            difficulty="medium"
        )
    ]

    all_results = {}

    for agent_name, agent_function in available_agents.items():
        try:
            logger.info(f"Evaluating {agent_name}...")

            # Use appropriate benchmark if available, otherwise use quick tests
            if agent_name == "rag_agent":
                test_cases = create_benchmark_suite("rag")[:2]  # Limit to first 2 for quick test
            elif agent_name == "code_analysis_agent":
                test_cases = create_benchmark_suite("code_analysis")[:2]
            elif agent_name == "linkedin_agent":
                test_cases = create_benchmark_suite("linkedin_post")[:2]
            else:
                test_cases = quick_tests

            results = evaluator.evaluate_agent(
                agent_function=agent_function,
                test_cases=test_cases,
                agent_name=agent_name
            )

            all_results[agent_name] = results

            # Quick summary
            successful_results = [r for r in results if r.success]
            if successful_results:
                avg_score = sum(r.overall_score or 0 for r in successful_results) / len(successful_results)
                logger.info(f"✅ {agent_name}: {len(successful_results)}/{len(results)} tests passed, avg score: {avg_score:.3f}")
            else:
                logger.warning(f"⚠️ {agent_name}: All tests failed")

        except Exception as e:
            logger.error(f"❌ Failed to evaluate {agent_name}: {e}")

    # Generate summary report
    print("\n" + "="*80)
    print("AGENT EVALUATION SUMMARY")
    print("="*80)
    print(f"Evaluation completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Agents evaluated: {len(all_results)}")
    print()

    for agent_name, results in all_results.items():
        successful_results = [r for r in results if r.success]
        if successful_results:
            avg_accuracy = sum(r.accuracy_score or 0 for r in successful_results) / len(successful_results)
            avg_relevance = sum(r.relevance_score or 0 for r in successful_results) / len(successful_results)
            avg_clarity = sum(r.clarity_score or 0 for r in successful_results) / len(successful_results)
            avg_overall = sum(r.overall_score or 0 for r in successful_results) / len(successful_results)

            print(f"{agent_name.upper()}:")
            print(f"  Tests: {len(successful_results)}/{len(results)} passed")
            print(f"  Accuracy:  {avg_accuracy:.3f}")
            print(f"  Relevance: {avg_relevance:.3f}")
            print(f"  Clarity:   {avg_clarity:.3f}")
            print(f"  Overall:   {avg_overall:.3f}")
        else:
            print(f"{agent_name.upper()}: All tests failed")
        print()

    print("✅ Evaluation complete! Check evaluation/results/ for detailed JSON files.")
    print("="*80)


if __name__ == "__main__":
    safe_import_and_evaluate()
