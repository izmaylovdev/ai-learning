"""Core evaluation classes and interfaces."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Callable
import json
import os
import logging

from .metrics import MetricResult, EvaluationMetric


@dataclass
class EvaluationResult:
    """Results from evaluating an agent."""

    agent_name: str
    test_case_id: str
    question: str
    agent_response: str
    expected_response: Optional[str] = None

    # Metric scores (0.0 to 1.0)
    accuracy_score: Optional[float] = None
    relevance_score: Optional[float] = None
    clarity_score: Optional[float] = None

    # Detailed metric results
    metric_results: Dict[str, MetricResult] = field(default_factory=dict)

    # Overall score
    overall_score: Optional[float] = None

    # Execution metadata
    execution_time_ms: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)

    # Success/failure status
    success: bool = True
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "agent_name": self.agent_name,
            "test_case_id": self.test_case_id,
            "question": self.question,
            "agent_response": self.agent_response,
            "expected_response": self.expected_response,
            "accuracy_score": self.accuracy_score,
            "relevance_score": self.relevance_score,
            "clarity_score": self.clarity_score,
            "overall_score": self.overall_score,
            "execution_time_ms": self.execution_time_ms,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context,
            "success": self.success,
            "error_message": self.error_message,
            "metric_results": {k: v.to_dict() for k, v in self.metric_results.items()}
        }


@dataclass
class TestCase:
    """A single test case for agent evaluation."""

    id: str
    question: str
    expected_response: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    difficulty: str = "medium"  # easy, medium, hard

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "question": self.question,
            "expected_response": self.expected_response,
            "context": self.context,
            "tags": self.tags,
            "difficulty": self.difficulty
        }


class AgentEvaluator:
    """Main evaluator class for testing agents."""

    def __init__(
        self,
        metrics: List[EvaluationMetric],
        output_dir: str = "evaluation_results"
    ):
        self.metrics = metrics
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

    def evaluate_agent(
        self,
        agent_function: Callable[[str], str],
        test_cases: List[TestCase],
        agent_name: str
    ) -> List[EvaluationResult]:
        """
        Evaluate an agent against a set of test cases.

        Args:
            agent_function: Function that takes a question string and returns response string
            test_cases: List of test cases to evaluate
            agent_name: Name of the agent being tested

        Returns:
            List of evaluation results
        """
        results = []

        self.logger.info(f"Starting evaluation of {agent_name} with {len(test_cases)} test cases")

        for test_case in test_cases:
            try:
                start_time = datetime.now()

                # Get agent response
                agent_response = agent_function(test_case.question)

                end_time = datetime.now()
                execution_time_ms = (end_time - start_time).total_seconds() * 1000

                # Create evaluation result
                result = EvaluationResult(
                    agent_name=agent_name,
                    test_case_id=test_case.id,
                    question=test_case.question,
                    agent_response=agent_response,
                    expected_response=test_case.expected_response,
                    execution_time_ms=execution_time_ms,
                    context=test_case.context
                )

                # Calculate metrics
                self._calculate_metrics(result, test_case)

                results.append(result)

            except Exception as e:
                self.logger.error(f"Error evaluating test case {test_case.id}: {str(e)}")
                result = EvaluationResult(
                    agent_name=agent_name,
                    test_case_id=test_case.id,
                    question=test_case.question,
                    agent_response="",
                    expected_response=test_case.expected_response,
                    success=False,
                    error_message=str(e),
                    context=test_case.context
                )
                results.append(result)

        # Save results
        self._save_results(results, agent_name)

        return results

    def _calculate_metrics(self, result: EvaluationResult, test_case: TestCase) -> None:
        """Calculate all metrics for a result."""
        metric_scores = {}

        for metric in self.metrics:
            try:
                metric_result = metric.calculate(
                    question=test_case.question,
                    agent_response=result.agent_response,
                    expected_response=test_case.expected_response,
                    context=test_case.context
                )

                result.metric_results[metric.name] = metric_result
                metric_scores[metric.name] = metric_result.score

                # Set specific score fields for common metrics
                if metric.name.lower() == "accuracy":
                    result.accuracy_score = metric_result.score
                elif metric.name.lower() == "relevance":
                    result.relevance_score = metric_result.score
                elif metric.name.lower() == "clarity":
                    result.clarity_score = metric_result.score

            except Exception as e:
                self.logger.error(f"Error calculating {metric.name}: {str(e)}")
                result.metric_results[metric.name] = MetricResult(
                    score=0.0,
                    details={"error": str(e)}
                )

        # Calculate overall score as average of all metrics
        if metric_scores:
            result.overall_score = sum(metric_scores.values()) / len(metric_scores)

    def _save_results(self, results: List[EvaluationResult], agent_name: str) -> None:
        """Save evaluation results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{agent_name}_evaluation_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)

        # Convert results to dictionaries
        results_data = {
            "agent_name": agent_name,
            "timestamp": timestamp,
            "total_tests": len(results),
            "successful_tests": sum(1 for r in results if r.success),
            "failed_tests": sum(1 for r in results if not r.success),
            "average_score": self._calculate_average_score(results),
            "results": [r.to_dict() for r in results]
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Results saved to {filepath}")

    def _calculate_average_score(self, results: List[EvaluationResult]) -> Optional[float]:
        """Calculate average overall score across all results."""
        successful_results = [r for r in results if r.success and r.overall_score is not None]
        if not successful_results:
            return None
        return sum(r.overall_score for r in successful_results) / len(successful_results)

    def generate_report(self, results: List[EvaluationResult], agent_name: str) -> str:
        """Generate a human-readable evaluation report."""
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]

        report_lines = [
            f"Evaluation Report for {agent_name}",
            "=" * 60,
            f"Total Test Cases: {len(results)}",
            f"Successful: {len(successful_results)}",
            f"Failed: {len(failed_results)}",
            ""
        ]

        if successful_results:
            # Calculate average scores
            avg_accuracy = self._calculate_average_metric(successful_results, "accuracy_score")
            avg_relevance = self._calculate_average_metric(successful_results, "relevance_score")
            avg_clarity = self._calculate_average_metric(successful_results, "clarity_score")
            avg_overall = self._calculate_average_score(successful_results)
            avg_time = sum(r.execution_time_ms or 0 for r in successful_results) / len(successful_results)

            report_lines.extend([
                "Average Scores:",
                f"  Accuracy:  {avg_accuracy:.3f}" if avg_accuracy is not None else "  Accuracy:  N/A",
                f"  Relevance: {avg_relevance:.3f}" if avg_relevance is not None else "  Relevance: N/A",
                f"  Clarity:   {avg_clarity:.3f}" if avg_clarity is not None else "  Clarity:   N/A",
                f"  Overall:   {avg_overall:.3f}" if avg_overall is not None else "  Overall:   N/A",
                f"  Avg Time:  {avg_time:.1f} ms",
                ""
            ])

        # Add details for failed tests
        if failed_results:
            report_lines.extend([
                "Failed Test Cases:",
                "-" * 30
            ])
            for result in failed_results:
                report_lines.append(f"  {result.test_case_id}: {result.error_message}")
            report_lines.append("")

        return "\n".join(report_lines)

    def _calculate_average_metric(self, results: List[EvaluationResult], metric_name: str) -> Optional[float]:
        """Calculate average for a specific metric."""
        values = [getattr(r, metric_name) for r in results if getattr(r, metric_name) is not None]
        if not values:
            return None
        return sum(values) / len(values)
