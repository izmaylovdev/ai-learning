"""Metrics for evaluating agent performance."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional
import re
import logging
from difflib import SequenceMatcher


@dataclass
class MetricResult:
    """Result from a metric calculation."""
    score: float  # 0.0 to 1.0
    details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": self.score,
            "details": self.details
        }


class EvaluationMetric(ABC):
    """Base class for evaluation metrics."""

    def __init__(self, name: str, weight: float = 1.0):
        self.name = name
        self.weight = weight
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def calculate(
        self,
        question: str,
        agent_response: str,
        expected_response: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> MetricResult:
        """Calculate the metric score."""
        pass


class AccuracyMetric(EvaluationMetric):
    """Measures factual accuracy of the response."""

    def __init__(self, weight: float = 1.0):
        super().__init__("accuracy", weight)

    def calculate(
        self,
        question: str,
        agent_response: str,
        expected_response: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> MetricResult:
        """Calculate accuracy score."""
        details = {}

        if not agent_response or not agent_response.strip():
            return MetricResult(score=0.0, details={"error": "Empty response"})

        score = 0.0

        # If we have an expected response, calculate similarity
        if expected_response:
            similarity = self._calculate_similarity(agent_response, expected_response)
            score += similarity * 0.6  # 60% weight for similarity
            details["similarity_score"] = similarity
        else:
            # Without expected response, use heuristics
            score += 0.6  # Give benefit of doubt

        # Check for presence of factual content
        factual_score = self._assess_factual_content(agent_response, context)
        score += factual_score * 0.3  # 30% weight
        details["factual_content_score"] = factual_score

        # Check for hedging/uncertainty (good for accuracy)
        uncertainty_score = self._assess_uncertainty_handling(agent_response)
        score += uncertainty_score * 0.1  # 10% weight
        details["uncertainty_handling_score"] = uncertainty_score

        return MetricResult(score=min(score, 1.0), details=details)

    def _calculate_similarity(self, response: str, expected: str) -> float:
        """Calculate semantic similarity between responses."""
        matcher = SequenceMatcher(None, response.lower(), expected.lower())
        return matcher.ratio()

    def _assess_factual_content(self, response: str, context: Optional[Dict[str, Any]]) -> float:
        """Assess the factual content quality."""
        if context and "expected_facts" in context:
            expected_facts = context["expected_facts"]
            found_facts = 0
            for fact in expected_facts:
                if fact.lower() in response.lower():
                    found_facts += 1
            return found_facts / len(expected_facts) if expected_facts else 1.0

        # General heuristics for factual content
        has_specifics = bool(re.search(r'\b\d+\b', response)) or \
                       bool(re.search(r'\b[A-Z][a-z]+\b', response))

        return 0.8 if has_specifics else 0.5

    def _assess_uncertainty_handling(self, response: str) -> float:
        """Assess how well the agent handles uncertainty."""
        uncertainty_phrases = [
            "i don't know", "not sure", "unclear", "uncertain",
            "may be", "might be", "possibly", "probably",
            "i don't have enough information", "based on available information"
        ]

        response_lower = response.lower()

        if any(phrase in response_lower for phrase in uncertainty_phrases):
            return 1.0

        if len(response.split()) > 10:  # Detailed response
            return 0.8

        return 0.6  # Default moderate score


class RelevanceMetric(EvaluationMetric):
    """Measures how relevant the response is to the question."""

    def __init__(self, weight: float = 1.0):
        super().__init__("relevance", weight)

    def calculate(
        self,
        question: str,
        agent_response: str,
        expected_response: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> MetricResult:
        """Calculate relevance score."""
        details = {}

        if not agent_response or not agent_response.strip():
            return MetricResult(score=0.0, details={"error": "Empty response"})

        score = 0.0

        # Check keyword overlap between question and response
        keyword_score = self._calculate_keyword_overlap(question, agent_response)
        score += keyword_score * 0.5  # 50% weight
        details["keyword_overlap_score"] = keyword_score

        # Check if response directly addresses the question type
        question_type_score = self._assess_question_type_alignment(question, agent_response)
        score += question_type_score * 0.3  # 30% weight
        details["question_type_score"] = question_type_score

        # Check for off-topic content
        focus_score = self._assess_response_focus(question, agent_response)
        score += focus_score * 0.2  # 20% weight
        details["focus_score"] = focus_score

        return MetricResult(score=min(score, 1.0), details=details)

    def _calculate_keyword_overlap(self, question: str, response: str) -> float:
        """Calculate overlap of important keywords."""
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
                     "for", "of", "with", "by", "is", "are", "was", "were", "be",
                     "been", "have", "has", "had", "do", "does", "did", "will",
                     "would", "could", "should"}

        question_words = set(word.lower().strip(".,?!") for word in question.split()
                           if len(word) > 2 and word.lower() not in stop_words)
        response_words = set(word.lower().strip(".,?!") for word in response.split()
                           if len(word) > 2 and word.lower() not in stop_words)

        if not question_words:
            return 1.0

        overlap = len(question_words.intersection(response_words))
        return overlap / len(question_words)

    def _assess_question_type_alignment(self, question: str, response: str) -> float:
        """Check if response type matches question type."""
        question_lower = question.lower()
        response_lower = response.lower()

        if question_lower.startswith("what"):
            if any(word in response_lower for word in ["is", "are", "means", "refers to"]):
                return 1.0
            return 0.7
        elif question_lower.startswith("how"):
            if any(word in response_lower for word in ["step", "process", "method", "by"]):
                return 1.0
            return 0.7
        elif question_lower.startswith("why"):
            if any(word in response_lower for word in ["because", "since", "due to", "reason"]):
                return 1.0
            return 0.7
        elif question_lower.startswith(("when", "where")):
            return 0.8

        return 0.8

    def _assess_response_focus(self, question: str, response: str) -> float:
        """Assess if response stays focused on the question topic."""
        response_length = len(response.split())

        if response_length < 10:
            return 0.6  # Too brief might miss important details
        elif response_length > 200:
            # Very long responses might be unfocused
            question_words = set(word.lower() for word in question.split() if len(word) > 3)
            response_parts = [response[i:i+100] for i in range(0, len(response), 100)]

            relevant_parts = sum(1 for part in response_parts
                               if any(word in part.lower() for word in question_words))

            return relevant_parts / len(response_parts) if response_parts else 0.5
        else:
            return 1.0  # Good length


class ClarityMetric(EvaluationMetric):
    """Measures how clear and understandable the response is."""

    def __init__(self, weight: float = 1.0):
        super().__init__("clarity", weight)

    def calculate(
        self,
        question: str,
        agent_response: str,
        expected_response: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> MetricResult:
        """Calculate clarity score."""
        details = {}

        if not agent_response or not agent_response.strip():
            return MetricResult(score=0.0, details={"error": "Empty response"})

        score = 0.0

        # Check readability
        readability_score = self._assess_readability(agent_response)
        score += readability_score * 0.4  # 40% weight
        details["readability_score"] = readability_score

        # Check structure and organization
        structure_score = self._assess_structure(agent_response)
        score += structure_score * 0.3  # 30% weight
        details["structure_score"] = structure_score

        # Check for jargon and complexity
        complexity_score = self._assess_complexity(agent_response)
        score += complexity_score * 0.3  # 30% weight
        details["complexity_score"] = complexity_score

        return MetricResult(score=min(score, 1.0), details=details)

    def _assess_readability(self, response: str) -> float:
        """Assess readability using simple metrics."""
        sentences = re.split(r'[.!?]+', response)
        words = response.split()

        if not sentences or not words:
            return 0.0

        # Average sentence length (aim for 15-20 words)
        avg_sentence_length = len(words) / len([s for s in sentences if s.strip()])

        if 10 <= avg_sentence_length <= 25:
            return 1.0
        elif avg_sentence_length < 5 or avg_sentence_length > 40:
            return 0.3
        else:
            return 0.7

    def _assess_structure(self, response: str) -> float:
        """Assess organization and structure."""
        score = 0.0

        # Check for logical connectors
        connectors = ["first", "second", "then", "next", "finally", "however",
                     "therefore", "because", "since", "moreover", "furthermore"]
        connector_count = sum(1 for connector in connectors if connector in response.lower())

        if connector_count > 0:
            score += 0.5

        # Check for bullet points or numbering
        if re.search(r'^\s*[-*•]\s+', response, re.MULTILINE) or \
           re.search(r'^\s*\d+\.?\s+', response, re.MULTILINE):
            score += 0.5

        return min(score, 1.0)

    def _assess_complexity(self, response: str) -> float:
        """Assess appropriate complexity level."""
        words = response.split()

        if not words:
            return 0.0

        # Count complex words (long words as proxy)
        complex_words = sum(1 for word in words if len(word) > 8)
        complexity_ratio = complex_words / len(words)

        # Moderate complexity is good (10-30%)
        if 0.1 <= complexity_ratio <= 0.3:
            return 1.0
        elif complexity_ratio < 0.05:
            return 0.7  # Too simple might lack detail
        elif complexity_ratio > 0.5:
            return 0.3  # Too complex
        else:
            return 0.6
