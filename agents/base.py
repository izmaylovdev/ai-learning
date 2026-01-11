"""Base interface for all agents in the system.

This module defines the standard interface that all subagents must implement
to be dynamically loaded by the main LinkedIn post generation agent.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class AgentMetadata:
    """Metadata describing an agent's capabilities and purpose."""

    name: str
    description: str
    keywords: List[str]
    version: str = "1.0.0"
    author: str = ""

    def matches_topic(self, topic: str) -> bool:
        """
        Check if this agent is relevant for the given topic.

        Args:
            topic: The topic string to check against

        Returns:
            True if any keyword matches the topic (case-insensitive)
        """
        topic_lower = topic.lower()
        return any(keyword.lower() in topic_lower for keyword in self.keywords)


class BaseAgent(ABC):
    """Base class for all agents in the system."""

    @property
    @abstractmethod
    def metadata(self) -> AgentMetadata:
        """
        Return metadata describing this agent.

        Returns:
            AgentMetadata instance with agent information
        """
        pass

    @abstractmethod
    def gather_insights(self, topic: str, **kwargs) -> str:
        """
        Gather insights relevant to the given topic.

        Args:
            topic: The topic to gather insights about
            **kwargs: Additional parameters specific to the agent

        Returns:
            String containing insights gathered by the agent
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the agent is available and properly initialized.

        Returns:
            True if agent is ready to use, False otherwise
        """
        pass

    def initialize(self, **kwargs) -> None:
        """
        Initialize the agent with any required setup.

        This method can be overridden by subclasses that need initialization.

        Args:
            **kwargs: Initialization parameters specific to the agent
        """
        pass

    def validate_topic(self, topic: str) -> bool:
        """
        Validate if this agent can handle the given topic.

        Args:
            topic: The topic to validate

        Returns:
            True if agent can handle this topic
        """
        return self.metadata.matches_topic(topic)

