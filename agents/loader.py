"""Dynamic agent loader for discovering and loading subagents.

This module provides utilities to automatically discover and load all agents
from the agents directory that implement the BaseAgent interface.
"""

import importlib
import inspect
from pathlib import Path
from typing import List, Dict, Type, Optional, Any

from agents.base import BaseAgent, AgentMetadata


class AgentLoader:
    """Dynamically load agents from the agents directory."""

    def __init__(self, agents_dir: Optional[str] = None):
        """
        Initialize the agent loader.

        Args:
            agents_dir: Path to the agents directory. If None, uses the directory
                       where this module is located.
        """
        if agents_dir is None:
            # Use the directory where this module is located
            agents_dir = Path(__file__).parent
        self.agents_dir = Path(agents_dir)
        self._agent_classes: Dict[str, Type[BaseAgent]] = {}
        self._discover_agents()

    def _discover_agents(self) -> None:
        """Discover all agent classes in the agents directory."""
        if not self.agents_dir.exists():
            print(f"Warning: Agents directory {self.agents_dir} does not exist")
            return

        # Iterate through subdirectories in agents folder
        for item in self.agents_dir.iterdir():
            if item.is_dir() and not item.name.startswith('_'):
                self._load_agent_from_directory(item)

    def _load_agent_from_directory(self, directory: Path) -> None:
        """
        Load agent class from a directory.

        Args:
            directory: Path to the agent directory
        """
        agent_file = directory / "agent.py"
        if not agent_file.exists():
            return

        try:
            # Construct module path (e.g., agents.code_analysis.agent)
            module_path = f"agents.{directory.name}.agent"

            # Import the module
            module = importlib.import_module(module_path)

            # Find all classes that inherit from BaseAgent
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if (issubclass(obj, BaseAgent) and
                    obj is not BaseAgent and
                    obj.__module__ == module_path):

                    # Store the agent class
                    self._agent_classes[directory.name] = obj
                    print(f"Discovered agent: {name} from {module_path}")

        except ImportError as e:
            print(f"Warning: Could not load agent from {directory.name}: Missing dependency - {e}")
        except Exception as e:
            print(f"Warning: Could not load agent from {directory.name}: {e}")
            import traceback
            traceback.print_exc()

    def get_agent_classes(self) -> Dict[str, Type[BaseAgent]]:
        """
        Get all discovered agent classes.

        Returns:
            Dictionary mapping agent directory name to agent class
        """
        return self._agent_classes.copy()

    def get_agent_metadata(self) -> List[AgentMetadata]:
        """
        Get metadata for all discovered agents.

        Note: This creates temporary instances to access metadata.
        For production use, consider caching metadata separately.

        Returns:
            List of AgentMetadata objects
        """
        metadata_list = []
        for name, agent_class in self._agent_classes.items():
            try:
                # Create a temporary instance to get metadata
                # This is a simplified approach; in production you might want
                # to handle initialization parameters differently
                if hasattr(agent_class, 'metadata'):
                    # Try to access as class property or create minimal instance
                    try:
                        temp_instance = agent_class.__new__(agent_class)
                        metadata = temp_instance.metadata
                        metadata_list.append(metadata)
                    except:
                        # If can't create instance, skip
                        pass
            except Exception as e:
                print(f"Warning: Could not get metadata for {name}: {e}")

        return metadata_list

    def find_agents_for_topic(self, topic: str, agent_instances: Dict[str, BaseAgent]) -> List[str]:
        """
        Find which agents are relevant for a given topic.

        Args:
            topic: The topic to match against
            agent_instances: Dictionary of agent name to agent instance

        Returns:
            List of agent names that match the topic
        """
        matching_agents = []

        for name, agent in agent_instances.items():
            if agent.is_available() and agent.validate_topic(topic):
                matching_agents.append(name)

        return matching_agents

    def create_agent_instance(
        self,
        agent_name: str,
        init_params: Optional[Dict[str, Any]] = None
    ) -> Optional[BaseAgent]:
        """
        Create an instance of a specific agent.

        Args:
            agent_name: Name of the agent (directory name)
            init_params: Initialization parameters for the agent

        Returns:
            Agent instance or None if creation failed
        """
        if agent_name not in self._agent_classes:
            print(f"Agent {agent_name} not found")
            return None

        try:
            agent_class = self._agent_classes[agent_name]
            init_params = init_params or {}
            return agent_class(**init_params)
        except Exception as e:
            print(f"Error creating agent {agent_name}: {e}")
            return None


def get_all_agent_classes() -> Dict[str, Type[BaseAgent]]:
    """
    Convenience function to get all agent classes.

    Returns:
        Dictionary mapping agent name to agent class
    """
    loader = AgentLoader()
    return loader.get_agent_classes()


def discover_agents_for_topic(topic: str, agent_instances: Dict[str, BaseAgent]) -> List[str]:
    """
    Convenience function to find agents matching a topic.

    Args:
        topic: The topic to match
        agent_instances: Dictionary of agent instances

    Returns:
        List of matching agent names
    """
    loader = AgentLoader()
    return loader.find_agents_for_topic(topic, agent_instances)

