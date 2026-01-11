"""LinkedIn Post Generation Agent.
This agent orchestrates dynamically loaded subagents to generate 
professional LinkedIn posts about technical topics.
"""
from typing import Optional, Dict, Any, List
import sys
from pathlib import Path
# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from text_generation.generator import GeneratorInterface
from text_generation.get_generator import get_generator
from agents.loader import AgentLoader
from agents.base import BaseAgent
import config
class LinkedInPostAgent:
    """Main agent for generating LinkedIn posts using dynamically loaded subagents."""
    def __init__(
        self,
        generator: Optional[GeneratorInterface] = None,
        repo_root: str = ".",
        generator_backend: Optional[str] = None,
        generator_kwargs: Optional[Dict] = None,
    ):
        """
        Initialize LinkedIn Post Agent.
        Args:
            generator: Text generator for creating posts
            repo_root: Root directory of the repository for code analysis
            generator_backend: Override generator backend from config
            generator_kwargs: Additional kwargs for generator
        """
        self.generator_backend = generator_backend or config.GENERATOR_BACKEND
        self.generator_kwargs = generator_kwargs or {}
        self.repo_root = repo_root
        # Initialize main generator
        self.generator = generator or get_generator(self.generator_backend, **self.generator_kwargs)
        # Dynamically discover and load agents
        self.agent_loader = AgentLoader()
        self._agent_instances: Dict[str, BaseAgent] = {}
        self._agent_classes = self.agent_loader.get_agent_classes()
        print(f"\nDiscovered {len(self._agent_classes)} subagent(s):")
        for name in self._agent_classes.keys():
            print(f"  - {name}")
    def _get_or_create_agent(self, agent_name: str) -> Optional[BaseAgent]:
        """
        Lazy initialization of agents.
        Args:
            agent_name: Name of the agent to initialize
        Returns:
            Agent instance or None if initialization failed
        """
        if agent_name in self._agent_instances:
            return self._agent_instances[agent_name]
        if agent_name not in self._agent_classes:
            print(f"Agent {agent_name} not found")
            return None
        try:
            # Initialize agent with appropriate parameters
            if agent_name == "code_analysis":
                from langchain_google_genai import ChatGoogleGenerativeAI
                llm = ChatGoogleGenerativeAI(
                    model=config.GEMINI_MODEL,
                    google_api_key=config.GOOGLE_API_KEY,
                    temperature=config.GEMINI_TEMPERATURE,
                )
                agent = self.agent_loader.create_agent_instance(
                    agent_name,
                    init_params={"root_path": self.repo_root, "llm": llm}
                )
            elif agent_name == "learning_program_rag":
                agent = self.agent_loader.create_agent_instance(
                    agent_name,
                    init_params={
                        "generator_backend": self.generator_backend,
                        "generator_kwargs": self.generator_kwargs,
                    }
                )
            else:
                # Generic initialization for unknown agents
                agent = self.agent_loader.create_agent_instance(agent_name)
            if agent:
                self._agent_instances[agent_name] = agent
                print(f"Initialized {agent.metadata.name}")
            return agent
        except Exception as e:
            print(f"Warning: Could not initialize {agent_name} agent: {e}")
            return None
    def _find_relevant_agents(self, topic: str) -> List[str]:
        """
        Automatically determine which agents are relevant for a topic.
        Args:
            topic: The topic to analyze
        Returns:
            List of agent names that are relevant
        """
        relevant_agents = []
        for agent_name, agent_class in self._agent_classes.items():
            try:
                # Create a minimal instance just to check metadata
                temp_instance = agent_class.__new__(agent_class)
                if temp_instance.metadata.matches_topic(topic):
                    relevant_agents.append(agent_name)
            except:
                # If we can't check metadata, skip
                pass
        return relevant_agents
    def generate_post(
        self,
        topic: str,
        style: str = "professional",
        tone: str = "informative",
        length: str = "medium",
        include_hashtags: bool = True,
        auto_select_agents: bool = True,
        enabled_agents: Optional[List[str]] = None,
        disabled_agents: Optional[List[str]] = None,
        agent_params: Optional[Dict[str, Dict]] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate a LinkedIn post on the given topic.
        Args:
            topic: The main topic or theme for the post
            style: Writing style (professional, casual, technical, storytelling)
            tone: Tone of the post (informative, inspirational, thought-provoking)
            length: Post length (short, medium, long)
            include_hashtags: Whether to include relevant hashtags
            auto_select_agents: Automatically select agents based on topic keywords
            enabled_agents: List of specific agents to use (overrides auto-selection)
            disabled_agents: List of agents to exclude
            agent_params: Dictionary of agent-specific parameters
            verbose: Whether to print progress messages
        Returns:
            Dictionary containing the post and metadata
        """
        if verbose:
            print(f"\n{'='*80}")
            print(f"Generating LinkedIn Post")
            print(f"Topic: {topic}")
            print(f"Style: {style} | Tone: {tone} | Length: {length}")
            print(f"{'='*80}\n")
        # Determine which agents to use
        if enabled_agents:
            agents_to_use = enabled_agents
        elif auto_select_agents:
            agents_to_use = self._find_relevant_agents(topic)
        else:
            agents_to_use = list(self._agent_classes.keys())
        # Remove disabled agents
        if disabled_agents:
            agents_to_use = [a for a in agents_to_use if a not in disabled_agents]
        if verbose:
            print(f"Selected agents: {agents_to_use if agents_to_use else 'None'}\n")
        # Gather insights from all relevant agents
        context_parts = []
        agents_used = {}
        agent_params = agent_params or {}
        for agent_name in agents_to_use:
            agent = self._get_or_create_agent(agent_name)
            if agent and agent.is_available():
                if verbose:
                    print(f"Gathering insights from {agent.metadata.name}...")
                try:
                    # Get agent-specific parameters
                    params = agent_params.get(agent_name, {})
                    # Gather insights
                    insights = agent.gather_insights(topic, **params)
                    if insights and insights.strip():
                        context_parts.append(f"{agent.metadata.name} Insights:\n{insights}")
                        agents_used[agent_name] = True
                except Exception as e:
                    if verbose:
                        print(f"Error gathering insights from {agent_name}: {e}")
                    agents_used[agent_name] = False
        # Combine all context
        context = "\n\n---\n\n".join(context_parts) if context_parts else "No additional context gathered."
        # Generate the LinkedIn post
        if verbose:
            print("\nGenerating LinkedIn post...")
        post = self._generate_post_content(
            topic=topic,
            context=context,
            style=style,
            tone=tone,
            length=length,
            include_hashtags=include_hashtags,
        )
        # Prepare result
        result = {
            "post": post,
            "topic": topic,
            "style": style,
            "tone": tone,
            "length": length,
            "agents_used": agents_used,
            "context_gathered": bool(context_parts),
        }
        if verbose:
            print(f"\n{'='*80}")
            print("Generated LinkedIn Post:")
            print(f"{'='*80}\n")
            print(post)
            print(f"\n{'='*80}\n")
        return result
    def list_available_agents(self) -> Dict[str, Dict[str, Any]]:
        """
        List all available agents and their metadata.
        Returns:
            Dictionary mapping agent name to metadata information
        """
        agents_info = {}
        for agent_name, agent_class in self._agent_classes.items():
            try:
                temp_instance = agent_class.__new__(agent_class)
                metadata = temp_instance.metadata
                agents_info[agent_name] = {
                    "name": metadata.name,
                    "description": metadata.description,
                    "keywords": metadata.keywords,
                    "version": metadata.version,
                    "author": metadata.author,
                }
            except Exception as e:
                agents_info[agent_name] = {
                    "name": agent_name,
                    "description": "Metadata unavailable",
                    "error": str(e)
                }
        return agents_info
    def _generate_post_content(
        self,
        topic: str,
        context: str,
        style: str,
        tone: str,
        length: str,
        include_hashtags: bool,
    ) -> str:
        """
        Generate the actual post content using the generator.
        Args:
            topic: Main topic
            context: Gathered context from subagents
            style: Writing style
            tone: Tone of the post
            length: Desired length
            include_hashtags: Whether to include hashtags
        Returns:
            Generated LinkedIn post text
        """
        # Define length guidelines
        length_guide = {
            "short": "1-2 short paragraphs (100-150 words)",
            "medium": "2-3 paragraphs (200-300 words)",
            "long": "3-5 paragraphs (400-600 words)",
        }
        # Build the prompt
        prompt = f"""You are an expert LinkedIn content creator. Generate a compelling LinkedIn post based on the following requirements:
Topic: {topic}
Style: {style}
Tone: {tone}
Length: {length_guide.get(length, length)}
Context and Research:
{context}
Guidelines:
1. Write in a {style} style with a {tone} tone
2. Make it engaging and relevant for LinkedIn's professional audience
3. Use clear, concise language
4. Include a hook at the beginning to grab attention
5. Add value - share insights, lessons learned, or actionable takeaways
6. Use short paragraphs for better readability
7. {"Include 3-5 relevant hashtags at the end" if include_hashtags else "Do not include hashtags"}
8. Be authentic and relatable
Post:"""
        try:
            return self.generator.generate(prompt)
        except Exception as e:
            return f"Error generating post: {str(e)}"
