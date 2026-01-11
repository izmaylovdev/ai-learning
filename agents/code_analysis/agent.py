import os
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from langchain.agents import create_agent

from agents.base import BaseAgent, AgentMetadata
from agents.code_analysis.indexer import build_index, _tokenize, DescriptionAgent
from agents.code_analysis import tools


class CodebaseAgent(BaseAgent):
    """Agent for analyzing code repositories."""

    @property
    def metadata(self) -> AgentMetadata:
        """Return metadata describing this agent."""
        return AgentMetadata(
            name="Code Analysis Agent",
            description="Provide information about project codebase",
            keywords=[
                "code", "repository", "repo", "project", "implementation",
                "architecture", "structure", "codebase", "develop", "development",
                "programming", "software", "source code", "file", "function",
                "class", "module", "package", "design", "pattern"
            ],
            version="1.0.0",
            author="AI Learning System"
        )

    def is_available(self) -> bool:
        """Check if the agent is available and properly initialized."""
        return self.agent is not None and self.llm is not None

    def gather_insights(self, topic: str, **kwargs) -> str:
        """
        Gather insights from code analysis.

        Args:
            topic: The topic to analyze
            **kwargs: Additional parameters (max_files, use_cache)

        Returns:
            String with code insights
        """
        if not self.is_available():
            return "Code analysis agent not available."

        try:
            max_files = kwargs.get('max_files', 100)
            use_cache = kwargs.get('use_cache', True)

            # Build index if not already built
            if not self.index:
                self.build_index(max_files=max_files, use_cache=use_cache)

            # Query the agent
            query = f"Analyze the repository and provide insights relevant to: {topic}"
            result = self.agent.invoke({"input": query})

            if isinstance(result, dict) and "output" in result:
                return result["output"]
            return str(result)

        except Exception as e:
            return f"Error during code analysis: {str(e)}"

    def __init__(
        self,
        root_path: str = ".",
        llm: Optional[Any] = None,
        cache_dir: str = "output/code_analysis",
    ) -> None:
        if llm is None:
            raise ValueError("LLM is required. Please provide a LangChain-compatible language model.")

        self.root_path = os.path.abspath(root_path)
        self.llm = llm
        self.index: Dict[str, str] = {}
        self.agent = None
        self.cache_dir = cache_dir

        # Create cache directory if it doesn't exist
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

        tools.set_repo_root(self.root_path)

        self._create_agent()

    def _create_agent(self) -> None:
        langchain_tools = tools.get_all_tools()
        self.agent = create_agent(self.llm, tools=langchain_tools)

    def _get_cache_key(self, max_files: Optional[int]) -> str:
        """Generate a cache key based on root_path and max_files."""
        key_str = f"{self.root_path}:{max_files}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get the full path to the cache file."""
        return Path(self.cache_dir) / f"index_cache_{cache_key}.json"

    def _save_cache(self, cache_key: str) -> None:
        """Save the current index to cache file."""
        cache_path = self._get_cache_path(cache_key)
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'root_path': self.root_path,
                    'index': self.index,
                    'timestamp': os.path.getmtime(cache_path) if cache_path.exists() else None
                }, f, indent=2)
            print(f"Index cached to {cache_path}")
        except Exception as e:
            print(f"Warning: Failed to save cache: {e}")

    def _load_cache(self, cache_key: str) -> bool:
        """Load index from cache file if it exists. Returns True if successful."""
        cache_path = self._get_cache_path(cache_key)
        if not cache_path.exists():
            return False

        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if data.get('root_path') == self.root_path:
                    self.index = data.get('index', {})
                    print(f"Loaded index from cache: {cache_path}")
                    return True
        except Exception as e:
            print(f"Warning: Failed to load cache: {e}")

        return False

    def build_index(self, max_files: int | None = None, description_agent: Optional[DescriptionAgent] = None, use_cache: bool = True) -> None:
        """Build or load the file index.

        Args:
            max_files: Maximum number of files to index
            description_agent: Optional agent for generating file descriptions
            use_cache: If True, try to load from cache first and save after building
        """
        cache_key = self._get_cache_key(max_files)

        # Try to load from cache first
        if use_cache and self._load_cache(cache_key):
            return

        # Build index from scratch
        print("Building index from scratch...")
        mapping = build_index(self.root_path, max_files=max_files, description_agent=description_agent)
        self.index = mapping

        # Save to cache
        if use_cache:
            self._save_cache(cache_key)

    def _score(self, description: str, query_tokens: List[str]) -> float:
        desc_tokens = _tokenize(description)
        if not desc_tokens:
            return 0.0
        qset = set(query_tokens)
        dset = set(desc_tokens)
        overlap = qset & dset
        return len(overlap) / max(1, len(dset))

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        tokens = _tokenize(query)
        scored: List[Tuple[str, float]] = []
        for path, desc in self.index.items():
            s = self._score(desc, tokens)
            if s > 0:
                scored.append((path, s))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def answer_question(self, question: str, top_k: int = 5) -> str:
        if not self.index:
            raise RuntimeError("Index is empty. Call build_index() first.")

        hits = self.retrieve(question, top_k=top_k)

        context_parts = []
        for path, score in hits:
            desc = self.index.get(path, "")
            context_parts.append(f"File: {path}\nScore: {score:.4f}\nDescription: {desc}\n")
        context = "\n".join(context_parts)

        prompt = (
            f"You are a chat assistant. Use the following file information to answer the question. About codebase. \n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            f"Provide a concise answer and mention which files you referenced."
        )

        response = self.agent.invoke({"input": prompt})
        return response.get("output", str(response))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("question", help="Question about the repository codebase")
    parser.add_argument("--root", default=".")
    parser.add_argument("--max-files", type=int, default=500)
    args = parser.parse_args()

    agent = CodebaseAgent(root_path=args.root)
    print("Building index...")
    agent.build_index(max_files=args.max_files)
    print(agent.answer_question(args.question))
