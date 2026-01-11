"""Indexing utilities extracted from agent.py.

Provides:
- A tokenizer helper
- build_index(root_path, max_files, description_agent) -> Dict[file_path, description]
- DescriptionAgent: accepts file content and returns a short description using an LLM when available

This version scans every file under root_path (skipping README.md) and extracts a short
description (by default uses the first non-empty line or first 200 characters). If a
DescriptionAgent is provided, it will be used to generate descriptions via an LLM or
callable.
"""
from __future__ import annotations

import importlib
import os
import re
import fnmatch
from typing import Callable, Dict, List, Optional

# Try to dynamically import LangChain pieces if available
_LLMChain = None
_PromptTemplate = None
_HAS_LANGCHAIN = False
try:
    _lc = importlib.import_module("langchain")
    _LLMChain = getattr(_lc, "LLMChain", None)
    _PromptTemplate = getattr(_lc, "PromptTemplate", None)
    _HAS_LANGCHAIN = bool(_LLMChain and _PromptTemplate)
except Exception:
    _HAS_LANGCHAIN = False


def _read_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def _tokenize(text: str) -> List[str]:
    tokens = re.split(r"\W+", text.lower())
    return [t for t in tokens if len(t) > 1]


def _make_description(text: str, max_chars: int = 200) -> str:
    # Prefer first non-empty line; otherwise first max_chars chars
    for line in text.splitlines():
        s = line.strip()
        if s:
            return s if len(s) <= max_chars else s[:max_chars].rstrip() + "..."
    # fallback to leading chars
    s = text.strip()
    return s[:max_chars].rstrip() + ("..." if len(s) > max_chars else "") if s else ""


def _read_agentignore(root_path: str) -> List[str]:
    """Read .agentignore file and return list of patterns to ignore.

    The .agentignore file should contain one pattern per line.
    Lines starting with # are treated as comments and ignored.
    Empty lines are ignored.

    Args:
        root_path: The root directory to look for .agentignore file

    Returns:
        List of patterns to ignore
    """
    ignore_file = os.path.join(root_path, ".agentignore")
    patterns = []

    if not os.path.exists(ignore_file):
        return patterns

    try:
        with open(ignore_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if line and not line.startswith("#"):
                    patterns.append(line)
    except Exception as e:
        print(f"Warning: Failed to read .agentignore: {e}")

    return patterns


def _should_ignore(path: str, root_path: str, patterns: List[str]) -> bool:
    """Check if a path should be ignored based on .agentignore patterns.

    Supports:
    - Exact file/directory names (e.g., "node_modules")
    - Wildcard patterns (e.g., "*.pyc", "test_*.py")
    - Directory patterns (e.g., "build/", "dist/")
    - Path patterns (e.g., "docs/*.md")

    Args:
        path: Absolute path to check
        root_path: Root directory for relative path calculation
        patterns: List of ignore patterns from .agentignore

    Returns:
        True if the path should be ignored, False otherwise
    """
    if not patterns:
        return False

    # Get relative path from root
    try:
        rel_path = os.path.relpath(path, root_path)
    except ValueError:
        # Paths on different drives on Windows
        return False

    # Normalize path separators for cross-platform compatibility
    rel_path = rel_path.replace(os.sep, "/")

    for pattern in patterns:
        # Normalize pattern separators
        pattern = pattern.replace(os.sep, "/")

        # Check if pattern matches the full path
        if fnmatch.fnmatch(rel_path, pattern):
            return True

        # Check if pattern matches any component of the path
        if fnmatch.fnmatch(os.path.basename(path), pattern):
            return True

        # Check if any parent directory matches the pattern
        path_parts = rel_path.split("/")
        for part in path_parts:
            if fnmatch.fnmatch(part, pattern.rstrip("/")):
                return True

    return False


class DescriptionAgent:
    """Agent that produces a short description from file content.

    Initialization options:
    - llm: a LangChain-compatible LLM instance (optional). If provided and LangChain is
      available, the agent will create an LLMChain and use it to generate descriptions.
    - llm_callable: a simple callable(content: str) -> str that will be used instead of
      LangChain when supplied. This is useful for testing or for alternative LLM wrappers.

    The agent exposes a single method `generate_description(content: str) -> str`.
    """

    def __init__(self, llm: Optional[object] = None, llm_callable: Optional[Callable[[str], str]] = None) -> None:
        self.llm = llm
        self.llm_callable = llm_callable
        self._chain = None

        if _HAS_LANGCHAIN and llm is not None:
            # create a small prompt template for concise descriptions
            template = (
                "You are a concise summarizer for code/project files.\n"
                "Given the file content, produce a one-sentence description (<= 500 characters) "
                "that explains the file's purpose. Return only the description text, no headers.\n\n"
                "File content:\n{content}\n\nDescription:"
            )
            try:
                prompt = _PromptTemplate(template=template, input_variables=["content"]) if _PromptTemplate is not None else None
                if prompt is not None and _LLMChain is not None:
                    # instantiate LLMChain dynamically
                    self._chain = _LLMChain(llm=llm, prompt=prompt)
            except Exception:
                self._chain = None

    def generate_description(self, content: str) -> str:
        """Generate a short description from file content using the configured LLM or fallback.

        Preference order:
        1. If a LangChain LLMChain was created, use chain.run(content=...)
        2. Else if llm_callable is provided, call llm_callable(content)
        3. Else fallback to simple rule-based `_make_description`
        """
        # 1) LangChain LLMChain
        if self._chain is not None:
            try:
                # Many LLMChain instances support run()
                return str(self._chain.run(content=content)).strip()
            except Exception:
                pass

        # 2) generic callable
        if self.llm_callable is not None and callable(self.llm_callable):
            try:
                return str(self.llm_callable(content)).strip()
            except Exception:
                pass

        # 3) fallback
        return _make_description(content)


def build_index(root_path: str = ".", max_files: int | None = None, description_agent: Optional[DescriptionAgent] = None) -> Dict[str, str]:
    """Scan files under root_path, skip README.md (case-insensitive), and return mapping
    file_path -> short description.

    If `description_agent` is provided, it will be called with the file content to produce a
    description. Otherwise the rule-based `_make_description` is used.

    Files and directories can be excluded by creating a .agentignore file in the root_path.
    """
    root_path = os.path.abspath(root_path)

    # Read .agentignore patterns
    ignore_patterns = _read_agentignore(root_path)
    if ignore_patterns:
        print(f"Loaded {len(ignore_patterns)} ignore patterns from .agentignore")

    mapping: Dict[str, str] = {}
    files_seen = 0
    for dirpath, dirnames, filenames in os.walk(root_path):
        # skip common ignored directories
        if ".git" in dirpath or "venv" in dirpath or "__pycache__" in dirpath:
            continue

        # Check if the current directory should be ignored
        if _should_ignore(dirpath, root_path, ignore_patterns):
            dirnames.clear()  # Don't recurse into this directory
            continue

        # Filter out ignored subdirectories to prevent os.walk from entering them
        dirnames[:] = [d for d in dirnames if not _should_ignore(os.path.join(dirpath, d), root_path, ignore_patterns)]

        for fn in filenames:
            # skip README.md (case-insensitive)
            if fn.lower() == "readme.md":
                continue

            path = os.path.join(dirpath, fn)

            # Check if file should be ignored
            if _should_ignore(path, root_path, ignore_patterns):
                continue

            text = _read_file(path)
            if not text:
                continue
            if description_agent is not None:
                try:
                    desc = description_agent.generate_description(text)
                except Exception:
                    desc = _make_description(text)
            else:
                desc = _make_description(text)
            mapping[path] = desc
            files_seen += 1
            if max_files and files_seen >= max_files:
                break
        if max_files and files_seen >= max_files:
            break

    return mapping
