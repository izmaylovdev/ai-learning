"""LangChain tools for code analysis operations."""
from __future__ import annotations
import glob
import os
from typing import Optional, Dict
from langchain.tools import tool
from .indexer import build_index

# Global variables to store repository root path and file index
_REPO_ROOT = os.path.abspath(".")
_FILE_INDEX: Dict[str, str] = {}


def initialize_code_analysis(root_path: Optional[str] = None) -> None:
    """Initialize code analysis tools with repository root path and build file index.

    Args:
        root_path: The absolute path to the repository root directory (default: current directory)
    """
    global _REPO_ROOT, _FILE_INDEX
    _REPO_ROOT = os.path.abspath(root_path or ".")
    print(f"Code analysis initialized with root: {_REPO_ROOT}")
    print("Building repository index...")
    _FILE_INDEX = build_index(_REPO_ROOT)
    print(f"Indexed {len(_FILE_INDEX)} files")


def set_repo_root(root_path: str) -> None:
    """Set the repository root path for all tools and rebuild index.

    Args:
        root_path: The absolute path to the repository root directory.
    """
    global _REPO_ROOT, _FILE_INDEX
    _REPO_ROOT = os.path.abspath(root_path)
    print(f"Repository root updated to: {_REPO_ROOT}")
    print("Rebuilding repository index...")
    _FILE_INDEX = build_index(_REPO_ROOT)
    print(f"Indexed {len(_FILE_INDEX)} files")

@tool
def read_file(path: str, max_chars: int = 10000) -> str:
    """Read the contents of a file from the repository.

    Args:
        path: Path to the file (relative to repository root or absolute)
        max_chars: Maximum number of characters to read

    Returns:
        File contents or error message
    """
    # Resolve absolute path; allow relative paths relative to repo root
    if not os.path.isabs(path):
        abs_path = os.path.abspath(os.path.join(_REPO_ROOT, path))
    else:
        abs_path = os.path.abspath(path)
    # Ensure the file is inside the repo root to avoid exfiltration
    try:
        common = os.path.commonpath([_REPO_ROOT, abs_path])
    except Exception:
        return f"Error: invalid path '{path}'."
    if common != _REPO_ROOT:
        return f"Error: access to paths outside the repository root is not allowed: '{path}'"
    if not os.path.exists(abs_path):
        return f"Error: file not found: '{path}'"
    if os.path.isdir(abs_path):
        return f"Error: requested path is a directory: '{path}'"
    try:
        with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read(max_chars)
    except Exception as e:
        return f"Error reading file '{path}': {e}"
    # Truncate with an explicit notice if the file was longer than max_chars
    if len(text) >= max_chars:
        return text + "\n\n...[truncated]"
    return text

@tool
def list_files(directory: str = ".", pattern: str = "*") -> str:
    """List files in a directory matching a pattern.

    Args:
        directory: Directory path (relative to repository root or absolute)
        pattern: Glob pattern for filtering files (default: "*")

    Returns:
        Newline-separated list of matching files or error message
    """
    # Resolve absolute path
    if not os.path.isabs(directory):
        abs_dir = os.path.abspath(os.path.join(_REPO_ROOT, directory))
    else:
        abs_dir = os.path.abspath(directory)
    # Ensure directory is inside repo root
    try:
        common = os.path.commonpath([_REPO_ROOT, abs_dir])
    except Exception:
        return f"Error: invalid directory path '{directory}'."
    if common != _REPO_ROOT:
        return f"Error: access to paths outside the repository root is not allowed: '{directory}'"
    if not os.path.exists(abs_dir):
        return f"Error: directory not found: '{directory}'"
    if not os.path.isdir(abs_dir):
        return f"Error: path is not a directory: '{directory}'"
    try:
        search_pattern = os.path.join(abs_dir, pattern)
        files = glob.glob(search_pattern)
        # Make paths relative to root for cleaner output
        rel_files = []
        for f in files:
            try:
                rel = os.path.relpath(f, _REPO_ROOT)
                rel_files.append(rel)
            except ValueError:
                rel_files.append(f)
        if not rel_files:
            return f"No files found matching pattern '{pattern}' in '{directory}'"
        return "\n".join(sorted(rel_files))
    except Exception as e:
        return f"Error listing files: {e}"

@tool
def search_repository(query: str) -> str:
    """Search through the repository index to find relevant files based on a query.

    This tool searches file paths and descriptions to find files related to your query.
    Use this to discover relevant files before reading them.

    Args:
        query: Search query (keywords or concepts to find in file paths and descriptions)

    Returns:
        List of relevant files with their descriptions
    """
    if not _FILE_INDEX:
        return "Error: Repository index not initialized. Please initialize code analysis first."

    query_lower = query.lower()
    query_parts = query_lower.split()

    results = []
    for file_path, description in _FILE_INDEX.items():
        # Get relative path for display
        try:
            rel_path = os.path.relpath(file_path, _REPO_ROOT)
        except ValueError:
            rel_path = file_path

        # Calculate relevance score
        score = 0
        search_text = (rel_path + " " + description).lower()

        for part in query_parts:
            if part in search_text:
                score += search_text.count(part)

        if score > 0:
            results.append((score, rel_path, description))

    if not results:
        return f"No files found matching query: '{query}'"

    # Sort by relevance score (highest first)
    results.sort(reverse=True, key=lambda x: x[0])

    # Format output
    output_lines = [f"Found {len(results)} relevant file(s):\n"]
    for score, path, desc in results[:20]:  # Limit to top 20 results
        output_lines.append(f"📄 {path}")
        output_lines.append(f"   {desc}\n")

    if len(results) > 20:
        output_lines.append(f"... and {len(results) - 20} more files")

    return "\n".join(output_lines)

def get_all_tools() -> list:
    return [read_file, list_files, search_repository]
