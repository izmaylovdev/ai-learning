from __future__ import annotations
import glob
import os
from langchain.tools import tool

# Global variable to store repository root path
_REPO_ROOT = os.path.abspath("")
def set_repo_root(root_path: str) -> None:
    """Set the repository root path for all tools.
    Args:
        root_path: The absolute path to the repository root directory.
    """
    global _REPO_ROOT
    _REPO_ROOT = os.path.abspath(root_path)

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

def get_all_tools() -> list:
    return [read_file, list_files]
