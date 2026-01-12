"""code_analysis package
Exports analyze_code tool for querying the repository codebase.
"""

from importlib import import_module

try:
    _mod = import_module(f"{__name__}.agent")
    analyze_code = getattr(_mod, "analyze_code")
except Exception:  # pragma: no cover
    # If import fails (static analysis / partial environment), provide a safe fallback.
    analyze_code = None

__all__ = ["analyze_code"]
