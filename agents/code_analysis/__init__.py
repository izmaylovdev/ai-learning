"""code_analysis package
Exports CodebaseAgent for querying the repository codebase.
"""

from importlib import import_module

try:
    _mod = import_module(f"{__name__}.agent")
    CodebaseAgent = getattr(_mod, "CodebaseAgent")
except Exception:  # pragma: no cover
    # If import fails (static analysis / partial environment), provide a safe fallback.
    CodebaseAgent = None

__all__ = ["CodebaseAgent"]
