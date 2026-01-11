"""learning_program_rag package
Exports RAGAgent for answering questions using RAG (Retrieval-Augmented Generation).
"""

from importlib import import_module

try:
    _mod = import_module(f"{__name__}.agent")
    RAGAgent = getattr(_mod, "RAGAgent")
except Exception:  # pragma: no cover
    # If import fails (static analysis / partial environment), provide a safe fallback.
    RAGAgent = None

__all__ = ["RAGAgent"]

