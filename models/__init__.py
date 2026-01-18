"""Models package for AI learning project."""
from .llm_studio_model import llm_studio_model
from .gemini_model import gemini_model, gemini_pro_model

default_model = gemini_model

__all__ = [
    "llm_studio_model",
    "gemini_model",
    "gemini_pro_model",
    "default_model",
]
