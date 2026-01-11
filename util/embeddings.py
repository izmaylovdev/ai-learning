"""Embedding implementations."""

from typing import List
from langchain_huggingface import HuggingFaceEmbeddings
import config
from util.interfaces import EmbeddingInterface


class HuggingFaceEmbeddingModel(EmbeddingInterface):
    """HuggingFace embedding model implementation."""

    def __init__(self, model_name: str = None):
        """
        Initialize HuggingFace embedding model.

        Args:
            model_name: Name of the HuggingFace model to use (default from config)
        """
        self.model_name = model_name or config.EMBEDDING_MODEL
        print(f"Loading embedding model: {self.model_name}")
        self.model = HuggingFaceEmbeddings(model_name=self.model_name)
        self.dimension = config.EMBEDDING_DIMENSION

    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query text."""
        return self.model.embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple documents."""
        return self.model.embed_documents(texts)

    def get_dimension(self) -> int:
        """Get the dimension of the embedding vectors."""
        return self.dimension


# Factory function to get embedding model
def get_embedding_model(model_type: str = "huggingface", **kwargs) -> EmbeddingInterface:
    """
    Factory function to get an embedding model instance.

    Args:
        model_type: Type of embedding model ("huggingface")
        **kwargs: Additional arguments for the model

    Returns:
        EmbeddingInterface implementation
    """
    if model_type == "huggingface":
        return HuggingFaceEmbeddingModel(**kwargs)
    else:
        raise ValueError(f"Unknown embedding model type: {model_type}")

