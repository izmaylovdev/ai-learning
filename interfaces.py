"""Interfaces for RAG components."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any


class EmbeddingInterface(ABC):
    """Interface for embedding models."""
    
    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a single query text.
        
        Args:
            text: Input text to embed
            
        Returns:
            List of float values representing the embedding vector
        """
        pass
    
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple documents.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors.
        
        Returns:
            Dimension size
        """
        pass


class VectorStoreInterface(ABC):
    """Interface for vector database operations."""
    
    @abstractmethod
    def search(
        self, 
        query_vector: List[float], 
        top_k: int = 5,
        source_filter: str = None,
        score_threshold: float = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in the database.
        
        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            source_filter: Optional filter by source filename
            score_threshold: Minimum similarity score
            
        Returns:
            List of dictionaries containing text, metadata, and scores
        """
        pass
    
    @abstractmethod
    def verify_collection(self) -> bool:
        """
        Verify that the collection exists and is accessible.
        
        Returns:
            True if collection exists, False otherwise
        """
        pass
    
    @abstractmethod
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the collection.
        
        Returns:
            Dictionary with collection metadata
        """
        pass
    
    @abstractmethod
    def list_sources(self) -> List[Dict[str, str]]:
        """
        List all available sources in the collection.
        
        Returns:
            List of dictionaries with source name and type
        """
        pass

