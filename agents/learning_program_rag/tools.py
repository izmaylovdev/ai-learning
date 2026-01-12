"""LangChain tools for RAG operations."""
from __future__ import annotations
from typing import Optional
from langchain.tools import tool

import config
from util.embeddings import get_embedding_model
from util.vector_stores import get_vector_store

# Global RAG components
_embeddings = None
_vector_store = None


def initialize_rag_components(embedding_backend: Optional[str] = None, vector_store_backend: Optional[str] = None) -> None:
    """Initialize RAG components for tools.

    Args:
        embedding_backend: Embedding model backend (default from config)
        vector_store_backend: Vector store backend (default from config)
    """
    global _embeddings, _vector_store

    embedding_backend = embedding_backend or config.EMBEDDING_BACKEND
    vector_store_backend = vector_store_backend or config.VECTOR_STORE_BACKEND

    _embeddings = get_embedding_model(embedding_backend)
    _vector_store = get_vector_store(vector_store_backend)

    print(f"RAG components initialized: {embedding_backend} + {vector_store_backend}")


@tool
def search_knowledge_base(query: str, top_k: int = 3) -> str:
    """Search the knowledge base for relevant information.

    Args:
        query: The search query or question
        top_k: Number of relevant chunks to retrieve (default: 3)

    Returns:
        Formatted context from relevant documents
    """
    if _embeddings is None or _vector_store is None:
        return "Error: RAG components not initialized. Please initialize first."

    try:
        # Generate query embedding
        query_vector = _embeddings.embed_query(query)

        # Search in vector store
        results = _vector_store.search(
            query_vector=query_vector,
            top_k=top_k,
            score_threshold=config.SIMILARITY_THRESHOLD
        )

        if not results:
            return "No relevant information found in the knowledge base."

        # Format results
        context_parts = []
        for i, chunk in enumerate(results, 1):
            source = chunk['source']
            text = chunk['text']
            score = chunk['score']
            context_parts.append(
                f"[Source {i}: {source} (relevance: {score:.2f})]\n{text}\n"
            )

        return "\n".join(context_parts)

    except Exception as e:
        return f"Error searching knowledge base: {str(e)}"


@tool
def search_by_source(query: str, source_filter: str, top_k: int = 3) -> str:
    """Search the knowledge base filtered by specific source.

    Args:
        query: The search query or question
        source_filter: Filter by source filename
        top_k: Number of relevant chunks to retrieve (default: 3)

    Returns:
        Formatted context from relevant documents in the specified source
    """
    if _embeddings is None or _vector_store is None:
        return "Error: RAG components not initialized. Please initialize first."

    try:
        # Generate query embedding
        query_vector = _embeddings.embed_query(query)

        # Search in vector store with source filter
        results = _vector_store.search(
            query_vector=query_vector,
            top_k=top_k,
            source_filter=source_filter,
            score_threshold=config.SIMILARITY_THRESHOLD
        )

        if not results:
            return f"No relevant information found in source: {source_filter}"

        # Format results
        context_parts = []
        for i, chunk in enumerate(results, 1):
            source = chunk['source']
            text = chunk['text']
            score = chunk['score']
            context_parts.append(
                f"[Source {i}: {source} (relevance: {score:.2f})]\n{text}\n"
            )

        return "\n".join(context_parts)

    except Exception as e:
        return f"Error searching by source: {str(e)}"


@tool
def list_available_sources() -> str:
    """List all available sources in the knowledge base.

    Returns:
        Newline-separated list of available sources
    """
    if _vector_store is None:
        return "Error: Vector store not initialized."

    try:
        sources_list = _vector_store.list_sources()

        if not sources_list:
            return "No sources found in the knowledge base."

        result = "Available sources in the knowledge base:\n"
        result += "-" * 60 + "\n"
        for item in sources_list:
            result += f"  - {item['source']} ({item['type']})\n"
        result += f"\nTotal: {len(sources_list)} sources"

        return result

    except Exception as e:
        return f"Error listing sources: {str(e)}"


def get_all_tools() -> list:
    """Get all RAG tools."""
    return [search_knowledge_base, search_by_source, list_available_sources]

