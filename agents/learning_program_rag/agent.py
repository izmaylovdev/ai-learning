"""RAG Agent for answering questions using Retrieval-Augmented Generation."""
from typing import List, Dict, Optional

import config
from agents.base import BaseAgent, AgentMetadata
from util.interfaces import EmbeddingInterface, VectorStoreInterface
from util.embeddings import get_embedding_model
from text_generation.generator import GeneratorInterface
from util.vector_stores import get_vector_store
from text_generation.get_generator import get_generator


class RAGAgent(BaseAgent):
    """Answer questions using RAG with modular components."""

    @property
    def metadata(self) -> AgentMetadata:
        """Return metadata describing this agent."""
        return AgentMetadata(
            name="Learning Program RAG Agent",
            description="Uses Retrieval-Augmented Generation to provide information from learning materials, courses, and documentation",
            keywords=[
                "rag", "retrieval", "augmented", "generation", "database",
                "vector", "embedding", "ai", "ml", "machine learning",
                "llm", "learning", "course", "tutorial", "pattern",
                "knowledge", "documentation", "material", "resource",
                "education", "training", "concept", "theory"
            ],
            version="1.0.0",
            author="AI Learning System"
        )

    def is_available(self) -> bool:
        """Check if the agent is available and properly initialized."""
        return (self.embeddings is not None and
                self.vector_store is not None and
                self.generator is not None)

    def gather_insights(self, topic: str, **kwargs) -> str:
        """
        Gather insights from learning resources using RAG.

        Args:
            topic: The topic to research
            **kwargs: Additional parameters (top_k, source_filter)

        Returns:
            String with RAG insights
        """
        if not self.is_available():
            return "RAG agent not available."

        try:
            top_k = kwargs.get('top_k', 3)
            source_filter = kwargs.get('source_filter', None)

            # Query the RAG system
            query = f"Provide information about: {topic}"
            result = self.answer_question(
                question=query,
                top_k=top_k,
                source_filter=source_filter,
                show_sources=False,
                verbose=False
            )

            answer = result.get("answer", "No information found.")
            sources = result.get("sources", [])

            # Format with sources
            formatted = f"{answer}\n\nSources used:\n"
            for src in sources[:3]:  # Limit to top 3 sources
                formatted += f"- {src['source']} (relevance: {src['score']:.2f})\n"

            return formatted

        except Exception as e:
            return f"Error during RAG lookup: {str(e)}"

    def __init__(
        self,
        embedding_model: Optional[EmbeddingInterface] = None,
        vector_store: Optional[VectorStoreInterface] = None,
        generator: Optional[GeneratorInterface] = None,
        embedding_backend: Optional[str] = None,
        vector_store_backend: Optional[str] = None,
        generator_backend: Optional[str] = None,
        generator_kwargs: Optional[Dict] = None,
    ):
        """
        Initialize RAG system with pluggable components.

        Args:
            embedding_model: Embedding model implementation (default: from config)
            vector_store: Vector database implementation (default: from config)
            generator: Text generator implementation (default: from config)
            embedding_backend: Override embedding backend from config
            vector_store_backend: Override vector store backend from config
            generator_backend: Override generator backend from config
            generator_kwargs: Additional kwargs for generator
        """
        print("Initializing RAG Agent...")

        # Use config defaults unless overridden
        self.embedding_backend = embedding_backend or config.EMBEDDING_BACKEND
        self.vector_store_backend = vector_store_backend or config.VECTOR_STORE_BACKEND
        self.generator_backend = generator_backend or config.GENERATOR_BACKEND
        self.generator_kwargs = generator_kwargs or {}

        # Ensure generator kwargs include sensible defaults from config
        # For HuggingFace generator, ensure max_new_tokens is set
        if self.generator_backend.lower() == "huggingface":
            self.generator_kwargs.setdefault("max_new_tokens", config.HUGGINGFACE_MAX_NEW_TOKENS)

        # Initialize components with defaults if not provided
        self.embeddings = embedding_model or get_embedding_model(self.embedding_backend)
        self.vector_store = vector_store or get_vector_store(self.vector_store_backend)
        self.generator = generator or get_generator(self.generator_backend, **self.generator_kwargs)

    def retrieve_relevant_chunks(
        self,
        query: str,
        top_k: Optional[int] = None,
        source_filter: Optional[str] = None
    ) -> List[Dict]:
        """
        Retrieve relevant text chunks using the vector store.

        Args:
            query: The user's question
            top_k: Number of results to retrieve (default from config)
            source_filter: Optional filter by source filename

        Returns:
            List of dictionaries containing text and metadata
        """
        if top_k is None:
            top_k = config.TOP_K_RESULTS

        try:
            # Generate query embedding
            query_vector = self.embeddings.embed_query(query)

            # Search in vector store
            results = self.vector_store.search(
                query_vector=query_vector,
                top_k=top_k,
                source_filter=source_filter,
                score_threshold=config.SIMILARITY_THRESHOLD
            )

            return results

        except Exception as e:
            print(f"Error retrieving chunks: {e}")
            import traceback
            traceback.print_exc()
            return []

    def format_context(self, chunks: List[Dict]) -> str:
        """Format retrieved chunks into context for the LLM."""
        if not chunks:
            return "No relevant context found."

        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk['source']
            text = chunk['text']
            score = chunk['score']
            context_parts.append(
                f"[Source {i}: {source} (relevance: {score:.2f})]\n{text}\n"
            )

        return "\n".join(context_parts)

    def generate_answer(self, question: str, context: str) -> str:
        """Generate answer using generator with retrieved context."""
        return self.generator.generate_answer(question, context)

    def answer_question(
        self,
        question: str,
        top_k: Optional[int] = None,
        source_filter: Optional[str] = None,
        show_sources: bool = True,
        verbose: bool = True
    ) -> Dict:
        """
        Answer a question using RAG.

        Args:
            question: The user's question
            top_k: Number of relevant chunks to retrieve
            source_filter: Optional filter by source filename
            show_sources: Whether to show source documents
            verbose: Whether to print progress messages

        Returns:
            Dictionary with answer and metadata
        """
        if verbose:
            print(f"\nQuestion: {question}")
            print("-" * 80)

        # Step 1: Retrieve relevant chunks
        if verbose:
            print(f"Retrieving top {top_k or config.TOP_K_RESULTS} relevant passages...")
        chunks = self.retrieve_relevant_chunks(question, top_k, source_filter)

        if verbose:
            print(f"Found {len(chunks)} relevant passages")

        # Step 2: Format context
        context = self.format_context(chunks)

        # Step 3: Generate answer
        if verbose:
            print("Generating answer...")
        answer = self.generate_answer(question, context)

        # Prepare result
        result = {
            'answer': answer,
            'sources': [{'source': c['source'], 'score': c['score']} for c in chunks],
            'chunks_used': len(chunks)
        }

        # Display result
        if verbose:
            print("\nAnswer:")
            print("=" * 80)
            print(answer)
            print("=" * 80)

            if show_sources:
                print("\nSources used:")
                sources_dict = {}
                for chunk in chunks:
                    source = chunk['source']
                    score = chunk['score']
                    if source not in sources_dict or score > sources_dict[source]:
                        sources_dict[source] = score

                for source, score in sorted(sources_dict.items(), key=lambda x: x[1], reverse=True):
                    print(f"  - {source} (relevance: {score:.2f})")

        return result

    def list_sources(self) -> List[Dict]:
        """
        List all available sources in the collection.

        Returns:
            List of dictionaries with source information
        """
        try:
            sources_list = self.vector_store.list_sources()

            print("\nAvailable sources:")
            print("-" * 80)
            for item in sources_list:
                print(f"  - {item['source']} ({item['type']})")
            print(f"\nTotal: {len(sources_list)} sources")

            return sources_list

        except Exception as e:
            print(f"Error listing sources: {e}")
            return []


if __name__ == "__main__":
    # Simple test when running agent.py directly
    agent = RAGAgent()
    result = agent.answer_question("What is RAG?")
    print(f"\nResult: {result}")

