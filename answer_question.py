"""Answer questions using RAG (Retrieval-Augmented Generation)."""

from typing import List, Dict

import config
from interfaces import EmbeddingInterface, VectorStoreInterface
from embeddings import get_embedding_model
from text_generation.generator import GeneratorInterface
from vector_stores import get_vector_store
from text_generation.get_generator import get_generator

class RAGQuestionAnswerer:
    """Answer questions using RAG with modular components."""

    def __init__(
        self,
        embedding_model: EmbeddingInterface = None,
        vector_store: VectorStoreInterface = None,
        generator: GeneratorInterface = None
    ):
        """
        Initialize RAG system with pluggable components.

        Args:
            embedding_model: Embedding model implementation (default: HuggingFace)
            vector_store: Vector database implementation (default: Qdrant)
            generator: Text generator implementation (default: HuggingFace)
        """
        print("Initializing RAG Question Answering System...")

        # Initialize components with defaults if not provided
        self.embeddings = embedding_model or get_embedding_model("huggingface")
        self.vector_store = vector_store or get_vector_store("qdrant")
        self.generator = generator or get_generator("gemini")

    def retrieve_relevant_chunks(self, query: str, top_k: int = None, source_filter: str = None) -> List[Dict]:
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

    def answer_question(self, question: str, top_k: int = None, source_filter: str = None,
                       show_sources: bool = True) -> Dict:
        """
        Answer a question using RAG.

        Args:
            question: The user's question
            top_k: Number of relevant chunks to retrieve
            source_filter: Optional filter by source filename
            show_sources: Whether to show source documents

        Returns:
            Dictionary with answer and metadata
        """
        print(f"\nQuestion: {question}")
        print("-" * 80)

        # Step 1: Retrieve relevant chunks
        print(f"Retrieving top {top_k or config.TOP_K_RESULTS} relevant passages...")
        chunks = self.retrieve_relevant_chunks(question, top_k, source_filter)

        print(f"Found {len(chunks)} relevant passages")

        # Step 2: Format context
        context = self.format_context(chunks)

        # Step 3: Generate answer
        print("Generating answer...")
        answer = self.generate_answer(question, context)

        # Prepare result
        result = {
            'answer': answer,
            'sources': [{'source': c['source'], 'score': c['score']} for c in chunks],
            'chunks_used': len(chunks)
        }

        # Display result
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



    def list_sources(self):
        """List all available sources in the collection."""
        try:
            sources_list = self.vector_store.list_sources()

            print("\nAvailable sources:")
            print("-" * 80)
            for item in sources_list:
                print(f"  - {item['source']} ({item['type']})")
            print(f"\nTotal: {len(sources_list)} sources")

        except Exception as e:
            print(f"Error listing sources: {e}")


def main():
    """Main function to run RAG question answering."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Answer questions using RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Answer a question (will prompt if not provided)
  python answer_question.py
  
  # Answer a single question
  python answer_question.py -q "What is RAG?"
  
  # Answer with more context
  python answer_question.py -q "What is RAG?" -k 10
  
  # Filter by source
  python answer_question.py -q "What is RAG?" -s "RAG Intro.pdf"
  
  # List available sources
  python answer_question.py --list-sources
        """
    )

    parser.add_argument(
        '-q', '--question',
        type=str,
        help='Question to answer (if not provided, will prompt for input)'
    )
    parser.add_argument(
        '-k', '--top-k',
        type=int,
        default=None,
        help=f'Number of relevant passages to retrieve (default: {config.TOP_K_RESULTS})'
    )
    parser.add_argument(
        '-s', '--source',
        type=str,
        default=None,
        help='Filter results by source filename'
    )
    parser.add_argument(
        '--no-sources',
        action='store_true',
        help='Don\'t show source documents'
    )
    parser.add_argument(
        '--list-sources',
        action='store_true',
        help='List all available sources and exit'
    )

    args = parser.parse_args()

    # Initialize RAG system
    rag = RAGQuestionAnswerer()

    # List sources if requested
    if args.list_sources:
        rag.list_sources()
        return

    # Get question from args or prompt user
    question = args.question
    if not question:
        print("\n" + "=" * 80)
        print("RAG Question Answering System")
        print("=" * 80)
        question = input("\n🤔 Your question: ").strip()

        if not question:
            print("No question provided. Exiting.")
            return

    # Answer the question
    rag.answer_question(
        question=question,
        top_k=args.top_k,
        source_filter=args.source,
        show_sources=not args.no_sources
    )


if __name__ == "__main__":
    main()

