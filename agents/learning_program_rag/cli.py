"""CLI wrapper for the RAG Agent to answer questions from command line."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.learning_program_rag.agent import RAGAgent
import config


def main():
    """Main function to run RAG question answering from CLI."""
    parser = argparse.ArgumentParser(
        description="Answer questions using RAG (Retrieval-Augmented Generation)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Answer a question (will prompt if not provided)
  python -m learning_program_rag.cli
  
  # Answer a single question
  python -m learning_program_rag.cli -q "What is RAG?"
  
  # Answer with more context
  python -m learning_program_rag.cli -q "What is RAG?" -k 10
  
  # Filter by source
  python -m learning_program_rag.cli -q "What is RAG?" -s "RAG Intro.pdf"
  
  # List available sources
  python -m learning_program_rag.cli --list-sources
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

    # Backend override flags
    parser.add_argument(
        '--embedding-backend',
        type=str,
        default=None,
        help='Override embedding backend (e.g., huggingface, openai)'
    )
    parser.add_argument(
        '--vector-store-backend',
        type=str,
        default=None,
        help='Override vector store backend (e.g., qdrant, chroma)'
    )
    parser.add_argument(
        '--generator',
        type=str,
        default=None,
        help='Override generator backend (e.g., gemini, huggingface)'
    )
    parser.add_argument(
        '--generator-max-tokens',
        type=int,
        default=None,
        help='Override max tokens/new tokens for the generator (backend-specific)'
    )

    args = parser.parse_args()

    # Build generator kwargs from CLI and config
    generator_kwargs = {}
    if args.generator_max_tokens is not None:
        # Map CLI flag to the appropriate generator kwarg name
        # HuggingFace expects max_new_tokens, Gemini expects max_output_tokens
        generator_backend = args.generator or config.GENERATOR_BACKEND
        if generator_backend.lower() == 'huggingface':
            generator_kwargs['max_new_tokens'] = args.generator_max_tokens
        else:
            generator_kwargs['max_output_tokens'] = args.generator_max_tokens

    # Initialize RAG agent
    rag_agent = RAGAgent(
        embedding_backend=args.embedding_backend,
        vector_store_backend=args.vector_store_backend,
        generator_backend=args.generator,
        generator_kwargs=generator_kwargs if generator_kwargs else None,
    )

    # List sources if requested
    if args.list_sources:
        rag_agent.list_sources()
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
    rag_agent.answer_question(
        question=question,
        top_k=args.top_k,
        source_filter=args.source,
        show_sources=not args.no_sources,
        verbose=True
    )


if __name__ == "__main__":
    main()

