"""Get and display data suggestions for a query."""

import sys
from typing import List, Dict

import config
from interfaces import EmbeddingInterface, VectorStoreInterface
from embeddings import get_embedding_model
from vector_stores import get_vector_store


class QueryDataRetriever:
    """Retrieve and display data suggestions for queries."""

    def __init__(
        self,
        embedding_model: EmbeddingInterface = None,
        vector_store: VectorStoreInterface = None
    ):
        """
        Initialize query data retriever.

        Args:
            embedding_model: Embedding model implementation (default: HuggingFace)
            vector_store: Vector database implementation (default: Qdrant)
        """
        print("Initializing Query Data Retriever...")

        # Initialize components with defaults if not provided
        self.embeddings = embedding_model or get_embedding_model("huggingface")
        self.vector_store = vector_store or get_vector_store("qdrant")

    def get_suggestions(
        self,
        query: str,
        top_k: int = None,
        source_filter: str = None,
        min_score: float = None
    ) -> List[Dict]:
        """
        Get data suggestions for a query.

        Args:
            query: The user's query
            top_k: Number of results to retrieve (default from config)
            source_filter: Optional filter by source filename
            min_score: Minimum similarity score (default from config)

        Returns:
            List of dictionaries containing text and metadata
        """
        if top_k is None:
            top_k = config.TOP_K_RESULTS
        if min_score is None:
            min_score = config.SIMILARITY_THRESHOLD

        try:
            # Generate query embedding
            query_vector = self.embeddings.embed_query(query)

            # Search in vector store
            results = self.vector_store.search(
                query_vector=query_vector,
                top_k=top_k,
                source_filter=source_filter,
                score_threshold=min_score
            )

            return results

        except Exception as e:
            print(f"Error retrieving suggestions: {e}")
            import traceback
            traceback.print_exc()
            return []

    def display_suggestions(
        self,
        query: str,
        top_k: int = None,
        source_filter: str = None,
        show_full_text: bool = True
    ):
        """
        Display data suggestions for a query in console.

        Args:
            query: The user's query
            top_k: Number of results to retrieve
            source_filter: Optional filter by source filename
            show_full_text: Whether to show full text or just a preview
        """
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print(f"{'='*80}\n")

        # Get suggestions
        print(f"Retrieving top {top_k or config.TOP_K_RESULTS} suggestions...")
        if source_filter:
            print(f"Filtering by source: {source_filter}")
        
        suggestions = self.get_suggestions(query, top_k, source_filter)

        if not suggestions:
            print("\n❌ No suggestions found for this query.")
            print("Try:")
            print("  - Adjusting the query")
            print("  - Lowering the similarity threshold in config.py")
            print("  - Removing the source filter")
            return

        print(f"\n✓ Found {len(suggestions)} relevant suggestions\n")

        # Display each suggestion
        for i, suggestion in enumerate(suggestions, 1):
            source = suggestion['source']
            text = suggestion['text']
            doc_type = suggestion['type']
            score = suggestion['score']

            print(f"{'─'*80}")
            print(f"Suggestion #{i}")
            print(f"{'─'*80}")
            print(f"📄 Source: {source}")
            print(f"📝 Type: {doc_type}")
            print(f"⭐ Relevance Score: {score:.4f}")
            print(f"\n{'Content:'}")
            print(f"{'─'*80}")
            
            if show_full_text:
                print(text)
            else:
                # Show preview (first 300 characters)
                preview = text[:300] + "..." if len(text) > 300 else text
                print(preview)
                if len(text) > 300:
                    print(f"\n[...truncated, {len(text) - 300} more characters]")
            
            print()

        # Show summary of sources
        print(f"{'='*80}")
        print("Summary of Sources Used:")
        print(f"{'='*80}")
        
        sources_dict = {}
        for suggestion in suggestions:
            source = suggestion['source']
            score = suggestion['score']
            if source not in sources_dict:
                sources_dict[source] = {'max_score': score, 'count': 0}
            sources_dict[source]['max_score'] = max(sources_dict[source]['max_score'], score)
            sources_dict[source]['count'] += 1

        for source, info in sorted(sources_dict.items(), key=lambda x: x[1]['max_score'], reverse=True):
            print(f"  📌 {source}")
            print(f"     - Chunks: {info['count']}")
            print(f"     - Best relevance: {info['max_score']:.4f}")

        print(f"\n{'='*80}\n")

    def list_sources(self):
        """List all available sources in the collection."""
        try:
            sources_list = self.vector_store.list_sources()

            print("\n" + "="*80)
            print("Available sources:")
            print("="*80)
            for item in sources_list:
                print(f"  📄 {item['source']} ({item['type']})")
            print(f"\nTotal: {len(sources_list)} sources")
            print("="*80 + "\n")

        except Exception as e:
            print(f"Error listing sources: {e}")


def main():
    """Main function to retrieve and display data for queries."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Get and display data suggestions for queries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '-q', '--query',
        type=str,
        help='Query to search for (if not provided, will prompt for input)'
    )
    parser.add_argument(
        '-k', '--top-k',
        type=int,
        default=None,
        help=f'Number of suggestions to retrieve (default: {config.TOP_K_RESULTS})'
    )
    parser.add_argument(
        '-s', '--source',
        type=str,
        default=None,
        help='Filter results by source filename'
    )
    parser.add_argument(
        '--preview',
        action='store_true',
        help='Show only text previews instead of full content'
    )
    parser.add_argument(
        '--list-sources',
        action='store_true',
        help='List all available sources and exit'
    )

    args = parser.parse_args()

    # Initialize retriever
    retriever = QueryDataRetriever()

    # List sources if requested
    if args.list_sources:
        retriever.list_sources()
        return

    # Get query from args or prompt user
    query = args.query
    if not query:
        print("\nEnter your query (or 'quit' to exit):")
        query = input("> ").strip()
        if query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            return

    if not query:
        print("Error: No query provided")
        sys.exit(1)

    # Display suggestions
    retriever.display_suggestions(
        query=query,
        top_k=args.top_k,
        source_filter=args.source,
        show_full_text=not args.preview
    )

    # Interactive mode if no query was provided via args
    if not args.query:
        while True:
            print("\nEnter another query (or 'quit' to exit):")
            query = input("> ").strip()
            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            if query:
                retriever.display_suggestions(
                    query=query,
                    top_k=args.top_k,
                    source_filter=args.source,
                    show_full_text=not args.preview
                )


if __name__ == "__main__":
    main()

