"""Test retrieval from the vector store. Accepts a query and prints results."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import config
from util.embeddings import get_embedding_model
from util.vector_stores import get_vector_store


def show_collection_info(vector_store) -> bool:
    """Show collection info and return True if collection is healthy."""
    if not vector_store.verify_collection():
        print("Collection does not exist. Please run reindex first.")
        return False

    info = vector_store.get_collection_info()
    print(f"  Collection: {info['name']}")
    print(f"  Points:     {info.get('points_count', 'unknown')}")

    if info.get("error"):
        print(f"  Error:      {info['error']}")
        return False

    sources = vector_store.list_sources()
    if sources:
        print(f"  Sources:    {len(sources)}")
        for s in sources:
            print(f"    - {s['source']} ({s['type']})")
    print()
    return True


def do_reindex():
    """Clear the database and reindex all documents."""
    from agents.learning_program_rag.reindex import main as reindex_main

    reindex_main()


def test_retrieval(query: str, top_k: int = 3) -> None:
    """Run a retrieval query and print results.

    Args:
        query: The search query string
        top_k: Number of results to retrieve
    """
    print(f"\n{'=' * 60}")
    print(f"Query: {query}")
    print(f"Top K: {top_k}")
    print(f"{'=' * 60}\n")

    # Initialize components
    print("Initializing embedding model and vector store...")
    embeddings = get_embedding_model(config.EMBEDDING_BACKEND)
    vector_store = get_vector_store(config.VECTOR_STORE_BACKEND)

    # Show collection info
    if not show_collection_info(vector_store):
        return

    # Generate query embedding
    print("Generating query embedding...")
    query_vector = embeddings.embed_query(query)

    # Search
    print("Searching vector store...\n")
    try:
        results = vector_store.search(
            query_vector=query_vector,
            top_k=top_k,
            score_threshold=config.SIMILARITY_THRESHOLD,
        )
    except Exception as e:
        error_msg = str(e)
        print(f"Error during search: {error_msg}\n")
        if "OutputTooSmall" in error_msg or "500" in error_msg:
            print("This is a known Qdrant index corruption issue.")
            print("The HNSW index needs to be rebuilt.")
            answer = input("Would you like to reindex now? (y/n): ").strip().lower()
            if answer == "y":
                do_reindex()
                print("\nReindex complete. Please run your query again.")
        return

    if not results:
        print("No results found.")
        return

    print(f"Found {len(results)} result(s):\n")
    for i, chunk in enumerate(results, 1):
        print(f"--- Result {i} ---")
        print(f"  Source: {chunk['source']}")
        print(f"  Type:  {chunk.get('type', 'N/A')}")
        print(f"  Score: {chunk['score']:.4f}")
        print(f"  Text:\n{chunk['text'][:500]}")
        if len(chunk['text']) > 500:
            print(f"  ... ({len(chunk['text'])} chars total)")
        print()


def main():
    query = input("Enter your search query: ").strip()
    if not query:
        print("No query provided. Exiting.")
        return

    test_retrieval(query=query)


if __name__ == "__main__":
    main()
