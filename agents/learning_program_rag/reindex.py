"""Script to clear the vector database and reindex all documents."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

import config


def clear_database():
    """Clear the Qdrant collection by deleting and recreating it."""
    print(f"Connecting to Qdrant at {config.QDRANT_HOST}:{config.QDRANT_PORT}")
    client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)

    try:
        # Check if collection exists
        collections = client.get_collections().collections
        collection_names = [c.name for c in collections]

        if config.COLLECTION_NAME in collection_names:
            # Get current count before deletion
            info = client.get_collection(config.COLLECTION_NAME)
            print(f"Collection '{config.COLLECTION_NAME}' found with {info.points_count} points")

            # Delete the collection
            print(f"Deleting collection: {config.COLLECTION_NAME}")
            client.delete_collection(config.COLLECTION_NAME)
            print("Collection deleted successfully")
        else:
            print(f"Collection '{config.COLLECTION_NAME}' does not exist")

        # Recreate the collection
        print(f"Creating fresh collection: {config.COLLECTION_NAME}")
        client.create_collection(
            collection_name=config.COLLECTION_NAME,
            vectors_config=VectorParams(
                size=config.EMBEDDING_DIMENSION,
                distance=Distance.COSINE
            )
        )
        print("Collection created successfully")

    except Exception as e:
        print(f"Error clearing database: {e}")
        sys.exit(1)


def reindex_all():
    """Reindex all documents (PDFs, videos, HTML)."""
    from .data_extraction.extract_pdf import PDFProcessor, main as index_pdfs
    from .data_extraction.extract_video import VideoProcessor, main as index_videos
    from .data_extraction.extract_html import HTMLProcessor, main as index_html

    print("\n" + "="*60)
    print("Reindexing PDFs...")
    print("="*60)
    index_pdfs()

    print("\n" + "="*60)
    print("Reindexing Videos...")
    print("="*60)
    index_videos()

    print("\n" + "="*60)
    print("Reindexing HTML files...")
    print("="*60)
    index_html()

    print("\n" + "="*60)
    print("All documents reindexed!")
    print("="*60)


def main():
    """Main function to clear database and reindex everything."""
    print("\n" + "="*60)
    print("CLEAR DATABASE AND REINDEX")
    print("="*60 + "\n")

    # Step 1: Clear database
    print("Step 1: Clearing database...")
    clear_database()

    # Step 2: Reindex all documents
    print("\nStep 2: Reindexing all documents...")
    reindex_all()

    print("\n✓ Database cleared and all documents reindexed successfully!")


if __name__ == "__main__":
    main()

