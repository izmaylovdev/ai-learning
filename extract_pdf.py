import os
from pathlib import Path
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def extract_text_from_pdf(pdf_path: str) -> tuple[str, Dict]:
    """
    Extract text from a PDF file.

    Args:
        pdf_path (str): Path to the PDF file

    Returns:
        tuple: (extracted text content, metadata dict)
    """
    try:
        reader = PdfReader(pdf_path)
        text_content = []

        for page_num, page in enumerate(reader.pages, 1):
            text = page.extract_text()
            if text.strip():
                text_content.append(text)

        full_text = '\n'.join(text_content)

        metadata = {
            'source': os.path.basename(pdf_path),
            'file_path': pdf_path,
            'total_pages': len(reader.pages)
        }

        return full_text, metadata

    except Exception as e:
        print(f"Error processing {pdf_path}: {str(e)}")
        return None, None


def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """
    Split text into chunks using RecursiveCharacterTextSplitter.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = text_splitter.split_text(text)
    print(f"Created {len(chunks)} chunks from text")
    return chunks


def initialize_vector_store(collection_name: str = "pdf_documents", model_name: str = "all-MiniLM-L6-v2",
                            qdrant_url: str = "http://localhost:6333") -> tuple[QdrantClient, SentenceTransformer]:
    """
    Initialize or connect to Qdrant vector store.
    """
    # Initialize the embedding model
    model = SentenceTransformer(model_name)
    embedding_dim = model.get_sentence_embedding_dimension()

    # Initialize Qdrant client
    client = QdrantClient(url=qdrant_url)

    # Check if collection exists, if not create it
    collections = client.get_collections().collections
    collection_exists = any(col.name == collection_name for col in collections)

    if collection_exists:
        print(f"Connected to existing Qdrant collection: {collection_name}")
    else:
        print(f"Creating new Qdrant collection: {collection_name}")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE)
        )

    return client, model


def store_chunks_in_vector_db(
        chunks: List[str],
        metadata: Dict,
        collection_name: str = "pdf_documents",
        model_name: str = "all-MiniLM-L6-v2",
        qdrant_url: str = "http://localhost:6333"
) -> None:
    # Initialize the vector store
    client, model = initialize_vector_store(collection_name, model_name, qdrant_url)

    # Get current collection count for unique IDs
    collection_info = client.get_collection(collection_name)
    existing_count = collection_info.points_count

    # Generate embeddings for chunks
    print(f"Generating embeddings for {len(chunks)} chunks...")
    embeddings = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)

    # Prepare points for Qdrant
    points = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        chunk_metadata = metadata.copy()
        chunk_metadata['chunk_index'] = i
        chunk_metadata['chunk_total'] = len(chunks)
        chunk_metadata['text'] = chunk

        point = PointStruct(
            id=existing_count + i,
            vector=embedding.tolist(),
            payload=chunk_metadata
        )
        points.append(point)

    # Upload points to Qdrant
    client.upsert(
        collection_name=collection_name,
        points=points
    )

    print(f"Successfully stored {len(chunks)} chunks in Qdrant vector database")

    # Get updated count
    collection_info = client.get_collection(collection_name)
    print(f"Vector database now contains {collection_info.points_count} total documents")


def process_pdf_to_vector_db(
        pdf_path: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        collection_name: str = "pdf_documents",
        model_name: str = "all-MiniLM-L6-v2",
        qdrant_url: str = "http://localhost:6333"
) -> None:
    # Extract text from PDF
    text, metadata = extract_text_from_pdf(pdf_path)

    if text is None:
        print(f"Failed to extract text from {pdf_path}")
        return

    # Chunk the text
    chunks = chunk_text(text, chunk_size, chunk_overlap)

    # Store chunks in vector database
    store_chunks_in_vector_db(chunks, metadata, collection_name, model_name, qdrant_url)


def process_all_pdfs_in_folder(
        folder_path: str = "./data",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        collection_name: str = "pdf_documents",
        model_name: str = "all-MiniLM-L6-v2",
        qdrant_url: str = "http://localhost:6333"
) -> None:
    folder = Path(folder_path)
    pdf_files = list(folder.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {folder_path}")
        return

    print(f"\nFound {len(pdf_files)} PDF file(s) to process")
    print(f"{'=' * 80}\n")

    for pdf_file in pdf_files:
        print(f"\nProcessing: {pdf_file.name}")
        process_pdf_to_vector_db(
            str(pdf_file),
            chunk_size,
            chunk_overlap,
            collection_name,
            model_name,
            qdrant_url
        )

    print(f"\n{'=' * 80}")
    print("All PDFs processed successfully!")
    print(f"{'=' * 80}\n")


def query_vector_db(
        query_text: str,
        collection_name: str = "pdf_documents",
        model_name: str = "all-MiniLM-L6-v2",
        qdrant_url: str = "http://localhost:6333",
        n_results: int = 3
) -> List[Dict]:
    # Initialize the vector store
    client, model = initialize_vector_store(collection_name, model_name, qdrant_url)

    # Check if collection has documents
    collection_info = client.get_collection(collection_name)
    if collection_info.points_count == 0:
        print("No documents in the vector database!")
        return []

    # Generate embedding for the query
    query_embedding = model.encode([query_text], convert_to_numpy=True)[0]

    # Search the collection
    search_results = client.search(
        collection_name=collection_name,
        query_vector=query_embedding.tolist(),
        limit=n_results
    )

    print(f"\nQuery: {query_text}")
    print(f"{'=' * 80}\n")

    results = []
    for i, hit in enumerate(search_results):
        doc = hit.payload.get('text', '')
        metadata = {k: v for k, v in hit.payload.items() if k != 'text'}
        score = hit.score

        print(f"Result {i + 1} (Score: {score:.4f}):")
        print(f"Source: {metadata.get('source', 'Unknown')}")
        print(f"Chunk: {metadata.get('chunk_index', 0) + 1}/{metadata.get('chunk_total', 0)}")
        print(f"Content: {doc[:200]}...")
        print(f"{'-' * 80}\n")

        results.append({
            'document': doc,
            'metadata': metadata,
            'score': float(score)
        })

    return results


if __name__ == "__main__":
    # Process all PDFs in the data folder and store in Qdrant vector database
    print("Starting the extraction magic...")
    print("=" * 80)

    process_all_pdfs_in_folder(
        folder_path="./data",
        chunk_size=1000,
        chunk_overlap=200,
        collection_name="pdf_documents"
    )

    # Example query to test the RAG system
    print("\n" + "=" * 80)
    print("Testing RAG Query System")
    print("=" * 80)

    try:
        query_vector_db(
            query_text="What is RAG?",
            n_results=3
        )
    except Exception as e:
        print(f"Query test skipped (vector database might be empty): {e}")
