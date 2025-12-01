"""Extract and process PDF files for AI learning."""

import os
from pathlib import Path
from typing import List
import warnings
warnings.filterwarnings('ignore')

from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid

import config


class PDFProcessor:
    """Process PDF files: extract text and store in vector DB."""

    def __init__(self):
        """Initialize PDF processor."""
        print("Initializing PDF Processor...")

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len,
        )

        # Initialize embeddings
        print(f"Loading embedding model: {config.EMBEDDING_MODEL}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL
        )

        # Initialize Qdrant client
        print(f"Connecting to Qdrant at {config.QDRANT_HOST}:{config.QDRANT_PORT}")
        self.qdrant_client = QdrantClient(
            host=config.QDRANT_HOST,
            port=config.QDRANT_PORT
        )

        # Create collection if it doesn't exist
        self._ensure_collection()

    def _ensure_collection(self):
        """Ensure Qdrant collection exists."""
        try:
            collections = self.qdrant_client.get_collections().collections
            collection_names = [c.name for c in collections]

            if config.COLLECTION_NAME not in collection_names:
                print(f"Creating collection: {config.COLLECTION_NAME}")
                self.qdrant_client.create_collection(
                    collection_name=config.COLLECTION_NAME,
                    vectors_config=VectorParams(
                        size=config.EMBEDDING_DIMENSION,
                        distance=Distance.COSINE
                    )
                )
            else:
                print(f"Collection {config.COLLECTION_NAME} already exists")
        except Exception as e:
            print(f"Error with Qdrant collection: {e}")
            print("Make sure Qdrant is running (docker-compose up -d)")
            import sys
            sys.exit(1)

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file."""
        try:
            print(f"Extracting text from: {pdf_path}")
            reader = PdfReader(pdf_path)

            text = []
            for page_num, page in enumerate(reader.pages, 1):
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)

            full_text = "\n".join(text)
            print(f"Extracted {len(reader.pages)} pages, {len(full_text)} characters")
            return full_text
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            return ""

    def process_and_store(self, pdf_name: str, text: str):
        """Process text and store in Qdrant."""
        try:
            # Split text into chunks
            chunks = self.text_splitter.split_text(text)
            print(f"Split text into {len(chunks)} chunks")

            # Generate embeddings and store in Qdrant
            points = []
            for idx, chunk in enumerate(chunks):
                # Generate embedding
                embedding = self.embeddings.embed_query(chunk)

                # Create point
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        "source": pdf_name,
                        "type": "pdf_document",
                        "chunk_index": idx,
                        "text": chunk
                    }
                )
                points.append(point)

            # Upload to Qdrant
            self.qdrant_client.upsert(
                collection_name=config.COLLECTION_NAME,
                points=points
            )
            print(f"Stored {len(points)} chunks in Qdrant")

        except Exception as e:
            print(f"Error processing and storing text: {e}")

    def process_pdf(self, pdf_path: str):
        """Complete pipeline: extract text and store."""
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            print(f"Error: PDF file not found: {pdf_path}")
            return

        print(f"\n{'='*60}")
        print(f"Processing PDF: {pdf_path.name}")
        print(f"{'='*60}")

        # Step 1: Extract text
        text = self.extract_text_from_pdf(str(pdf_path))
        if not text:
            print("Failed to extract text. Skipping PDF.")
            return

        # Step 2: Process and store in vector DB
        self.process_and_store(pdf_path.name, text)

        print(f"\n✓ Successfully processed: {pdf_path.name}\n")


def main():
    """Main function to process all PDFs in data directory."""
    # Get all PDF files
    data_path = Path(config.DATA_DIR)
    pdf_files = [f for f in data_path.iterdir() if f.suffix.lower() == '.pdf']

    if not pdf_files:
        print(f"No PDF files found in {config.DATA_DIR}")
        return

    print(f"Found {len(pdf_files)} PDF file(s) to process")

    # Initialize processor
    processor = PDFProcessor()

    # Process each PDF
    for pdf_file in pdf_files:
        try:
            processor.process_pdf(pdf_file)
        except Exception as e:
            print(f"Error processing {pdf_file.name}: {e}")
            continue

    print("\n" + "="*60)
    print("All PDFs processed!")
    print("="*60)


if __name__ == "__main__":
    main()

