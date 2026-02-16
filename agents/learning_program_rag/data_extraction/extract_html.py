"""Extract and process HTML files for AI learning."""

import sys
from pathlib import Path
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

# Add project root to path so we can import config
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid

import config
from util.embeddings import get_embedding_model


class HTMLProcessor:
    """Process HTML files: extract text and store in vector DB."""

    def __init__(self):
        """Initialize HTML processor."""
        print("Initializing HTML Processor...")

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len,
        )

        # Initialize embeddings using modular architecture
        self.embeddings = get_embedding_model("huggingface")

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
            sys.exit(1)

    def extract_text_from_html(self, html_path: str) -> tuple[str, Optional[str]]:
        """Extract text from HTML file.

        Returns:
            tuple: (extracted_text, title) - title may be None if not found
        """
        try:
            print(f"Extracting text from: {html_path}")

            with open(html_path, 'r', encoding='utf-8') as f:
                html_content = f.read()

            soup = BeautifulSoup(html_content, 'html.parser')

            # Extract title
            title = None
            title_tag = soup.find('title')
            if title_tag:
                title = title_tag.get_text(strip=True)

            # Remove script and style elements
            for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                element.decompose()

            # Extract text from body, or full document if no body
            body = soup.find('body')
            if body:
                text = body.get_text(separator='\n', strip=True)
            else:
                text = soup.get_text(separator='\n', strip=True)

            # Clean up whitespace
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            clean_text = '\n'.join(lines)

            print(f"Extracted {len(clean_text)} characters" + (f", title: '{title}'" if title else ""))
            return clean_text, title

        except Exception as e:
            print(f"Error extracting text from HTML: {e}")
            return "", None

    def process_and_store(self, html_name: str, text: str, title: Optional[str] = None):
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

                # Create point with metadata
                payload = {
                    "source": html_name,
                    "type": "html_document",
                    "chunk_index": idx,
                    "text": chunk
                }
                if title:
                    payload["title"] = title

                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload=payload
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

    def process_html(self, html_path: str):
        """Complete pipeline: extract text and store."""
        html_path = Path(html_path)
        if not html_path.exists():
            print(f"Error: HTML file not found: {html_path}")
            return

        print(f"\n{'='*60}")
        print(f"Processing HTML: {html_path.name}")
        print(f"{'='*60}")

        # Step 1: Extract text
        text, title = self.extract_text_from_html(str(html_path))
        if not text:
            print("Failed to extract text. Skipping HTML.")
            return

        # Step 2: Process and store in vector DB
        self.process_and_store(html_path.name, text, title)

        print(f"\n✓ Successfully processed: {html_path.name}\n")


def main():
    """Main function to process all HTML files in data directory."""
    # Get all HTML files
    data_path = Path(config.DATA_DIR)
    html_extensions = ['.html', '.htm']
    html_files = [
        f for f in data_path.iterdir()
        if f.is_file() and f.suffix.lower() in html_extensions
    ]

    if not html_files:
        print(f"No HTML files found in {config.DATA_DIR}")
        return

    print(f"Found {len(html_files)} HTML file(s) to process")

    # Initialize processor
    processor = HTMLProcessor()

    # Process each HTML file
    for html_file in html_files:
        try:
            processor.process_html(str(html_file))
        except Exception as e:
            print(f"Error processing {html_file.name}: {e}")
            continue

    print("\n" + "="*60)
    print("All HTML files processed!")
    print("="*60)


if __name__ == "__main__":
    main()

