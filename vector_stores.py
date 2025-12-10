"""Vector store implementations."""

from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
import config
from interfaces import VectorStoreInterface


class QdrantVectorStore(VectorStoreInterface):
    """Qdrant vector database implementation."""

    def __init__(
        self,
        host: str = None,
        port: int = None,
        collection_name: str = None
    ):
        """
        Initialize Qdrant vector store.

        Args:
            host: Qdrant host (default from config)
            port: Qdrant port (default from config)
            collection_name: Collection name (default from config)
        """
        self.host = host or config.QDRANT_HOST
        self.port = port or config.QDRANT_PORT
        self.collection_name = collection_name or config.COLLECTION_NAME

        print(f"Connecting to Qdrant at {self.host}:{self.port}")
        self.client = QdrantClient(host=self.host, port=self.port)

    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        source_filter: str = None,
        score_threshold: float = None
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors in the database."""
        try:
            # Prepare search filter if specified
            search_filter = None
            if source_filter:
                search_filter = Filter(
                    must=[
                        FieldCondition(
                            key="source",
                            match=MatchValue(value=source_filter)
                        )
                    ]
                )

            # Search in Qdrant - first try with threshold if provided
            if score_threshold is not None:
                search_results = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    limit=top_k,
                    query_filter=search_filter,
                    with_payload=True,
                    score_threshold=score_threshold
                ).points

                # If no results found with threshold, try without threshold
                if len(search_results) == 0:
                    print(f"No results found with similarity threshold {score_threshold}, trying without threshold...")
                    search_results = self.client.query_points(
                        collection_name=self.collection_name,
                        query=query_vector,
                        limit=top_k,
                        query_filter=search_filter,
                        with_payload=True
                    ).points
            else:
                search_results = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    limit=top_k,
                    query_filter=search_filter,
                    with_payload=True
                ).points

            # Extract results
            results = []
            for hit in search_results:
                results.append({
                    'text': hit.payload['text'],
                    'source': hit.payload['source'],
                    'type': hit.payload['type'],
                    'score': hit.score
                })

            return results

        except Exception as e:
            print(f"Error searching vector store: {e}")
            import traceback
            traceback.print_exc()
            return []

    def verify_collection(self) -> bool:
        """Verify that the collection exists."""
        try:
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            return self.collection_name in collection_names
        except Exception as e:
            print(f"Error verifying collection: {e}")
            return False

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return {
                'name': self.collection_name,
                'points_count': collection_info.points_count,
                'status': 'active'
            }
        except Exception as e:
            print(f"Error getting collection info: {e}")
            return {'name': self.collection_name, 'error': str(e)}

    def list_sources(self) -> List[Dict[str, str]]:
        """List all available sources in the collection."""
        try:
            # Get a sample of points to find unique sources
            scroll_result = self.client.scroll(
                collection_name=self.collection_name,
                limit=1000,
                with_payload=True
            )

            sources = {}
            for point in scroll_result[0]:
                source = point.payload['source']
                doc_type = point.payload['type']
                if source not in sources:
                    sources[source] = doc_type

            return [
                {'source': source, 'type': doc_type}
                for source, doc_type in sorted(sources.items())
            ]

        except Exception as e:
            print(f"Error listing sources: {e}")
            return []


# Factory function to get vector store
def get_vector_store(store_type: str = "qdrant", **kwargs) -> VectorStoreInterface:
    """
    Factory function to get a vector store instance.

    Args:
        store_type: Type of vector store ("qdrant")
        **kwargs: Additional arguments for the store

    Returns:
        VectorStoreInterface implementation
    """
    if store_type == "qdrant":
        return QdrantVectorStore(**kwargs)
    else:
        raise ValueError(f"Unknown vector store type: {store_type}")

