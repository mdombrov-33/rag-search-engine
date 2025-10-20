from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import Distance, VectorParams

from app.config import get_settings
from app.monitoring.logger import logger

settings = get_settings()


class QdrantInitializer:
    """Initialize and manage Qdrant collections with robust error handling."""

    def __init__(self, client: QdrantClient):
        self.client = client
        self.collection_name = settings.QDRANT_COLLECTION
        self.vector_size = settings.VECTOR_SIZE
        self.distance = settings.distance

    def initialize_collection(self, recreate: bool = False) -> bool:
        """
        Initialize Qdrant collection.

        Args:
            recreate: If True, delete and recreate collection

        Returns:
            True if successful
        """
        try:
            exists = self._collection_exists()

            if exists and not recreate:
                logger.info(f"✓ Collection '{self.collection_name}' exists")
                self._log_collection_info()
                return True

            if exists and recreate:
                logger.warning(f"Recreating collection '{self.collection_name}'")
                try:
                    self.client.delete_collection(self.collection_name)
                except UnexpectedResponse as e:
                    logger.error(f"Qdrant API error while deleting collection: {e}")
                    return False

            logger.info(f"Creating collection '{self.collection_name}'...")
            try:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=self.distance,
                    ),
                )
            except UnexpectedResponse as e:
                logger.error(f"Qdrant API error while creating collection: {e}")
                return False

            logger.info(f"✓ Collection '{self.collection_name}' created successfully")
            return True

        except Exception as e:
            logger.error(f"✗ Failed to initialize collection: {e}")
            return False

    def _collection_exists(self) -> bool:
        """Check if collection exists"""
        try:
            collections = self.client.get_collections().collections
            return any(c.name == self.collection_name for c in collections)
        except UnexpectedResponse as e:
            logger.error(f"Qdrant API error while checking collection existence: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error while checking collection existence: {e}")
            return False

    def _log_collection_info(self) -> None:
        """Log collection information"""
        try:
            info = self.client.get_collection(self.collection_name)
            logger.info(f"  - Vectors: {info.vectors_count}")
            logger.info(f"  - Points: {info.points_count}")
            logger.info(f"  - Status: {info.status}")
        except UnexpectedResponse as e:
            logger.warning(f"Qdrant API error while retrieving collection info: {e}")
        except Exception as e:
            logger.warning(f"Could not retrieve collection info: {e}")

    def health_check(self) -> bool:
        """Check if Qdrant is accessible"""
        try:
            self.client.get_collections()
            logger.info("✓ Qdrant connection healthy")
            return True
        except UnexpectedResponse as e:
            logger.error(f"Qdrant API error during health check: {e}")
            return False
        except Exception as e:
            logger.error(f"Qdrant connection failed: {e}")
            return False

    def get_collection_stats(self) -> dict:
        """Get collection statistics"""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status.value if info.status else "unknown",
                "indexed_vectors_count": getattr(info, "indexed_vectors_count", None),
            }
        except UnexpectedResponse as e:
            logger.error(f"Qdrant API error while getting stats: {e}")
            return {}
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {}
