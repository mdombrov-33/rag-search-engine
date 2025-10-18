from functools import lru_cache

from qdrant_client import QdrantClient

from app.config import get_settings
from app.services.document_service import DocumentService  # Add this
from app.services.qdrant_init import QdrantInitializer

settings = get_settings()


@lru_cache()
def get_qdrant_client() -> QdrantClient:
    """Cached Qdrant client"""
    return QdrantClient(url=settings.QDRANT_URL, timeout=60, https=True, port=443)


@lru_cache()
def get_qdrant_initializer() -> QdrantInitializer:
    """Cached Qdrant initializer"""
    client = get_qdrant_client()
    return QdrantInitializer(client)


@lru_cache()
def get_document_service() -> DocumentService:
    """Cached document service"""
    return DocumentService()
