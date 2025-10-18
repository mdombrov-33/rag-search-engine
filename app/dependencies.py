# app/dependencies.py
from functools import lru_cache

from qdrant_client import QdrantClient

from app.config import get_settings
from app.services.qdrant_init import QdrantInitializer
from app.services.vector_store import VectorStore

settings = get_settings()


@lru_cache()
def get_qdrant_client() -> QdrantClient:
    """Cached Qdrant client"""
    return QdrantClient(url=settings.QDRANT_URL, timeout=60)


@lru_cache()
def get_qdrant_initializer() -> QdrantInitializer:
    """Cached Qdrant initializer"""
    client = get_qdrant_client()
    return QdrantInitializer(client)


@lru_cache()
def get_vector_store() -> VectorStore:
    """Cached vector store"""
    client = get_qdrant_client()
    return VectorStore(client)
