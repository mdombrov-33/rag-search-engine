from functools import lru_cache

from qdrant_client import QdrantClient

from app.config import get_settings
from app.services.embedding import EmbeddingService
from app.services.qdrant_init import QdrantInitializer
from app.services.query_enhancement import QueryEnhancementService
from app.services.reranking import RerankingService
from app.services.search import SearchService
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


@lru_cache()
def get_embedding_service() -> EmbeddingService:
    """Cached embedding service"""
    return EmbeddingService()


@lru_cache()
def get_query_enhancement_service() -> QueryEnhancementService:
    """Cached query enhancement service"""
    return QueryEnhancementService()


@lru_cache()
def get_reranking_service() -> RerankingService:
    """Cached reranking service"""
    return RerankingService()


@lru_cache()
def get_search_service() -> SearchService:
    """Cached search service"""
    vector_store = get_vector_store()
    embedding_service = get_embedding_service()
    query_enhancement_service = get_query_enhancement_service()
    reranking_service = get_reranking_service()
    return SearchService(
        vector_store=vector_store,
        embedding_service=embedding_service,
        query_enhancement_service=query_enhancement_service,
        reranking_service=reranking_service,
    )
