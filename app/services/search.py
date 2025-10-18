import time

from app.config import get_settings
from app.models.search import SearchRequest, SearchResponse
from app.monitoring.logger import logger
from app.services.embedding import EmbeddingService
from app.services.query_enhancement import QueryEnhancementService
from app.services.reranking import RerankingService
from app.services.vector_store import VectorStore

settings = get_settings()


class SearchService:
    """Orchestrates the complete search pipeline"""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_service: EmbeddingService,
        query_enhancement_service: QueryEnhancementService | None = None,
        reranking_service: RerankingService | None = None,
    ):
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.query_enhancement_service = query_enhancement_service
        self.reranking_service = reranking_service

    async def search(self, request: SearchRequest) -> SearchResponse:
        """
        Execute full search pipeline: enhance → embed → search → rerank.

        Args:
            request: SearchRequest with query and options

        Returns:
            SearchResponse with results and metadata
        """
        start_time = time.time()

        try:
            # 1. Query Enhancement
            enhanced_query = request.query
            if request.enhance_query and self.query_enhancement_service:
                enhanced_query = await self.query_enhancement_service.enhance_query(request.query)
                logger.info(f"Enhanced query: '{request.query}' → '{enhanced_query}'")

            # 2. Generate embedding
            query_vector = await self.embedding_service.embed(enhanced_query)
            logger.info("Generated query embedding")

            # 3. Vector search
            raw_results = self.vector_store.search(
                query_vector=query_vector,
                limit=request.limit * 2,  # Get more for reranking
                score_threshold=request.threshold,
                filters=request.filters,
            )

            # 4. Reranking
            final_results = raw_results
            if request.rerank and self.reranking_service:
                final_results = await self.reranking_service.rerank(
                    query=enhanced_query, results=raw_results, top_k=request.limit
                )

            # 5. Build response
            processing_time = int((time.time() - start_time) * 1000)

            return SearchResponse(
                query=request.query,
                enhanced_query=enhanced_query if request.enhance_query else None,
                results=final_results[: request.limit],
                total_results=len(final_results),
                search_type=request.search_type,
                processing_time_ms=processing_time,
                metadata={
                    "enhance_query": request.enhance_query,
                    "rerank": request.rerank,
                    "threshold": request.threshold,
                    "filters_applied": request.filters is not None,
                },
            )

        except Exception as e:
            processing_time = int((time.time() - start_time) * 1000)
            logger.error(f"Search failed: {e}")
            return SearchResponse(
                query=request.query,
                results=[],
                total_results=0,
                search_type=request.search_type,
                processing_time_ms=processing_time,
                metadata={"error": str(e)},
            )
