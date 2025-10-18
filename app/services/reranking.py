from typing import List

from app.models.search import SearchResult


class RerankingService:
    async def rerank(
        self, query: str, results: List[SearchResult], top_k: int = 10
    ) -> List[SearchResult]:
        return results[:top_k]

    async def rerank_with_cross_encoder(
        self,
        query: str,
        results: List[SearchResult],
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> List[SearchResult]:
        return results

    def get_available_models(self) -> List[str]:
        return [
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "cross-encoder/ms-marco-TinyBERT-L-2-v2",
            "cross-encoder/qnli-electra-base",
        ]
