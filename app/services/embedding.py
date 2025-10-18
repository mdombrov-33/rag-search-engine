from typing import List, Optional

from app.config import get_settings

settings = get_settings()


class EmbeddingService:
    async def embed(self, text: str, model: Optional[str] = None) -> List[float]:
        return [0.0] * settings.VECTOR_SIZE

    async def embed_batch(self, texts: List[str], model: Optional[str] = None) -> List[List[float]]:
        return [[0.0] * settings.VECTOR_SIZE for _ in texts]

    def get_available_models(self) -> List[str]:
        return ["text-embedding-3-small"]

    def get_model_dimensions(self, model: str) -> int:
        return settings.VECTOR_SIZE
