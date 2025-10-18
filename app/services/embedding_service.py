from typing import List

import tiktoken
from openai import OpenAI

from app.config import get_settings
from app.monitoring.logger import logger

settings = get_settings()


class EmbeddingService:
    """Handles text embedding using OpenAI"""

    def __init__(self) -> None:
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.EMBEDDING_MODEL or "text-embedding-ada-002"
        self.tokenizer = tiktoken.encoding_for_model(self.model)
        self.max_tokens = 8000  # Adjust based on model limits, safe limit under 8192

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts, batched by token limit.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        all_embeddings = []

        batches = self._batch_texts_by_tokens(texts)

        for batch in batches:
            try:
                response = self.client.embeddings.create(input=batch, model=self.model)
                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)
                logger.info(f"Embedded batch of {len(batch)} texts")
            except Exception as e:
                logger.error(f"Failed to embed batch: {e}")
                raise

        logger.info(f"Generated {len(all_embeddings)} embeddings total")
        return all_embeddings

    def _batch_texts_by_tokens(self, texts: List[str]) -> List[List[str]]:
        """Split texts into batches under token limit"""
        batches: list[list[str]] = []
        current_batch: list[str] = []
        current_tokens = 0

        for text in texts:
            text_tokens = len(self.tokenizer.encode(text))
            if current_tokens + text_tokens > self.max_tokens:
                if current_batch:
                    batches.append(current_batch)
                    current_batch = []
                    current_tokens = 0
                if text_tokens > self.max_tokens:
                    logger.warning(f"Text exceeds token limit: {text_tokens} tokens")
            current_batch.append(text)
            current_tokens += text_tokens

        if current_batch:
            batches.append(current_batch)

        return batches

    async def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text string to embed

        Returns:
            Embedding vector
        """
        embeddings = await self.embed_texts([text])
        return embeddings[0] if embeddings else []

    def get_embedding_dimensions(self) -> int:
        """Get the dimensionality of embeddings for this model"""
        model_dims = {
            "text-embedding-ada-002": 1536,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
        }
        return model_dims.get(self.model, 1536)
