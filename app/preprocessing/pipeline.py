from typing import List

from app.config import get_settings
from app.models.document import Chunk
from app.monitoring.logger import logger
from app.preprocessing.chunking import ChunkingStrategy
from app.preprocessing.enrichment import ChunkEnricher
from app.preprocessing.text_cleaner import TextCleaner
from app.preprocessing.tokenizer import TokenizationPipeline

settings = get_settings()


class PreprocessingPipeline:
    """Full text preprocessing orchestration"""

    def __init__(self):
        self.cleaner = TextCleaner()
        self.tokenizer = TokenizationPipeline()
        # self.normalizer = TextNormalizer()  # Skip for semantic search. Deal with keywords later
        self.chunker = ChunkingStrategy()
        self.enricher = ChunkEnricher()

    def process_document(
        self,
        text: str,
        document_id: str,
        chunk_strategy: str = "recursive",
        normalize: bool = False,  # Default False for semantic search
    ) -> List[Chunk]:
        """
        Process document through full pipeline.

        Args:
            text: Raw document text
            document_id: Unique document identifier
            chunk_strategy: Chunking method (fixed, semantic, etc.)
            normalize: Whether to apply normalization (False for semantic)

        Returns:
            List of enriched chunks
        """
        # 1: Clean text (basic cleaning only for semantic)
        cleaned_text = self.cleaner.clean_text(text)
        # if normalize:  # Uncomment for keyword search later
        #     cleaned_text = self.normalizer.normalize(cleaned_text, strategy="lemmatize")

        # 2: Chunk based on strategy
        chunk_strategy = chunk_strategy or settings.CHUNK_STRATEGY
        chunks = self._apply_chunking(cleaned_text, document_id, chunk_strategy)

        # 3: Enrich chunks
        enriched_chunks = []
        for chunk in chunks:
            enriched_metadata = self.enricher.enrich_chunk(chunk.text, chunk.metadata)
            chunk.metadata = enriched_metadata
            enriched_chunks.append(chunk)

        logger.info(f"Processed document {document_id}: {len(enriched_chunks)} chunks")
        return enriched_chunks

    def _apply_chunking(self, text: str, document_id: str, strategy: str) -> List[Chunk]:
        """Apply selected chunking strategy"""
        if strategy == "fixed":
            return self.chunker.fixed_size_chunking(
                text, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP, document_id
            )
        elif strategy == "semantic":
            return self.chunker.semantic_chunking(text, document_id=document_id)
        elif strategy == "recursive":
            return self.chunker.recursive_character_chunking(
                text, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP, document_id
            )
        elif strategy == "sentence":
            return self.chunker.sentence_chunking(text, document_id)
        elif strategy == "sliding":
            return self.chunker.sliding_window_chunking(text, document_id=document_id)
        else:
            logger.warning(f"Unknown chunk strategy '{strategy}'; using fixed")
            return self.chunker.fixed_size_chunking(
                text, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP, document_id
            )
