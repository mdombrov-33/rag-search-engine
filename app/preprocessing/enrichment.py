import re
from collections import Counter
from typing import Any

from app.monitoring.logger import logger
from app.preprocessing.pos_tagger import POSTagger


class ChunkEnricher:
    """Enrich chunks with linguistic and semantic metadata"""

    def __init__(self):
        self.pos_tagger = POSTagger()
        if self.pos_tagger.nlp:
            self.pos_tagger.nlp.max_length = 2000000

    def enrich_chunk(self, chunk_text: str, chunk_metadata: dict[str, Any]) -> dict[str, Any]:
        """Enrich a single chunk with metadata"""
        enriched = chunk_metadata.copy()

        # POS-based enrichment
        pos_dist = self.pos_tagger.get_pos_distribution(chunk_text)
        enriched["pos_distribution"] = pos_dist

        nouns_verbs = self.pos_tagger.extract_nouns_verbs(chunk_text)
        enriched["key_nouns"] = nouns_verbs["nouns"][:10]  # Top 10 nouns
        enriched["key_verbs"] = nouns_verbs["verbs"][:5]  # Top 5 verbs

        # Text statistics
        enriched["has_questions"] = "?" in chunk_text
        enriched["has_numbers"] = bool(re.search(r"\d", chunk_text))
        enriched["capitalized_words"] = len([w for w in chunk_text.split() if w.istitle()])

        # Keyword extraction (simple frequency-based)
        words = re.findall(r"\b\w+\b", chunk_text.lower())
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
        }
        keywords = [w for w in words if w not in stop_words and len(w) > 3]
        enriched["keywords"] = [k for k, _ in Counter(keywords).most_common(10)]

        # Readability score (simple heuristic)
        avg_word_len = sum(len(w) for w in words) / len(words) if words else 0
        enriched["avg_word_length"] = round(avg_word_len, 2)

        return enriched

    def enrich_chunks(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Enrich multiple chunks"""
        enriched_chunks = []
        for chunk in chunks:
            try:
                enriched_metadata = self.enrich_chunk(chunk["text"], chunk.get("metadata", {}))
                chunk["metadata"] = enriched_metadata
                enriched_chunks.append(chunk)
            except Exception as e:
                logger.error(f"Failed to enrich chunk {chunk.get('chunk_id')}: {e}")
                enriched_chunks.append(chunk)  # return original if enrichment fails
        return enriched_chunks
