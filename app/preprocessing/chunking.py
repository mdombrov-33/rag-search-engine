import nltk  # type: ignore
import spacy

from app.models.document import Chunk
from app.monitoring.logger import logger
from app.preprocessing.tokenizer import TokenizationPipeline

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


class ChunkingStrategy:
    """
    Multiple text chunking strategies.
    Each strategy preserves text integrity and semantic boundaries.
    """

    def __init__(self) -> None:
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("✓ spaCy model loaded")
        except OSError:
            logger.warning("spaCy model not found. Run: python -m spacy download en_core_web_sm")

        self.tokenizer = TokenizationPipeline()

    def chunk_text(
        self,
        text: str,
        strategy: str = "semantic_sentences",
        document_id: str = "",
        chunk_size: int = 300,
        chunk_overlap: int = 50,
    ) -> list[Chunk]:
        """
        Unified interface for all chunking strategies.

        Args:
            text: Text to chunk
            strategy: Chunking strategy name
            document_id: Document identifier
            chunk_size: Target size (meaning varies by strategy)
            chunk_overlap: Overlap size (for applicable strategies)

        Returns:
            List of Chunk objects

        Strategies:
            - 'semantic_sentences': Smart sentence-based chunking (RECOMMENDED)
            - 'fixed': Fixed-size word chunks with overlap
            - 'recursive': LangChain-style recursive splitting
            - 'sentence': One sentence per chunk
            - 'paragraph': One paragraph per chunk
            - 'sliding': Sliding window over words
        """
        logger.info(f"Chunking with strategy='{strategy}', size={chunk_size}")

        if strategy == "semantic_sentences":
            return self.semantic_sentence_chunking(text, chunk_size, document_id)
        elif strategy == "fixed":
            return self.fixed_size_chunking(text, chunk_size, chunk_overlap, document_id)
        elif strategy == "recursive":
            return self.recursive_character_chunking(text, chunk_size, chunk_overlap, document_id)
        elif strategy == "sentence":
            return self.sentence_chunking(text, document_id)
        elif strategy == "paragraph":
            return self.paragraph_chunking(text, document_id)
        elif strategy == "sliding":
            step = max(1, chunk_size - chunk_overlap)
            return self.sliding_window_chunking(text, chunk_size, step, document_id)
        else:
            logger.warning(f"Unknown strategy '{strategy}', defaulting to semantic_sentences")
            return self.semantic_sentence_chunking(text, chunk_size, document_id)

    def semantic_sentence_chunking(
        self, text: str, max_words: int = 300, document_id: str = ""
    ) -> list[Chunk]:
        """
        RECOMMENDED: Smart sentence-based chunking.
        Groups sentences until reaching max_words, preserving sentence boundaries.

        Best for: Resumes, articles, structured documents
        """
        sentences = self.tokenizer.tokenize_sentences_nltk(text)
        chunks = []

        current_sentences: list[str] = []
        current_word_count = 0
        chunk_index = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sentence_words = len(self.tokenizer.tokenize_words_nltk(sentence))

            if current_word_count + sentence_words > max_words and current_sentences:
                chunk_text = " ".join(current_sentences)

                chunks.append(
                    Chunk(
                        chunk_id=f"semantic_{chunk_index}",
                        document_id=document_id,
                        text=chunk_text,
                        chunk_index=chunk_index,
                        chunk_type="semantic_sentences",
                        word_count=current_word_count,
                        sentence_count=len(current_sentences),
                        metadata={
                            "max_words": max_words,
                            "strategy": "semantic_sentences",
                        },
                    )
                )

                chunk_index += 1
                current_sentences = []
                current_word_count = 0

            current_sentences.append(sentence)
            current_word_count += sentence_words

        if current_sentences:
            chunk_text = " ".join(current_sentences)
            chunks.append(
                Chunk(
                    chunk_id=f"semantic_{chunk_index}",
                    document_id=document_id,
                    text=chunk_text,
                    chunk_index=chunk_index,
                    chunk_type="semantic_sentences",
                    word_count=current_word_count,
                    sentence_count=len(current_sentences),
                    metadata={
                        "max_words": max_words,
                        "strategy": "semantic_sentences",
                    },
                )
            )

        logger.info(f"✓ Created {len(chunks)} semantic sentence chunks")
        return chunks

    def fixed_size_chunking(
        self, text: str, chunk_size: int = 200, overlap: int = 50, document_id: str = ""
    ) -> list[Chunk]:
        """
        Fixed-size word chunks with overlap.

        Best for: When you need consistent chunk sizes
        """
        words = self.tokenizer.tokenize_words_nltk(text)
        chunks = []

        if chunk_size <= overlap:
            logger.warning(f"chunk_size ({chunk_size}) <= overlap ({overlap}), adjusting")
            overlap = max(0, chunk_size - 1)

        step = max(1, chunk_size - overlap)

        for i in range(0, len(words), step):
            chunk_words = words[i : i + chunk_size]
            chunk_text = " ".join(chunk_words)

            chunks.append(
                Chunk(
                    chunk_id=f"fixed_{i}",
                    document_id=document_id,
                    text=chunk_text,
                    chunk_index=i // step,
                    chunk_type="fixed",
                    word_count=len(chunk_words),
                    sentence_count=len(self.tokenizer.tokenize_sentences_nltk(chunk_text)),
                    metadata={
                        "start_word": i,
                        "end_word": i + chunk_size,
                        "overlap_size": overlap,
                        "strategy": "fixed",
                    },
                )
            )

        logger.info(f"✓ Created {len(chunks)} fixed-size chunks")
        return chunks

    def recursive_character_chunking(
        self, text: str, chunk_size: int = 500, chunk_overlap: int = 50, document_id: str = ""
    ) -> list[Chunk]:
        """
        Recursive character-based splitting (LangChain-style).
        Tries to split at natural boundaries (paragraphs, sentences, spaces).

        Best for: Long documents with varied structure
        """
        separators = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]

        splits = self._recursive_split(text, separators, chunk_size)
        merged_chunks = self._merge_splits_with_overlap(splits, chunk_size, chunk_overlap)

        chunks = []
        for i, chunk_text in enumerate(merged_chunks):
            chunks.append(
                Chunk(
                    chunk_id=f"recursive_{i}",
                    document_id=document_id,
                    text=chunk_text,
                    chunk_index=i,
                    chunk_type="recursive",
                    word_count=len(self.tokenizer.tokenize_words_nltk(chunk_text)),
                    sentence_count=len(self.tokenizer.tokenize_sentences_nltk(chunk_text)),
                    metadata={
                        "chunk_size_target": chunk_size,
                        "overlap": chunk_overlap,
                        "strategy": "recursive",
                    },
                )
            )

        logger.info(f"✓ Created {len(chunks)} recursive chunks")
        return chunks

    def _recursive_split(self, text: str, separators: list[str], chunk_size: int) -> list[str]:
        """Recursively split text using separators"""
        if not separators or len(text) <= chunk_size:
            return [text] if text.strip() else []

        separator = separators[0]
        remaining_separators = separators[1:]

        splits = text.split(separator) if separator else [text]

        good_splits = []
        for split in splits:
            split = split.strip()
            if not split:
                continue

            if len(split) > chunk_size and remaining_separators:
                good_splits.extend(self._recursive_split(split, remaining_separators, chunk_size))
            else:
                good_splits.append(split)

        return good_splits

    def _merge_splits_with_overlap(
        self, splits: list[str], chunk_size: int, chunk_overlap: int
    ) -> list[str]:
        """Merge splits into chunks respecting size and overlap"""
        chunks: list[str] = []
        current_chunk: list[str] = []
        current_length = 0

        for split in splits:
            split_length = len(split)

            if split_length > chunk_size:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_length = 0
                chunks.append(split)
                continue

            if current_length + split_length + len(current_chunk) > chunk_size:
                if current_chunk:
                    chunk_text = " ".join(current_chunk)
                    chunks.append(chunk_text)

                    if chunk_overlap > 0:
                        overlap_text = chunk_text[-chunk_overlap:]
                        current_chunk = [overlap_text, split]
                        current_length = len(overlap_text) + split_length
                    else:
                        current_chunk = [split]
                        current_length = split_length
                else:
                    current_chunk = [split]
                    current_length = split_length
            else:
                current_chunk.append(split)
                current_length += split_length

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def sentence_chunking(self, text: str, document_id: str = "") -> list[Chunk]:
        """
        One sentence per chunk.

        Best for: Fine-grained search, Q&A systems
        """
        sentences = self.tokenizer.tokenize_sentences_nltk(text)
        chunks = []

        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue

            chunks.append(
                Chunk(
                    chunk_id=f"sentence_{i}",
                    document_id=document_id,
                    text=sentence,
                    chunk_index=i,
                    chunk_type="sentence",
                    word_count=len(self.tokenizer.tokenize_words_nltk(sentence)),
                    sentence_count=1,
                    metadata={"strategy": "sentence"},
                )
            )

        logger.info(f"✓ Created {len(chunks)} sentence chunks")
        return chunks

    def paragraph_chunking(self, text: str, document_id: str = "") -> list[Chunk]:
        """
        One paragraph per chunk.

        Best for: Structured documents with clear paragraphs
        """
        paragraphs = text.split("\n\n")
        chunks = []
        chunk_index = 0

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            if self.nlp:
                doc = self.nlp(paragraph)
                sentence_count = len(list(doc.sents))
            else:
                sentence_count = len(self.tokenizer.tokenize_sentences_nltk(paragraph))

            chunks.append(
                Chunk(
                    chunk_id=f"paragraph_{chunk_index}",
                    document_id=document_id,
                    text=paragraph,
                    chunk_index=chunk_index,
                    chunk_type="paragraph",
                    word_count=len(self.tokenizer.tokenize_words_nltk(paragraph)),
                    sentence_count=sentence_count,
                    metadata={"strategy": "paragraph"},
                )
            )
            chunk_index += 1

        logger.info(f"✓ Created {len(chunks)} paragraph chunks")
        return chunks

    def sliding_window_chunking(
        self, text: str, window_size: int = 150, step_size: int = 75, document_id: str = ""
    ) -> list[Chunk]:
        """
        Sliding window over words.

        Best for: Dense coverage, when context overlap is critical
        """
        words = self.tokenizer.tokenize_words_nltk(text)
        chunks = []

        if len(words) < window_size:
            return [
                Chunk(
                    chunk_id="sliding_0",
                    document_id=document_id,
                    text=text,
                    chunk_index=0,
                    chunk_type="sliding",
                    word_count=len(words),
                    sentence_count=len(self.tokenizer.tokenize_sentences_nltk(text)),
                    metadata={
                        "window_size": window_size,
                        "step_size": step_size,
                        "strategy": "sliding",
                    },
                )
            ]

        for i in range(0, len(words) - window_size + 1, step_size):
            chunk_words = words[i : i + window_size]
            chunk_text = " ".join(chunk_words)

            chunks.append(
                Chunk(
                    chunk_id=f"sliding_{i}",
                    document_id=document_id,
                    text=chunk_text,
                    chunk_index=i // step_size,
                    chunk_type="sliding",
                    word_count=len(chunk_words),
                    sentence_count=len(self.tokenizer.tokenize_sentences_nltk(chunk_text)),
                    metadata={
                        "window_start": i,
                        "window_end": i + window_size,
                        "step_size": step_size,
                        "strategy": "sliding",
                    },
                )
            )

        logger.info(f"✓ Created {len(chunks)} sliding window chunks")
        return chunks


# Convenience function for backward compatibility
def get_chunking_strategy() -> ChunkingStrategy:
    """Get a configured ChunkingStrategy instance"""
    return ChunkingStrategy()
