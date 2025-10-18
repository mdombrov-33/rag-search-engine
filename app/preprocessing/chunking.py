import nltk  # type: ignore
import spacy
from sklearn.cluster import KMeans  # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore

from app.models.document import Chunk

nltk.download("punkt_tab", quiet=True)


class RecursiveCharacterTextSplitter:
    """Manual implementation of LangChain's RecursiveCharacterTextSplitter"""

    def __init__(
        self, chunk_size: int = 500, chunk_overlap: int = 100, separators: list[str] | None = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def create_documents(self, texts: list[str]) -> list[dict]:
        """Split text into documents"""
        documents = []
        for text in texts:
            chunks = self._split_text(text)
            for chunk in chunks:
                documents.append({"page_content": chunk})
        return documents

    def _split_text(self, text: str) -> list[str]:
        """Recursive splitting logic"""
        return self._recursive_split(text, self.separators[:])

    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        if not separators:
            return [text]

        separator = separators[0]
        if separator == "":
            return list(text)

        splits = text.split(separator)
        good_splits = []

        for split in splits:
            if len(split) <= self.chunk_size:
                good_splits.append(split)
            else:
                # Recurse with next separator
                good_splits.extend(self._recursive_split(split, separators[1:]))

        # Merge small chunks
        merged = []
        current = ""
        for split in good_splits:
            if len(current + split) <= self.chunk_size:
                current += split + separator if current else split
            else:
                if current:
                    merged.append(current.rstrip(separator))
                current = split

        if current:
            merged.append(current)

        return merged


class ChunkingStrategy:
    """Multiple chunking strategies using various libraries"""

    def __init__(self) -> None:
        # Load spaCy model
        self.nlp = spacy.load("en_core_web_sm")

    def fixed_size_chunking(
        self, text: str, chunk_size: int = 200, overlap: int = 50, document_id: str = ""
    ) -> list[Chunk]:
        """Fixed-size chunks with overlap using NLTK"""
        words = nltk.word_tokenize(text)
        chunks = []

        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i : i + chunk_size]
            chunk_text = " ".join(chunk_words)

            chunks.append(
                Chunk(
                    chunk_id=f"fixed_{i}",
                    document_id=document_id,
                    text=chunk_text,
                    chunk_index=i // (chunk_size - overlap),
                    chunk_type="fixed",
                    word_count=len(chunk_words),
                    sentence_count=len(nltk.sent_tokenize(chunk_text)),
                    metadata={"start_word": i, "end_word": i + chunk_size, "overlap_size": overlap},
                )
            )

        return chunks

    def recursive_character_chunking(
        self, text: str, chunk_size: int = 500, chunk_overlap: int = 100, document_id: str = ""
    ) -> list[Chunk]:
        """Recursive character chunking using manual implementation"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

        documents = text_splitter.create_documents([text])
        chunks = []

        for i, doc in enumerate(documents):
            chunk_text = doc["page_content"]

            chunks.append(
                Chunk(
                    chunk_id=f"recursive_{i}",
                    document_id=document_id,
                    text=chunk_text,
                    chunk_index=i,
                    chunk_type="recursive",
                    word_count=len(chunk_text.split()),
                    sentence_count=len(nltk.sent_tokenize(chunk_text)),
                    metadata={"chunk_size_target": chunk_size, "overlap": chunk_overlap},
                )
            )

        return chunks

    def sentence_chunking(self, text: str, document_id: str = "") -> list[Chunk]:
        """Sentence-based chunking using NLTK"""
        sentences = nltk.sent_tokenize(text)
        chunks = []

        for i, sentence in enumerate(sentences):
            chunks.append(
                Chunk(
                    chunk_id=f"sentence_{i}",
                    document_id=document_id,
                    text=sentence,
                    chunk_index=i,
                    chunk_type="sentence",
                    word_count=len(sentence.split()),
                    sentence_count=1,
                    metadata={},
                )
            )

        return chunks

    def document_structure_chunking(self, text: str, document_id: str = "") -> list[Chunk]:
        """Document-structured chunking using spaCy"""
        chunks = []

        for i, paragraph in enumerate(text.split("\n\n")):
            cleaned_para = paragraph.strip()
            if not cleaned_para:
                continue

            # Use spaCy to count sentences
            doc = self.nlp(cleaned_para)
            sentence_count = len(list(doc.sents))

            chunks.append(
                Chunk(
                    chunk_id=f"doc_struct_{i}",
                    document_id=document_id,
                    text=cleaned_para,
                    chunk_index=i,
                    chunk_type="document_structure",
                    word_count=len(cleaned_para.split()),
                    sentence_count=sentence_count,
                    metadata={"paragraph_number": i},
                )
            )

        return chunks

    def semantic_chunking(
        self, text: str, num_clusters: int = 3, document_id: str = ""
    ) -> list[Chunk]:
        """Simple semantic chunking using TF-IDF + K-means clustering"""
        sentences = nltk.sent_tokenize(text)

        if len(sentences) <= num_clusters:
            # Not enough sentences for clustering, return as-is
            return self.sentence_chunking(text, document_id)

        # Vectorize sentences
        vectorizer = TfidfVectorizer(stop_words="english")
        X = vectorizer.fit_transform(sentences)

        # Cluster sentences
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X)

        # Group sentences by cluster
        cluster_groups: dict[int, list[str]] = {}
        for i, cluster_id in enumerate(clusters):
            if cluster_id not in cluster_groups:
                cluster_groups[cluster_id] = []
            cluster_groups[cluster_id].append(sentences[i])

        # Create chunks from clusters
        chunks = []
        for cluster_id, cluster_sentences in cluster_groups.items():
            chunk_text = " ".join(cluster_sentences)

            chunks.append(
                Chunk(
                    chunk_id=f"semantic_{cluster_id}",
                    document_id=document_id,
                    text=chunk_text,
                    chunk_index=cluster_id,
                    chunk_type="semantic",
                    word_count=len(chunk_text.split()),
                    sentence_count=len(cluster_sentences),
                    metadata={"num_clusters": num_clusters, "cluster_method": "tfidf_kmeans"},
                )
            )

        return chunks

    def sliding_window_chunking(
        self, text: str, window_size: int = 150, step_size: int = 75, document_id: str = ""
    ) -> list[Chunk]:
        """Sliding window chunks (dense coverage, good for QA)"""
        words = text.split()
        chunks = []

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
                    sentence_count=len(nltk.sent_tokenize(chunk_text)),
                    metadata={
                        "window_start": i,
                        "window_end": i + window_size,
                        "step_size": step_size,
                    },
                )
            )

        return chunks
