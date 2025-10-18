from typing import List

from nltk.corpus import stopwords  # type: ignore
from nltk.stem import PorterStemmer, WordNetLemmatizer  # type: ignore

from app.preprocessing.text_cleaner import TextCleaner
from app.preprocessing.tokenizer import TokenizationPipeline


class TextNormalizer:
    """Text normalization with multiple strategies (for keyword search; skipped for semantic)"""

    def __init__(self, language: str = "english"):
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stopwords = set(stopwords.words(language))
        self.custom_stopwords: set[str] = set()
        self.cleaner = TextCleaner()

    # Remove clean_text method (moved to TextCleaner)

    def lowercase(self, tokens: List[str]) -> List[str]:
        """Convert to lowercase"""
        return [t.lower() for t in tokens]

    def lemmatize(self, tokens: List[str], pos_tags: List[str] | None = None) -> List[str]:
        """Lemmatization"""
        if pos_tags:
            # Context-aware lemmatization using POS tags
            return [
                self.lemmatizer.lemmatize(token, self._get_wordnet_pos(pos))
                for token, pos in zip(tokens, pos_tags, strict=False)
            ]
        return [self.lemmatizer.lemmatize(t) for t in tokens]

    def stem(self, tokens: List[str]) -> List[str]:
        """Stemming"""
        return [self.stemmer.stem(t) for t in tokens]

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove common words"""
        all_stopwords = self.stopwords | self.custom_stopwords
        return [t for t in tokens if t.lower() not in all_stopwords]

    def add_custom_stopwords(self, words: List[str]):
        """Add domain-specific stopwords"""
        self.custom_stopwords.update(words)

    def normalize_pipeline(
        self, text: str, lemmatize: bool = True, remove_stops: bool = True
    ) -> List[str]:
        """Complete normalization pipeline (for keyword search)"""
        text = self.cleaner.clean_text(text)

        tokenizer = TokenizationPipeline()
        tokens = tokenizer.tokenize_words_nltk(text)

        tokens = self.lowercase(tokens)

        if remove_stops:
            tokens = self.remove_stopwords(tokens)

        if lemmatize:
            tokens = self.lemmatize(tokens)

        return tokens

    @staticmethod
    def _get_wordnet_pos(treebank_tag: str) -> str:
        """Convert Penn Treebank POS tag to WordNet POS tag"""
        if treebank_tag.startswith("J"):
            return "a"  # adjective
        elif treebank_tag.startswith("V"):
            return "v"  # verb
        elif treebank_tag.startswith("N"):
            return "n"  # noun
        elif treebank_tag.startswith("R"):
            return "r"  # adverb
        else:
            return "n"  # default to noun
