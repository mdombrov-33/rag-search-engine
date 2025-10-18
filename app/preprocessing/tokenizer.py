import spacy
from nltk.tokenize import sent_tokenize, word_tokenize  # type: ignore


class TokenizationPipeline:
    """Multi-strategy tokenization with linguistic analysis"""

    def __init__(self) -> None:
        self.nlp = spacy.load("en_core_web_sm")

    def tokenize_words_nltk(self, text: str) -> list[str]:
        """Basic word tokenization using NLTK"""
        return word_tokenize(text)  # type: ignore

    def tokenize_sentences_nltk(self, text: str) -> list[str]:
        """Sentence boundary detection"""
        return sent_tokenize(text)  # type: ignore

    def tokenize_spacy(self, text: str) -> list[str]:
        """Advanced tokenization with spaCy (handles contractions better)"""
        doc = self.nlp(text)
        return [token.text for token in doc]

    def get_pos_tags(self, text: str) -> list[tuple[str, str]]:
        """Extract Part-of-Speech tags"""
        doc = self.nlp(text)
        return [(token.text, token.pos_) for token in doc]

    def extract_entities(self, text: str) -> list[tuple[str, str]]:
        """Named Entity Recognition"""
        doc = self.nlp(text)
        return [(ent.text, ent.label_) for ent in doc.ents]

    def get_linguistic_features(self, text: str) -> dict:
        """Comprehensive linguistic analysis"""
        doc = self.nlp(text)
        return {
            "tokens": [token.text for token in doc],
            "lemmas": [token.lemma_ for token in doc],
            "pos_tags": [(token.text, token.pos_) for token in doc],
            "entities": [(ent.text, ent.label_) for ent in doc.ents],
            "noun_chunks": [chunk.text for chunk in doc.noun_chunks],
            "sentence_count": len(list(doc.sents)),
        }
