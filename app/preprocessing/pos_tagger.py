import spacy

from app.monitoring.logger import logger


class POSTagger:
    """POS tagging using spaCy"""

    def __init__(self) -> None:
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found; POS tagging disabled")

    def tag_text(self, text: str) -> list[tuple[str, str]]:
        """Tag tokens with POS labels"""
        if not self.nlp:
            return []
        doc = self.nlp(text)
        return [(token.text, token.pos_) for token in doc]

    def get_pos_distribution(self, text: str) -> dict[str, int]:
        """Count POS tags in text"""
        if not self.nlp:
            return {}
        doc = self.nlp(text)
        pos_counts: dict[str, int] = {}
        for token in doc:
            pos = token.pos_
            pos_counts[pos] = pos_counts.get(pos, 0) + 1
        return pos_counts

    def extract_nouns_verbs(self, text: str) -> dict[str, list[str]]:
        """Extract nouns and verbs for enrichment"""
        if not self.nlp:
            return {"nouns": [], "verbs": []}
        doc = self.nlp(text)
        nouns = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN"]]
        verbs = [token.text for token in doc if token.pos_ == "VERB"]
        return {"nouns": nouns, "verbs": verbs}
