import re

from app.monitoring.logger import logger


class TextCleaner:
    """Handles basic text cleaning and validation"""

    def clean_text(self, text: str) -> str:
        """Basic cleaning for all text processing"""
        if not text.strip():
            logger.warning("Input text is empty or whitespace-only")
            return ""

        # Remove URLs
        text = re.sub(r"http\S+|www\S+", "", text)
        # Remove email addresses
        text = re.sub(r"\S+@\S+", "", text)
        # Remove special characters but keep sentence structure
        text = re.sub(r"[^\w\s.,!?;:-]", "", text)
        # Normalize whitespace
        text = " ".join(text.split())
        return text

    def validate_text(self, text: str, min_length: int = 10) -> bool:
        """Validate text for processing"""
        if len(text.strip()) < min_length:
            logger.warning(f"Text too short: {len(text)} chars (min: {min_length})")
            return False
        return True
