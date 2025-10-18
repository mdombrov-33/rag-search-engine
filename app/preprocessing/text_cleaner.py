import re

from app.monitoring.logger import logger


class TextCleaner:
    """Handles basic text cleaning and validation"""

    def clean_text(self, text: str) -> str:
        """Basic cleaning for all text processing"""
        if not text.strip():
            logger.warning("Input text is empty or whitespace-only")
            return ""

        logger.info(f"Cleaning text: {len(text)} chars, first 200 chars: {text[:200]}")

        # Remove URLs
        text = re.sub(r"http\S+|www\S+", "", text)
        # Remove email addresses
        text = re.sub(r"\S+@\S+", "", text)
        # Fix OCR artifacts: add spaces between concatenated words (lowercase + uppercase)
        text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
        # Normalize whitespace
        text = " ".join(text.split())

        logger.info(f"Cleaned text: {len(text)} chars, first 200 chars: {text[:200]}")
        return text

    def validate_text(self, text: str, min_length: int = 10) -> bool:
        if len(text.strip()) < min_length:
            logger.warning(f"Text too short: {len(text)} chars (min: {min_length})")
            return False
        return True
