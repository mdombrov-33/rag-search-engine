from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class ContentType(str, Enum):
    """Supported file types"""

    PDF = "application/pdf"
    DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    TXT = "text/plain"
    CSV = "text/csv"


class DocumentMetadata(BaseModel):
    """Document metadata"""

    filename: str
    content_type: ContentType
    file_size: int
    upload_date: datetime = Field(default_factory=datetime.now)
    chunk_count: int = Field(default=0, ge=0)
    processing_time_ms: int = Field(default=0, ge=0)

    @field_validator("file_size")
    def validate_file_size(cls, v: int) -> int:
        max_size = 10_000_000  # 10MB
        if v > max_size:
            raise ValueError(f"File size exceeds {max_size} bytes")
        if v <= 0:
            raise ValueError("File size must be positive")
        return v


class Chunk(BaseModel):
    """Text chunk with metadata"""

    chunk_id: str
    document_id: str
    text: str
    chunk_index: int = Field(ge=0)
    chunk_type: str
    word_count: int = Field(gt=0)
    sentence_count: int = Field(gt=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("text")
    def text_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Chunk text cannot be empty")
        return v

    @field_validator("word_count", "sentence_count")
    def counts_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("Counts must be positive")
        return v
