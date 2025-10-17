"""
Application configuration settings
"""

from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration"""

    # Application
    APP_NAME: str = "RAG Search Engine"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    ENVIRONMENT: str = "development"  # development, staging, production

    # API Keys
    OPENAI_API_KEY: Optional[str] = None
    LANGFUSE_PUBLIC_KEY: Optional[str] = None
    LANGFUSE_SECRET_KEY: Optional[str] = None

    # LLM Configuration
    LLM_MODEL: str = "gpt-5"
    LLM_TEMPERATURE: float = 0.3
    MAX_TOKENS: int = 1000

    # Embeddings
    EMBEDDING_PROVIDER: str = "local"  # 'local' or 'openai'
    EMBEDDING_MODEL: str = "local-minilm"  # Local: 'local-minilm'; OpenAI: 'text-embedding-3-small'

    # Qdrant
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_COLLECTION: str = "documents"
    QDRANT_API_KEY: Optional[str] = None
    VECTOR_SIZE: int = 384  # Depends on embedding model; 384 for local-minilm

    # LangFuse
    LANGFUSE_HOST: str = "https://cloud.langfuse.com"
    LANGFUSE_ENABLED: bool = True

    # Basic Processing (for document upload)
    CHUNK_SIZE: int = 200
    CHUNK_OVERLAP: int = 50
    CHUNK_STRATEGY: str = "semantic"  # fixed, semantic, recursive, sliding, paragraph
    MAX_FILE_SIZE: int = 10_000_000  # 10MB
    SUPPORTED_EXTENSIONS: list = [".pdf", ".docx", ".txt", ".csv"]

    # Placeholders for future features
    # DEFAULT_SEARCH_LIMIT: int = 5
    # DEFAULT_SEARCH_TYPE: str = "hybrid"  # bm25, semantic, hybrid
    # PROMETHEUS_PORT: int = 9090
    # LOG_LEVEL: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=True, extra="ignore"
    )

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == "production"

    @property
    def is_development(self) -> bool:
        return self.ENVIRONMENT == "development"


@lru_cache()
def get_settings() -> Settings:
    """Cached settings instance"""
    return Settings()
