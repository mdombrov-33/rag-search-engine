from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration"""

    # Application
    APP_NAME: str = "RAG Search Engine"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    ENVIRONMENT: str = "development"  # development, staging, production
    LOG_LEVEL: str = "INFO"

    # API Keys
    OPENAI_API_KEY: str | None = None
    LANGFUSE_PUBLIC_KEY: str | None = None
    LANGFUSE_SECRET_KEY: str | None = None

    # LLM Configuration
    LLM_MODEL: str = "gpt-5"
    LLM_TEMPERATURE: float = 0.3
    MAX_TOKENS: int = 1000

    # Embeddings
    EMBEDDING_MODEL: str = "text-embedding-3-small"

    # Qdrant
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_COLLECTION: str = "documents"
    QDRANT_API_KEY: str | None = None
    VECTOR_SIZE: int = 1536
    DISTANCE_METRIC: str = "COSINE"  # COSINE, EUCLID, DOT

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
