from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    """Search request parameters"""

    query: str = Field(..., min_length=1, max_length=500, description="Search query")
    limit: int = Field(default=5, ge=1, le=50, description="Number of results")
    threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Minimum similarity score")
    search_type: str = Field(default="hybrid", pattern="^(bm25|semantic|hybrid)$")
    enhance_query: bool = Field(default=False, description="Apply query enhancement")
    rerank: bool = Field(default=True, description="Apply reranking")
    filters: dict | None = Field(default=None, description="Metadata filters")

    class Config:
        schema_extra = {
            "example": {
                "query": "What are the benefits of machine learning?",
                "limit": 5,
                "threshold": 0.7,
                "search_type": "hybrid",
                "enhance_query": True,
                "rerank": True,
            }
        }


class SearchResult(BaseModel):
    """Single search result"""

    chunk_id: str
    text: str
    score: float = Field(..., ge=0.0, le=1.0)
    metadata: dict

    class Config:
        schema_extra = {
            "example": {
                "chunk_id": "doc_123_chunk_5",
                "text": "Machine learning enables computers to learn from data...",
                "score": 0.95,
                "metadata": {"filename": "ml_intro.pdf", "chunk_index": 5, "search_type": "hybrid"},
            }
        }


class SearchResponse(BaseModel):
    """Search response with results"""

    query: str
    results: list[SearchResult]
    total_found: int
    search_time_ms: int
    metadata: dict = {}
