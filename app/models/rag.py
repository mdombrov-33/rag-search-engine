from pydantic import BaseModel, Field

from .search import SearchResult


class RAGRequest(BaseModel):
    """RAG generation request"""

    query: str = Field(..., min_length=1, max_length=500)
    limit: int = Field(default=5, ge=1, le=20, description="Chunks to retrieve")
    model: str = Field(default="gpt-5", description="LLM model")
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    rerank: bool = Field(default=True)
    compress_context: bool = Field(default=False, description="Apply context compression")

    class Config:
        schema_extra = {
            "example": {
                "query": "Explain how neural networks work",
                "limit": 5,
                "model": "gpt-5",
                "temperature": 0.3,
                "rerank": True,
                "compress_context": False,
            }
        }


class RAGResponse(BaseModel):
    """RAG generation response"""

    query: str
    answer: str
    sources: list[SearchResult]
    citations: dict[int, SearchResult]
    model: str
    generation_time_ms: int
    metadata: dict = {}

    class Config:
        schema_extra = {
            "example": {
                "query": "Explain neural networks",
                "answer": "Neural networks are computational models inspired by the brain [1]...",
                "sources": [],
                "citations": {},
                "model": "gpt-5",
                "generation_time_ms": 1250,
                "metadata": {"reranked": True, "total_chunks": 5},
            }
        }
