from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.dependencies import get_qdrant_initializer, get_vector_store
from app.services.qdrant_init import QdrantInitializer
from app.services.vector_store import VectorStore

settings = get_settings()

app = FastAPI(
    title="RAG Search Engine",
    description="Production-grade RAG system with advanced NLP",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check() -> dict:
    """Health check endpoint."""
    return {"status": "healthy", "version": settings.VERSION}


@app.get("/vector-store-stats")
async def vector_store_stats(
    vector_store: VectorStore = Depends(get_vector_store),  # noqa: B008
) -> dict:
    """Get vector store statistics"""
    count = vector_store.count_documents()
    return {"document_count": count}


@app.post("/init-qdrant")
async def init_qdrant(
    initializer: QdrantInitializer = Depends(get_qdrant_initializer),  # noqa: B008
) -> dict:
    """Initialize Qdrant collection for testing"""
    success = initializer.initialize_collection(recreate=False)
    health = initializer.health_check()
    stats = initializer.get_collection_stats()
    return {"initialized": success, "healthy": health, "stats": stats}
