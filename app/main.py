import os
import tempfile

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.dependencies import get_document_service, get_qdrant_initializer
from app.services.document_service import DocumentService
from app.services.qdrant_init import QdrantInitializer

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


@app.post("/init-qdrant")
async def init_qdrant(
    initializer: QdrantInitializer = Depends(get_qdrant_initializer),  # noqa: B008
) -> dict:
    """Initialize Qdrant collection for testing"""
    success = initializer.initialize_collection(recreate=False)
    health = initializer.health_check()
    stats = initializer.get_collection_stats()
    return {"initialized": success, "healthy": health, "stats": stats}


@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),  # noqa: B008
    chunk_strategy: str = Form(default="semantic_sentences"),
    chunk_size: int = Form(default=300),
    chunk_overlap: int = Form(default=50),
    document_service: DocumentService = Depends(get_document_service),  # noqa: B008
) -> dict:
    """
    Upload and process a document.

    Supports PDF, DOCX, TXT, CSV files up to 10MB.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    if file.size is None or file.size > settings.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413, detail=f"File too large. Max size: {settings.MAX_FILE_SIZE} bytes"
        )

    allowed_extensions = [ext.strip(".") for ext in settings.SUPPORTED_EXTENSIONS]
    file_ext = file.filename.split(".")[-1].lower() if "." in file.filename else ""
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}",
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as temp_file:
        temp_file.write(await file.read())
        temp_file_path = temp_file.name

    try:
        result = await document_service.upload_document(
            temp_file_path, file.filename, chunk_strategy, chunk_size, chunk_overlap
        )
        return result
    finally:
        os.unlink(temp_file_path)
