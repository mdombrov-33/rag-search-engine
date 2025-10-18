import os
import tempfile

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.config import get_settings
from app.dependencies import get_document_service
from app.services.document_service import DocumentService

settings = get_settings()
router = APIRouter()


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),  # noqa: B008
    chunk_strategy: str = Form(
        settings.CHUNK_STRATEGY,
        description="Chunking strategy: 'semantic_sentences' (recommended), "
        "'fixed', 'recursive', 'sentence', 'paragraph', 'sliding'",
    ),
    chunk_size: int = Form(
        settings.CHUNK_SIZE,
        description="Target chunk size (words for semantic_sentences/fixed, "
        "characters for recursive)",
    ),
    chunk_overlap: int = Form(
        settings.CHUNK_OVERLAP, description="Overlap between chunks (words/characters)"
    ),
    document_service: DocumentService = Depends(get_document_service),  # noqa: B008
) -> dict:
    """
    Upload and process a document.

    Supports PDF, DOCX, TXT, CSV files up to 10MB.

    **Chunking Strategies:**
    - `semantic_sentences`: Smart sentence-based chunking (RECOMMENDED for coherent text segments)
    - `fixed`: Fixed-size word chunks with overlap
    - `recursive`: LangChain-style recursive character splitting
    - `sentence`: One sentence per chunk
    - `paragraph`: One paragraph per chunk
    - `sliding`: Sliding window over words
    """
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="Filename is required")

        if file.size and file.size > settings.MAX_FILE_SIZE:
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
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        result = await document_service.upload_document(
            temp_file_path, file.filename, chunk_strategy, chunk_size, chunk_overlap
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}") from e
    finally:
        if "temp_file_path" in locals():
            try:
                os.unlink(temp_file_path)
            except OSError:
                pass
