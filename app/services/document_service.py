# app/services/document_service.py
import uuid
from pathlib import Path

import aiofiles
import pandas as pd
import pdfplumber
from docx import Document as DocxDocument

from app.config import get_settings
from app.models.document import ContentType, DocumentMetadata
from app.monitoring.logger import logger
from app.preprocessing.pipeline import PreprocessingPipeline
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStore

settings = get_settings()


class DocumentService:
    """Handles document upload, processing, and storage"""

    def __init__(self):
        self.pipeline = PreprocessingPipeline()
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStore()

    async def upload_document(self, file_path: str, filename: str) -> dict:
        """
        Upload and process a document.

        Args:
            file_path: Path to uploaded file
            filename: Original filename

        Returns:
            Processing result
        """
        document_id = str(uuid.uuid4())

        try:
            text = await self._extract_text(file_path, filename)
            if not text.strip():
                raise ValueError("No text extracted from document")

            logger.info(f"Extracted text for {filename}: {len(text)} chars")

            chunks = self.pipeline.process_document(text, document_id, normalize=False)

            chunk_texts = [chunk.text for chunk in chunks]
            logger.info(
                f"Processing {len(chunks)} chunks with texts: {[len(t) for t in chunk_texts]} chars"
            )
            embeddings = await self.embedding_service.embed_texts(chunk_texts)

            documents = [
                {
                    "id": str(uuid.uuid4()),
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "text": chunk.text,
                    "metadata": chunk.metadata,
                }
                for chunk in chunks
            ]

            response = self.vector_store.add_documents(documents, embeddings)

            metadata = DocumentMetadata(
                filename=filename,
                content_type=self._get_content_type(filename),
                file_size=Path(file_path).stat().st_size,
                chunk_count=len(chunks),
                processing_time_ms=response.processing_time_ms,  # Assuming this is in the response
            )

            logger.info(f"Uploaded document {document_id}: {len(chunks)} chunks")
            return {
                "document_id": document_id,
                "metadata": metadata,
                "chunks_added": len(chunks),
                "success": response.success,
            }

        except Exception as e:
            logger.error(f"Failed to upload document {filename}: {e}")
            raise

    async def _extract_text(self, file_path: str, filename: str) -> str:
        """Extract text from file based on extension"""
        ext = Path(filename).suffix.lower()

        if ext == ".pdf":
            return self._extract_pdf(file_path)
        elif ext == ".docx":
            return self._extract_docx(file_path)
        elif ext == ".txt":
            async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                return await f.read()
        elif ext == ".csv":
            return await self._extract_csv(file_path)  # Make async for consistency
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    def _extract_pdf(self, file_path: str) -> str:
        """Extract text from PDF using pdfplumber"""
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        logger.info(
            f"Extracted text from PDF {file_path}: {len(text)} chars, first 200 chars: {text[:200]}"
        )
        return text

    def _extract_docx(self, file_path: str) -> str:
        """Extract text from DOCX"""
        doc = DocxDocument(file_path)
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text

    async def _extract_csv(self, file_path: str) -> str:
        """Extract text from CSV using pandas"""
        try:
            df = pd.read_csv(file_path)
            # Convert DataFrame to readable text
            text_parts = []
            for col in df.columns:
                text_parts.append(f"{col}: {'; '.join(df[col].astype(str))}")
            return " | ".join(text_parts)
        except Exception as e:
            logger.warning(f"Failed to parse CSV with pandas, falling back to basic: {e}")
            # Fallback to basic text reading
            async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                content = await f.read()
                return content.replace(",", " ").replace("\n", " ")

    def _get_content_type(self, filename: str) -> ContentType:
        """Get MIME type enum from filename"""
        ext = Path(filename).suffix.lower()
        mapping = {
            ".pdf": ContentType.PDF,
            ".docx": ContentType.DOCX,
            ".txt": ContentType.TXT,
            ".csv": ContentType.CSV,
        }
        return mapping.get(ext, ContentType.TXT)  # default to TXT for unknown types
