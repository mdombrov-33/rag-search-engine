import time
from typing import cast

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue, PointStruct

from app.config import get_settings
from app.models.search import AddDocumentsResponse, DeleteDocumentsResponse, SearchResult
from app.monitoring.logger import logger

settings = get_settings()


class VectorStore:
    """Qdrant vector store operations"""

    def __init__(self, client: QdrantClient | None = None):
        if client:
            self.client = client
        else:
            self.client = QdrantClient(
                url=settings.QDRANT_URL,
            )
        self.collection_name = settings.QDRANT_COLLECTION

    def add_documents(
        self, documents: list[dict], embeddings: list[list[float]]
    ) -> AddDocumentsResponse:
        """
        Add documents to vector store.

        Args:
            documents: List of document dictionaries with text and metadata
            embeddings: Corresponding embeddings

        Returns:
            AddDocumentsResponse with operation details
        """

        start_time = time.time()

        try:
            points = []
            for i, (doc, embedding) in enumerate(zip(documents, embeddings, strict=True)):
                point = PointStruct(
                    id=doc.get("document_id", f"doc_{i}"),
                    vector=embedding,
                    payload={
                        "text": doc.get("text", ""),
                        "document_id": doc.get("document_id", ""),
                        "metadata": doc.get("metadata", {}),
                    },
                )
                points.append(point)

            self.client.upsert(collection_name=self.collection_name, points=points)

            processing_time = int((time.time() - start_time) * 1000)
            total_points = self.count_documents()

            logger.info(f"✓ Added {len(points)} documents to vector store")
            return AddDocumentsResponse(
                success=True,
                documents_added=len(points),
                total_points=total_points,
                processing_time_ms=processing_time,
                errors=[],
            )

        except Exception as e:
            processing_time = int((time.time() - start_time) * 1000)
            error_msg = str(e)
            logger.error(f"✗ Failed to add documents: {error_msg}")
            return AddDocumentsResponse(
                success=False,
                documents_added=0,
                total_points=self.count_documents(),
                processing_time_ms=processing_time,
                errors=[error_msg],
            )

    def search(
        self,
        query_vector: list[float],
        limit: int = 10,
        score_threshold: float = 0.0,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        """
        Search for similar vectors.

        Args:
            query_vector: Query embedding
            limit: Max results
            score_threshold: Minimum similarity score
            filters: Metadata filters

        Returns:
            List of SearchResult objects
        """
        try:
            search_filter = None
            if filters:
                search_filter = self._build_filter(filters)

            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=search_filter,
            )

            search_results = []
            for result in results:
                payload = result.payload or {}
                search_results.append(
                    SearchResult(
                        chunk_id=str(result.id),
                        text=payload.get("text", ""),
                        score=float(result.score),
                        metadata=payload.get("metadata", {}),
                    )
                )

            return search_results

        except Exception as e:
            logger.error(f"✗ Search failed: {e}")
            return []

    def _build_filter(self, filters: dict) -> Filter:
        """Build Qdrant filter from dict"""
        conditions = []
        for key, value in filters.items():
            conditions.append(FieldCondition(key=f"metadata.{key}", match=MatchValue(value=value)))
        return Filter(must=cast(list, conditions))

    def delete_by_document_id(self, document_id: str) -> DeleteDocumentsResponse:
        """Delete all points belonging to a document"""
        import time

        start_time = time.time()

        try:
            points_before = self.count_documents()

            self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
                ),
            )

            processing_time = int((time.time() - start_time) * 1000)
            points_after = self.count_documents()
            points_deleted = points_before - points_after

            logger.info(f"✓ Deleted document: {document_id} ({points_deleted} points)")
            return DeleteDocumentsResponse(
                success=True,
                documents_deleted=1,  # we deleted one document (even if it had multiple chunks)
                total_points_remaining=points_after,
                processing_time_ms=processing_time,
                errors=[],
            )

        except Exception as e:
            processing_time = int((time.time() - start_time) * 1000)
            error_msg = str(e)
            logger.error(f"✗ Failed to delete document {document_id}: {error_msg}")
            return DeleteDocumentsResponse(
                success=False,
                documents_deleted=0,
                total_points_remaining=self.count_documents(),
                processing_time_ms=processing_time,
                errors=[error_msg],
            )

    def count_documents(self) -> int:
        """Get total document count"""
        try:
            info = self.client.get_collection(self.collection_name)
            return info.points_count or 0
        except Exception as e:
            logger.error(f"Error counting documents: {e}")
            return 0
