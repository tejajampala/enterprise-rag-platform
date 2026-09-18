"""Qdrant collection management and explicit dense/sparse search paths."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any

from qdrant_client import QdrantClient, models

from consumer_finance_rag.config import QdrantSettings
from consumer_finance_rag.models import Chunk, RetrievedChunk
from consumer_finance_rag.sparse import BM25SparseEncoder

POINT_NAMESPACE = uuid.UUID("73cf7db2-c6f4-43f0-b68d-6af773734caf")


class QdrantHybridStore:
    """One Qdrant collection with HNSW dense and sparse BM25 named vectors."""

    def __init__(self, settings: QdrantSettings, client: QdrantClient | None = None) -> None:
        self.settings = settings
        self.client = client or QdrantClient(url=settings.url)

    def recreate_collection(self, dense_size: int) -> None:
        """Create a fresh development collection.

        Dense vectors use Qdrant's HNSW index for ANN search. Sparse vectors use
        Qdrant's inverted sparse index and carry corpus-fitted BM25 weights.
        """

        if self.client.collection_exists(self.settings.collection_name):
            self.client.delete_collection(self.settings.collection_name)
        self.client.create_collection(
            collection_name=self.settings.collection_name,
            vectors_config={
                self.settings.dense_vector_name: models.VectorParams(
                    size=dense_size,
                    distance=models.Distance.COSINE,
                    hnsw_config=models.HnswConfigDiff(
                        m=self.settings.hnsw_m,
                        ef_construct=self.settings.hnsw_ef_construct,
                    ),
                )
            },
            sparse_vectors_config={
                self.settings.sparse_vector_name: models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=False),
                    modifier=models.Modifier.NONE,
                )
            },
        )

    def upsert(
        self,
        chunks: Sequence[Chunk],
        dense_vectors: Sequence[Sequence[float]],
        sparse_encoder: BM25SparseEncoder,
    ) -> None:
        if len(chunks) != len(dense_vectors):
            raise ValueError("chunks and dense_vectors must have equal lengths")
        points: list[models.PointStruct] = []
        for chunk, dense in zip(chunks, dense_vectors, strict=True):
            sparse = sparse_encoder.encode_document(chunk.text)
            payload = {
                "document_id": chunk.document_id,
                "chunk_id": chunk.chunk_id,
                "document_version": chunk.document_version,
                "title": chunk.metadata.get("title"),
                "category": chunk.metadata.get("category"),
                "source_url": chunk.metadata.get("source_url"),
                "source_id": chunk.metadata.get("source_id"),
                "chunk_hash": chunk.chunk_hash,
                "ordinal": chunk.ordinal,
                "text": chunk.text,
            }
            points.append(
                models.PointStruct(
                    id=str(uuid.uuid5(POINT_NAMESPACE, chunk.chunk_id)),
                    vector={
                        self.settings.dense_vector_name: list(dense),
                        self.settings.sparse_vector_name: models.SparseVector(
                            indices=sparse.indices, values=sparse.values
                        ),
                    },
                    payload=payload,
                )
            )
        self.client.upsert(collection_name=self.settings.collection_name, points=points, wait=True)

    def search_dense(self, vector: list[float], limit: int) -> list[RetrievedChunk]:
        result = self.client.query_points(
            collection_name=self.settings.collection_name,
            query=vector,
            using=self.settings.dense_vector_name,
            limit=limit,
            with_payload=True,
            search_params=models.SearchParams(hnsw_ef=max(limit * 2, 32), exact=False),
        )
        return self._to_retrieved(result.points, rank_kind="dense")

    def search_sparse(
        self, query: str, encoder: BM25SparseEncoder, limit: int
    ) -> list[RetrievedChunk]:
        sparse = encoder.encode_query(query)
        if not sparse.indices:
            return []
        result = self.client.query_points(
            collection_name=self.settings.collection_name,
            query=models.SparseVector(indices=sparse.indices, values=sparse.values),
            using=self.settings.sparse_vector_name,
            limit=limit,
            with_payload=True,
        )
        return self._to_retrieved(result.points, rank_kind="sparse")

    def indexed_chunk_hashes(self) -> dict[str, str]:
        """Read stable chunk identities from the live collection without loading vectors/text."""

        identities: dict[str, str] = {}
        offset: Any | None = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.settings.collection_name,
                limit=256,
                offset=offset,
                with_payload=["chunk_id", "chunk_hash"],
                with_vectors=False,
            )
            for point in points:
                payload = point.payload or {}
                chunk_id = payload.get("chunk_id")
                chunk_hash = payload.get("chunk_hash")
                if not isinstance(chunk_id, str) or not isinstance(chunk_hash, str):
                    raise ValueError("Qdrant point is missing string chunk_id/chunk_hash payloads")
                if chunk_id in identities:
                    raise ValueError(f"Qdrant collection contains duplicate chunk ID: {chunk_id}")
                identities[chunk_id] = chunk_hash
            if offset is None:
                return identities

    @staticmethod
    def _to_retrieved(points: Sequence[Any], *, rank_kind: str) -> list[RetrievedChunk]:
        retrieved: list[RetrievedChunk] = []
        for rank, point in enumerate(points, start=1):
            payload = point.payload or {}
            chunk = Chunk(
                chunk_id=str(payload["chunk_id"]),
                document_id=str(payload["document_id"]),
                document_version=str(payload["document_version"]),
                text=str(payload["text"]),
                ordinal=int(payload["ordinal"]),
                chunk_hash=str(payload["chunk_hash"]),
                metadata={
                    "title": payload.get("title"),
                    "category": payload.get("category"),
                    "source_url": payload.get("source_url"),
                    "source_id": payload.get("source_id"),
                },
            )
            retrieved.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=float(point.score),
                    dense_rank=rank if rank_kind == "dense" else None,
                    sparse_rank=rank if rank_kind == "sparse" else None,
                )
            )
        return retrieved
