"""Typed domain models shared by ingestion, retrieval, and generation."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Document(BaseModel):
    """A normalized source document with a deterministic identity and hash."""

    model_config = ConfigDict(frozen=True)

    document_id: str
    source_id: str
    title: str
    source_url: str | None = None
    category: str
    text: str
    source_version: str
    content_hash: str


class Chunk(BaseModel):
    """A citation-ready passage derived from a versioned document."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    document_id: str
    document_version: str
    text: str
    ordinal: int = Field(ge=0)
    chunk_hash: str
    metadata: dict[str, Any]


class RetrievedChunk(BaseModel):
    """A chunk plus scores accumulated during retrieval."""

    model_config = ConfigDict(frozen=True)

    chunk: Chunk
    score: float
    dense_rank: int | None = None
    sparse_rank: int | None = None
    rerank_score: float | None = None


class Citation(BaseModel):
    """A user-visible pointer from an answer back to evidence."""

    model_config = ConfigDict(frozen=True)

    citation_id: str
    document_title: str
    source_url: str | None
    chunk_id: str


class StructuredContext(BaseModel):
    """Prompt-ready evidence and its citation catalogue."""

    model_config = ConfigDict(frozen=True)

    text: str
    citations: list[Citation]
    chunks: list[RetrievedChunk]


class RetrievalMetadata(BaseModel):
    """Inspectable retrieval settings and actual candidate counts."""

    model_config = ConfigDict(frozen=True)

    strategy: str = "Qdrant dense HNSW/ANN + sparse BM25, RRF, reranking"
    dense_candidates: int
    sparse_candidates: int
    fused_candidates: int
    reranked: int
    final_contexts: int


class ModelMetadata(BaseModel):
    """Models used for one answer."""

    model_config = ConfigDict(frozen=True)

    embedding_model: str
    reranker_model: str
    generation_model: str


class RAGResponse(BaseModel):
    """Complete response returned by the application service."""

    model_config = ConfigDict(frozen=True)

    question: str
    answer: str
    citations: list[Citation]
    retrieved_chunks: list[RetrievedChunk]
    retrieval_metadata: RetrievalMetadata
    model_metadata: ModelMetadata
    generation_metadata: dict[str, Any] = Field(default_factory=dict)
