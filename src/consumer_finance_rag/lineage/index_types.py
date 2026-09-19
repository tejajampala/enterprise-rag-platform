"""Immutable textual-corpus types used before managed index construction."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any
from uuid import UUID

from consumer_finance_rag.sparse import BM25State


@dataclass(frozen=True, slots=True)
class ActiveDocumentVersion:
    """Latest immutable version of one ACTIVE logical source document."""

    document_pk: UUID
    document_version_id: UUID
    version_number: int
    source_key: str
    source_id: str
    source_url: str | None
    title: str
    category: str
    source_version: str
    normalized_text: str
    content_hash: str
    metadata_hash: str
    version_hash: str
    compatibility_document_id: str


@dataclass(frozen=True, slots=True)
class ActiveChunk:
    """One citation-ready active chunk without vector or sparse-index data."""

    document_pk: UUID
    document_version_id: UUID
    version_number: int
    source_key: str
    source_id: str
    source_url: str | None
    title: str
    category: str
    source_version: str
    content_hash: str
    compatibility_document_id: str
    chunk_pk: UUID
    compatibility_chunk_id: str
    ordinal: int
    chunk_hash: str
    text: str
    chunking_fingerprint: str


@dataclass(frozen=True, slots=True)
class ActiveChunkManifest:
    """Exact deterministic active textual corpus requested for a future index."""

    corpus_id: UUID
    corpus_key: str
    source_snapshot_hash: str
    chunking_fingerprint: str
    documents: tuple[ActiveDocumentVersion, ...]
    chunks: tuple[ActiveChunk, ...]
    document_count: int
    chunk_count: int
    manifest_hash: str


class BM25PlanAction(StrEnum):
    """How a future generation should obtain its complete BM25 state."""

    REUSE = "REUSE"
    REFIT = "REFIT"


@dataclass(frozen=True, slots=True)
class PreviousBM25GenerationState:
    """Detached BM25 fields projected from the optional ACTIVE index generation."""

    index_state_id: UUID
    bm25_state: Mapping[str, Any] | None
    bm25_state_hash: str | None
    bm25_corpus_hash: str | None
    index_configuration: Mapping[str, Any] | None
    index_configuration_fingerprint: str | None
    chunking_fingerprint: str
    chunk_count: int


@dataclass(frozen=True, slots=True)
class BM25GenerationPlan:
    """Verified complete BM25 state for one desired active chunk corpus."""

    action: BM25PlanAction
    corpus_hash: str
    state: BM25State
    state_hash: str
    logical_document_count: int
    chunk_count: int
    bm25_k1: float
    bm25_b: float
    reason: str
    ordered_chunk_ids: tuple[str, ...]
