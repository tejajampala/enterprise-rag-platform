"""Immutable textual-corpus types used before managed index construction."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID


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
