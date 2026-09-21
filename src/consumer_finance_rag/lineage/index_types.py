"""Immutable textual-corpus types used before managed index construction."""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from enum import StrEnum
from typing import Any
from uuid import UUID

from consumer_finance_rag.access import DocumentAccessPolicy, public_access_policy
from consumer_finance_rag.lineage.fingerprints import (
    PAYLOAD_SCHEMA_VERSION,
    POINT_ID_FORMAT_VERSION,
)
from consumer_finance_rag.sparse import BM25State

MANAGED_POINT_NAMESPACE = UUID("cc63cb7b-a905-5cf0-a0df-a6290ee69960")


def managed_point_id(chunk_pk: UUID) -> UUID:
    """Derive the canonical generation-independent point ID for a lineage chunk."""

    return uuid.uuid5(
        MANAGED_POINT_NAMESPACE,
        f"{POINT_ID_FORMAT_VERSION}:{chunk_pk}",
    )


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
    access_policy: DocumentAccessPolicy = dataclass_field(
        default_factory=public_access_policy
    )


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
    access_policy: DocumentAccessPolicy = dataclass_field(
        default_factory=public_access_policy
    )


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


class DensePlanAction(StrEnum):
    """How one current chunk obtains its dense vector."""

    REUSED = "REUSED"
    COMPUTED = "COMPUTED"


@dataclass(frozen=True, slots=True)
class PreviousDenseGenerationState:
    """Detached dense-vector facts from the optional ACTIVE generation."""

    index_state_id: UUID
    qdrant_collection: str
    embedding_fingerprint: str
    embedding_dimension: int | None
    dense_vector_name: str | None
    generation: int


@dataclass(frozen=True, slots=True)
class DenseReuseCandidate:
    """Prior ACTIVE membership eligible for exact-content physical validation."""

    previous_chunk_pk: UUID
    chunk_hash: str
    text: str
    qdrant_point_id: UUID
    compatibility_chunk_id: str


@dataclass(frozen=True, slots=True)
class PreparedDenseChunk:
    """One immutable current-chunk vector plus its reuse provenance."""

    lineage_chunk_pk: UUID
    compatibility_chunk_id: str
    chunk_hash: str
    action: DensePlanAction
    dense_vector: tuple[float, ...]
    dense_source_chunk_pk: UUID | None
    dense_source_point_id: UUID | None
    reason: str


@dataclass(frozen=True, slots=True)
class DenseGenerationPlan:
    """Complete dense-vector plan in exact active-manifest order."""

    embedding_fingerprint: str
    embedding_dimension: int
    chunks: tuple[PreparedDenseChunk, ...]

    @property
    def reused_count(self) -> int:
        """Return the number of entries copied from the prior ACTIVE generation."""

        return sum(chunk.action is DensePlanAction.REUSED for chunk in self.chunks)

    @property
    def computed_count(self) -> int:
        """Return the number of entries produced by the embedding provider."""

        return sum(chunk.action is DensePlanAction.COMPUTED for chunk in self.chunks)


@dataclass(frozen=True, slots=True)
class PreparedIndexIdentity:
    """Detached build identity applied to every point in one future generation."""

    ingestion_run_id: UUID
    index_state_id: UUID
    generation: int
    chunking_fingerprint: str
    embedding_fingerprint: str
    embedding_dimension: int
    point_id_format_version: str = POINT_ID_FORMAT_VERSION
    payload_schema_version: str = PAYLOAD_SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class ManagedPointInput:
    """One fully prepared managed Qdrant point with no ORM or model provider state."""

    chunk: ActiveChunk
    qdrant_point_id: UUID
    dense_vector: tuple[float, ...]
    sparse_indices: tuple[int, ...]
    sparse_values: tuple[float, ...]
    ingestion_run_id: UUID
    index_state_id: UUID
    index_generation: int
    payload: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class PreparedIndexMembership:
    """Future index-state membership and dense-vector provenance for one point."""

    chunk_pk: UUID
    qdrant_point_id: UUID
    dense_action: DensePlanAction
    dense_source_chunk_pk: UUID | None


@dataclass(frozen=True, slots=True)
class PreparedManagedGeneration:
    """Complete immutable point inputs for a future managed-index publisher."""

    ingestion_run_id: UUID
    index_state_id: UUID
    generation: int
    embedding_dimension: int
    document_count: int
    chunk_count: int
    bm25_action: BM25PlanAction
    points: tuple[ManagedPointInput, ...]
    memberships: tuple[PreparedIndexMembership, ...]

    @property
    def reused_dense_count(self) -> int:
        """Return how many prepared memberships reuse a prior dense vector."""

        return sum(
            membership.dense_action is DensePlanAction.REUSED
            for membership in self.memberships
        )

    @property
    def computed_dense_count(self) -> int:
        """Return how many prepared memberships contain a newly computed vector."""

        return sum(
            membership.dense_action is DensePlanAction.COMPUTED
            for membership in self.memberships
        )


@dataclass(frozen=True, slots=True)
class StoredSparseVector:
    """Detached sparse vector returned by physical Qdrant inspection."""

    indices: tuple[int, ...]
    values: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class ManagedStoredPoint:
    """Detached Qdrant point used for inspection and physical validation."""

    point_id: UUID
    payload: Mapping[str, Any]
    dense_vector: tuple[float, ...] | None
    sparse_vector: StoredSparseVector | None


@dataclass(frozen=True, slots=True)
class RetrievedDenseVector:
    """Dense-vector reuse input detached from Qdrant SDK classes."""

    point_id: UUID
    dense_vector: tuple[float, ...]
    lineage_chunk_pk: UUID
    chunk_hash: str
    text: str
    compatibility_chunk_id: str


@dataclass(frozen=True, slots=True)
class DenseVectorRetrievalIssue:
    """One returned Qdrant point that could not provide a reusable dense record."""

    point_id: UUID
    reason: str


@dataclass(frozen=True, slots=True)
class DenseVectorRetrievalResult:
    """Requested dense-vector records plus explicit missing point identities."""

    records: tuple[RetrievedDenseVector, ...]
    missing_point_ids: tuple[UUID, ...]
    issues: tuple[DenseVectorRetrievalIssue, ...] = ()


@dataclass(frozen=True, slots=True)
class ManagedCollectionMetadata:
    """Detached physical collection configuration relevant to managed validation."""

    collection_name: str
    dense_vector_names: tuple[str, ...]
    sparse_vector_names: tuple[str, ...]
    dense_dimension: int | None
    distance_metric: str | None
    hnsw_m: int | None
    hnsw_ef_construct: int | None
    sparse_modifier: str | None
    sparse_on_disk: bool | None


@dataclass(frozen=True, slots=True)
class ManagedCollectionValidationResult:
    """Successful physical integrity result for an immutable generation collection."""

    collection_name: str
    point_count: int
    point_ids: tuple[UUID, ...]
