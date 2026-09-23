"""Pure alignment and point preparation for a future managed-index publisher."""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Real
from types import MappingProxyType
from uuid import UUID

from consumer_finance_rag.access import access_policy_payload
from consumer_finance_rag.lineage.fingerprints import (
    PAYLOAD_SCHEMA_VERSION,
    POINT_ID_FORMAT_VERSION,
)
from consumer_finance_rag.lineage.index_types import (
    ActiveChunk,
    ActiveChunkManifest,
    BM25GenerationPlan,
    DenseGenerationPlan,
    DensePlanAction,
    ManagedPointInput,
    PreparedDenseChunk,
    PreparedIndexIdentity,
    PreparedIndexMembership,
    PreparedManagedGeneration,
    managed_point_id,
)
from consumer_finance_rag.pii import pii_summary_payload
from consumer_finance_rag.sparse import (
    BM25SparseEncoder,
    BM25State,
    SparseVectorData,
    bm25_corpus_hash,
    bm25_state_hash,
)


class ManagedPointPreparationError(ValueError):
    """Base exception for invalid inputs to managed point preparation."""


class PlanAlignmentError(ManagedPointPreparationError):
    """Raised when manifest, dense, sparse, and build identities do not align."""


def prepare_managed_generation(
    manifest: ActiveChunkManifest,
    dense_plan: DenseGenerationPlan,
    bm25_plan: BM25GenerationPlan,
    identity: PreparedIndexIdentity,
) -> PreparedManagedGeneration:
    """Build deterministic point inputs without database, Qdrant, or model calls."""

    _validate_manifest_and_identity(manifest, dense_plan, identity)
    _validate_dense_alignment(manifest, dense_plan)
    _validate_bm25_alignment(manifest, bm25_plan)

    encoder = BM25SparseEncoder(bm25_plan.state)
    points: list[ManagedPointInput] = []
    memberships: list[PreparedIndexMembership] = []
    seen_point_ids: set[UUID] = set()
    for chunk, dense in zip(manifest.chunks, dense_plan.chunks, strict=True):
        sparse = encoder.encode_document(chunk.text)
        _validate_sparse_vector(sparse, bm25_plan.state)
        point_id = managed_point_id(chunk.chunk_pk)
        if point_id in seen_point_ids:
            raise PlanAlignmentError(
                f"Multiple active chunks derive the same managed point ID: {point_id}"
            )
        seen_point_ids.add(point_id)
        payload = _current_payload(chunk, identity)
        points.append(
            ManagedPointInput(
                chunk=chunk,
                qdrant_point_id=point_id,
                dense_vector=dense.dense_vector,
                sparse_indices=tuple(sparse.indices),
                sparse_values=tuple(sparse.values),
                ingestion_run_id=identity.ingestion_run_id,
                index_state_id=identity.index_state_id,
                index_generation=identity.generation,
                payload=MappingProxyType(payload),
            )
        )
        memberships.append(
            PreparedIndexMembership(
                chunk_pk=chunk.chunk_pk,
                qdrant_point_id=point_id,
                dense_action=dense.action,
                dense_source_chunk_pk=dense.dense_source_chunk_pk,
            )
        )

    return PreparedManagedGeneration(
        ingestion_run_id=identity.ingestion_run_id,
        index_state_id=identity.index_state_id,
        generation=identity.generation,
        embedding_dimension=identity.embedding_dimension,
        document_count=manifest.document_count,
        chunk_count=manifest.chunk_count,
        bm25_action=bm25_plan.action,
        points=tuple(points),
        memberships=tuple(memberships),
    )


def _validate_manifest_and_identity(
    manifest: ActiveChunkManifest,
    dense_plan: DenseGenerationPlan,
    identity: PreparedIndexIdentity,
) -> None:
    if manifest.document_count != len(manifest.documents):
        raise PlanAlignmentError(
            "Manifest document_count does not match its active document tuple"
        )
    if manifest.chunk_count != len(manifest.chunks):
        raise PlanAlignmentError(
            "Manifest chunk_count does not match its active chunk tuple"
        )
    if manifest.chunk_count == 0:
        raise ManagedPointPreparationError(
            "Cannot prepare an empty managed physical generation"
        )
    chunk_pks = [chunk.chunk_pk for chunk in manifest.chunks]
    if len(set(chunk_pks)) != len(chunk_pks):
        raise PlanAlignmentError("Manifest contains duplicate lineage chunk IDs")
    if identity.generation <= 0:
        raise ManagedPointPreparationError("Index generation must be positive")
    if (
        isinstance(identity.embedding_dimension, bool)
        or not isinstance(identity.embedding_dimension, int)
        or identity.embedding_dimension <= 0
    ):
        raise ManagedPointPreparationError("Embedding dimension must be a positive integer")
    if not identity.chunking_fingerprint:
        raise ManagedPointPreparationError("Chunking fingerprint cannot be empty")
    if not identity.embedding_fingerprint:
        raise ManagedPointPreparationError("Embedding fingerprint cannot be empty")
    if identity.chunking_fingerprint != manifest.chunking_fingerprint:
        raise PlanAlignmentError(
            "Prepared index chunking fingerprint does not match the manifest"
        )
    if any(
        chunk.chunking_fingerprint != identity.chunking_fingerprint
        for chunk in manifest.chunks
    ):
        raise PlanAlignmentError(
            "An active chunk has a different chunking fingerprint from the build identity"
        )
    if identity.embedding_fingerprint != dense_plan.embedding_fingerprint:
        raise PlanAlignmentError(
            "Prepared index embedding fingerprint does not match the dense plan"
        )
    if identity.embedding_dimension != dense_plan.embedding_dimension:
        raise PlanAlignmentError(
            "Prepared index embedding dimension does not match the dense plan"
        )
    if identity.point_id_format_version != POINT_ID_FORMAT_VERSION:
        raise ManagedPointPreparationError(
            "Prepared index point-ID format is not supported: "
            f"{identity.point_id_format_version}"
        )
    if identity.payload_schema_version != PAYLOAD_SCHEMA_VERSION:
        raise ManagedPointPreparationError(
            "Prepared index payload schema is not supported: "
            f"{identity.payload_schema_version}"
        )


def _validate_dense_alignment(
    manifest: ActiveChunkManifest,
    dense_plan: DenseGenerationPlan,
) -> None:
    if (
        isinstance(dense_plan.embedding_dimension, bool)
        or not isinstance(dense_plan.embedding_dimension, int)
        or dense_plan.embedding_dimension <= 0
    ):
        raise ManagedPointPreparationError(
            "Dense plan embedding dimension must be a positive integer"
        )
    if len(dense_plan.chunks) != manifest.chunk_count:
        raise PlanAlignmentError(
            "Dense plan must contain exactly one entry per active manifest chunk"
        )
    for position, (chunk, dense) in enumerate(
        zip(manifest.chunks, dense_plan.chunks, strict=True)
    ):
        _validate_dense_entry(chunk, dense, dense_plan.embedding_dimension, position)


def _validate_dense_entry(
    chunk: ActiveChunk,
    dense: PreparedDenseChunk,
    expected_dimension: int,
    position: int,
) -> None:
    prefix = f"Dense plan entry {position}"
    if dense.lineage_chunk_pk != chunk.chunk_pk:
        raise PlanAlignmentError(f"{prefix} lineage chunk does not match the manifest")
    if dense.compatibility_chunk_id != chunk.compatibility_chunk_id:
        raise PlanAlignmentError(
            f"{prefix} compatibility chunk ID does not match the manifest"
        )
    if dense.chunk_hash != chunk.chunk_hash:
        raise PlanAlignmentError(f"{prefix} chunk hash does not match the manifest")
    if len(dense.dense_vector) != expected_dimension:
        raise PlanAlignmentError(
            f"{prefix} dense dimension {len(dense.dense_vector)} "
            f"does not match {expected_dimension}"
        )
    if not _valid_numeric_vector(dense.dense_vector):
        raise ManagedPointPreparationError(
            f"{prefix} contains a non-numeric or non-finite dense value"
        )
    if dense.action is DensePlanAction.REUSED:
        if dense.dense_source_chunk_pk is None:
            raise PlanAlignmentError(
                f"{prefix} REUSED provenance requires a source chunk"
            )
    elif dense.action is DensePlanAction.COMPUTED:
        if dense.dense_source_chunk_pk is not None:
            raise PlanAlignmentError(
                f"{prefix} COMPUTED provenance cannot reference a source chunk"
            )
    else:
        raise ManagedPointPreparationError(
            f"{prefix} has unsupported dense action {dense.action!r}"
        )


def _validate_bm25_alignment(
    manifest: ActiveChunkManifest,
    plan: BM25GenerationPlan,
) -> None:
    state = plan.state
    _validate_bm25_state(state)
    if plan.logical_document_count != manifest.document_count:
        raise PlanAlignmentError(
            "BM25 logical document count does not match the manifest"
        )
    if plan.chunk_count != manifest.chunk_count:
        raise PlanAlignmentError("BM25 chunk count does not match the manifest")
    if state.document_count != manifest.chunk_count:
        raise PlanAlignmentError(
            "BM25 state document count does not match the manifest chunk count"
        )
    ordered_ids = tuple(chunk.compatibility_chunk_id for chunk in manifest.chunks)
    if plan.ordered_chunk_ids != ordered_ids:
        raise PlanAlignmentError(
            "BM25 ordered chunk identities do not match manifest order"
        )
    corpus_hash = bm25_corpus_hash(tuple(chunk.text for chunk in manifest.chunks))
    if plan.corpus_hash != corpus_hash:
        raise PlanAlignmentError("BM25 plan corpus hash does not match manifest texts")
    if state.corpus_hash != corpus_hash:
        raise PlanAlignmentError("BM25 state corpus hash does not match manifest texts")
    if bm25_state_hash(state) != plan.state_hash:
        raise PlanAlignmentError("BM25 state hash does not match the plan state hash")
    if state.k1 != plan.bm25_k1 or state.b != plan.bm25_b:
        raise PlanAlignmentError("BM25 plan parameters do not match its selected state")


def _validate_bm25_state(state: BM25State) -> None:
    if (
        isinstance(state.document_count, bool)
        or not isinstance(state.document_count, int)
        or state.document_count <= 0
    ):
        raise ManagedPointPreparationError("BM25 document count must be positive")
    if not math.isfinite(state.average_document_length) or state.average_document_length <= 0:
        raise ManagedPointPreparationError(
            "BM25 average document length must be positive and finite"
        )
    if (
        not math.isfinite(state.k1)
        or state.k1 <= 0
        or not math.isfinite(state.b)
        or not 0 <= state.b <= 1
    ):
        raise ManagedPointPreparationError("BM25 k1/b values are invalid")
    vocabulary_indices = tuple(state.vocabulary.values())
    if any(
        isinstance(index, bool) or not isinstance(index, int) or index < 0
        for index in vocabulary_indices
    ):
        raise ManagedPointPreparationError(
            "BM25 vocabulary indices must be non-negative integers"
        )
    if set(vocabulary_indices) != set(range(len(vocabulary_indices))):
        raise ManagedPointPreparationError(
            "BM25 vocabulary indices must be unique and contiguous"
        )
    if set(state.document_frequencies) != set(state.vocabulary):
        raise ManagedPointPreparationError(
            "BM25 document-frequency terms must exactly match the vocabulary"
        )
    if any(
        isinstance(frequency, bool)
        or not isinstance(frequency, int)
        or frequency <= 0
        or frequency > state.document_count
        for frequency in state.document_frequencies.values()
    ):
        raise ManagedPointPreparationError(
            "BM25 document frequencies must be valid corpus counts"
        )


def _validate_sparse_vector(vector: SparseVectorData, state: BM25State) -> None:
    if len(vector.indices) != len(vector.values):
        raise ManagedPointPreparationError(
            "Sparse vector indices and values must have equal lengths"
        )
    if len(set(vector.indices)) != len(vector.indices):
        raise ManagedPointPreparationError("Sparse vector indices must be unique")
    valid_indices = set(state.vocabulary.values())
    if any(
        isinstance(index, bool)
        or not isinstance(index, int)
        or index < 0
        or index not in valid_indices
        for index in vector.indices
    ):
        raise ManagedPointPreparationError(
            "Sparse vector index is invalid for the selected BM25 vocabulary"
        )
    if not _valid_numeric_vector(vector.values, allow_empty=True):
        raise ManagedPointPreparationError(
            "Sparse vector contains a non-numeric or non-finite value"
        )


def _valid_numeric_vector(
    values: Sequence[float],
    *,
    allow_empty: bool = False,
) -> bool:
    return (allow_empty or bool(values)) and all(
        not isinstance(value, bool)
        and isinstance(value, Real)
        and math.isfinite(float(value))
        for value in values
    )


def _current_payload(
    chunk: ActiveChunk,
    identity: PreparedIndexIdentity,
) -> dict[str, object]:
    return {
        "document_id": chunk.compatibility_document_id,
        "chunk_id": chunk.compatibility_chunk_id,
        "document_version": chunk.source_version,
        "title": chunk.title,
        "category": chunk.category,
        "source_url": chunk.source_url,
        "source_id": chunk.source_id,
        "source_key": chunk.source_key,
        "chunk_hash": chunk.chunk_hash,
        "ordinal": chunk.ordinal,
        "text": chunk.text,
        "content_hash": chunk.content_hash,
        "source_version": chunk.source_version,
        "lineage_document_pk": str(chunk.document_pk),
        "lineage_document_version_id": str(chunk.document_version_id),
        "lineage_chunk_pk": str(chunk.chunk_pk),
        "lineage_version_number": chunk.version_number,
        "ingestion_run_id": str(identity.ingestion_run_id),
        "index_generation": identity.generation,
        "index_state_id": str(identity.index_state_id),
        "chunking_fingerprint": chunk.chunking_fingerprint,
        "point_id_format_version": identity.point_id_format_version,
        "payload_schema_version": identity.payload_schema_version,
        **access_policy_payload(chunk.access_policy),
        **pii_summary_payload(chunk.pii_summary),
    }
