"""Read-only dense-vector reuse and computation planning."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from numbers import Real
from typing import Protocol
from uuid import UUID

from consumer_finance_rag.lineage.index_types import (
    ActiveChunk,
    ActiveChunkManifest,
    DenseGenerationPlan,
    DensePlanAction,
    DenseReuseCandidate,
    DenseVectorRetrievalIssue,
    DenseVectorRetrievalResult,
    PreparedDenseChunk,
    PreviousDenseGenerationState,
    RetrievedDenseVector,
)
from consumer_finance_rag.providers import EmbeddingProvider


class DenseGenerationError(RuntimeError):
    """Base exception for a dense-vector planning failure."""


class CorruptActiveDenseGenerationError(DenseGenerationError):
    """Raised when persisted ACTIVE generation metadata cannot be trusted."""


class DenseReuseInvariantError(DenseGenerationError):
    """Raised for impossible or ambiguous lineage/Qdrant reuse state."""


class EmbeddingDimensionMismatchError(DenseGenerationError):
    """Raised when newly computed vectors do not share the required dimension."""


class DenseReuseCandidateReader(Protocol):
    """Minimal read-only lineage dependency used by dense planning."""

    def find_reuse_candidates_for_hashes(
        self,
        index_state_id: UUID,
        chunk_hashes: Sequence[str],
    ) -> list[DenseReuseCandidate]: ...


class DenseVectorReader(Protocol):
    """Minimal read-only physical-store dependency used by dense planning."""

    def retrieve_dense_vectors(
        self,
        collection_name: str,
        point_ids: Sequence[UUID],
    ) -> DenseVectorRetrievalResult: ...


class DenseGenerationPlanner:
    """Prepare one vector per active chunk without mutating lineage or Qdrant."""

    def __init__(
        self,
        candidate_reader: DenseReuseCandidateReader,
        vector_reader: DenseVectorReader,
        embedding_provider: EmbeddingProvider,
        *,
        retrieval_batch_size: int = 256,
    ) -> None:
        if retrieval_batch_size <= 0:
            raise ValueError("retrieval_batch_size must be positive")
        self._candidate_reader = candidate_reader
        self._vector_reader = vector_reader
        self._embedding_provider = embedding_provider
        self._retrieval_batch_size = retrieval_batch_size

    def plan(
        self,
        manifest: ActiveChunkManifest,
        requested_embedding_fingerprint: str,
        previous: PreviousDenseGenerationState | None = None,
    ) -> DenseGenerationPlan:
        """Reuse validated exact vectors and batch-compute every remaining chunk."""

        _validate_manifest(manifest)
        if not requested_embedding_fingerprint.strip():
            raise DenseGenerationError("Requested embedding fingerprint cannot be empty")

        target_dimension: int | None = None
        reusable: dict[UUID, tuple[tuple[float, ...], DenseReuseCandidate]] = {}
        compute_reasons: dict[UUID, str] = {}

        if previous is None:
            compute_reasons = {
                chunk.chunk_pk: "no_previous_active_generation"
                for chunk in manifest.chunks
            }
        else:
            _validate_previous(previous)
            if previous.embedding_fingerprint != requested_embedding_fingerprint:
                compute_reasons = {
                    chunk.chunk_pk: "embedding_fingerprint_changed"
                    for chunk in manifest.chunks
                }
            else:
                target_dimension = previous.embedding_dimension
                reusable, compute_reasons = self._resolve_reuse(
                    manifest,
                    previous,
                )

        computed_chunks = [
            chunk for chunk in manifest.chunks if chunk.chunk_pk not in reusable
        ]
        computed_vectors: dict[UUID, tuple[float, ...]] = {}
        if computed_chunks:
            computed_vectors, inferred_dimension = self._compute_vectors(computed_chunks)
            if target_dimension is None:
                target_dimension = inferred_dimension
            elif inferred_dimension != target_dimension:
                raise EmbeddingDimensionMismatchError(
                    "Computed dense vector dimension "
                    f"{inferred_dimension} != previous ACTIVE dimension {target_dimension}"
                )

        if target_dimension is None:
            raise DenseGenerationError(
                "Cannot determine embedding dimension for an empty first-generation manifest"
            )

        prepared: list[PreparedDenseChunk] = []
        for chunk in manifest.chunks:
            reused = reusable.get(chunk.chunk_pk)
            if reused is not None:
                vector, source = reused
                prepared.append(
                    PreparedDenseChunk(
                        lineage_chunk_pk=chunk.chunk_pk,
                        compatibility_chunk_id=chunk.compatibility_chunk_id,
                        chunk_hash=chunk.chunk_hash,
                        action=DensePlanAction.REUSED,
                        dense_vector=vector,
                        dense_source_chunk_pk=source.previous_chunk_pk,
                        dense_source_point_id=source.qdrant_point_id,
                        reason="exact_active_chunk_vector_validated",
                    )
                )
            else:
                prepared.append(
                    PreparedDenseChunk(
                        lineage_chunk_pk=chunk.chunk_pk,
                        compatibility_chunk_id=chunk.compatibility_chunk_id,
                        chunk_hash=chunk.chunk_hash,
                        action=DensePlanAction.COMPUTED,
                        dense_vector=computed_vectors[chunk.chunk_pk],
                        dense_source_chunk_pk=None,
                        dense_source_point_id=None,
                        reason=compute_reasons.get(
                            chunk.chunk_pk, "no_exact_previous_chunk"
                        ),
                    )
                )

        return DenseGenerationPlan(
            embedding_fingerprint=requested_embedding_fingerprint,
            embedding_dimension=target_dimension,
            chunks=tuple(prepared),
        )

    def _resolve_reuse(
        self,
        manifest: ActiveChunkManifest,
        previous: PreviousDenseGenerationState,
    ) -> tuple[
        dict[UUID, tuple[tuple[float, ...], DenseReuseCandidate]],
        dict[UUID, str],
    ]:
        candidates = self._candidate_reader.find_reuse_candidates_for_hashes(
            previous.index_state_id,
            tuple(chunk.chunk_hash for chunk in manifest.chunks),
        )
        grouped: dict[str, list[DenseReuseCandidate]] = defaultdict(list)
        for candidate in candidates:
            grouped[candidate.chunk_hash].append(candidate)
        for values in grouped.values():
            values.sort(
                key=lambda candidate: (
                    candidate.compatibility_chunk_id,
                    str(candidate.previous_chunk_pk),
                )
            )

        selected: dict[UUID, DenseReuseCandidate] = {}
        reasons: dict[UUID, str] = {}
        for chunk in manifest.chunks:
            exact_hash = grouped.get(chunk.chunk_hash, [])
            mismatched = [candidate for candidate in exact_hash if candidate.text != chunk.text]
            if mismatched:
                raise DenseReuseInvariantError(
                    "Identical chunk_hash maps to different text for current chunk "
                    f"{chunk.chunk_pk} and prior chunk {mismatched[0].previous_chunk_pk}"
                )
            if exact_hash:
                selected[chunk.chunk_pk] = exact_hash[0]
            else:
                reasons[chunk.chunk_pk] = "no_exact_previous_chunk"

        requested_points = tuple(
            sorted({candidate.qdrant_point_id for candidate in selected.values()}, key=str)
        )
        records, missing, issues = self._retrieve_batches(
            previous.qdrant_collection,
            requested_points,
        )
        records_by_id: dict[UUID, RetrievedDenseVector] = {}
        requested_set = set(requested_points)
        for record in records:
            if record.point_id not in requested_set:
                raise DenseReuseInvariantError(
                    f"Qdrant returned unrequested point {record.point_id}"
                )
            if record.point_id in records_by_id:
                raise DenseReuseInvariantError(
                    f"Qdrant returned duplicate point {record.point_id}"
                )
            records_by_id[record.point_id] = record

        issue_ids = {issue.point_id for issue in issues}
        missing_ids = set(missing)
        reusable: dict[UUID, tuple[tuple[float, ...], DenseReuseCandidate]] = {}
        for chunk in manifest.chunks:
            chosen_candidate = selected.get(chunk.chunk_pk)
            if chosen_candidate is None:
                continue
            if chosen_candidate.qdrant_point_id in missing_ids:
                reasons[chunk.chunk_pk] = "previous_vector_missing"
                continue
            if chosen_candidate.qdrant_point_id in issue_ids:
                reasons[chunk.chunk_pk] = "previous_vector_invalid"
                continue
            returned_record = records_by_id.get(chosen_candidate.qdrant_point_id)
            if returned_record is None:
                reasons[chunk.chunk_pk] = "previous_vector_missing"
                continue
            if not _record_matches(
                returned_record,
                chosen_candidate,
                chunk,
                previous.embedding_dimension,
            ):
                reasons[chunk.chunk_pk] = "previous_vector_invalid"
                continue
            reusable[chunk.chunk_pk] = (
                returned_record.dense_vector,
                chosen_candidate,
            )
        return reusable, reasons

    def _retrieve_batches(
        self,
        collection_name: str,
        point_ids: tuple[UUID, ...],
    ) -> tuple[
        list[RetrievedDenseVector],
        list[UUID],
        list[DenseVectorRetrievalIssue],
    ]:
        records: list[RetrievedDenseVector] = []
        missing: list[UUID] = []
        issues: list[DenseVectorRetrievalIssue] = []
        for start in range(0, len(point_ids), self._retrieval_batch_size):
            batch = point_ids[start : start + self._retrieval_batch_size]
            try:
                result = self._vector_reader.retrieve_dense_vectors(
                    collection_name,
                    batch,
                )
            except Exception as exc:
                raise DenseGenerationError(
                    "Failed to read previous ACTIVE dense vectors from Qdrant"
                ) from exc
            records.extend(result.records)
            missing.extend(result.missing_point_ids)
            issues.extend(result.issues)
        return records, missing, issues

    def _compute_vectors(
        self,
        chunks: Sequence[ActiveChunk],
    ) -> tuple[dict[UUID, tuple[float, ...]], int]:
        try:
            raw_vectors = self._embedding_provider.embed_documents(
                [chunk.text for chunk in chunks]
            )
        except Exception as exc:
            raise DenseGenerationError("Embedding provider failed") from exc
        if len(raw_vectors) != len(chunks):
            raise DenseGenerationError(
                "Embedding provider returned a different number of vectors than texts"
            )
        vectors: dict[UUID, tuple[float, ...]] = {}
        dimension: int | None = None
        for chunk, raw_vector in zip(chunks, raw_vectors, strict=True):
            vector = _validated_vector(raw_vector)
            if dimension is None:
                dimension = len(vector)
            elif len(vector) != dimension:
                raise EmbeddingDimensionMismatchError(
                    "Embedding provider returned inconsistent dense vector dimensions"
                )
            vectors[chunk.chunk_pk] = vector
        if dimension is None:
            raise DenseGenerationError("Embedding provider returned no vectors")
        return vectors, dimension


def _validate_manifest(manifest: ActiveChunkManifest) -> None:
    if manifest.chunk_count != len(manifest.chunks):
        raise DenseGenerationError(
            "Manifest chunk_count does not equal its active chunk tuple length"
        )
    chunk_ids = [chunk.chunk_pk for chunk in manifest.chunks]
    if len(set(chunk_ids)) != len(chunk_ids):
        raise DenseGenerationError("Manifest contains duplicate lineage chunk IDs")


def _validate_previous(previous: PreviousDenseGenerationState) -> None:
    if not previous.qdrant_collection.strip():
        raise CorruptActiveDenseGenerationError(
            "ACTIVE dense generation is missing its Qdrant collection"
        )
    if not previous.embedding_fingerprint.strip():
        raise CorruptActiveDenseGenerationError(
            "ACTIVE dense generation is missing its embedding fingerprint"
        )
    if (
        isinstance(previous.embedding_dimension, bool)
        or not isinstance(previous.embedding_dimension, int)
        or previous.embedding_dimension <= 0
    ):
        raise CorruptActiveDenseGenerationError(
            "ACTIVE dense generation has no valid embedding dimension"
        )
    if previous.dense_vector_name is None or not previous.dense_vector_name.strip():
        raise CorruptActiveDenseGenerationError(
            "ACTIVE dense generation is missing its dense vector name"
        )
    if (
        isinstance(previous.generation, bool)
        or not isinstance(previous.generation, int)
        or previous.generation <= 0
    ):
        raise CorruptActiveDenseGenerationError(
            "ACTIVE dense generation number must be positive"
        )


def _record_matches(
    record: RetrievedDenseVector,
    candidate: DenseReuseCandidate,
    current: ActiveChunk,
    expected_dimension: int | None,
) -> bool:
    if (
        record.point_id != candidate.qdrant_point_id
        or record.lineage_chunk_pk != candidate.previous_chunk_pk
        or record.chunk_hash != candidate.chunk_hash
        or record.text != candidate.text
        or record.text != current.text
        or record.compatibility_chunk_id != candidate.compatibility_chunk_id
    ):
        return False
    if expected_dimension is None or len(record.dense_vector) != expected_dimension:
        return False
    return _is_valid_vector(record.dense_vector)


def _validated_vector(raw_vector: Sequence[float]) -> tuple[float, ...]:
    if not _is_valid_vector(raw_vector):
        raise DenseGenerationError(
            "Embedding provider returned an empty, non-numeric, or non-finite vector"
        )
    return tuple(float(value) for value in raw_vector)


def _is_valid_vector(vector: Sequence[float]) -> bool:
    return bool(vector) and all(
        not isinstance(value, bool)
        and isinstance(value, Real)
        and math.isfinite(float(value))
        for value in vector
    )
