"""Safe physical Qdrant operations for immutable managed index generations."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any
from uuid import UUID

from qdrant_client import QdrantClient, models

from consumer_finance_rag.access import access_policy_payload
from consumer_finance_rag.lineage.fingerprints import (
    POINT_ID_FORMAT_VERSION,
    ManagedIndexConfiguration,
)
from consumer_finance_rag.lineage.index_types import (
    DenseVectorRetrievalIssue,
    DenseVectorRetrievalResult,
    ManagedCollectionMetadata,
    ManagedCollectionValidationResult,
    ManagedPointInput,
    ManagedStoredPoint,
    RetrievedDenseVector,
    StoredSparseVector,
)
from consumer_finance_rag.lineage.index_types import (
    managed_point_id as managed_point_id,
)
from consumer_finance_rag.pii import pii_summary_payload

PROTECTED_COLLECTION_NAMES = frozenset(
    {"consumer_finance_phase1", "consumer_finance_evaluation"}
)
COLLECTION_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,254}$")
FLOAT_REL_TOLERANCE = 1e-5
FLOAT_ABS_TOLERANCE = 1e-6


class ManagedQdrantError(RuntimeError):
    """Base exception for expected managed physical-store failures."""


class UnsafeManagedCollectionNameError(ManagedQdrantError):
    """Raised when an operation targets a protected or unmanaged collection name."""


class ManagedCollectionCollisionError(ManagedQdrantError):
    """Raised instead of replacing an existing physical collection."""


class ManagedCollectionNotFoundError(ManagedQdrantError):
    """Raised when a requested managed generation collection does not exist."""


class ManagedVectorValidationError(ManagedQdrantError):
    """Raised before Qdrant receives malformed prepared vector data."""


class ManagedCollectionValidationError(ManagedQdrantError):
    """Raised when physical collection configuration or point contents are incomplete."""


def managed_generation_collection_name(prefix: str, generation: int) -> str:
    """Return the canonical non-destructive physical name for one generation."""

    _validate_prefix(prefix)
    if generation <= 0:
        raise UnsafeManagedCollectionNameError("Index generation must be positive")
    name = f"{prefix}_g{generation:06d}"
    if len(name) > 255:
        raise UnsafeManagedCollectionNameError(
            "Managed generation collection name exceeds 255 characters"
        )
    return name


class ManagedQdrantStore:
    """Physical adapter that stores already-prepared dense and sparse vectors."""

    def __init__(
        self,
        client: QdrantClient,
        *,
        managed_prefix: str,
        configuration: ManagedIndexConfiguration,
        embedding_dimension: int,
    ) -> None:
        _validate_prefix(managed_prefix)
        if embedding_dimension <= 0:
            raise ManagedVectorValidationError("Embedding dimension must be positive")
        if configuration.point_id_format_version != POINT_ID_FORMAT_VERSION:
            raise ManagedVectorValidationError(
                "Managed point-ID format is not supported by this adapter: "
                f"{configuration.point_id_format_version}"
            )
        self._client = client
        self._managed_prefix = managed_prefix
        self._configuration = configuration
        self._embedding_dimension = embedding_dimension

    def collection_name(self, generation: int) -> str:
        """Return this managed corpus's canonical name for a generation."""

        return managed_generation_collection_name(self._managed_prefix, generation)

    def create_generation_collection(self, collection_name: str) -> None:
        """Create one new immutable physical collection without replacing collisions."""

        self._require_managed_name(collection_name)
        if self._client.collection_exists(collection_name):
            raise ManagedCollectionCollisionError(
                f"Managed generation collection already exists: {collection_name}"
            )
        created = self._client.create_collection(
            collection_name=collection_name,
            vectors_config={
                self._configuration.dense_vector_name: models.VectorParams(
                    size=self._embedding_dimension,
                    distance=_distance(self._configuration.distance_metric),
                    hnsw_config=models.HnswConfigDiff(
                        m=self._configuration.hnsw_m,
                        ef_construct=self._configuration.hnsw_ef_construct,
                    ),
                )
            },
            sparse_vectors_config={
                self._configuration.sparse_vector_name: models.SparseVectorParams(
                    index=models.SparseIndexParams(
                        on_disk=self._configuration.sparse_on_disk
                    ),
                    modifier=_modifier(self._configuration.sparse_modifier),
                )
            },
        )
        if not created:
            raise ManagedQdrantError(
                f"Qdrant did not confirm collection creation: {collection_name}"
            )

    def delete_generation_collection(self, collection_name: str) -> None:
        """Delete one explicit managed generation; callers own cleanup policy."""

        self._require_managed_name(collection_name)
        if not self._client.collection_exists(collection_name):
            raise ManagedCollectionNotFoundError(
                f"Managed generation collection does not exist: {collection_name}"
            )
        deleted = self._client.delete_collection(collection_name)
        if not deleted:
            raise ManagedQdrantError(
                f"Qdrant did not confirm collection deletion: {collection_name}"
            )

    def upsert_points(
        self,
        collection_name: str,
        points: Sequence[ManagedPointInput],
        *,
        batch_size: int = 128,
    ) -> None:
        """Validate all inputs, then upsert deterministic batches with ``wait=True``."""

        generation = self._require_existing_collection(collection_name)
        if batch_size <= 0:
            raise ManagedVectorValidationError("batch_size must be positive")
        point_structs: list[models.PointStruct] = []
        seen_ids: set[UUID] = set()
        for point in points:
            if point.qdrant_point_id in seen_ids:
                raise ManagedVectorValidationError(
                    f"Duplicate prepared point ID: {point.qdrant_point_id}"
                )
            seen_ids.add(point.qdrant_point_id)
            point_structs.append(self._to_point_struct(point, generation=generation))

        for start in range(0, len(point_structs), batch_size):
            self._client.upsert(
                collection_name=collection_name,
                points=point_structs[start : start + batch_size],
                wait=True,
            )

    def retrieve_dense_vectors(
        self,
        collection_name: str,
        point_ids: Sequence[UUID],
    ) -> DenseVectorRetrievalResult:
        """Retrieve named dense vectors and bounded validation payload fields."""

        self._require_existing_collection(collection_name)
        requested = tuple(point_ids)
        records = self._client.retrieve(
            collection_name=collection_name,
            ids=list(requested),
            with_payload=["lineage_chunk_pk", "chunk_hash", "text", "chunk_id"],
            with_vectors=[self._configuration.dense_vector_name],
        )
        detached: list[RetrievedDenseVector] = []
        issues: list[DenseVectorRetrievalIssue] = []
        returned_ids: set[UUID] = set()
        for record in records:
            point_id = _as_uuid(record.id, "Qdrant point ID")
            returned_ids.add(point_id)
            payload = record.payload or {}
            try:
                dense = _named_dense_vector(
                    record.vector,
                    self._configuration.dense_vector_name,
                    self._embedding_dimension,
                )
                detached.append(
                    RetrievedDenseVector(
                        point_id=point_id,
                        dense_vector=dense,
                        lineage_chunk_pk=_as_uuid(
                            payload.get("lineage_chunk_pk"), "lineage_chunk_pk"
                        ),
                        chunk_hash=_payload_string(payload, "chunk_hash"),
                        text=_payload_string(payload, "text"),
                        compatibility_chunk_id=_payload_string(payload, "chunk_id"),
                    )
                )
            except (ManagedVectorValidationError, TypeError, ValueError) as exc:
                issues.append(
                    DenseVectorRetrievalIssue(point_id=point_id, reason=str(exc))
                )
        detached.sort(key=lambda item: str(item.point_id))
        missing = tuple(
            sorted(
                (point_id for point_id in requested if point_id not in returned_ids),
                key=str,
            )
        )
        return DenseVectorRetrievalResult(
            records=tuple(detached),
            missing_point_ids=missing,
            issues=tuple(sorted(issues, key=lambda item: str(item.point_id))),
        )

    def scroll_all_points(
        self,
        collection_name: str,
        *,
        page_size: int = 256,
    ) -> tuple[ManagedStoredPoint, ...]:
        """Read every point with both named vectors, following all scroll pages."""

        self._require_existing_collection(collection_name)
        if page_size <= 0:
            raise ManagedVectorValidationError("page_size must be positive")
        points: list[ManagedStoredPoint] = []
        offset: Any | None = None
        while True:
            records, offset = self._client.scroll(
                collection_name=collection_name,
                limit=page_size,
                offset=offset,
                with_payload=True,
                with_vectors=[
                    self._configuration.dense_vector_name,
                    self._configuration.sparse_vector_name,
                ],
            )
            points.extend(self._detach_point(record) for record in records)
            if offset is None:
                break
        return tuple(sorted(points, key=lambda point: str(point.point_id)))

    def count_points(self, collection_name: str) -> int:
        """Return Qdrant's exact physical point count."""

        self._require_existing_collection(collection_name)
        return int(
            self._client.count(collection_name=collection_name, exact=True).count
        )

    def inspect_collection(self, collection_name: str) -> ManagedCollectionMetadata:
        """Detach managed dense/sparse configuration from Qdrant SDK models."""

        self._require_existing_collection(collection_name)
        info = self._client.get_collection(collection_name)
        vectors = info.config.params.vectors
        sparse_vectors = info.config.params.sparse_vectors or {}
        dense_map = vectors if isinstance(vectors, Mapping) else {}
        dense = dense_map.get(self._configuration.dense_vector_name)
        sparse = sparse_vectors.get(self._configuration.sparse_vector_name)
        dense_hnsw = dense.hnsw_config if dense is not None else None
        sparse_index = sparse.index if sparse is not None else None
        return ManagedCollectionMetadata(
            collection_name=collection_name,
            dense_vector_names=tuple(sorted(str(name) for name in dense_map)),
            sparse_vector_names=tuple(sorted(str(name) for name in sparse_vectors)),
            dense_dimension=int(dense.size) if dense is not None else None,
            distance_metric=(
                str(dense.distance.value).lower() if dense is not None else None
            ),
            hnsw_m=dense_hnsw.m if dense_hnsw is not None else None,
            hnsw_ef_construct=(
                dense_hnsw.ef_construct if dense_hnsw is not None else None
            ),
            sparse_modifier=(
                str(sparse.modifier.value).lower()
                if sparse is not None and sparse.modifier is not None
                else None
            ),
            sparse_on_disk=(
                sparse_index.on_disk if sparse_index is not None else None
            ),
        )

    def validate_generation_collection(
        self,
        collection_name: str,
        expected_points: Sequence[ManagedPointInput],
    ) -> ManagedCollectionValidationResult:
        """Validate physical configuration, identities, payloads, and prepared vectors."""

        self._require_managed_name(collection_name)
        if not self._client.collection_exists(collection_name):
            raise ManagedCollectionValidationError(
                f"Managed generation collection does not exist: {collection_name}"
            )
        expected: dict[UUID, ManagedPointInput] = {}
        generation = self._generation_from_name(collection_name)
        for point in expected_points:
            self._validate_point(point, generation=generation)
            if point.qdrant_point_id in expected:
                raise ManagedCollectionValidationError(
                    f"Expected point set contains duplicate ID: {point.qdrant_point_id}"
                )
            expected[point.qdrant_point_id] = point

        issues = self._collection_configuration_issues(
            self.inspect_collection(collection_name)
        )
        actual_count = self.count_points(collection_name)
        if actual_count != len(expected):
            issues.append(
                f"point count {actual_count} does not equal expected {len(expected)}"
            )
        actual_points = {
            point.point_id: point for point in self.scroll_all_points(collection_name)
        }
        missing = sorted(set(expected) - set(actual_points), key=str)
        unexpected = sorted(set(actual_points) - set(expected), key=str)
        if missing:
            issues.append(f"missing point IDs: {', '.join(map(str, missing))}")
        if unexpected:
            issues.append(f"unexpected point IDs: {', '.join(map(str, unexpected))}")
        for point_id in sorted(set(expected) & set(actual_points), key=str):
            issues.extend(
                self._point_validation_issues(
                    expected[point_id],
                    actual_points[point_id],
                )
            )
        if issues:
            raise ManagedCollectionValidationError(
                f"Managed Qdrant collection {collection_name} failed validation:\n- "
                + "\n- ".join(issues)
            )
        return ManagedCollectionValidationResult(
            collection_name=collection_name,
            point_count=actual_count,
            point_ids=tuple(sorted(actual_points, key=str)),
        )

    def _to_point_struct(
        self, point: ManagedPointInput, *, generation: int
    ) -> models.PointStruct:
        self._validate_point(point, generation=generation)
        return models.PointStruct(
            id=point.qdrant_point_id,
            vector={
                self._configuration.dense_vector_name: [
                    float(value) for value in point.dense_vector
                ],
                self._configuration.sparse_vector_name: models.SparseVector(
                    indices=list(point.sparse_indices),
                    values=[float(value) for value in point.sparse_values],
                ),
            },
            payload=deepcopy(dict(point.payload)),
        )

    def _validate_point(self, point: ManagedPointInput, *, generation: int) -> None:
        if point.qdrant_point_id != managed_point_id(point.chunk.chunk_pk):
            raise ManagedVectorValidationError(
                "Prepared point ID does not match its lineage chunk identity"
            )
        if point.index_generation != generation:
            raise ManagedVectorValidationError(
                f"Point generation {point.index_generation} does not match collection "
                f"generation {generation}"
            )
        expected_payload = _point_payload(point, self._configuration)
        if dict(point.payload) != expected_payload:
            raise ManagedVectorValidationError(
                "Prepared point payload does not match its current chunk/build identity"
            )
        _validate_dense_vector(point.dense_vector, self._embedding_dimension)
        _validate_sparse_vector(point.sparse_indices, point.sparse_values)

    def _detach_point(self, record: Any) -> ManagedStoredPoint:
        vector = record.vector
        return ManagedStoredPoint(
            point_id=_as_uuid(record.id, "Qdrant point ID"),
            payload=deepcopy(record.payload or {}),
            dense_vector=_optional_named_dense_vector(
                vector,
                self._configuration.dense_vector_name,
            ),
            sparse_vector=_optional_named_sparse_vector(
                vector,
                self._configuration.sparse_vector_name,
            ),
        )

    def _collection_configuration_issues(
        self, metadata: ManagedCollectionMetadata
    ) -> list[str]:
        issues: list[str] = []
        if self._configuration.dense_vector_name not in metadata.dense_vector_names:
            issues.append(
                f"missing dense vector {self._configuration.dense_vector_name!r}"
            )
        if self._configuration.sparse_vector_name not in metadata.sparse_vector_names:
            issues.append(
                f"missing sparse vector {self._configuration.sparse_vector_name!r}"
            )
        if metadata.dense_dimension != self._embedding_dimension:
            issues.append(
                f"dense dimension {metadata.dense_dimension} != {self._embedding_dimension}"
            )
        if metadata.distance_metric != self._configuration.distance_metric.lower():
            issues.append(
                f"distance {metadata.distance_metric!r} != "
                f"{self._configuration.distance_metric.lower()!r}"
            )
        if metadata.hnsw_m is not None and metadata.hnsw_m != self._configuration.hnsw_m:
            issues.append(f"HNSW m {metadata.hnsw_m} != {self._configuration.hnsw_m}")
        if (
            metadata.hnsw_ef_construct is not None
            and metadata.hnsw_ef_construct != self._configuration.hnsw_ef_construct
        ):
            issues.append(
                f"HNSW ef_construct {metadata.hnsw_ef_construct} != "
                f"{self._configuration.hnsw_ef_construct}"
            )
        if metadata.sparse_modifier != self._configuration.sparse_modifier.lower():
            issues.append(
                f"sparse modifier {metadata.sparse_modifier!r} != "
                f"{self._configuration.sparse_modifier.lower()!r}"
            )
        if (
            metadata.sparse_on_disk is not None
            and metadata.sparse_on_disk != self._configuration.sparse_on_disk
        ):
            issues.append(
                f"sparse on_disk {metadata.sparse_on_disk} != "
                f"{self._configuration.sparse_on_disk}"
            )
        return issues

    def _point_validation_issues(
        self,
        expected: ManagedPointInput,
        actual: ManagedStoredPoint,
    ) -> list[str]:
        issues: list[str] = []
        expected_payload = expected.payload
        for key, value in expected_payload.items():
            if actual.payload.get(key) != value:
                issues.append(
                    f"point {actual.point_id} payload {key!r} mismatch"
                )
        if actual.dense_vector is None:
            issues.append(f"point {actual.point_id} is missing its dense vector")
        else:
            try:
                _validate_dense_vector(
                    actual.dense_vector,
                    self._embedding_dimension,
                )
            except ManagedVectorValidationError as exc:
                issues.append(f"point {actual.point_id} dense vector: {exc}")
        if actual.sparse_vector is None:
            issues.append(f"point {actual.point_id} is missing its sparse vector")
        else:
            try:
                _validate_sparse_vector(
                    actual.sparse_vector.indices,
                    actual.sparse_vector.values,
                )
            except ManagedVectorValidationError as exc:
                issues.append(f"point {actual.point_id} sparse vector: {exc}")
            else:
                if actual.sparse_vector.indices != expected.sparse_indices:
                    issues.append(f"point {actual.point_id} sparse indices mismatch")
                elif not _float_sequences_close(
                    actual.sparse_vector.values,
                    expected.sparse_values,
                ):
                    issues.append(f"point {actual.point_id} sparse values mismatch")
        return issues

    def _require_managed_name(self, collection_name: str) -> None:
        self._generation_from_name(collection_name)

    def _generation_from_name(self, collection_name: str) -> int:
        if collection_name in PROTECTED_COLLECTION_NAMES:
            raise UnsafeManagedCollectionNameError(
                f"Protected Phase 1/2 collection cannot be managed: {collection_name}"
            )
        pattern = re.compile(rf"^{re.escape(self._managed_prefix)}_g([0-9]{{6,}})$")
        match = pattern.fullmatch(collection_name)
        if match is None or not COLLECTION_NAME_PATTERN.fullmatch(collection_name):
            raise UnsafeManagedCollectionNameError(
                f"Collection is not a canonical managed generation name: {collection_name}"
            )
        generation = int(match.group(1))
        if generation <= 0:
            raise UnsafeManagedCollectionNameError("Index generation must be positive")
        return generation

    def _require_existing_collection(self, collection_name: str) -> int:
        generation = self._generation_from_name(collection_name)
        if not self._client.collection_exists(collection_name):
            raise ManagedCollectionNotFoundError(
                f"Managed generation collection does not exist: {collection_name}"
            )
        return generation


def _point_payload(
    point: ManagedPointInput,
    configuration: ManagedIndexConfiguration,
) -> dict[str, Any]:
    chunk = point.chunk
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
        "ingestion_run_id": str(point.ingestion_run_id),
        "index_generation": point.index_generation,
        "index_state_id": str(point.index_state_id),
        "chunking_fingerprint": chunk.chunking_fingerprint,
        "point_id_format_version": configuration.point_id_format_version,
        "payload_schema_version": configuration.payload_schema_version,
        **access_policy_payload(chunk.access_policy),
        **pii_summary_payload(chunk.pii_summary),
    }


def _validate_dense_vector(vector: Sequence[float], expected_dimension: int) -> None:
    if len(vector) != expected_dimension:
        raise ManagedVectorValidationError(
            f"Dense vector dimension {len(vector)} != {expected_dimension}"
        )
    if not vector:
        raise ManagedVectorValidationError("Dense vector cannot be empty")
    for value in vector:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ManagedVectorValidationError("Dense vector values must be numeric")
        if not math.isfinite(float(value)):
            raise ManagedVectorValidationError("Dense vector values must be finite")


def _validate_sparse_vector(
    indices: Sequence[int], values: Sequence[float]
) -> None:
    if len(indices) != len(values):
        raise ManagedVectorValidationError(
            "Sparse vector indices and values must have equal lengths"
        )
    if len(set(indices)) != len(indices):
        raise ManagedVectorValidationError("Sparse vector indices must be unique")
    for index in indices:
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ManagedVectorValidationError(
                "Sparse vector indices must be non-negative integers"
            )
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ManagedVectorValidationError("Sparse vector values must be numeric")
        if not math.isfinite(float(value)):
            raise ManagedVectorValidationError("Sparse vector values must be finite")


def _named_dense_vector(
    vectors: Any,
    vector_name: str,
    expected_dimension: int,
) -> tuple[float, ...]:
    dense = _optional_named_dense_vector(vectors, vector_name)
    if dense is None:
        raise ManagedVectorValidationError(f"Qdrant point is missing {vector_name!r}")
    _validate_dense_vector(dense, expected_dimension)
    return dense


def _optional_named_dense_vector(
    vectors: Any,
    vector_name: str,
) -> tuple[float, ...] | None:
    if not isinstance(vectors, Mapping):
        return None
    raw = vectors.get(vector_name)
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return None
    return tuple(float(value) for value in raw)


def _optional_named_sparse_vector(
    vectors: Any,
    vector_name: str,
) -> StoredSparseVector | None:
    if not isinstance(vectors, Mapping):
        return None
    raw = vectors.get(vector_name)
    if not isinstance(raw, models.SparseVector):
        return None
    return StoredSparseVector(
        indices=tuple(raw.indices),
        values=tuple(float(value) for value in raw.values),
    )


def _payload_string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise ManagedVectorValidationError(
            f"Qdrant payload field {key!r} must be a string"
        )
    return value


def _as_uuid(value: object, label: str) -> UUID:
    if isinstance(value, UUID):
        return value
    try:
        return UUID(str(value))
    except (TypeError, ValueError) as exc:
        raise ManagedVectorValidationError(f"{label} is not a UUID: {value!r}") from exc


def _float_sequences_close(
    actual: Sequence[float],
    expected: Sequence[float],
) -> bool:
    return len(actual) == len(expected) and all(
        math.isclose(
            float(actual_value),
            float(expected_value),
            rel_tol=FLOAT_REL_TOLERANCE,
            abs_tol=FLOAT_ABS_TOLERANCE,
        )
        for actual_value, expected_value in zip(actual, expected, strict=True)
    )


def _distance(value: str) -> models.Distance:
    normalized = value.lower()
    distances = {
        "cosine": models.Distance.COSINE,
        "dot": models.Distance.DOT,
        "euclid": models.Distance.EUCLID,
        "manhattan": models.Distance.MANHATTAN,
    }
    try:
        return distances[normalized]
    except KeyError as exc:
        raise ManagedVectorValidationError(
            f"Unsupported Qdrant distance metric: {value}"
        ) from exc


def _modifier(value: str) -> models.Modifier:
    normalized = value.lower()
    modifiers = {"none": models.Modifier.NONE, "idf": models.Modifier.IDF}
    try:
        return modifiers[normalized]
    except KeyError as exc:
        raise ManagedVectorValidationError(
            f"Unsupported Qdrant sparse modifier: {value}"
        ) from exc


def _validate_prefix(prefix: str) -> None:
    if not COLLECTION_NAME_PATTERN.fullmatch(prefix):
        raise UnsafeManagedCollectionNameError(
            "Managed collection prefix must contain only letters, digits, '_' or '-'"
        )
    if prefix in PROTECTED_COLLECTION_NAMES:
        raise UnsafeManagedCollectionNameError(
            f"Protected Phase 1/2 collection name cannot be a managed prefix: {prefix}"
        )
