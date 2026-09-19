from __future__ import annotations

import ast
import math
from dataclasses import replace
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest
from qdrant_client import QdrantClient, models

from consumer_finance_rag.lineage.fingerprints import ManagedIndexConfiguration
from consumer_finance_rag.lineage.index_types import ActiveChunk, ManagedPointInput
from consumer_finance_rag.lineage.managed_vector_store import (
    FLOAT_ABS_TOLERANCE,
    ManagedCollectionCollisionError,
    ManagedCollectionValidationError,
    ManagedQdrantStore,
    ManagedVectorValidationError,
    UnsafeManagedCollectionNameError,
    managed_generation_collection_name,
    managed_point_id,
)

MANAGED_PREFIX = "consumer_finance_managed"
EMBEDDING_DIMENSION = 3


def managed_configuration() -> ManagedIndexConfiguration:
    return ManagedIndexConfiguration(
        embedding_fingerprint="e" * 64,
        chunking_fingerprint="c" * 64,
        bm25_k1=1.2,
        bm25_b=0.75,
        dense_vector_name="managed_dense",
        sparse_vector_name="managed_sparse",
        distance_metric="cosine",
        hnsw_m=24,
        hnsw_ef_construct=128,
        sparse_modifier="none",
        sparse_on_disk=False,
        managed_index_format_version="managed-hybrid-index-v1",
    )


@pytest.fixture
def client() -> QdrantClient:
    return QdrantClient(":memory:")


@pytest.fixture
def store(client: QdrantClient) -> ManagedQdrantStore:
    return ManagedQdrantStore(
        client,
        managed_prefix=MANAGED_PREFIX,
        configuration=managed_configuration(),
        embedding_dimension=EMBEDDING_DIMENSION,
    )


def active_chunk(index: int = 0) -> ActiveChunk:
    document_pk = UUID(f"00000000-0000-0000-0000-{index + 1:012d}")
    document_version_id = UUID(f"10000000-0000-0000-0000-{index + 1:012d}")
    chunk_pk = UUID(f"20000000-0000-0000-0000-{index + 1:012d}")
    return ActiveChunk(
        document_pk=document_pk,
        document_version_id=document_version_id,
        version_number=index + 1,
        source_key=f"https://example.test/source-{index}",
        source_id=f"source-{index}",
        source_url=f"https://example.test/source-{index}",
        title=f"Consumer guidance {index}",
        category="credit-cards",
        source_version=f"2026-09-{index + 1:02d}",
        content_hash=f"{index + 1:064x}",
        compatibility_document_id=f"document-{index}",
        chunk_pk=chunk_pk,
        compatibility_chunk_id=f"chunk-{index}",
        ordinal=index,
        chunk_hash=f"{index + 101:064x}",
        text=f"Consumer finance evidence {index}",
        chunking_fingerprint="c" * 64,
    )


def prepared_point(index: int = 0, *, generation: int = 1) -> ManagedPointInput:
    chunk = active_chunk(index)
    ingestion_run_id = UUID("30000000-0000-0000-0000-000000000001")
    index_state_id = UUID(f"40000000-0000-0000-0000-{generation:012d}")
    return ManagedPointInput(
        chunk=chunk,
        qdrant_point_id=managed_point_id(chunk.chunk_pk),
        dense_vector=(1.0, float(index + 1), 0.5),
        sparse_indices=(index, index + 10),
        sparse_values=(0.25 + index, 0.75 + index),
        ingestion_run_id=ingestion_run_id,
        index_state_id=index_state_id,
        index_generation=generation,
        payload={
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
            "ingestion_run_id": str(ingestion_run_id),
            "index_generation": generation,
            "index_state_id": str(index_state_id),
            "chunking_fingerprint": chunk.chunking_fingerprint,
            "point_id_format_version": "lineage-chunk-uuid5-v1",
            "payload_schema_version": "managed-qdrant-payload-v1",
        },
    )


def create_and_upsert(
    store: ManagedQdrantStore,
    points: list[ManagedPointInput],
    *,
    generation: int = 1,
) -> str:
    collection_name = store.collection_name(generation)
    store.create_generation_collection(collection_name)
    store.upsert_points(collection_name, points)
    return collection_name


def raw_vectors(point: ManagedPointInput) -> dict[str, Any]:
    return {
        "managed_dense": list(point.dense_vector),
        "managed_sparse": models.SparseVector(
            indices=list(point.sparse_indices),
            values=list(point.sparse_values),
        ),
    }


def test_create_managed_collection_configures_named_dense_sparse_and_hnsw(
    store: ManagedQdrantStore,
) -> None:
    name = store.collection_name(1)

    store.create_generation_collection(name)
    metadata = store.inspect_collection(name)

    assert name == "consumer_finance_managed_g000001"
    assert metadata.dense_vector_names == ("managed_dense",)
    assert metadata.sparse_vector_names == ("managed_sparse",)
    assert metadata.dense_dimension == EMBEDDING_DIMENSION
    assert metadata.distance_metric == "cosine"
    assert metadata.hnsw_m == 24
    assert metadata.hnsw_ef_construct == 128
    assert metadata.sparse_modifier == "none"
    assert metadata.sparse_on_disk is False


def test_existing_collection_collision_is_non_destructive(
    store: ManagedQdrantStore,
) -> None:
    name = store.collection_name(1)
    store.create_generation_collection(name)

    with pytest.raises(ManagedCollectionCollisionError, match="already exists"):
        store.create_generation_collection(name)

    assert store.count_points(name) == 0


@pytest.mark.parametrize(
    "protected_name",
    ["consumer_finance_phase1", "consumer_finance_evaluation"],
)
def test_phase_one_and_evaluation_names_are_rejected(
    store: ManagedQdrantStore,
    protected_name: str,
) -> None:
    with pytest.raises(UnsafeManagedCollectionNameError, match="Protected"):
        store.create_generation_collection(protected_name)
    with pytest.raises(UnsafeManagedCollectionNameError, match="Protected"):
        store.delete_generation_collection(protected_name)


@pytest.mark.parametrize(
    ("prefix", "generation"),
    [("unsafe prefix", 1), ("consumer_finance_phase1", 1), (MANAGED_PREFIX, 0)],
)
def test_managed_collection_name_rejects_unsafe_input(
    prefix: str,
    generation: int,
) -> None:
    with pytest.raises(UnsafeManagedCollectionNameError):
        managed_generation_collection_name(prefix, generation)


def test_managed_point_id_is_deterministic_and_generation_independent() -> None:
    chunk_pk = uuid4()

    first = managed_point_id(chunk_pk)
    second = managed_point_id(chunk_pk)

    assert first == second
    assert first.version == 5
    assert first != managed_point_id(uuid4())
    generation_one = prepared_point(generation=1)
    generation_two = replace(
        generation_one,
        index_generation=2,
        index_state_id=uuid4(),
    )
    assert generation_one.qdrant_point_id == generation_two.qdrant_point_id


def test_one_point_upsert_contains_legacy_and_lineage_payload(
    client: QdrantClient,
    store: ManagedQdrantStore,
) -> None:
    point = prepared_point()
    name = create_and_upsert(store, [point])

    record = client.retrieve(name, [point.qdrant_point_id], with_payload=True)[0]
    payload = record.payload or {}

    assert payload["document_id"] == point.chunk.compatibility_document_id
    assert payload["chunk_id"] == point.chunk.compatibility_chunk_id
    assert payload["document_version"] == point.chunk.source_version
    assert payload["title"] == point.chunk.title
    assert payload["category"] == point.chunk.category
    assert payload["source_url"] == point.chunk.source_url
    assert payload["source_id"] == point.chunk.source_id
    assert payload["chunk_hash"] == point.chunk.chunk_hash
    assert payload["ordinal"] == point.chunk.ordinal
    assert payload["text"] == point.chunk.text
    assert payload["content_hash"] == point.chunk.content_hash
    assert payload["source_version"] == point.chunk.source_version
    assert payload["lineage_document_pk"] == str(point.chunk.document_pk)
    assert payload["lineage_document_version_id"] == str(
        point.chunk.document_version_id
    )
    assert payload["lineage_chunk_pk"] == str(point.chunk.chunk_pk)
    assert payload["ingestion_run_id"] == str(point.ingestion_run_id)
    assert payload["index_generation"] == point.index_generation
    assert payload["index_state_id"] == str(point.index_state_id)
    assert payload["chunking_fingerprint"] == point.chunk.chunking_fingerprint
    assert payload["point_id_format_version"] == "lineage-chunk-uuid5-v1"
    assert payload["payload_schema_version"] == "managed-qdrant-payload-v1"


def test_batched_upsert_waits_and_preserves_all_points(
    client: QdrantClient,
    store: ManagedQdrantStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = store.collection_name(1)
    store.create_generation_collection(name)
    points = [prepared_point(index) for index in range(5)]
    calls: list[tuple[int, bool]] = []
    original_upsert = client.upsert

    def recording_upsert(*args: Any, **kwargs: Any):  # type: ignore[no-untyped-def]
        calls.append((len(kwargs["points"]), kwargs["wait"]))
        return original_upsert(*args, **kwargs)

    monkeypatch.setattr(client, "upsert", recording_upsert)

    store.upsert_points(name, points, batch_size=2)

    assert calls == [(2, True), (2, True), (1, True)]
    assert store.count_points(name) == 5


def test_retrieve_dense_vector_and_payload_identity_are_detached(
    store: ManagedQdrantStore,
) -> None:
    first = prepared_point(0)
    second = prepared_point(1)
    missing = uuid4()
    name = create_and_upsert(store, [first, second])

    result = store.retrieve_dense_vectors(
        name,
        [second.qdrant_point_id, missing, first.qdrant_point_id],
    )

    assert {record.point_id for record in result.records} == {
        first.qdrant_point_id,
        second.qdrant_point_id,
    }
    assert result.missing_point_ids == (missing,)
    record = next(item for item in result.records if item.point_id == first.qdrant_point_id)
    assert isinstance(record.dense_vector, tuple)
    assert len(record.dense_vector) == EMBEDDING_DIMENSION
    assert record.lineage_chunk_pk == first.chunk.chunk_pk
    assert record.chunk_hash == first.chunk.chunk_hash
    assert record.text == first.chunk.text
    assert record.compatibility_chunk_id == first.chunk.compatibility_chunk_id


def test_retrieve_dense_vectors_reports_one_unusable_returned_point(
    client: QdrantClient,
    store: ManagedQdrantStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = store.collection_name(1)
    store.create_generation_collection(name)
    point_id = uuid4()
    monkeypatch.setattr(
        client,
        "retrieve",
        lambda **_kwargs: [
            models.Record(
                id=str(point_id),
                payload={
                    "lineage_chunk_pk": str(uuid4()),
                    "chunk_hash": "a" * 64,
                    "text": "Reusable text",
                    "chunk_id": "chunk-1",
                },
                vector={},
            )
        ],
    )

    result = store.retrieve_dense_vectors(name, [point_id])

    assert result.records == ()
    assert result.missing_point_ids == ()
    assert len(result.issues) == 1
    assert result.issues[0].point_id == point_id
    assert "missing" in result.issues[0].reason


def test_scroll_all_points_follows_pagination_and_returns_both_vectors(
    store: ManagedQdrantStore,
) -> None:
    points = [prepared_point(index) for index in range(5)]
    name = create_and_upsert(store, points)

    stored = store.scroll_all_points(name, page_size=1)

    assert len(stored) == 5
    assert [str(point.point_id) for point in stored] == sorted(
        str(point.qdrant_point_id) for point in points
    )
    assert all(point.dense_vector is not None for point in stored)
    assert all(point.sparse_vector is not None for point in stored)


def test_validator_succeeds_for_complete_collection(store: ManagedQdrantStore) -> None:
    points = [prepared_point(index) for index in range(3)]
    name = create_and_upsert(store, points)

    result = store.validate_generation_collection(name, points)

    assert result.collection_name == name
    assert result.point_count == 3
    assert set(result.point_ids) == {point.qdrant_point_id for point in points}


def test_validator_detects_physical_collection_configuration_mismatch(
    client: QdrantClient,
    store: ManagedQdrantStore,
) -> None:
    name = store.collection_name(1)
    client.create_collection(
        name,
        vectors_config={
            "managed_dense": models.VectorParams(
                size=2,
                distance=models.Distance.DOT,
                hnsw_config=models.HnswConfigDiff(m=8, ef_construct=32),
            )
        },
        sparse_vectors_config={
            "wrong_sparse": models.SparseVectorParams(modifier=models.Modifier.NONE)
        },
    )

    with pytest.raises(
        ManagedCollectionValidationError,
        match="missing sparse vector",
    ) as error:
        store.validate_generation_collection(name, [])

    message = str(error.value)
    assert "dense dimension 2 != 3" in message
    assert "distance 'dot' != 'cosine'" in message
    assert "HNSW m 8 != 24" in message
    assert "HNSW ef_construct 32 != 128" in message


def test_validator_detects_missing_point(store: ManagedQdrantStore) -> None:
    points = [prepared_point(0), prepared_point(1)]
    name = create_and_upsert(store, [points[0]])

    with pytest.raises(ManagedCollectionValidationError, match="missing point IDs"):
        store.validate_generation_collection(name, points)


def test_validator_detects_unexpected_point(store: ManagedQdrantStore) -> None:
    points = [prepared_point(0), prepared_point(1)]
    name = create_and_upsert(store, points)

    with pytest.raises(ManagedCollectionValidationError, match="unexpected point IDs"):
        store.validate_generation_collection(name, [points[0]])


def test_validator_detects_payload_mismatch(
    client: QdrantClient,
    store: ManagedQdrantStore,
) -> None:
    point = prepared_point()
    name = create_and_upsert(store, [point])
    client.overwrite_payload(
        name,
        payload={"chunk_id": "wrong"},
        points=[point.qdrant_point_id],
        wait=True,
    )

    with pytest.raises(ManagedCollectionValidationError, match=r"payload .* mismatch"):
        store.validate_generation_collection(name, [point])


@pytest.mark.parametrize("missing_vector", ["dense", "sparse"])
def test_validator_detects_missing_named_vector_where_constructible(
    client: QdrantClient,
    store: ManagedQdrantStore,
    missing_vector: str,
) -> None:
    point = prepared_point()
    name = store.collection_name(1)
    store.create_generation_collection(name)
    vectors = raw_vectors(point)
    vectors.pop(f"managed_{missing_vector}")
    client.upsert(
        name,
        points=[
            models.PointStruct(
                id=point.qdrant_point_id,
                vector=vectors,
                payload={
                    "chunk_id": point.chunk.compatibility_chunk_id,
                    "lineage_chunk_pk": str(point.chunk.chunk_pk),
                },
            )
        ],
        wait=True,
    )

    with pytest.raises(
        ManagedCollectionValidationError,
        match=f"missing its {missing_vector} vector",
    ):
        store.validate_generation_collection(name, [point])


@pytest.mark.parametrize(
    "dense_vector",
    [
        (),
        (1.0, 2.0),
        (1.0, 2.0, float("nan")),
        (1.0, 2.0, float("inf")),
    ],
    ids=["empty", "wrong-dimension", "nan", "infinity"],
)
def test_invalid_dense_vectors_are_rejected_before_upsert(
    store: ManagedQdrantStore,
    dense_vector: tuple[float, ...],
) -> None:
    name = store.collection_name(1)
    store.create_generation_collection(name)
    point = replace(prepared_point(), dense_vector=dense_vector)

    with pytest.raises(ManagedVectorValidationError, match="Dense vector"):
        store.upsert_points(name, [point])

    assert store.count_points(name) == 0


@pytest.mark.parametrize(
    ("indices", "values"),
    [
        ((1,), (0.1, 0.2)),
        ((-1,), (0.1,)),
        ((1, 1), (0.1, 0.2)),
        ((1,), (float("nan"),)),
        ((1,), (float("inf"),)),
    ],
    ids=["length", "negative", "duplicate", "nan", "infinity"],
)
def test_malformed_sparse_vectors_are_rejected_before_upsert(
    store: ManagedQdrantStore,
    indices: tuple[int, ...],
    values: tuple[float, ...],
) -> None:
    name = store.collection_name(1)
    store.create_generation_collection(name)
    point = replace(
        prepared_point(),
        sparse_indices=indices,
        sparse_values=values,
    )

    with pytest.raises(ManagedVectorValidationError, match="Sparse vector"):
        store.upsert_points(name, [point])

    assert store.count_points(name) == 0


def test_sparse_validation_uses_float32_appropriate_tolerance(
    store: ManagedQdrantStore,
) -> None:
    point = prepared_point()
    name = create_and_upsert(store, [point])
    within_tolerance = replace(
        point,
        sparse_values=(
            point.sparse_values[0] + FLOAT_ABS_TOLERANCE / 2,
            point.sparse_values[1],
        ),
    )

    store.validate_generation_collection(name, [within_tolerance])

    outside_tolerance = replace(
        point,
        sparse_values=(point.sparse_values[0] + 0.01, point.sparse_values[1]),
    )
    with pytest.raises(ManagedCollectionValidationError, match="sparse values mismatch"):
        store.validate_generation_collection(name, [outside_tolerance])


def test_old_and_new_generations_coexist_and_delete_is_generation_scoped(
    client: QdrantClient,
    store: ManagedQdrantStore,
) -> None:
    first_name = create_and_upsert(store, [prepared_point(0, generation=1)])
    second_name = create_and_upsert(
        store,
        [prepared_point(0, generation=2)],
        generation=2,
    )

    assert client.collection_exists(first_name)
    assert client.collection_exists(second_name)
    store.delete_generation_collection(first_name)

    assert not client.collection_exists(first_name)
    assert client.collection_exists(second_name)
    assert store.count_points(second_name) == 1


def test_phase_one_and_evaluation_collections_remain_untouched(
    client: QdrantClient,
    store: ManagedQdrantStore,
) -> None:
    for protected_name in ("consumer_finance_phase1", "consumer_finance_evaluation"):
        client.create_collection(
            protected_name,
            vectors_config=models.VectorParams(size=1, distance=models.Distance.COSINE),
        )
    name = create_and_upsert(store, [prepared_point()])

    assert store.count_points(name) == 1
    assert client.collection_exists("consumer_finance_phase1")
    assert client.collection_exists("consumer_finance_evaluation")
    assert client.count("consumer_finance_phase1", exact=True).count == 0
    assert client.count("consumer_finance_evaluation", exact=True).count == 0


def test_adapter_has_no_embedding_bm25_or_database_dependencies() -> None:
    path = Path("src/consumer_finance_rag/lineage/managed_vector_store.py")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    forbidden = {
        "sqlalchemy",
        "consumer_finance_rag.providers",
        "consumer_finance_rag.sparse",
        "consumer_finance_rag.lineage.models",
        "consumer_finance_rag.lineage.repository",
        "consumer_finance_rag.lineage.bm25_generation",
    }
    assert not {
        name
        for name in imported
        if any(name == root or name.startswith(f"{root}.") for root in forbidden)
    }


def test_retrieved_dense_values_are_finite_after_qdrant_cosine_storage(
    store: ManagedQdrantStore,
) -> None:
    point = prepared_point()
    name = create_and_upsert(store, [point])

    retrieved = store.retrieve_dense_vectors(name, [point.qdrant_point_id]).records[0]

    assert all(math.isfinite(value) for value in retrieved.dense_vector)
    assert len(retrieved.dense_vector) == EMBEDDING_DIMENSION
