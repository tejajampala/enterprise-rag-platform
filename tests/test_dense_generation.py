from __future__ import annotations

import ast
import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4, uuid5

import pytest

from consumer_finance_rag.lineage.dense_generation import (
    CorruptActiveDenseGenerationError,
    DenseGenerationError,
    DenseGenerationPlanner,
    DenseReuseInvariantError,
    EmbeddingDimensionMismatchError,
)
from consumer_finance_rag.lineage.index_types import (
    ActiveChunk,
    ActiveChunkManifest,
    DensePlanAction,
    DenseReuseCandidate,
    DenseVectorRetrievalIssue,
    DenseVectorRetrievalResult,
    PreviousDenseGenerationState,
    RetrievedDenseVector,
)

NAMESPACE = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
FINGERPRINT = "f" * 64
NEW_FINGERPRINT = "e" * 64


def uid(value: str) -> UUID:
    return uuid5(NAMESPACE, value)


def active_chunk(
    text: str,
    ordinal: int,
    *,
    chunk_hash: str | None = None,
    chunk_pk: UUID | None = None,
    compatibility_chunk_id: str | None = None,
    chunking_fingerprint: str = "c" * 64,
) -> ActiveChunk:
    digest = chunk_hash or f"{ordinal + 1:064x}"
    return ActiveChunk(
        document_pk=uid(f"document-{ordinal}"),
        document_version_id=uid(f"version-{ordinal}"),
        version_number=1,
        source_key=f"source:{ordinal}",
        source_id=f"source-{ordinal}",
        source_url=f"https://example.test/{ordinal}",
        title=f"Title {ordinal}",
        category="credit-cards",
        source_version="2026-09-19",
        content_hash=f"{ordinal + 100:064x}",
        compatibility_document_id=f"doc-{ordinal}",
        chunk_pk=chunk_pk or uid(f"chunk-{ordinal}-{text}"),
        compatibility_chunk_id=compatibility_chunk_id or f"compat-{ordinal}",
        ordinal=ordinal,
        chunk_hash=digest,
        text=text,
        chunking_fingerprint=chunking_fingerprint,
    )


def manifest(*chunks: ActiveChunk) -> ActiveChunkManifest:
    return ActiveChunkManifest(
        corpus_id=uid("corpus"),
        corpus_key="consumer-finance",
        source_snapshot_hash="s" * 64,
        chunking_fingerprint="c" * 64,
        documents=(),
        chunks=chunks,
        document_count=0,
        chunk_count=len(chunks),
        manifest_hash="m" * 64,
    )


def previous(**overrides: Any) -> PreviousDenseGenerationState:
    values: dict[str, Any] = {
        "index_state_id": uid("index-state"),
        "qdrant_collection": "managed_g000001",
        "embedding_fingerprint": FINGERPRINT,
        "embedding_dimension": 2,
        "dense_vector_name": "dense_hnsw",
        "generation": 1,
    }
    values.update(overrides)
    return PreviousDenseGenerationState(**values)


def candidate(
    chunk: ActiveChunk,
    *,
    previous_chunk_pk: UUID | None = None,
    compatibility_chunk_id: str | None = None,
    text: str | None = None,
) -> DenseReuseCandidate:
    selected_compatibility_id = compatibility_chunk_id or chunk.compatibility_chunk_id
    source_pk = previous_chunk_pk or uid(f"prior-{selected_compatibility_id}")
    return DenseReuseCandidate(
        previous_chunk_pk=source_pk,
        chunk_hash=chunk.chunk_hash,
        text=chunk.text if text is None else text,
        qdrant_point_id=uid(f"point-{source_pk}"),
        compatibility_chunk_id=selected_compatibility_id,
    )


def record(
    source: DenseReuseCandidate,
    *,
    vector: tuple[float, ...] = (0.1, 0.2),
    lineage_chunk_pk: UUID | None = None,
    chunk_hash: str | None = None,
    text: str | None = None,
    compatibility_chunk_id: str | None = None,
) -> RetrievedDenseVector:
    return RetrievedDenseVector(
        point_id=source.qdrant_point_id,
        dense_vector=vector,
        lineage_chunk_pk=lineage_chunk_pk or source.previous_chunk_pk,
        chunk_hash=chunk_hash or source.chunk_hash,
        text=source.text if text is None else text,
        compatibility_chunk_id=(
            compatibility_chunk_id or source.compatibility_chunk_id
        ),
    )


class FakeCandidates:
    def __init__(self, candidates: Sequence[DenseReuseCandidate] = ()) -> None:
        self.candidates = list(candidates)
        self.calls: list[tuple[UUID, tuple[str, ...]]] = []

    def find_reuse_candidates_for_hashes(
        self, index_state_id: UUID, chunk_hashes: Sequence[str]
    ) -> list[DenseReuseCandidate]:
        self.calls.append((index_state_id, tuple(chunk_hashes)))
        requested = set(chunk_hashes)
        return [item for item in self.candidates if item.chunk_hash in requested]


class FakeVectors:
    def __init__(
        self,
        records: Sequence[RetrievedDenseVector] = (),
        *,
        missing: Sequence[UUID] = (),
        issues: Sequence[DenseVectorRetrievalIssue] = (),
        error: Exception | None = None,
        errors: Sequence[Exception] = (),
    ) -> None:
        self.records = {item.point_id: item for item in records}
        self.missing = set(missing)
        self.issues = {item.point_id: item for item in issues}
        self.error = error
        self.errors = list(errors)
        self.read_calls: list[tuple[str, tuple[UUID, ...]]] = []
        self.mutation_calls = 0

    def retrieve_dense_vectors(
        self, collection_name: str, point_ids: Sequence[UUID]
    ) -> DenseVectorRetrievalResult:
        requested = tuple(point_ids)
        self.read_calls.append((collection_name, requested))
        if self.errors:
            raise self.errors.pop(0)
        if self.error is not None:
            raise self.error
        return DenseVectorRetrievalResult(
            records=tuple(
                self.records[point_id]
                for point_id in requested
                if point_id in self.records
            ),
            missing_point_ids=tuple(
                point_id
                for point_id in requested
                if point_id in self.missing
                or (point_id not in self.records and point_id not in self.issues)
            ),
            issues=tuple(
                self.issues[point_id]
                for point_id in requested
                if point_id in self.issues
            ),
        )

    def create_generation_collection(self, _name: str) -> None:
        self.mutation_calls += 1
        raise AssertionError("dense planning must not create Qdrant collections")

    def upsert_points(self, _name: str, _points: Sequence[object]) -> None:
        self.mutation_calls += 1
        raise AssertionError("dense planning must not upsert Qdrant points")

    def delete_generation_collection(self, _name: str) -> None:
        self.mutation_calls += 1
        raise AssertionError("dense planning must not delete Qdrant collections")


class FakeEmbeddings:
    def __init__(self, vectors: Sequence[Sequence[float]]) -> None:
        self.vectors = [list(vector) for vector in vectors]
        self.calls: list[tuple[str, ...]] = []

    @property
    def model_name(self) -> str:
        return "fake-embedding"

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(tuple(texts))
        return [list(vector) for vector in self.vectors]

    def embed_query(self, _text: str) -> list[float]:
        raise AssertionError("dense generation must use document encoding")


def planner(
    candidates: FakeCandidates,
    vectors: FakeVectors,
    embeddings: FakeEmbeddings,
    *,
    batch_size: int = 256,
    max_attempts: int = 3,
    backoff_seconds: float = 0.25,
    sleeper: Callable[[float], None] = lambda _delay: None,
) -> DenseGenerationPlanner:
    return DenseGenerationPlanner(
        candidates,
        vectors,
        embeddings,
        retrieval_batch_size=batch_size,
        retrieval_max_attempts=max_attempts,
        retrieval_backoff_seconds=backoff_seconds,
        sleeper=sleeper,
    )


def test_first_generation_computes_all_in_manifest_order_and_infers_dimension() -> None:
    second = active_chunk("second", 1)
    first = active_chunk("first", 0)
    candidates = FakeCandidates()
    vectors = FakeVectors()
    embeddings = FakeEmbeddings(((2.0, 3.0, 4.0), (5.0, 6.0, 7.0)))

    plan = planner(candidates, vectors, embeddings).plan(
        manifest(second, first), FINGERPRINT
    )

    assert plan.embedding_dimension == 3
    assert [entry.lineage_chunk_pk for entry in plan.chunks] == [
        second.chunk_pk,
        first.chunk_pk,
    ]
    assert [entry.action for entry in plan.chunks] == [
        DensePlanAction.COMPUTED,
        DensePlanAction.COMPUTED,
    ]
    assert plan.reused_count == 0
    assert plan.computed_count == 2
    assert embeddings.calls == [("second", "first")]
    assert candidates.calls == []
    assert vectors.read_calls == []
    assert all(entry.dense_source_chunk_pk is None for entry in plan.chunks)


def test_all_reusable_skips_embedding_and_preserves_provenance() -> None:
    chunks = (active_chunk("one", 0), active_chunk("two", 1))
    sources = tuple(candidate(chunk) for chunk in chunks)
    candidates = FakeCandidates(sources)
    vectors = FakeVectors(tuple(record(source) for source in sources))
    embeddings = FakeEmbeddings(())

    plan = planner(candidates, vectors, embeddings).plan(
        manifest(*chunks), FINGERPRINT, previous()
    )

    assert plan.reused_count == 2
    assert plan.computed_count == 0
    assert embeddings.calls == []
    assert [entry.dense_source_chunk_pk for entry in plan.chunks] == [
        source.previous_chunk_pk for source in sources
    ]
    assert [entry.dense_source_point_id for entry in plan.chunks] == [
        source.qdrant_point_id for source in sources
    ]
    assert vectors.mutation_calls == 0


def test_partial_reuse_embeds_only_non_reused_chunks() -> None:
    reused_chunk = active_chunk("stable", 0)
    changed_chunk = active_chunk("changed", 1)
    source = candidate(reused_chunk)
    embeddings = FakeEmbeddings(((0.3, 0.4),))

    plan = planner(
        FakeCandidates((source,)),
        FakeVectors((record(source),)),
        embeddings,
    ).plan(manifest(reused_chunk, changed_chunk), FINGERPRINT, previous())

    assert [entry.action for entry in plan.chunks] == [
        DensePlanAction.REUSED,
        DensePlanAction.COMPUTED,
    ]
    assert embeddings.calls == [("changed",)]
    assert plan.chunks[1].dense_source_chunk_pk is None


def test_same_hash_with_different_text_is_an_explicit_invariant_failure() -> None:
    current = active_chunk("current", 0)
    source = candidate(current, text="different")

    with pytest.raises(DenseReuseInvariantError, match="different text"):
        planner(FakeCandidates((source,)), FakeVectors(), FakeEmbeddings(())).plan(
            manifest(current), FINGERPRINT, previous()
        )


def test_compatibility_id_and_ordinal_alone_never_enable_reuse() -> None:
    current = active_chunk("current", 0, chunk_hash="1" * 64)
    unrelated = candidate(
        active_chunk(
            "other",
            0,
            chunk_hash="2" * 64,
            compatibility_chunk_id=current.compatibility_chunk_id,
        )
    )
    embeddings = FakeEmbeddings(((0.1, 0.2),))

    plan = planner(
        FakeCandidates((unrelated,)), FakeVectors(), embeddings
    ).plan(manifest(current), FINGERPRINT, previous())

    assert plan.chunks[0].action is DensePlanAction.COMPUTED
    assert plan.chunks[0].reason == "no_exact_previous_chunk"
    assert embeddings.calls == [("current",)]


@pytest.mark.parametrize("reverse", [False, True])
def test_duplicate_candidates_choose_deterministic_business_order(reverse: bool) -> None:
    current = active_chunk("same", 0)
    later = candidate(current, compatibility_chunk_id="z-source")
    earlier = candidate(current, compatibility_chunk_id="a-source")
    sources = [later, earlier]
    if reverse:
        sources.reverse()
    vectors = FakeVectors(tuple(record(source) for source in sources))

    plan = planner(
        FakeCandidates(sources), vectors, FakeEmbeddings(())
    ).plan(manifest(current), FINGERPRINT, previous())

    assert plan.chunks[0].dense_source_chunk_pk == earlier.previous_chunk_pk
    assert vectors.read_calls[0][1] == (earlier.qdrant_point_id,)


def test_changed_embedding_fingerprint_computes_all_without_old_reads() -> None:
    current = active_chunk("current", 0)
    source = candidate(current)
    candidates = FakeCandidates((source,))
    vectors = FakeVectors((record(source),))
    embeddings = FakeEmbeddings(((1.0, 2.0, 3.0),))

    plan = planner(candidates, vectors, embeddings).plan(
        manifest(current), NEW_FINGERPRINT, previous()
    )

    assert plan.embedding_dimension == 3
    assert plan.chunks[0].action is DensePlanAction.COMPUTED
    assert plan.chunks[0].reason == "embedding_fingerprint_changed"
    assert candidates.calls == []
    assert vectors.read_calls == []


def test_reused_dimension_mismatch_falls_back_to_computation() -> None:
    current = active_chunk("current", 0)
    source = candidate(current)
    embeddings = FakeEmbeddings(((0.8, 0.9),))

    plan = planner(
        FakeCandidates((source,)),
        FakeVectors((record(source, vector=(0.1, 0.2, 0.3)),)),
        embeddings,
    ).plan(manifest(current), FINGERPRINT, previous())

    assert plan.chunks[0].action is DensePlanAction.COMPUTED
    assert plan.chunks[0].reason == "previous_vector_invalid"
    assert embeddings.calls == [("current",)]


def test_non_finite_returned_previous_vector_falls_back_to_computation() -> None:
    current = active_chunk("current", 0)
    source = candidate(current)

    plan = planner(
        FakeCandidates((source,)),
        FakeVectors((record(source, vector=(math.nan, 0.2)),)),
        FakeEmbeddings(((0.8, 0.9),)),
    ).plan(manifest(current), FINGERPRINT, previous())

    assert plan.chunks[0].action is DensePlanAction.COMPUTED
    assert plan.chunks[0].reason == "previous_vector_invalid"


def test_computed_fallback_must_match_previous_dimension() -> None:
    current = active_chunk("current", 0)

    with pytest.raises(EmbeddingDimensionMismatchError, match="previous ACTIVE"):
        planner(
            FakeCandidates(),
            FakeVectors(),
            FakeEmbeddings(((0.1, 0.2, 0.3),)),
        ).plan(manifest(current), FINGERPRINT, previous())


def test_first_generation_rejects_inconsistent_computed_dimensions() -> None:
    chunks = (active_chunk("one", 0), active_chunk("two", 1))

    with pytest.raises(EmbeddingDimensionMismatchError, match="inconsistent"):
        planner(
            FakeCandidates(),
            FakeVectors(),
            FakeEmbeddings(((0.1, 0.2), (0.3, 0.4, 0.5))),
        ).plan(manifest(*chunks), FINGERPRINT)


@pytest.mark.parametrize(
    ("result_kind", "reason"),
    [
        ("missing", "previous_vector_missing"),
        ("missing_dense", "previous_vector_invalid"),
        ("non_finite", "previous_vector_invalid"),
    ],
)
def test_unavailable_individual_candidate_computes_fallback(
    result_kind: str, reason: str
) -> None:
    current = active_chunk("current", 0)
    source = candidate(current)
    missing: tuple[UUID, ...] = ()
    issues: tuple[DenseVectorRetrievalIssue, ...] = ()
    if result_kind == "missing":
        missing = (source.qdrant_point_id,)
    else:
        issues = (
            DenseVectorRetrievalIssue(
                source.qdrant_point_id,
                "missing vector" if result_kind == "missing_dense" else "not finite",
            ),
        )
    store = FakeVectors(missing=missing, issues=issues)

    plan = planner(
        FakeCandidates((source,)), store, FakeEmbeddings(((0.1, 0.2),))
    ).plan(manifest(current), FINGERPRINT, previous())

    assert plan.chunks[0].action is DensePlanAction.COMPUTED
    assert plan.chunks[0].reason == reason


def test_qdrant_infrastructure_failure_aborts_instead_of_computing() -> None:
    current = active_chunk("current", 0)
    source = candidate(current)
    embeddings = FakeEmbeddings(((0.1, 0.2),))

    with pytest.raises(DenseGenerationError, match="Qdrant"):
        planner(
            FakeCandidates((source,)),
            FakeVectors(error=ConnectionError("offline")),
            embeddings,
        ).plan(manifest(current), FINGERPRINT, previous())
    assert embeddings.calls == []


def test_transient_qdrant_read_retries_then_reuses_without_embedding() -> None:
    current = active_chunk("current", 0)
    source = candidate(current)
    vectors = FakeVectors(
        (record(source),),
        errors=(ConnectionError("starting"), TimeoutError("not ready")),
    )
    delays: list[float] = []
    embeddings = FakeEmbeddings(())

    plan = planner(
        FakeCandidates((source,)),
        vectors,
        embeddings,
        max_attempts=3,
        backoff_seconds=0.1,
        sleeper=delays.append,
    ).plan(manifest(current), FINGERPRINT, previous())

    assert plan.chunks[0].action is DensePlanAction.REUSED
    assert len(vectors.read_calls) == 3
    assert delays == [0.1, 0.2]
    assert embeddings.calls == []


def test_transient_qdrant_read_retry_is_bounded_and_never_recomputes() -> None:
    current = active_chunk("current", 0)
    source = candidate(current)
    vectors = FakeVectors(error=ConnectionError("offline"))
    delays: list[float] = []
    embeddings = FakeEmbeddings(((0.1, 0.2),))

    with pytest.raises(DenseGenerationError, match="Qdrant"):
        planner(
            FakeCandidates((source,)),
            vectors,
            embeddings,
            max_attempts=3,
            backoff_seconds=0.1,
            sleeper=delays.append,
        ).plan(manifest(current), FINGERPRINT, previous())

    assert len(vectors.read_calls) == 3
    assert delays == [0.1, 0.2]
    assert embeddings.calls == []


def test_non_transient_qdrant_read_error_is_not_retried_or_recomputed() -> None:
    current = active_chunk("current", 0)
    source = candidate(current)
    vectors = FakeVectors(error=ValueError("invalid stored response"))
    delays: list[float] = []
    embeddings = FakeEmbeddings(((0.1, 0.2),))

    with pytest.raises(DenseGenerationError, match="Qdrant"):
        planner(
            FakeCandidates((source,)),
            vectors,
            embeddings,
            max_attempts=3,
            backoff_seconds=0.1,
            sleeper=delays.append,
        ).plan(manifest(current), FINGERPRINT, previous())

    assert len(vectors.read_calls) == 1
    assert delays == []
    assert embeddings.calls == []


def test_metadata_only_new_lineage_chunk_reuses_exact_prior_text() -> None:
    current = active_chunk("unchanged text", 0, chunk_pk=uuid4())
    source = candidate(current, previous_chunk_pk=uuid4())

    plan = planner(
        FakeCandidates((source,)),
        FakeVectors((record(source),)),
        FakeEmbeddings(()),
    ).plan(manifest(current), FINGERPRINT, previous())

    assert current.chunk_pk != source.previous_chunk_pk
    assert plan.chunks[0].action is DensePlanAction.REUSED


def test_chunking_change_reuses_survivor_computes_added_and_omits_deleted() -> None:
    survivor = active_chunk("survivor", 0, chunking_fingerprint="n" * 64)
    added = active_chunk("added", 1, chunking_fingerprint="n" * 64)
    deleted = active_chunk("deleted", 2)
    survivor_source = candidate(survivor)
    deleted_source = candidate(deleted)

    plan = planner(
        FakeCandidates((deleted_source, survivor_source)),
        FakeVectors((record(survivor_source), record(deleted_source))),
        FakeEmbeddings(((0.7, 0.8),)),
    ).plan(manifest(survivor, added), FINGERPRINT, previous())

    assert [entry.action for entry in plan.chunks] == [
        DensePlanAction.REUSED,
        DensePlanAction.COMPUTED,
    ]
    assert all(entry.lineage_chunk_pk != deleted.chunk_pk for entry in plan.chunks)


@pytest.mark.parametrize(
    "override",
    [
        {"lineage_chunk_pk": uuid4()},
        {"chunk_hash": "9" * 64},
        {"text": "wrong payload text"},
        {"compatibility_chunk_id": "wrong-compatibility-id"},
    ],
)
def test_returned_point_identity_mismatch_prevents_reuse(
    override: dict[str, object]
) -> None:
    current = active_chunk("current", 0)
    source = candidate(current)
    bad_record = record(source, **override)  # type: ignore[arg-type]

    plan = planner(
        FakeCandidates((source,)),
        FakeVectors((bad_record,)),
        FakeEmbeddings(((0.4, 0.5),)),
    ).plan(manifest(current), FINGERPRINT, previous())

    assert plan.chunks[0].action is DensePlanAction.COMPUTED
    assert plan.chunks[0].reason == "previous_vector_invalid"


def test_qdrant_reads_are_bounded_batches_not_per_chunk_calls() -> None:
    chunks = tuple(active_chunk(f"chunk {index}", index) for index in range(5))
    sources = tuple(candidate(chunk) for chunk in chunks)
    store = FakeVectors(tuple(record(source) for source in sources))

    plan = planner(
        FakeCandidates(sources), store, FakeEmbeddings(()), batch_size=2
    ).plan(manifest(*chunks), FINGERPRINT, previous())

    assert plan.reused_count == 5
    assert [len(call[1]) for call in store.read_calls] == [2, 2, 1]


@pytest.mark.parametrize(
    "overrides",
    [
        {"qdrant_collection": ""},
        {"embedding_fingerprint": ""},
        {"embedding_dimension": None},
        {"embedding_dimension": 0},
        {"dense_vector_name": None},
        {"generation": 0},
    ],
)
def test_corrupt_previous_active_metadata_fails_explicitly(
    overrides: dict[str, object]
) -> None:
    current = active_chunk("current", 0)

    with pytest.raises(CorruptActiveDenseGenerationError):
        planner(FakeCandidates(), FakeVectors(), FakeEmbeddings(())).plan(
            manifest(current), FINGERPRINT, previous(**overrides)
        )


@pytest.mark.parametrize("vector", [(), (math.nan, 0.1), (math.inf, 0.1)])
def test_new_computed_vectors_must_be_positive_dimension_and_finite(
    vector: tuple[float, ...]
) -> None:
    current = active_chunk("current", 0)

    with pytest.raises(DenseGenerationError, match=r"non-finite|empty"):
        planner(
            FakeCandidates(), FakeVectors(), FakeEmbeddings((vector,))
        ).plan(manifest(current), FINGERPRINT)


def test_planner_module_has_no_bm25_or_persistence_mutation_dependencies() -> None:
    path = Path("src/consumer_finance_rag/lineage/dense_generation.py")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}

    assert "BM25SparseEncoder" not in names
    assert not {
        "create_building_index_state",
        "record_index_state_chunk",
        "transition_index_state",
        "create_generation_collection",
        "upsert_points",
        "delete_generation_collection",
    } & attributes
