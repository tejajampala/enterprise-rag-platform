from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

import pytest
from qdrant_client import QdrantClient, models
from sqlalchemy import create_engine, func, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from consumer_finance_rag.config import AppSettings, load_settings
from consumer_finance_rag.lineage.fingerprints import (
    ManagedIndexConfigurationIdentity,
    embedding_fingerprint,
    managed_index_configuration_identity_from_settings,
)
from consumer_finance_rag.lineage.managed_retrieval import (
    CorruptActiveManagedIndexError,
    ManagedCollectionUnavailableError,
    ManagedEmbeddingCompatibilityError,
    ManagedRetrievalError,
    NoActiveManagedIndexError,
    build_managed_retrieval_runtime,
    build_managed_retrieval_runtime_for_generation,
)
from consumer_finance_rag.lineage.managed_vector_store import (
    managed_generation_collection_name,
)
from consumer_finance_rag.lineage.models import Base, IndexState
from consumer_finance_rag.lineage.repository import (
    InvalidLineageStateError,
    LineageRepository,
)
from consumer_finance_rag.models import RetrievedChunk
from consumer_finance_rag.observability import (
    RetrievalMetrics,
    RetrievalTelemetrySink,
)
from consumer_finance_rag.sparse import (
    BM25SparseEncoder,
    bm25_state_hash,
    bm25_state_to_json_dict,
)

CORPUS_KEY = "consumer-finance-managed"
PREFIX = "consumer_finance_managed"
CHUNKING_FINGERPRINT = "c" * 64
EMBEDDING_DIMENSION = 2


class FakeEmbeddingProvider:
    def __init__(self, model_name: str, *, dimension: int = EMBEDDING_DIMENSION) -> None:
        self._model_name = model_name
        self.dimension = dimension
        self.document_calls = 0
        self.query_calls = 0

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.document_calls += 1
        raise AssertionError(f"managed retrieval must not embed documents: {texts!r}")

    def embed_query(self, text: str) -> list[float]:
        self.query_calls += 1
        return [1.0, 0.0]


class FakeReranker:
    def __init__(self, model_name: str = "fake-reranker") -> None:
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        del query
        return list(candidates[:top_k])


class RecordingTelemetrySink:
    def __init__(self) -> None:
        self.events: list[RetrievalMetrics] = []

    def emit(self, metrics: RetrievalMetrics) -> None:
        self.events.append(metrics)


def settings() -> AppSettings:
    base = load_settings("config/config.yaml")
    return base.model_copy(
        update={
            "managed_index": base.managed_index.model_copy(
                update={
                    "dense_vector_name": "managed_dense",
                    "sparse_vector_name": "managed_sparse",
                }
            )
        }
    )


def database() -> Engine:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    return engine


def embedding_fp(app_settings: AppSettings) -> str:
    configured = app_settings.models.embedding
    return embedding_fingerprint(
        provider=configured.provider,
        model=configured.model,
        trust_remote_code=configured.trust_remote_code,
        device=configured.device,
    )


def configuration_identity(
    app_settings: AppSettings,
) -> ManagedIndexConfigurationIdentity:
    return managed_index_configuration_identity_from_settings(
        app_settings.managed_index,
        embedding_fingerprint=embedding_fp(app_settings),
        chunking_fingerprint=CHUNKING_FINGERPRINT,
    )


def seed_generation(
    engine: Engine,
    app_settings: AppSettings,
    *,
    generation: int = 1,
    status: str = "ACTIVE",
) -> tuple[UUID, UUID, BM25SparseEncoder]:
    encoder = BM25SparseEncoder.fit(["fraud payment help"])
    identity = configuration_identity(app_settings)
    with Session(engine) as session, session.begin():
        repository = LineageRepository(session)
        corpus = repository.get_or_create_corpus(CORPUS_KEY, PREFIX)
        run = repository.start_ingestion_run(
            corpus.corpus_id,
            authoritative_snapshot=True,
            snapshot_hash="a" * 64,
        )
        repository.finish_ingestion_run(
            run.run_id,
            "SUCCEEDED",
            datetime.now(UTC),
            1,
            0,
            0,
            0,
        )
        state = repository.create_building_index_state(
            corpus.corpus_id,
            generation,
            run.run_id,
            managed_generation_collection_name(PREFIX, generation),
            "a" * 64,
            identity.configuration.embedding_fingerprint,
            CHUNKING_FINGERPRINT,
            document_count=1,
            chunk_count=1,
        )
        repository.update_index_build_metadata(
            state.index_state_id,
            bm25_state=bm25_state_to_json_dict(encoder.state),
            bm25_state_hash=bm25_state_hash(encoder.state),
            bm25_corpus_hash=encoder.state.corpus_hash,
            index_configuration=identity.payload,
            index_configuration_fingerprint=identity.fingerprint,
            embedding_dimension=EMBEDDING_DIMENSION,
            document_count=1,
            chunk_count=1,
        )
        if status == "READY":
            repository.transition_index_state(state.index_state_id, "BUILDING", "READY")
        elif status == "ACTIVE":
            repository.transition_index_state(state.index_state_id, "BUILDING", "READY")
            repository.transition_index_state(
                state.index_state_id,
                "READY",
                "ACTIVE",
                occurred_at=datetime.now(UTC),
            )
        elif status == "RETIRED":
            repository.transition_index_state(state.index_state_id, "BUILDING", "READY")
            repository.transition_index_state(
                state.index_state_id,
                "READY",
                "ACTIVE",
                occurred_at=datetime.now(UTC),
            )
            repository.transition_index_state(
                state.index_state_id,
                "ACTIVE",
                "RETIRED",
                occurred_at=datetime.now(UTC),
            )
        elif status == "FAILED":
            repository.transition_index_state(
                state.index_state_id,
                "BUILDING",
                "FAILED",
                occurred_at=datetime.now(UTC),
                error_message="test failure",
            )
        elif status != "BUILDING":
            raise ValueError(f"unsupported test status: {status}")
        return state.index_state_id, corpus.corpus_id, encoder


def create_physical_collection(
    client: QdrantClient,
    app_settings: AppSettings,
    encoder: BM25SparseEncoder,
    *,
    generation: int = 1,
    dimension: int = EMBEDDING_DIMENSION,
    distance: models.Distance = models.Distance.COSINE,
    include_point: bool = True,
    dense_name: str | None = None,
    sparse_name: str | None = None,
) -> str:
    collection = managed_generation_collection_name(PREFIX, generation)
    dense = dense_name or app_settings.managed_index.dense_vector_name
    sparse = sparse_name or app_settings.managed_index.sparse_vector_name
    client.create_collection(
        collection_name=collection,
        vectors_config={
            dense: models.VectorParams(size=dimension, distance=distance)
        },
        sparse_vectors_config={
            sparse: models.SparseVectorParams(modifier=models.Modifier.NONE)
        },
    )
    if include_point:
        sparse_vector = encoder.encode_document("fraud payment help")
        client.upsert(
            collection_name=collection,
            wait=True,
            points=[
                models.PointStruct(
                    id=str(uuid4()),
                    vector={
                        dense: [1.0] + [0.0] * (dimension - 1),
                        sparse: models.SparseVector(
                            indices=sparse_vector.indices,
                            values=sparse_vector.values,
                        ),
                    },
                    payload={
                        "document_id": "doc-1",
                        "chunk_id": "chunk-1",
                        "document_version": "v1",
                        "title": "Fraud help",
                        "category": "payments-fraud",
                        "source_url": "https://example.test/fraud",
                        "source_id": "fraud",
                        "chunk_hash": "d" * 64,
                        "ordinal": 0,
                        "text": "fraud payment help",
                        "access_visibility": "PUBLIC",
                        "allowed_groups": [],
                        "contains_pii": False,
                        "pii_types": [],
                        "pii_detector_version": "deterministic-pii-v1",
                    },
                )
            ],
        )
    return collection


def build(
    engine: Engine,
    app_settings: AppSettings,
    client: QdrantClient,
    *,
    embedding: FakeEmbeddingProvider | None = None,
    reranker: FakeReranker | None = None,
    telemetry_sink: RetrievalTelemetrySink | None = None,
) -> Any:
    with Session(engine) as session:
        return build_managed_retrieval_runtime(
            app_settings,
            LineageRepository(session),
            qdrant_client=client,
            embedding_provider=embedding
            or FakeEmbeddingProvider(app_settings.models.embedding.model),
            reranker_provider=reranker or FakeReranker(),
            telemetry_sink=telemetry_sink,
        )


def active_state(engine: Engine) -> IndexState:
    with Session(engine) as session:
        state = session.scalar(select(IndexState).where(IndexState.status == "ACTIVE"))
        assert state is not None
        session.expunge(state)
        return state


def test_resolves_exact_active_generation_and_reconstructs_persisted_bm25(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app_settings = settings()
    engine = database()
    state_id, _corpus_id, encoder = seed_generation(engine, app_settings)
    client = QdrantClient(":memory:")
    collection = create_physical_collection(client, app_settings, encoder)
    monkeypatch.setattr(
        BM25SparseEncoder,
        "fit",
        classmethod(
            lambda _cls, *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("managed retrieval must never fit BM25")
            )
        ),
    )

    embedding = FakeEmbeddingProvider(app_settings.models.embedding.model)
    runtime = build(engine, app_settings, client, embedding=embedding)

    assert runtime.index_state_id == state_id
    assert runtime.generation == 1
    assert runtime.qdrant_collection == collection
    assert runtime.document_count == 1
    assert runtime.chunk_count == 1
    assert runtime.retrieval_service.collection_name == collection
    assert runtime.retrieval_service.sparse_encoder.state == encoder.state
    store = runtime.retrieval_service.store
    assert store.settings.collection_name == collection
    assert store.settings.dense_vector_name == "managed_dense"
    assert store.settings.sparse_vector_name == "managed_sparse"
    assert embedding.query_calls == 0
    assert embedding.document_calls == 0
    engine.dispose()


@pytest.mark.parametrize("status", ["BUILDING", "READY", "RETIRED", "FAILED"])
def test_non_active_generations_are_never_selected(status: str) -> None:
    app_settings = settings()
    engine = database()
    seed_generation(engine, app_settings, status=status)
    with pytest.raises(NoActiveManagedIndexError, match="no ACTIVE"):
        build(engine, app_settings, QdrantClient(":memory:"))
    engine.dispose()


def test_explicit_historical_runtime_accepts_only_current_compatible_retired() -> None:
    app_settings = settings()
    engine = database()
    _state_id, _corpus_id, encoder = seed_generation(
        engine, app_settings, status="RETIRED"
    )
    client = QdrantClient(":memory:")
    collection = create_physical_collection(client, app_settings, encoder)

    with Session(engine) as session:
        runtime = build_managed_retrieval_runtime_for_generation(
            app_settings,
            LineageRepository(session),
            1,
            qdrant_client=client,
            embedding_provider=FakeEmbeddingProvider(
                app_settings.models.embedding.model
            ),
            reranker_provider=FakeReranker(),
        )

    assert runtime.generation == 1
    assert runtime.qdrant_collection == collection
    engine.dispose()


def test_explicit_historical_runtime_rejects_failed_build() -> None:
    app_settings = settings()
    engine = database()
    seed_generation(engine, app_settings, status="FAILED")

    with Session(engine) as session, pytest.raises(
        ManagedRetrievalError, match="only ACTIVE or RETIRED"
    ):
        build_managed_retrieval_runtime_for_generation(
            app_settings,
            LineageRepository(session),
            1,
            qdrant_client=QdrantClient(":memory:"),
        )
    engine.dispose()


def test_uninitialized_corpus_and_multiple_active_failure_are_clear(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app_settings = settings()
    engine = database()
    with Session(engine) as session:
        repository = LineageRepository(session)
        with pytest.raises(NoActiveManagedIndexError, match="not initialized"):
            build_managed_retrieval_runtime(
                app_settings,
                repository,
                qdrant_client=QdrantClient(":memory:"),
                embedding_provider=FakeEmbeddingProvider(
                    app_settings.models.embedding.model
                ),
                reranker_provider=FakeReranker(),
            )

    seed_generation(engine, app_settings)

    def multiple_active(_repository: Any, _corpus_id: UUID) -> None:
        raise InvalidLineageStateError("more than one ACTIVE index state")

    monkeypatch.setattr(LineageRepository, "get_active_index_state", multiple_active)
    with pytest.raises(CorruptActiveManagedIndexError, match="more than one ACTIVE"):
        build(engine, app_settings, QdrantClient(":memory:"))
    engine.dispose()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("bm25_state_hash", "f" * 64, "BM25 state hash"),
        ("bm25_corpus_hash", "f" * 64, "BM25 corpus hash"),
        ("chunk_count", 2, "BM25 document_count"),
        ("index_configuration_fingerprint", "f" * 64, "configuration fingerprint"),
    ],
)
def test_corrupt_active_metadata_fails_before_qdrant(
    field: str,
    value: object,
    message: str,
) -> None:
    app_settings = settings()
    engine = database()
    seed_generation(engine, app_settings)
    with Session(engine) as session, session.begin():
        state = session.scalar(select(IndexState).where(IndexState.status == "ACTIVE"))
        assert state is not None
        setattr(state, field, value)

    with pytest.raises(CorruptActiveManagedIndexError, match=message):
        build(engine, app_settings, QdrantClient(":memory:"))
    engine.dispose()


def test_bm25_parameters_must_match_the_index_configuration() -> None:
    app_settings = settings()
    engine = database()
    seed_generation(engine, app_settings)
    with Session(engine) as session, session.begin():
        state = session.scalar(select(IndexState).where(IndexState.status == "ACTIVE"))
        assert state is not None
        payload = dict(state.bm25_state or {})
        payload["k1"] = 2.0
        altered = BM25SparseEncoder.fit(["fraud payment help"], k1=2.0).state
        state.bm25_state = payload
        state.bm25_state_hash = bm25_state_hash(altered)

    with pytest.raises(CorruptActiveManagedIndexError, match="k1/b"):
        build(engine, app_settings, QdrantClient(":memory:"))
    engine.dispose()


def test_active_collection_name_must_match_registered_prefix_and_generation() -> None:
    app_settings = settings()
    engine = database()
    seed_generation(engine, app_settings)
    with Session(engine) as session, session.begin():
        state = session.scalar(select(IndexState).where(IndexState.status == "ACTIVE"))
        assert state is not None
        state.qdrant_collection = "wrong_prefix_g000001"

    with pytest.raises(CorruptActiveManagedIndexError, match="prefix/generation"):
        build(engine, app_settings, QdrantClient(":memory:"))
    engine.dispose()


def test_embedding_fingerprint_and_exposed_dimension_are_compatible() -> None:
    app_settings = settings()
    engine = database()
    _state_id, _corpus_id, encoder = seed_generation(engine, app_settings)
    client = QdrantClient(":memory:")
    create_physical_collection(client, app_settings, encoder)

    runtime = build(
        engine,
        app_settings,
        client,
        embedding=FakeEmbeddingProvider(app_settings.models.embedding.model),
    )
    assert runtime.embedding_fingerprint == embedding_fp(app_settings)

    changed_embedding = app_settings.models.embedding.model_copy(
        update={"model": "example/incompatible-model"}
    )
    changed_settings = app_settings.model_copy(
        update={
            "models": app_settings.models.model_copy(
                update={"embedding": changed_embedding}
            )
        }
    )
    with pytest.raises(ManagedEmbeddingCompatibilityError, match="incompatible"):
        build(
            engine,
            changed_settings,
            client,
            embedding=FakeEmbeddingProvider(changed_embedding.model),
        )
    with pytest.raises(ManagedEmbeddingCompatibilityError, match="dimension"):
        build(
            engine,
            app_settings,
            client,
            embedding=FakeEmbeddingProvider(
                app_settings.models.embedding.model,
                dimension=3,
            ),
        )
    engine.dispose()


def test_query_time_top_k_hnsw_and_reranker_changes_do_not_invalidate_index() -> None:
    app_settings = settings()
    engine = database()
    _state_id, _corpus_id, encoder = seed_generation(engine, app_settings)
    client = QdrantClient(":memory:")
    create_physical_collection(client, app_settings, encoder)
    changed = app_settings.model_copy(
        update={
            "retrieval": app_settings.retrieval.model_copy(
                update={
                    "dense_candidates": 7,
                    "sparse_candidates": 6,
                    "fused_candidates": 4,
                    "final_contexts": 2,
                    "rrf_k": 41,
                    "hnsw_ef": 75,
                }
            ),
            "models": app_settings.models.model_copy(
                update={
                    "reranker": app_settings.models.reranker.model_copy(
                        update={"model": "example/new-reranker"}
                    )
                }
            ),
        }
    )
    runtime = build(
        engine,
        changed,
        client,
        reranker=FakeReranker("example/new-reranker"),
    )
    assert runtime.retrieval_service.settings.rrf_k == 41
    assert runtime.retrieval_service.reranker.model_name == "example/new-reranker"
    assert runtime.retrieval_service.store.query_hnsw_ef == 75
    engine.dispose()


@pytest.mark.parametrize(
    ("physical", "message"),
    [
        ("missing", "unavailable"),
        ("dimension", "dense dimension"),
        ("count", "point count"),
        ("distance", "distance"),
        ("dense_name", "missing dense vector"),
        ("sparse_name", "missing sparse vector"),
    ],
)
def test_bounded_physical_validation_failures_are_clear(
    physical: str,
    message: str,
) -> None:
    app_settings = settings()
    engine = database()
    _state_id, _corpus_id, encoder = seed_generation(engine, app_settings)
    client = QdrantClient(":memory:")
    if physical != "missing":
        create_physical_collection(
            client,
            app_settings,
            encoder,
            dimension=3 if physical == "dimension" else EMBEDDING_DIMENSION,
            distance=(
                models.Distance.DOT
                if physical == "distance"
                else models.Distance.COSINE
            ),
            include_point=physical != "count",
            dense_name="wrong_dense" if physical == "dense_name" else None,
            sparse_name="wrong_sparse" if physical == "sparse_name" else None,
        )

    with pytest.raises(ManagedCollectionUnavailableError, match=message):
        build(engine, app_settings, client)
    engine.dispose()


def test_managed_runtime_uses_existing_dense_sparse_rrf_and_rerank_path() -> None:
    app_settings = settings()
    engine = database()
    _state_id, _corpus_id, encoder = seed_generation(engine, app_settings)
    client = QdrantClient(":memory:")
    create_physical_collection(client, app_settings, encoder)
    embedding = FakeEmbeddingProvider(app_settings.models.embedding.model)
    runtime = build(engine, app_settings, client, embedding=embedding)

    trace = runtime.retrieval_service.retrieve_debug("fraud payment", top_k=1)

    assert embedding.query_calls == 1
    assert embedding.document_calls == 0
    assert [item.chunk.chunk_id for item in trace.dense] == ["chunk-1"]
    assert [item.chunk.chunk_id for item in trace.sparse] == ["chunk-1"]
    assert [item.chunk.chunk_id for item in trace.fused] == ["chunk-1"]
    assert [item.chunk.chunk_id for item in trace.reranked] == ["chunk-1"]
    assert trace.sparse[0].sparse_rank == 1
    engine.dispose()


def test_resolution_performs_no_database_or_qdrant_mutations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app_settings = settings()
    engine = database()
    _state_id, _corpus_id, encoder = seed_generation(engine, app_settings)
    client = QdrantClient(":memory:")
    create_physical_collection(client, app_settings, encoder)
    telemetry_sink = RecordingTelemetrySink()
    with Session(engine) as session:
        count_before = session.scalar(select(func.count()).select_from(IndexState))
        for method in ("create_collection", "delete_collection", "upsert"):
            monkeypatch.setattr(
                client,
                method,
                lambda *_args, _method=method, **_kwargs: (_ for _ in ()).throw(
                    AssertionError(f"resolver must not call {_method}")
                ),
            )
        runtime = build_managed_retrieval_runtime(
            app_settings,
            LineageRepository(session),
            qdrant_client=client,
            embedding_provider=FakeEmbeddingProvider(
                app_settings.models.embedding.model
            ),
            reranker_provider=FakeReranker(),
            telemetry_sink=telemetry_sink,
        )
        assert runtime.generation == 1
        runtime.retrieval_service.retrieve("fraud payment", top_k=1)
        assert telemetry_sink.events[-1].generation == 1
        assert not session.new
        assert not session.dirty
        assert not session.deleted
        assert session.scalar(select(func.count()).select_from(IndexState)) == count_before
    engine.dispose()


def test_runtime_remains_pinned_until_explicit_rebuild() -> None:
    app_settings = settings()
    engine = database()
    _first_id, corpus_id, encoder = seed_generation(engine, app_settings)
    client = QdrantClient(":memory:")
    first_collection = create_physical_collection(client, app_settings, encoder)
    first_sink = RecordingTelemetrySink()
    first_runtime = build(
        engine,
        app_settings,
        client,
        telemetry_sink=first_sink,
    )

    identity = configuration_identity(app_settings)
    with Session(engine) as session, session.begin():
        repository = LineageRepository(session)
        first = repository.get_active_index_state(corpus_id)
        assert first is not None
        repository.transition_index_state(
            first.index_state_id,
            "ACTIVE",
            "RETIRED",
            occurred_at=datetime.now(UTC),
        )
        run = repository.get_latest_successful_ingestion_run(corpus_id)
        assert run is not None
        second = repository.create_building_index_state(
            corpus_id,
            2,
            run.run_id,
            managed_generation_collection_name(PREFIX, 2),
            "a" * 64,
            identity.configuration.embedding_fingerprint,
            CHUNKING_FINGERPRINT,
            document_count=1,
            chunk_count=1,
        )
        repository.update_index_build_metadata(
            second.index_state_id,
            bm25_state=bm25_state_to_json_dict(encoder.state),
            bm25_state_hash=bm25_state_hash(encoder.state),
            bm25_corpus_hash=encoder.state.corpus_hash,
            index_configuration=identity.payload,
            index_configuration_fingerprint=identity.fingerprint,
            embedding_dimension=EMBEDDING_DIMENSION,
            document_count=1,
            chunk_count=1,
        )
        repository.transition_index_state(second.index_state_id, "BUILDING", "READY")
        repository.transition_index_state(
            second.index_state_id,
            "READY",
            "ACTIVE",
            occurred_at=datetime.now(UTC),
        )
    second_collection = create_physical_collection(
        client,
        app_settings,
        encoder,
        generation=2,
    )

    assert first_runtime.generation == 1
    assert first_runtime.qdrant_collection == first_collection
    assert first_runtime.retrieval_service.store.settings.collection_name == first_collection
    first_runtime.retrieval_service.retrieve("fraud payment", top_k=1)
    assert first_sink.events[-1].generation == 1
    assert first_sink.events[-1].index_state_id == first_runtime.index_state_id
    assert first_sink.events[-1].qdrant_collection == first_collection

    second_runtime = build(engine, app_settings, client)
    assert second_runtime.generation == 2
    assert second_runtime.qdrant_collection == second_collection
    assert second_runtime.retrieval_service.store.settings.collection_name == second_collection
    engine.dispose()
