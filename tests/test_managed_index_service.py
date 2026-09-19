from __future__ import annotations

from collections.abc import Callable, Generator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

import pytest
from sqlalchemy import Engine, create_engine, event, func, select
from sqlalchemy.orm import Session, sessionmaker

from consumer_finance_rag.config import (
    ChunkingSettings,
    EmbeddingModelSettings,
    ManagedIndexSettings,
)
from consumer_finance_rag.hashing import sha256_text
from consumer_finance_rag.lineage.chunk_materialization import (
    ChunkMaterializationService,
)
from consumer_finance_rag.lineage.fingerprints import metadata_hash, version_hash
from consumer_finance_rag.lineage.index_types import (
    ActiveChunkManifest,
    DenseVectorRetrievalResult,
    ManagedCollectionMetadata,
    ManagedCollectionValidationResult,
    ManagedPointInput,
    RetrievedDenseVector,
)
from consumer_finance_rag.lineage.indexing import (
    ManagedIndexActivationConflictError,
    ManagedIndexBuildError,
    ManagedIndexPublishOutcome,
    ManagedIndexService,
    ManagedIndexStaleLineageError,
)
from consumer_finance_rag.lineage.models import (
    Base,
    DocumentVersion,
    IndexState,
    IndexStateChunk,
    IngestionRun,
    LineageChunk,
    LineageDocument,
)
from consumer_finance_rag.lineage.repository import LineageRepository

CHUNKING = ChunkingSettings(chunk_size=45, overlap=0)
EMBEDDING = EmbeddingModelSettings(
    model="test-embedding-v1",
    device="cpu",
    trust_remote_code=True,
)
INDEX_SETTINGS = ManagedIndexSettings()
START = datetime(2026, 9, 19, 0, 0, tzinfo=UTC)


@pytest.fixture
def engine() -> Generator[Engine]:
    sqlite_engine = create_engine("sqlite+pysqlite:///:memory:")

    @event.listens_for(sqlite_engine, "connect")
    def enable_foreign_keys(dbapi_connection: Any, _record: Any) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(sqlite_engine)
    yield sqlite_engine
    sqlite_engine.dispose()


@pytest.fixture
def sessions(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, class_=Session, expire_on_commit=False)


class FakeEmbeddingProvider:
    def __init__(self, model_name: str = EMBEDDING.model, dimension: int = 3) -> None:
        self._model_name = model_name
        self.dimension = dimension
        self.calls: list[tuple[str, ...]] = []
        self.error: Exception | None = None

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(tuple(texts))
        if self.error is not None:
            raise self.error
        return [
            [float(index + 1), float(len(text)), 0.5][: self.dimension]
            for index, text in enumerate(texts)
        ]

    def embed_query(self, _text: str) -> list[float]:
        raise AssertionError("managed publication must use document embeddings")


class FakeManagedStoreFactory:
    def __init__(self) -> None:
        self.collections: dict[str, dict[str, Any]] = {}
        self.create_calls: list[str] = []
        self.upsert_calls: list[str] = []
        self.validate_calls: list[str] = []
        self.delete_calls: list[str] = []
        self.fail_create = False
        self.fail_upsert = False
        self.fail_validate = False
        self.after_validate: Callable[[], None] | None = None

    def __call__(
        self,
        managed_prefix: str,
        configuration: Any,
        embedding_dimension: int,
    ) -> FakeManagedStore:
        return FakeManagedStore(
            self,
            managed_prefix,
            configuration,
            embedding_dimension,
        )


class FakeManagedStore:
    def __init__(
        self,
        owner: FakeManagedStoreFactory,
        managed_prefix: str,
        configuration: Any,
        embedding_dimension: int,
    ) -> None:
        self.owner = owner
        self.prefix = managed_prefix
        self.configuration = configuration
        self.dimension = embedding_dimension

    def create_generation_collection(self, collection_name: str) -> None:
        self.owner.create_calls.append(collection_name)
        if self.owner.fail_create:
            raise RuntimeError("create failed")
        if collection_name in self.owner.collections:
            raise RuntimeError("collection collision")
        self.owner.collections[collection_name] = {
            "configuration": self.configuration,
            "dimension": self.dimension,
            "points": {},
        }

    def upsert_points(
        self,
        collection_name: str,
        points: Sequence[ManagedPointInput],
        *,
        batch_size: int = 128,
    ) -> None:
        del batch_size
        self.owner.upsert_calls.append(collection_name)
        if self.owner.fail_upsert:
            raise RuntimeError("upsert failed")
        collection = self.owner.collections[collection_name]
        collection["points"] = {point.qdrant_point_id: point for point in points}

    def retrieve_dense_vectors(
        self,
        collection_name: str,
        point_ids: Sequence[UUID],
    ) -> DenseVectorRetrievalResult:
        collection = self.owner.collections[collection_name]
        points: dict[UUID, ManagedPointInput] = collection["points"]
        records = []
        missing = []
        for point_id in point_ids:
            point = points.get(point_id)
            if point is None:
                missing.append(point_id)
                continue
            records.append(
                RetrievedDenseVector(
                    point_id=point_id,
                    dense_vector=point.dense_vector,
                    lineage_chunk_pk=point.chunk.chunk_pk,
                    chunk_hash=point.chunk.chunk_hash,
                    text=point.chunk.text,
                    compatibility_chunk_id=point.chunk.compatibility_chunk_id,
                )
            )
        return DenseVectorRetrievalResult(tuple(records), tuple(missing))

    def inspect_collection(self, collection_name: str) -> ManagedCollectionMetadata:
        if collection_name not in self.owner.collections:
            raise RuntimeError("collection missing")
        collection = self.owner.collections[collection_name]
        config = collection["configuration"]
        return ManagedCollectionMetadata(
            collection_name=collection_name,
            dense_vector_names=(config.dense_vector_name,),
            sparse_vector_names=(config.sparse_vector_name,),
            dense_dimension=collection["dimension"],
            distance_metric=config.distance_metric,
            hnsw_m=config.hnsw_m,
            hnsw_ef_construct=config.hnsw_ef_construct,
            sparse_modifier=config.sparse_modifier,
            sparse_on_disk=config.sparse_on_disk,
        )

    def count_points(self, collection_name: str) -> int:
        return len(self.owner.collections[collection_name]["points"])

    def validate_generation_collection(
        self,
        collection_name: str,
        expected_points: Sequence[ManagedPointInput],
    ) -> ManagedCollectionValidationResult:
        self.owner.validate_calls.append(collection_name)
        if self.owner.fail_validate:
            raise RuntimeError("validation failed")
        actual = self.owner.collections[collection_name]["points"]
        assert set(actual) == {point.qdrant_point_id for point in expected_points}
        callback = self.owner.after_validate
        if callback is not None:
            self.owner.after_validate = None
            callback()
        return ManagedCollectionValidationResult(
            collection_name=collection_name,
            point_count=len(actual),
            point_ids=tuple(sorted(actual, key=str)),
        )

    def delete_generation_collection(self, collection_name: str) -> None:
        self.owner.delete_calls.append(collection_name)
        del self.owner.collections[collection_name]


@dataclass(frozen=True)
class LineageFixture:
    corpus_id: UUID
    run_id: UUID
    document_pk: UUID
    manifest: ActiveChunkManifest


def create_lineage(
    sessions: sessionmaker[Session],
    *,
    text: str = "Stable credit card paragraph.\n\nOriginal mortgage paragraph.",
    title: str = "Original title",
) -> LineageFixture:
    with sessions() as session, session.begin():
        repository = LineageRepository(session)
        corpus = repository.create_corpus(
            "consumer-finance-managed", "consumer_finance_managed"
        )
        run = repository.start_ingestion_run(
            corpus.corpus_id,
            authoritative_snapshot=False,
            snapshot_hash=sha256_text("observed-one"),
        )
        document = repository.create_document(
            corpus.corpus_id,
            "cfpb:source-one",
            "source-one",
            "https://consumerfinance.gov/source-one",
            run.run_id,
        )
        create_version(
            repository,
            document.document_pk,
            run.run_id,
            1,
            text,
            title,
        )
        repository.finish_ingestion_run(
            run.run_id,
            "SUCCEEDED",
            START,
            1,
            0,
            0,
            0,
        )
        corpus_id = corpus.corpus_id
        run_id = run.run_id
        document_pk = document.document_pk
    manifest = materialize(sessions, CHUNKING)
    return LineageFixture(corpus_id, run_id, document_pk, manifest)


def create_version(
    repository: LineageRepository,
    document_pk: UUID,
    run_id: UUID,
    number: int,
    text: str,
    title: str,
) -> DocumentVersion:
    content = sha256_text(text)
    metadata = metadata_hash(
        title=title,
        source_url="https://consumerfinance.gov/source-one",
        category="mortgages",
        source_version=f"v{number}",
    )
    return repository.create_document_version(
        document_pk,
        number,
        content,
        metadata,
        version_hash(content_hash=content, metadata_hash=metadata),
        f"source-one:{content}",
        title,
        "mortgages",
        "https://consumerfinance.gov/source-one",
        f"v{number}",
        text,
        run_id,
    )


def advance_version(
    sessions: sessionmaker[Session],
    fixture: LineageFixture,
    *,
    text: str,
    title: str,
    number: int = 2,
) -> tuple[UUID, ActiveChunkManifest]:
    with sessions() as session, session.begin():
        repository = LineageRepository(session)
        run = repository.start_ingestion_run(
            fixture.corpus_id,
            authoritative_snapshot=False,
            snapshot_hash=sha256_text(f"observed-{number}"),
        )
        repository.mark_document_seen(
            fixture.document_pk,
            run.run_id,
            "source-one",
            "https://consumerfinance.gov/source-one",
        )
        create_version(
            repository,
            fixture.document_pk,
            run.run_id,
            number,
            text,
            title,
        )
        repository.finish_ingestion_run(
            run.run_id,
            "SUCCEEDED",
            START + timedelta(minutes=number),
            0,
            1,
            0,
            0,
        )
        run_id = run.run_id
    return run_id, materialize(sessions, CHUNKING)


def materialize(
    sessions: sessionmaker[Session],
    chunking: ChunkingSettings,
) -> ActiveChunkManifest:
    with sessions() as session, session.begin():
        return ChunkMaterializationService(
            session,
            chunk_size=chunking.chunk_size,
            overlap=chunking.overlap,
        ).materialize_active_manifest("consumer-finance-managed")


def service(
    sessions: sessionmaker[Session],
    stores: FakeManagedStoreFactory,
    embedding: FakeEmbeddingProvider,
    *,
    embedding_settings: EmbeddingModelSettings = EMBEDDING,
    chunking: ChunkingSettings = CHUNKING,
    index_settings: ManagedIndexSettings = INDEX_SETTINGS,
) -> ManagedIndexService:
    return ManagedIndexService(
        session_factory=sessions,
        embedding_provider=embedding,
        embedding_settings=embedding_settings,
        chunking_settings=chunking,
        managed_index_settings=index_settings,
        store_factory=stores,
        clock=lambda: START + timedelta(hours=1),
    )


def states(sessions: sessionmaker[Session]) -> list[IndexState]:
    with sessions() as session:
        return list(session.scalars(select(IndexState).order_by(IndexState.generation)))


def test_first_generation_publishes_complete_active_index(
    sessions: sessionmaker[Session],
) -> None:
    fixture = create_lineage(sessions)
    stores = FakeManagedStoreFactory()
    embedding = FakeEmbeddingProvider()

    result = service(sessions, stores, embedding).publish(
        fixture.manifest, fixture.run_id
    )

    assert result.outcome is ManagedIndexPublishOutcome.PUBLISHED
    assert result.generation == 1
    assert result.collection == "consumer_finance_managed_g000001"
    assert result.bm25_action.value == "REFIT"
    assert result.dense_computed_count == fixture.manifest.chunk_count
    assert result.dense_reused_count == 0
    persisted = states(sessions)
    assert len(persisted) == 1
    assert persisted[0].status == "ACTIVE"
    assert persisted[0].embedding_dimension == 3
    assert persisted[0].bm25_state is not None
    assert persisted[0].index_configuration is not None
    with sessions() as session:
        memberships = list(session.scalars(select(IndexStateChunk)))
        assert len(memberships) == fixture.manifest.chunk_count
        assert all(item.dense_action == "COMPUTED" for item in memberships)
        assert all(item.dense_source_chunk_pk is None for item in memberships)
    assert stores.create_calls == [result.collection]
    assert stores.upsert_calls == [result.collection]
    assert stores.validate_calls == [result.collection]
    assert stores.delete_calls == []


def test_identical_second_publication_returns_no_index_change(
    sessions: sessionmaker[Session],
) -> None:
    fixture = create_lineage(sessions)
    stores = FakeManagedStoreFactory()
    embedding = FakeEmbeddingProvider()
    publisher = service(sessions, stores, embedding)
    first = publisher.publish(fixture.manifest, fixture.run_id)
    calls_after_first = list(stores.create_calls)

    second = publisher.publish(fixture.manifest, fixture.run_id)

    assert second.outcome is ManagedIndexPublishOutcome.NO_INDEX_CHANGE
    assert second.index_state_id == first.index_state_id
    assert second.generation == 1
    assert stores.create_calls == calls_after_first
    assert len(states(sessions)) == 1


def test_metadata_only_generation_reuses_dense_and_bm25_with_current_payload(
    sessions: sessionmaker[Session],
) -> None:
    fixture = create_lineage(sessions)
    stores = FakeManagedStoreFactory()
    embedding = FakeEmbeddingProvider()
    publisher = service(sessions, stores, embedding)
    first = publisher.publish(fixture.manifest, fixture.run_id)
    initial_embedding_calls = len(embedding.calls)
    run_id, current = advance_version(
        sessions,
        fixture,
        text="Stable credit card paragraph.\n\nOriginal mortgage paragraph.",
        title="Updated title",
    )

    result = publisher.publish(current, run_id)

    assert result.generation == 2
    assert result.bm25_action.value == "REUSE"
    assert result.dense_reused_count == current.chunk_count
    assert result.dense_computed_count == 0
    assert len(embedding.calls) == initial_embedding_calls
    new_points: dict[UUID, ManagedPointInput] = stores.collections[result.collection][
        "points"
    ]
    assert all(point.payload["title"] == "Updated title" for point in new_points.values())
    assert {point.chunk.chunk_pk for point in new_points.values()} == {
        chunk.chunk_pk for chunk in current.chunks
    }
    persisted = states(sessions)
    assert [state.status for state in persisted] == ["RETIRED", "ACTIVE"]
    assert first.collection in stores.collections
    assert result.collection in stores.collections
    with sessions() as session:
        new_memberships = list(
            session.scalars(
                select(IndexStateChunk).where(
                    IndexStateChunk.index_state_id == result.index_state_id
                )
            )
        )
        assert all(item.dense_action == "REUSED" for item in new_memberships)
        assert all(item.dense_source_chunk_pk is not None for item in new_memberships)


def test_changed_text_refits_bm25_and_mixes_dense_provenance(
    sessions: sessionmaker[Session],
) -> None:
    fixture = create_lineage(sessions)
    stores = FakeManagedStoreFactory()
    embedding = FakeEmbeddingProvider()
    publisher = service(sessions, stores, embedding)
    publisher.publish(fixture.manifest, fixture.run_id)
    run_id, current = advance_version(
        sessions,
        fixture,
        text="Stable credit card paragraph.\n\nChanged mortgage paragraph.",
        title="Changed text title",
    )

    result = publisher.publish(current, run_id)

    assert result.bm25_action.value == "REFIT"
    assert result.dense_reused_count >= 1
    assert result.dense_computed_count >= 1
    assert result.dense_reused_count + result.dense_computed_count == current.chunk_count


def test_document_addition_and_deletion_publish_only_current_manifest(
    sessions: sessionmaker[Session],
) -> None:
    fixture = create_lineage(sessions)
    stores = FakeManagedStoreFactory()
    embedding = FakeEmbeddingProvider()
    publisher = service(sessions, stores, embedding)
    publisher.publish(fixture.manifest, fixture.run_id)
    with sessions() as session, session.begin():
        repository = LineageRepository(session)
        run = repository.start_ingestion_run(
            fixture.corpus_id,
            authoritative_snapshot=False,
            snapshot_hash=sha256_text("addition"),
        )
        added = repository.create_document(
            fixture.corpus_id,
            "cfpb:source-two",
            "source-two",
            "https://consumerfinance.gov/source-two",
            run.run_id,
        )
        content = "New bank account paragraph."
        content_digest = sha256_text(content)
        metadata_digest = metadata_hash(
            title="Bank account guidance",
            source_url="https://consumerfinance.gov/source-two",
            category="bank-accounts",
            source_version="v1",
        )
        repository.create_document_version(
            added.document_pk,
            1,
            content_digest,
            metadata_digest,
            version_hash(
                content_hash=content_digest,
                metadata_hash=metadata_digest,
            ),
            f"source-two:{content_digest}",
            "Bank account guidance",
            "bank-accounts",
            "https://consumerfinance.gov/source-two",
            "v1",
            content,
            run.run_id,
        )
        repository.finish_ingestion_run(
            run.run_id,
            "SUCCEEDED",
            START + timedelta(minutes=2),
            1,
            0,
            1,
            0,
        )
        addition_run_id = run.run_id
        added_pk = added.document_pk
    added_manifest = materialize(sessions, CHUNKING)

    addition = publisher.publish(added_manifest, addition_run_id)

    assert addition.bm25_action.value == "REFIT"
    assert addition.dense_reused_count == fixture.manifest.chunk_count
    assert addition.dense_computed_count == (
        added_manifest.chunk_count - fixture.manifest.chunk_count
    )
    with sessions() as session, session.begin():
        repository = LineageRepository(session)
        run = repository.start_ingestion_run(
            fixture.corpus_id,
            authoritative_snapshot=False,
            snapshot_hash=sha256_text("deletion"),
        )
        repository.mark_document_deleted(added_pk, run.run_id)
        repository.mark_document_seen(
            fixture.document_pk,
            run.run_id,
            "source-one",
            "https://consumerfinance.gov/source-one",
        )
        repository.finish_ingestion_run(
            run.run_id,
            "SUCCEEDED",
            START + timedelta(minutes=3),
            0,
            0,
            1,
            1,
        )
        deletion_run_id = run.run_id
    deleted_manifest = materialize(sessions, CHUNKING)

    deletion = publisher.publish(deleted_manifest, deletion_run_id)

    assert deletion.generation == 3
    points: dict[UUID, ManagedPointInput] = stores.collections[deletion.collection][
        "points"
    ]
    assert len(points) == deleted_manifest.chunk_count
    assert all(point.chunk.source_key != "cfpb:source-two" for point in points.values())


def test_alternate_chunking_configuration_publishes_new_generation(
    sessions: sessionmaker[Session],
) -> None:
    fixture = create_lineage(sessions)
    stores = FakeManagedStoreFactory()
    embedding = FakeEmbeddingProvider()
    service(sessions, stores, embedding).publish(fixture.manifest, fixture.run_id)
    changed_chunking = ChunkingSettings(chunk_size=70, overlap=0)
    changed_manifest = materialize(sessions, changed_chunking)

    result = service(
        sessions,
        stores,
        embedding,
        chunking=changed_chunking,
    ).publish(changed_manifest, fixture.run_id)

    assert result.generation == 2
    assert result.bm25_action.value == "REFIT"
    assert result.chunk_count == changed_manifest.chunk_count
    assert result.dense_reused_count + result.dense_computed_count == result.chunk_count


def test_failed_noop_health_check_builds_replacement_generation(
    sessions: sessionmaker[Session],
) -> None:
    fixture = create_lineage(sessions)
    stores = FakeManagedStoreFactory()
    embedding = FakeEmbeddingProvider()
    publisher = service(sessions, stores, embedding)
    first = publisher.publish(fixture.manifest, fixture.run_id)
    points: dict[UUID, ManagedPointInput] = stores.collections[first.collection]["points"]
    points.pop(next(iter(points)))

    replacement = publisher.publish(fixture.manifest, fixture.run_id)

    assert replacement.outcome is ManagedIndexPublishOutcome.PUBLISHED
    assert replacement.generation == 2
    assert states(sessions)[0].status == "RETIRED"
    assert states(sessions)[1].status == "ACTIVE"


@pytest.mark.parametrize("failure", ["create", "upsert", "validate", "embedding"])
def test_pre_ready_failure_marks_candidate_failed_and_keeps_no_partial_active(
    sessions: sessionmaker[Session],
    failure: str,
) -> None:
    fixture = create_lineage(sessions)
    stores = FakeManagedStoreFactory()
    embedding = FakeEmbeddingProvider()
    if failure == "create":
        stores.fail_create = True
    elif failure == "upsert":
        stores.fail_upsert = True
    elif failure == "validate":
        stores.fail_validate = True
    else:
        embedding.error = RuntimeError("embedding failed")

    with pytest.raises(ManagedIndexBuildError):
        service(sessions, stores, embedding).publish(fixture.manifest, fixture.run_id)

    persisted = states(sessions)
    assert len(persisted) == 1
    assert persisted[0].status == "FAILED"
    assert persisted[0].error_message
    assert stores.delete_calls == []


@pytest.mark.parametrize("failure", ["create", "upsert", "validate", "embedding"])
def test_every_pre_activation_failure_keeps_previous_generation_active(
    sessions: sessionmaker[Session],
    failure: str,
) -> None:
    fixture = create_lineage(sessions)
    stores = FakeManagedStoreFactory()
    original_embedding = FakeEmbeddingProvider()
    original = service(sessions, stores, original_embedding).publish(
        fixture.manifest, fixture.run_id
    )
    changed_settings = INDEX_SETTINGS.model_copy(
        update={"hnsw_m": INDEX_SETTINGS.hnsw_m + 2}
    )
    selected_embedding = original_embedding
    selected_embedding_settings = EMBEDDING
    if failure == "create":
        stores.fail_create = True
    elif failure == "upsert":
        stores.fail_upsert = True
    elif failure == "validate":
        stores.fail_validate = True
    else:
        selected_embedding_settings = EMBEDDING.model_copy(
            update={"model": "failed-embedding-v2"}
        )
        selected_embedding = FakeEmbeddingProvider(model_name="failed-embedding-v2")
        selected_embedding.error = RuntimeError("embedding failed")

    with pytest.raises(ManagedIndexBuildError):
        service(
            sessions,
            stores,
            selected_embedding,
            embedding_settings=selected_embedding_settings,
            index_settings=changed_settings,
        ).publish(fixture.manifest, fixture.run_id)

    persisted = states(sessions)
    assert persisted[0].index_state_id == original.index_state_id
    assert persisted[0].status == "ACTIVE"
    assert persisted[1].status == "FAILED"
    assert original.collection in stores.collections
    assert stores.delete_calls == []


def test_generation_numbers_remain_monotonic_after_failed_build(
    sessions: sessionmaker[Session],
) -> None:
    fixture = create_lineage(sessions)
    stores = FakeManagedStoreFactory()
    embedding = FakeEmbeddingProvider()
    publisher = service(sessions, stores, embedding)
    publisher.publish(fixture.manifest, fixture.run_id)
    changed = INDEX_SETTINGS.model_copy(
        update={"hnsw_m": INDEX_SETTINGS.hnsw_m + 1}
    )
    stores.fail_validate = True
    with pytest.raises(ManagedIndexBuildError):
        service(
            sessions,
            stores,
            embedding,
            index_settings=changed,
        ).publish(fixture.manifest, fixture.run_id)
    stores.fail_validate = False
    another_change = INDEX_SETTINGS.model_copy(
        update={"hnsw_m": INDEX_SETTINGS.hnsw_m + 2}
    )

    result = service(
        sessions,
        stores,
        embedding,
        index_settings=another_change,
    ).publish(fixture.manifest, fixture.run_id)

    assert result.generation == 3
    assert [state.generation for state in states(sessions)] == [1, 2, 3]
    assert "consumer_finance_managed_g000002" in stores.collections
    assert stores.delete_calls == []


def test_collection_collision_fails_without_deleting_existing_artifact(
    sessions: sessionmaker[Session],
) -> None:
    fixture = create_lineage(sessions)
    stores = FakeManagedStoreFactory()
    stores.collections["consumer_finance_managed_g000001"] = {
        "configuration": INDEX_SETTINGS,
        "dimension": 3,
        "points": {},
    }

    with pytest.raises(ManagedIndexBuildError):
        service(sessions, stores, FakeEmbeddingProvider()).publish(
            fixture.manifest, fixture.run_id
        )

    assert states(sessions)[0].status == "FAILED"
    assert stores.delete_calls == []
    assert "consumer_finance_managed_g000001" in stores.collections


def test_embedding_configuration_change_computes_all_but_reuses_bm25(
    sessions: sessionmaker[Session],
) -> None:
    fixture = create_lineage(sessions)
    stores = FakeManagedStoreFactory()
    first_embedding = FakeEmbeddingProvider()
    service(sessions, stores, first_embedding).publish(fixture.manifest, fixture.run_id)
    changed_settings = EMBEDDING.model_copy(update={"model": "test-embedding-v2"})
    changed_embedding = FakeEmbeddingProvider(model_name="test-embedding-v2")

    result = service(
        sessions,
        stores,
        changed_embedding,
        embedding_settings=changed_settings,
    ).publish(fixture.manifest, fixture.run_id)

    assert result.generation == 2
    assert result.bm25_action.value == "REUSE"
    assert result.dense_computed_count == fixture.manifest.chunk_count
    assert result.dense_reused_count == 0
    assert len(stores.collections) == 2


@pytest.mark.parametrize(
    ("settings", "expected_bm25"),
    [
        (
            INDEX_SETTINGS.model_copy(
                update={"hnsw_m": INDEX_SETTINGS.hnsw_m + 8}
            ),
            "REUSE",
        ),
        (
            INDEX_SETTINGS.model_copy(
                update={"bm25_k1": INDEX_SETTINGS.bm25_k1 + 0.3}
            ),
            "REFIT",
        ),
    ],
)
def test_index_configuration_changes_publish_new_generation_with_expected_reuse(
    sessions: sessionmaker[Session],
    settings: ManagedIndexSettings,
    expected_bm25: str,
) -> None:
    fixture = create_lineage(sessions)
    stores = FakeManagedStoreFactory()
    embedding = FakeEmbeddingProvider()
    service(sessions, stores, embedding).publish(fixture.manifest, fixture.run_id)
    calls = len(embedding.calls)

    result = service(
        sessions,
        stores,
        embedding,
        index_settings=settings,
    ).publish(fixture.manifest, fixture.run_id)

    assert result.generation == 2
    assert result.bm25_action.value == expected_bm25
    assert result.dense_reused_count == fixture.manifest.chunk_count
    assert len(embedding.calls) == calls


def test_lineage_advancement_blocks_activation_and_marks_candidate_failed(
    sessions: sessionmaker[Session],
) -> None:
    fixture = create_lineage(sessions)
    stores = FakeManagedStoreFactory()

    def advance_lineage() -> None:
        with sessions() as session, session.begin():
            repository = LineageRepository(session)
            run = repository.start_ingestion_run(
                fixture.corpus_id,
                authoritative_snapshot=False,
                snapshot_hash=sha256_text("later-noop"),
            )
            repository.finish_ingestion_run(
                run.run_id,
                "NOOP",
                START + timedelta(days=1),
                0,
                0,
                1,
                0,
            )

    stores.after_validate = advance_lineage
    with pytest.raises(ManagedIndexStaleLineageError):
        service(sessions, stores, FakeEmbeddingProvider()).publish(
            fixture.manifest, fixture.run_id
        )

    assert states(sessions)[0].status == "FAILED"
    assert stores.delete_calls == []


def test_newer_active_blocks_activation_and_candidate_becomes_failed(
    sessions: sessionmaker[Session],
) -> None:
    fixture = create_lineage(sessions)
    stores = FakeManagedStoreFactory()
    embedding = FakeEmbeddingProvider()
    publisher = service(sessions, stores, embedding)
    first = publisher.publish(fixture.manifest, fixture.run_id)
    changed_settings = INDEX_SETTINGS.model_copy(
        update={"hnsw_m": INDEX_SETTINGS.hnsw_m + 1}
    )

    def replace_active() -> None:
        with sessions() as session, session.begin():
            old = session.get(IndexState, first.index_state_id)
            assert old is not None
            old.status = "RETIRED"
            old.retired_at = START + timedelta(hours=2)
            session.add(
                IndexState(
                    corpus_id=fixture.corpus_id,
                    generation=3,
                    ingestion_run_id=fixture.run_id,
                    status="ACTIVE",
                    qdrant_collection="consumer_finance_managed_g000003",
                    snapshot_hash=fixture.manifest.source_snapshot_hash,
                    embedding_fingerprint=old.embedding_fingerprint,
                    embedding_dimension=old.embedding_dimension,
                    chunking_fingerprint=old.chunking_fingerprint,
                    document_count=old.document_count,
                    chunk_count=old.chunk_count,
                    bm25_state=old.bm25_state,
                    bm25_state_hash=old.bm25_state_hash,
                    bm25_corpus_hash=old.bm25_corpus_hash,
                    index_configuration=old.index_configuration,
                    index_configuration_fingerprint=(
                        old.index_configuration_fingerprint
                    ),
                    activated_at=START + timedelta(hours=2),
                )
            )

    stores.after_validate = replace_active
    with pytest.raises(ManagedIndexActivationConflictError):
        service(
            sessions,
            stores,
            embedding,
            index_settings=changed_settings,
        ).publish(fixture.manifest, fixture.run_id)

    persisted = states(sessions)
    assert [state.status for state in persisted] == ["RETIRED", "FAILED", "ACTIVE"]
    assert stores.delete_calls == []


def test_ready_transaction_failure_keeps_old_active_and_rolls_back_memberships(
    sessions: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = create_lineage(sessions)
    stores = FakeManagedStoreFactory()
    embedding = FakeEmbeddingProvider()
    first = service(sessions, stores, embedding).publish(
        fixture.manifest, fixture.run_id
    )
    changed = INDEX_SETTINGS.model_copy(
        update={"hnsw_m": INDEX_SETTINGS.hnsw_m + 1}
    )

    def fail_batch(*_args: Any, **_kwargs: Any) -> list[IndexStateChunk]:
        raise RuntimeError("ready persistence failed")

    monkeypatch.setattr(
        LineageRepository,
        "record_index_state_chunks_batch",
        fail_batch,
    )
    with pytest.raises(ManagedIndexBuildError):
        service(
            sessions,
            stores,
            embedding,
            index_settings=changed,
        ).publish(fixture.manifest, fixture.run_id)

    persisted = states(sessions)
    assert persisted[0].index_state_id == first.index_state_id
    assert persisted[0].status == "ACTIVE"
    assert persisted[1].status == "FAILED"
    assert persisted[1].bm25_state is None
    with sessions() as session:
        count = session.scalar(
            select(func.count())
            .select_from(IndexStateChunk)
            .where(IndexStateChunk.index_state_id == persisted[1].index_state_id)
        )
        assert count == 0


def test_unexpected_activation_failure_leaves_candidate_ready_for_retry(
    sessions: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = create_lineage(sessions)
    stores = FakeManagedStoreFactory()
    embedding = FakeEmbeddingProvider()
    first = service(sessions, stores, embedding).publish(
        fixture.manifest, fixture.run_id
    )
    changed = INDEX_SETTINGS.model_copy(
        update={"hnsw_m": INDEX_SETTINGS.hnsw_m + 1}
    )
    original_transition = LineageRepository.transition_index_state

    def fail_candidate_activation(
        repository: LineageRepository,
        index_state_id: UUID,
        expected_current_status: str,
        target_status: str,
        **kwargs: Any,
    ) -> IndexState:
        if expected_current_status == "READY" and target_status == "ACTIVE":
            raise RuntimeError("transient database failure")
        return original_transition(
            repository,
            index_state_id,
            expected_current_status,
            target_status,
            **kwargs,
        )

    monkeypatch.setattr(
        LineageRepository,
        "transition_index_state",
        fail_candidate_activation,
    )
    with pytest.raises(RuntimeError, match="transient"):
        service(
            sessions,
            stores,
            embedding,
            index_settings=changed,
        ).publish(fixture.manifest, fixture.run_id)

    persisted = states(sessions)
    assert persisted[0].index_state_id == first.index_state_id
    assert persisted[0].status == "ACTIVE"
    assert persisted[1].status == "READY"
    assert stores.delete_calls == []


def test_publisher_does_not_mutate_source_lineage_or_create_ingestion_runs(
    sessions: sessionmaker[Session],
) -> None:
    fixture = create_lineage(sessions)
    with sessions() as session:
        before = (
            session.scalar(select(func.count()).select_from(IngestionRun)),
            session.scalar(select(func.count()).select_from(LineageDocument)),
            session.scalar(select(func.count()).select_from(DocumentVersion)),
            session.scalar(select(func.count()).select_from(LineageChunk)),
        )

    service(
        sessions, FakeManagedStoreFactory(), FakeEmbeddingProvider()
    ).publish(fixture.manifest, fixture.run_id)

    with sessions() as session:
        after = (
            session.scalar(select(func.count()).select_from(IngestionRun)),
            session.scalar(select(func.count()).select_from(LineageDocument)),
            session.scalar(select(func.count()).select_from(DocumentVersion)),
            session.scalar(select(func.count()).select_from(LineageChunk)),
        )
    assert after == before
