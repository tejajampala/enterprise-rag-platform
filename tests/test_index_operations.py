from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

import pytest
from qdrant_client import QdrantClient, models
from sqlalchemy import Engine, create_engine, event, func, select, text
from sqlalchemy.orm import Session, sessionmaker

from consumer_finance_rag.config import EmbeddingModelSettings, ManagedIndexSettings
from consumer_finance_rag.lineage.fingerprints import (
    ManagedIndexConfiguration,
    embedding_fingerprint,
    managed_index_configuration_identity,
)
from consumer_finance_rag.lineage.index_operations import (
    CleanupClassification,
    GenerationCleanupState,
    ManagedIndexOperationsService,
    plan_generation_cleanup,
)
from consumer_finance_rag.lineage.index_types import ActiveChunk, ManagedPointInput
from consumer_finance_rag.lineage.managed_vector_store import (
    ManagedQdrantStore,
    managed_generation_collection_name,
    managed_point_id,
)
from consumer_finance_rag.lineage.models import (
    Base,
    Corpus,
    DocumentVersion,
    DocumentVersionPII,
    IndexState,
    IndexStateChunk,
    IngestionRun,
    LineageChunk,
    LineageDocument,
)
from consumer_finance_rag.lineage.repository import LineageRepository
from consumer_finance_rag.lineage.restoration import (
    ManagedGenerationRestorationService,
    RestorationActivationConflictError,
    RestorationBuildError,
    RestorationSourceIneligibleError,
    RestorationUnavailablePhysicalCleanedError,
    RestorationValidationError,
)
from consumer_finance_rag.pii import DETERMINISTIC_PII_VERSION
from consumer_finance_rag.sparse import (
    BM25SparseEncoder,
    bm25_state_hash,
    bm25_state_to_json_dict,
)

CORPUS_KEY = "operations-test"
MANAGED_PREFIX = "consumer_finance_managed_ops"
NOW = datetime(2026, 9, 21, 12, tzinfo=UTC)
DIMENSION = 3


@dataclass(slots=True)
class OperationsEnvironment:
    engine: Engine
    sessions: sessionmaker[Session]
    client: QdrantClient
    settings: ManagedIndexSettings
    configuration: ManagedIndexConfiguration
    corpus_id: UUID
    run_id: UUID
    document_pk: UUID
    document_version_id: UUID
    chunk_pk: UUID

    def service(self) -> ManagedIndexOperationsService:
        return ManagedIndexOperationsService(
            session_factory=self.sessions,
            qdrant_client=self.client,
            corpus_key=CORPUS_KEY,
            settings=self.settings,
            clock=lambda: NOW,
        )


@pytest.fixture
def environment() -> Generator[OperationsEnvironment]:
    engine = create_engine("sqlite+pysqlite:///:memory:")

    @event.listens_for(engine, "connect")
    def enable_foreign_keys(dbapi_connection: Any, _record: Any) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine, class_=Session, expire_on_commit=False)
    client = QdrantClient(":memory:")
    settings = ManagedIndexSettings(
        dense_vector_name="managed_dense",
        sparse_vector_name="managed_sparse",
        hnsw_m=24,
        hnsw_ef_construct=128,
        retained_retired_generations=2,
        retired_cleanup_min_age_hours=24,
    )
    configuration = ManagedIndexConfiguration(
        embedding_fingerprint="e" * 64,
        chunking_fingerprint="c" * 64,
        bm25_k1=settings.bm25_k1,
        bm25_b=settings.bm25_b,
        dense_vector_name=settings.dense_vector_name,
        sparse_vector_name=settings.sparse_vector_name,
        distance_metric=settings.distance_metric,
        hnsw_m=settings.hnsw_m,
        hnsw_ef_construct=settings.hnsw_ef_construct,
        sparse_modifier=settings.sparse_modifier,
        sparse_on_disk=settings.sparse_on_disk,
        managed_index_format_version=settings.format_version,
    )
    corpus_id = uuid4()
    run_id = uuid4()
    document_pk = uuid4()
    document_version_id = uuid4()
    chunk_pk = uuid4()
    with sessions() as session, session.begin():
        session.add(
            Corpus(
                corpus_id=corpus_id,
                corpus_key=CORPUS_KEY,
                qdrant_collection_prefix=MANAGED_PREFIX,
            )
        )
        session.add(
            IngestionRun(
                run_id=run_id,
                corpus_id=corpus_id,
                status="SUCCEEDED",
                authoritative_snapshot=True,
                snapshot_hash="s" * 64,
                completed_at=NOW,
            )
        )
        session.add(
            LineageDocument(
                document_pk=document_pk,
                corpus_id=corpus_id,
                source_key="https://example.test/source",
                source_id="source",
                source_url="https://example.test/source",
                state="ACTIVE",
                first_seen_run_id=run_id,
                last_seen_run_id=run_id,
            )
        )
        session.add(
            DocumentVersion(
                document_version_id=document_version_id,
                document_pk=document_pk,
                version_number=1,
                content_hash="1" * 64,
                metadata_hash="2" * 64,
                version_hash="3" * 64,
                compatibility_document_id="document-1",
                title="Consumer guidance",
                category="credit-cards",
                source_url="https://example.test/source",
                source_version="v1",
                normalized_text="credit card dispute evidence",
                created_run_id=run_id,
            )
        )
        session.add(
            DocumentVersionPII(
                document_version_id=document_version_id,
                detector_version=DETERMINISTIC_PII_VERSION,
                classification_hash="4" * 64,
                pii_types=[],
                match_count=0,
            )
        )
        session.add(
            LineageChunk(
                chunk_pk=chunk_pk,
                document_version_id=document_version_id,
                chunking_fingerprint=configuration.chunking_fingerprint,
                compatibility_chunk_id="chunk-1",
                ordinal=0,
                chunk_hash="5" * 64,
                text="credit card dispute evidence",
                chunk_metadata={"title": "Consumer guidance"},
            )
        )
    yield OperationsEnvironment(
        engine=engine,
        sessions=sessions,
        client=client,
        settings=settings,
        configuration=configuration,
        corpus_id=corpus_id,
        run_id=run_id,
        document_pk=document_pk,
        document_version_id=document_version_id,
        chunk_pk=chunk_pk,
    )
    client.close()
    engine.dispose()


def add_generation(
    environment: OperationsEnvironment,
    generation: int,
    status: str,
    *,
    physical: bool = True,
    membership: bool = True,
    retired_at: datetime | None = None,
    failed_at: datetime | None = None,
    configuration: ManagedIndexConfiguration | None = None,
    collection: str | None = None,
) -> UUID:
    selected_configuration = configuration or environment.configuration
    identity = managed_index_configuration_identity(selected_configuration)
    encoder = BM25SparseEncoder.fit(
        ["credit card dispute evidence"],
        k1=selected_configuration.bm25_k1,
        b=selected_configuration.bm25_b,
    )
    collection_name = collection or managed_generation_collection_name(
        MANAGED_PREFIX, generation
    )
    index_state_id = uuid4()
    with environment.sessions() as session, session.begin():
        session.add(
            IndexState(
                index_state_id=index_state_id,
                corpus_id=environment.corpus_id,
                generation=generation,
                ingestion_run_id=environment.run_id,
                status=status,
                qdrant_collection=collection_name,
                snapshot_hash="s" * 64,
                bm25_corpus_hash=encoder.state.corpus_hash,
                bm25_state_hash=bm25_state_hash(encoder.state),
                bm25_state=bm25_state_to_json_dict(encoder.state),
                index_configuration=identity.payload,
                index_configuration_fingerprint=identity.fingerprint,
                embedding_fingerprint=selected_configuration.embedding_fingerprint,
                embedding_dimension=DIMENSION,
                chunking_fingerprint=selected_configuration.chunking_fingerprint,
                document_count=1,
                chunk_count=1,
                created_at=NOW - timedelta(days=10 - generation),
                activated_at=NOW - timedelta(days=9 - generation)
                if status in {"ACTIVE", "RETIRED"}
                else None,
                retired_at=retired_at,
                failed_at=failed_at,
                error_message="test failure" if status == "FAILED" else None,
            )
        )
        if membership:
            session.add(
                IndexStateChunk(
                    index_state_id=index_state_id,
                    chunk_pk=environment.chunk_pk,
                    qdrant_point_id=managed_point_id(environment.chunk_pk),
                    dense_action="COMPUTED",
                )
            )
    if physical:
        store = ManagedQdrantStore(
            environment.client,
            managed_prefix=MANAGED_PREFIX,
            configuration=selected_configuration,
            embedding_dimension=DIMENSION,
        )
        store.create_generation_collection(collection_name)
        store.upsert_points(
            collection_name,
            [managed_point(environment, index_state_id, generation, encoder)],
        )
    return index_state_id


def managed_point(
    environment: OperationsEnvironment,
    index_state_id: UUID,
    generation: int,
    encoder: BM25SparseEncoder | None = None,
) -> ManagedPointInput:
    sparse_encoder = encoder or BM25SparseEncoder.fit(["credit card dispute evidence"])
    sparse = sparse_encoder.encode_document("credit card dispute evidence")
    chunk = ActiveChunk(
        document_pk=environment.document_pk,
        document_version_id=environment.document_version_id,
        version_number=1,
        source_key="https://example.test/source",
        source_id="source",
        source_url="https://example.test/source",
        title="Consumer guidance",
        category="credit-cards",
        source_version="v1",
        content_hash="1" * 64,
        compatibility_document_id="document-1",
        chunk_pk=environment.chunk_pk,
        compatibility_chunk_id="chunk-1",
        ordinal=0,
        chunk_hash="5" * 64,
        text="credit card dispute evidence",
        chunking_fingerprint=environment.configuration.chunking_fingerprint,
    )
    return ManagedPointInput(
        chunk=chunk,
        qdrant_point_id=managed_point_id(chunk.chunk_pk),
        dense_vector=(1.0, 0.0, 0.5),
        sparse_indices=tuple(sparse.indices),
        sparse_values=tuple(sparse.values),
        ingestion_run_id=environment.run_id,
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
            "ingestion_run_id": str(environment.run_id),
            "index_generation": generation,
            "index_state_id": str(index_state_id),
            "chunking_fingerprint": chunk.chunking_fingerprint,
            "point_id_format_version": "lineage-chunk-uuid5-v1",
            "payload_schema_version": "managed-qdrant-payload-v3",
            "access_visibility": "PUBLIC",
            "allowed_groups": [],
            "contains_pii": False,
            "pii_types": [],
            "pii_detector_version": DETERMINISTIC_PII_VERSION,
        },
    )


def restoration_service(
    environment: OperationsEnvironment,
) -> tuple[ManagedGenerationRestorationService, ManagedIndexConfiguration]:
    embedding_settings = EmbeddingModelSettings(
        model="test/restoration-embedding",
        device="cpu",
        trust_remote_code=False,
    )
    fingerprint = embedding_fingerprint(
        provider=embedding_settings.provider,
        model=embedding_settings.model,
        trust_remote_code=embedding_settings.trust_remote_code,
        device=embedding_settings.device,
    )
    configuration = replace(
        environment.configuration,
        embedding_fingerprint=fingerprint,
    )
    return (
        ManagedGenerationRestorationService(
            session_factory=environment.sessions,
            qdrant_client=environment.client,
            corpus_key=CORPUS_KEY,
            managed_index_settings=environment.settings,
            embedding_settings=embedding_settings,
            clock=lambda: NOW,
        ),
        configuration,
    )


def test_status_orders_generations_and_identifies_active(
    environment: OperationsEnvironment,
) -> None:
    add_generation(environment, 3, "FAILED", physical=False, failed_at=NOW)
    add_generation(environment, 1, "RETIRED", retired_at=NOW - timedelta(days=4))
    add_generation(environment, 4, "ACTIVE")

    statuses = environment.service().status()

    assert [item.generation for item in statuses] == [1, 3, 4]
    assert statuses[-1].status == "ACTIVE"
    assert statuses[-1].cleanup_classification is CleanupClassification.ACTIVE_PROTECTED
    assert statuses[1].physical_present is False


def test_active_collection_passes_bounded_and_deep_validation(
    environment: OperationsEnvironment,
) -> None:
    add_generation(environment, 1, "ACTIVE")

    bounded = environment.service().validate()
    deep = environment.service().validate(deep=True)

    assert bounded.valid is True
    assert deep.valid is True
    assert deep.generations[0].deep is True


def test_multiple_active_generations_fail_validation(
    environment: OperationsEnvironment,
) -> None:
    add_generation(environment, 1, "ACTIVE")
    with environment.engine.begin() as connection:
        connection.execute(text("DROP INDEX uq_index_states_one_active_per_corpus"))
    add_generation(environment, 2, "ACTIVE")

    report = environment.service().validate()

    assert report.valid is False
    assert report.active_count == 2
    assert "exactly one ACTIVE" in report.issues[0]


def test_active_missing_collection_fails_validation(
    environment: OperationsEnvironment,
) -> None:
    add_generation(environment, 1, "ACTIVE", physical=False)

    result = environment.service().validate().generations[0]

    assert result.valid is False
    assert "ACTIVE Qdrant collection is missing" in result.issues


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("membership", "index_state_chunks count 0 != 1"),
        ("bm25_hash", "BM25 state hash"),
        ("bm25_corpus", "BM25 corpus hash"),
        ("configuration_hash", "configuration fingerprint"),
        ("payload_schema", "unsupported payload schema"),
    ],
)
def test_active_persisted_corruption_fails_validation(
    environment: OperationsEnvironment,
    mutation: str,
    expected: str,
) -> None:
    state_id = add_generation(environment, 1, "ACTIVE")
    with environment.sessions() as session, session.begin():
        state = session.get(IndexState, state_id)
        assert state is not None
        if mutation == "membership":
            membership = session.get(IndexStateChunk, (state_id, environment.chunk_pk))
            assert membership is not None
            session.delete(membership)
        elif mutation == "bm25_hash":
            state.bm25_state_hash = "f" * 64
        elif mutation == "bm25_corpus":
            state.bm25_corpus_hash = "f" * 64
        elif mutation == "configuration_hash":
            state.index_configuration_fingerprint = "f" * 64
        else:
            payload = dict(state.index_configuration or {})
            payload["payload_schema_version"] = "unsupported-v99"
            state.index_configuration = payload
            state.index_configuration_fingerprint = managed_index_configuration_identity(
                replace(environment.configuration, payload_schema_version="unsupported-v99")
            ).fingerprint

    result = environment.service().validate().generations[0]

    assert result.valid is False
    assert any(expected in issue for issue in result.issues)


def test_active_qdrant_point_count_mismatch_fails_validation(
    environment: OperationsEnvironment,
) -> None:
    add_generation(environment, 1, "ACTIVE")
    environment.client.delete(
        managed_generation_collection_name(MANAGED_PREFIX, 1),
        points_selector=models.PointIdsList(points=[managed_point_id(environment.chunk_pk)]),
        wait=True,
    )

    result = environment.service().validate().generations[0]

    assert any("Qdrant point count 0 != 1" in issue for issue in result.issues)


def test_active_bm25_corpus_hash_is_checked_against_membership_texts(
    environment: OperationsEnvironment,
) -> None:
    add_generation(environment, 1, "ACTIVE")
    with environment.sessions() as session, session.begin():
        chunk = session.get(LineageChunk, environment.chunk_pk)
        assert chunk is not None
        chunk.text = "externally corrupted text"

    result = environment.service().validate().generations[0]

    assert any("indexed membership texts" in issue for issue in result.issues)


@pytest.mark.parametrize(
    ("vectors", "sparse", "expected"),
    [
        (
            {"wrong_dense": models.VectorParams(size=3, distance=models.Distance.COSINE)},
            True,
            "missing dense",
        ),
        (
            {"managed_dense": models.VectorParams(size=3, distance=models.Distance.COSINE)},
            False,
            "missing sparse",
        ),
        (
            {"managed_dense": models.VectorParams(size=4, distance=models.Distance.COSINE)},
            True,
            "dense dimension",
        ),
        (
            {"managed_dense": models.VectorParams(size=3, distance=models.Distance.DOT)},
            True,
            "distance",
        ),
    ],
)
def test_active_qdrant_shape_mismatch_fails_validation(
    environment: OperationsEnvironment,
    vectors: dict[str, models.VectorParams],
    sparse: bool,
    expected: str,
) -> None:
    add_generation(environment, 1, "ACTIVE", physical=False)
    sparse_config = (
        {"managed_sparse": models.SparseVectorParams()} if sparse else None
    )
    environment.client.create_collection(
        managed_generation_collection_name(MANAGED_PREFIX, 1),
        vectors_config=vectors,
        sparse_vectors_config=sparse_config,
    )

    result = environment.service().validate().generations[0]

    assert result.valid is False
    assert any(expected in issue for issue in result.issues)


@pytest.mark.parametrize("field", ["access_visibility", "pii_detector_version"])
def test_deep_validation_checks_acl_and_pii_payloads(
    environment: OperationsEnvironment,
    field: str,
) -> None:
    state_id = add_generation(environment, 1, "ACTIVE")
    point = managed_point(environment, state_id, 1)
    payload = dict(point.payload)
    payload[field] = "invalid"
    environment.client.upsert(
        collection_name=managed_generation_collection_name(MANAGED_PREFIX, 1),
        points=[
            models.PointStruct(
                id=point.qdrant_point_id,
                vector={
                    "managed_dense": list(point.dense_vector),
                    "managed_sparse": models.SparseVector(
                        indices=list(point.sparse_indices),
                        values=list(point.sparse_values),
                    ),
                },
                payload=payload,
            )
        ],
        wait=True,
    )

    result = environment.service().validate(deep=True).generations[0]

    assert result.valid is False
    assert any("ACL payload" in issue or "PII detector" in issue for issue in result.issues)


def test_missing_retired_and_failed_physical_collections_are_allowed(
    environment: OperationsEnvironment,
) -> None:
    add_generation(
        environment,
        1,
        "RETIRED",
        physical=False,
        retired_at=NOW - timedelta(days=4),
    )
    add_generation(environment, 2, "FAILED", physical=False, failed_at=NOW)
    add_generation(environment, 3, "ACTIVE")

    report = environment.service().validate()

    assert report.valid is True
    assert "physical=cleaned" in report.generations[0].notes
    assert "FAILED history has no physical collection" in report.generations[1].notes


def test_failed_present_collection_is_reported_and_cleanup_eligible(
    environment: OperationsEnvironment,
) -> None:
    add_generation(environment, 1, "FAILED", failed_at=NOW)

    status = environment.service().status()[0]
    validation = environment.service().validate().generations[0]

    assert status.physical_present is True
    assert status.cleanup_classification is CleanupClassification.ELIGIBLE_FAILED
    assert validation.physical_present is True


def cleanup_state(generation: int, status: str, age_hours: int = 72) -> GenerationCleanupState:
    return GenerationCleanupState(
        index_state_id=UUID(f"00000000-0000-0000-0000-{generation:012d}"),
        generation=generation,
        status=status,
        qdrant_collection=managed_generation_collection_name(MANAGED_PREFIX, generation),
        retired_at=NOW - timedelta(hours=age_hours) if status == "RETIRED" else None,
        failed_at=NOW - timedelta(hours=age_hours) if status == "FAILED" else None,
    )


def test_cleanup_planner_retention_age_and_protected_statuses() -> None:
    states = (
        cleanup_state(1, "RETIRED", 72),
        cleanup_state(2, "RETIRED", 48),
        cleanup_state(3, "RETIRED", 1),
        cleanup_state(4, "ACTIVE"),
        cleanup_state(5, "READY"),
        cleanup_state(6, "BUILDING"),
    )
    physical = tuple(state.qdrant_collection for state in states)

    plan = plan_generation_cleanup(
        states,
        corpus_key=CORPUS_KEY,
        managed_prefix=MANAGED_PREFIX,
        physical_collections=physical,
        retained_retired_generations=2,
        retired_cleanup_min_age_hours=24,
        now=NOW,
    )

    classifications = {item.generation: item.classification for item in plan.candidates}
    assert classifications[1] is CleanupClassification.ELIGIBLE_RETIRED
    assert classifications[2] is CleanupClassification.RETAINED
    assert classifications[3] is CleanupClassification.RETAINED
    assert classifications[4] is CleanupClassification.ACTIVE_PROTECTED
    assert classifications[5] is CleanupClassification.READY_PROTECTED
    assert classifications[6] is CleanupClassification.BUILDING_PROTECTED


def test_retention_zero_still_protects_active_and_age_protects_recent() -> None:
    states = (cleanup_state(1, "RETIRED", 1), cleanup_state(2, "ACTIVE"))

    plan = plan_generation_cleanup(
        states,
        corpus_key=CORPUS_KEY,
        managed_prefix=MANAGED_PREFIX,
        physical_collections=tuple(state.qdrant_collection for state in states),
        retained_retired_generations=0,
        retired_cleanup_min_age_hours=24,
        now=NOW,
    )

    assert plan.candidates[0].classification is CleanupClassification.RECENT
    assert plan.candidates[1].classification is CleanupClassification.ACTIVE_PROTECTED


def test_missing_retired_history_does_not_consume_physical_retention_slot() -> None:
    states = (
        cleanup_state(1, "RETIRED", 72),
        cleanup_state(2, "RETIRED", 72),
    )

    plan = plan_generation_cleanup(
        states,
        corpus_key=CORPUS_KEY,
        managed_prefix=MANAGED_PREFIX,
        physical_collections=[states[0].qdrant_collection],
        retained_retired_generations=1,
        retired_cleanup_min_age_hours=0,
        now=NOW,
    )

    assert plan.candidates[0].classification is CleanupClassification.RETAINED
    assert plan.candidates[1].classification is CleanupClassification.PHYSICAL_MISSING


def test_status_validate_plan_and_unconfirmed_cleanup_are_read_only(
    environment: OperationsEnvironment,
) -> None:
    state_id = add_generation(
        environment,
        1,
        "RETIRED",
        retired_at=NOW - timedelta(days=4),
    )
    environment.settings = environment.settings.model_copy(
        update={"retained_retired_generations": 0}
    )
    service = environment.service()
    before = database_counts(environment)

    service.status()
    service.validate()
    plan = service.cleanup_plan()
    result = service.execute_cleanup(plan, confirmed=False)

    assert database_counts(environment) == before
    assert environment.client.collection_exists(plan.candidates[0].qdrant_collection)
    assert result.deleted_collections == ()
    assert state_id == plan.candidates[0].index_state_id


def test_confirmed_cleanup_preserves_all_postgresql_history_and_active_health(
    environment: OperationsEnvironment,
) -> None:
    retired_id = add_generation(
        environment,
        1,
        "RETIRED",
        retired_at=NOW - timedelta(days=4),
    )
    add_generation(environment, 2, "FAILED", physical=False, failed_at=NOW)
    add_generation(
        environment,
        3,
        "RETIRED",
        retired_at=NOW - timedelta(hours=1),
    )
    add_generation(environment, 4, "ACTIVE")
    environment.settings = environment.settings.model_copy(
        update={"retained_retired_generations": 1}
    )
    service = environment.service()
    before = database_counts(environment)

    plan = service.cleanup_plan()
    result = service.execute_cleanup(plan, confirmed=True)

    assert result.deleted_collections == (
        managed_generation_collection_name(MANAGED_PREFIX, 1),
    )
    assert not environment.client.collection_exists(result.deleted_collections[0])
    assert environment.client.collection_exists(
        managed_generation_collection_name(MANAGED_PREFIX, 3)
    )
    assert environment.client.collection_exists(
        managed_generation_collection_name(MANAGED_PREFIX, 4)
    )
    assert database_counts(environment) == before
    with environment.sessions() as session:
        retired = session.get(IndexState, retired_id)
        assert retired is not None and retired.status == "RETIRED"
    assert service.validate().valid is True
    cleaned = next(item for item in service.status() if item.generation == 1)
    assert cleaned.cleanup_classification is CleanupClassification.PHYSICAL_MISSING


def test_stale_cleanup_plan_rechecks_and_skips_changed_status(
    environment: OperationsEnvironment,
) -> None:
    state_id = add_generation(
        environment,
        1,
        "RETIRED",
        retired_at=NOW - timedelta(days=4),
    )
    environment.settings = environment.settings.model_copy(
        update={"retained_retired_generations": 0}
    )
    service = environment.service()
    plan = service.cleanup_plan()
    with environment.sessions() as session, session.begin():
        state = session.get(IndexState, state_id)
        assert state is not None
        state.status = "READY"

    result = service.execute_cleanup(plan, confirmed=True)

    assert result.deleted_collections == ()
    assert "READY_PROTECTED" in result.items[0].reason
    assert environment.client.collection_exists(plan.eligible[0].qdrant_collection)


def test_noncanonical_and_phase_one_two_collections_are_never_candidates() -> None:
    state = replace(cleanup_state(1, "RETIRED"), qdrant_collection="not_canonical")

    plan = plan_generation_cleanup(
        [state],
        corpus_key=CORPUS_KEY,
        managed_prefix=MANAGED_PREFIX,
        physical_collections=[
            "not_canonical",
            "consumer_finance_phase1",
            "consumer_finance_evaluation",
        ],
        retained_retired_generations=0,
        retired_cleanup_min_age_hours=0,
        now=NOW,
    )

    assert plan.eligible == ()
    assert plan.orphans == ()
    assert plan.candidates[0].classification is CleanupClassification.OTHER_PROTECTED


def test_orphan_is_detected_but_normal_cleanup_does_not_delete_it(
    environment: OperationsEnvironment,
) -> None:
    store = ManagedQdrantStore(
        environment.client,
        managed_prefix=MANAGED_PREFIX,
        configuration=environment.configuration,
        embedding_dimension=DIMENSION,
    )
    orphan = store.collection_name(99)
    store.create_generation_collection(orphan)

    plan = environment.service().cleanup_plan()
    result = environment.service().execute_cleanup(plan, confirmed=True)

    assert len(plan.orphans) == 1
    assert plan.orphans[0].qdrant_collection == orphan
    assert result.items == ()
    assert environment.client.collection_exists(orphan)


def test_failed_partial_collection_can_be_cleaned(
    environment: OperationsEnvironment,
) -> None:
    state_id = add_generation(
        environment,
        1,
        "FAILED",
        failed_at=NOW,
        membership=False,
    )
    service = environment.service()

    result = service.execute_cleanup(service.cleanup_plan(), confirmed=True)

    assert result.deleted_collections == (
        managed_generation_collection_name(MANAGED_PREFIX, 1),
    )
    with environment.sessions() as session:
        state = session.get(IndexState, state_id)
        assert state is not None and state.status == "FAILED"


def test_deletion_failure_is_reported_without_database_corruption(
    environment: OperationsEnvironment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    add_generation(environment, 1, "FAILED", failed_at=NOW)
    service = environment.service()
    before = database_counts(environment)
    monkeypatch.setattr(environment.client, "delete_collection", lambda _name: False)

    result = service.execute_cleanup(service.cleanup_plan(), confirmed=True)

    assert result.deleted_collections == ()
    assert "physical deletion failed" in result.items[0].reason
    assert database_counts(environment) == before


def test_one_deletion_failure_does_not_prevent_other_eligible_cleanup(
    environment: OperationsEnvironment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    add_generation(environment, 1, "FAILED", failed_at=NOW)
    add_generation(environment, 2, "FAILED", failed_at=NOW)
    service = environment.service()
    original_delete = environment.client.delete_collection

    def delete_with_one_failure(collection_name: str) -> bool:
        if collection_name.endswith("g000001"):
            return False
        return bool(original_delete(collection_name))

    monkeypatch.setattr(environment.client, "delete_collection", delete_with_one_failure)

    result = service.execute_cleanup(service.cleanup_plan(), confirmed=True)

    assert [item.deleted for item in result.items] == [False, True]
    assert environment.client.collection_exists(
        managed_generation_collection_name(MANAGED_PREFIX, 1)
    )
    assert not environment.client.collection_exists(
        managed_generation_collection_name(MANAGED_PREFIX, 2)
    )


def database_counts(environment: OperationsEnvironment) -> tuple[int, ...]:
    tables = (
        Corpus,
        IngestionRun,
        LineageDocument,
        DocumentVersion,
        DocumentVersionPII,
        LineageChunk,
        IndexState,
        IndexStateChunk,
    )
    with environment.sessions() as session:
        return tuple(
            int(session.scalar(select(func.count()).select_from(table)) or 0)
            for table in tables
        )


def test_retired_generation_produces_mutation_free_restore_plan(
    environment: OperationsEnvironment,
) -> None:
    service, configuration = restoration_service(environment)
    source_id = add_generation(
        environment,
        1,
        "RETIRED",
        retired_at=NOW - timedelta(days=2),
        configuration=configuration,
    )
    add_generation(environment, 2, "ACTIVE", configuration=configuration)
    before_db = database_counts(environment)
    before_collections = tuple(
        sorted(item.name for item in environment.client.get_collections().collections)
    )

    plan = service.plan(1)

    assert plan.source_index_state_id == source_id
    assert plan.source_generation == 1
    assert plan.current_active_generation == 2
    assert plan.proposed_generation == 3
    assert plan.proposed_target_collection.endswith("_g000003")
    assert plan.chunk_count == 1
    assert database_counts(environment) == before_db
    assert tuple(
        sorted(item.name for item in environment.client.get_collections().collections)
    ) == before_collections


@pytest.mark.parametrize("status", ["ACTIVE", "READY", "BUILDING", "FAILED"])
def test_restore_rejects_non_retired_sources(
    environment: OperationsEnvironment,
    status: str,
) -> None:
    service, configuration = restoration_service(environment)
    add_generation(
        environment,
        1,
        status,
        failed_at=NOW if status == "FAILED" else None,
        configuration=configuration,
    )
    if status != "ACTIVE":
        add_generation(environment, 2, "ACTIVE", configuration=configuration)

    with pytest.raises(RestorationSourceIneligibleError, match=status):
        service.plan(1)


def test_restore_reports_cleaned_historical_collection(
    environment: OperationsEnvironment,
) -> None:
    service, configuration = restoration_service(environment)
    add_generation(
        environment,
        1,
        "RETIRED",
        physical=False,
        retired_at=NOW - timedelta(days=2),
        configuration=configuration,
    )
    add_generation(environment, 2, "ACTIVE", configuration=configuration)

    with pytest.raises(
        RestorationUnavailablePhysicalCleanedError,
        match="RESTORATION_UNAVAILABLE_PHYSICAL_CLEANED",
    ):
        service.plan(1)


def test_restore_rejects_noncanonical_historical_collection(
    environment: OperationsEnvironment,
) -> None:
    service, configuration = restoration_service(environment)
    source_id = add_generation(
        environment,
        1,
        "RETIRED",
        retired_at=NOW - timedelta(days=2),
        configuration=configuration,
    )
    add_generation(environment, 2, "ACTIVE", configuration=configuration)
    with environment.sessions() as session, session.begin():
        source = session.get(IndexState, source_id)
        assert source is not None
        source.qdrant_collection = "noncanonical_historical_collection"

    with pytest.raises(RestorationValidationError, match="not canonical"):
        service.plan(1)


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("membership", "index_state_chunks"),
        ("bm25", "BM25 state hash"),
        ("configuration", "configuration fingerprint"),
        ("payload", "payload schema"),
    ],
)
def test_restore_rejects_corrupt_historical_generation(
    environment: OperationsEnvironment,
    mutation: str,
    expected: str,
) -> None:
    service, configuration = restoration_service(environment)
    source_id = add_generation(
        environment,
        1,
        "RETIRED",
        retired_at=NOW - timedelta(days=2),
        configuration=configuration,
    )
    add_generation(environment, 2, "ACTIVE", configuration=configuration)
    with environment.sessions() as session, session.begin():
        source = session.get(IndexState, source_id)
        assert source is not None
        if mutation == "membership":
            membership = session.get(
                IndexStateChunk,
                (source_id, environment.chunk_pk),
            )
            assert membership is not None
            session.delete(membership)
        elif mutation == "bm25":
            source.bm25_state_hash = "f" * 64
        elif mutation == "configuration":
            source.index_configuration_fingerprint = "f" * 64
        else:
            incompatible = replace(
                configuration,
                payload_schema_version="managed-qdrant-payload-v1",
            )
            identity = managed_index_configuration_identity(incompatible)
            source.index_configuration = identity.payload
            source.index_configuration_fingerprint = identity.fingerprint

    with pytest.raises(RestorationValidationError, match=expected):
        service.plan(1)


def test_restore_rejects_runtime_embedding_incompatibility(
    environment: OperationsEnvironment,
) -> None:
    service, compatible = restoration_service(environment)
    incompatible = replace(compatible, embedding_fingerprint="f" * 64)
    add_generation(
        environment,
        1,
        "RETIRED",
        retired_at=NOW - timedelta(days=2),
        configuration=incompatible,
    )
    add_generation(environment, 2, "ACTIVE", configuration=incompatible)

    with pytest.raises(RestorationValidationError, match="embedding fingerprint"):
        service.plan(1)


def test_confirmed_restore_copies_history_into_new_active_generation(
    environment: OperationsEnvironment,
) -> None:
    service, configuration = restoration_service(environment)
    source_id = add_generation(
        environment,
        1,
        "RETIRED",
        retired_at=NOW - timedelta(days=2),
        configuration=configuration,
    )
    previous_active_id = add_generation(
        environment,
        2,
        "ACTIVE",
        configuration=configuration,
    )
    source_collection = managed_generation_collection_name(MANAGED_PREFIX, 1)
    source_points_before = ManagedQdrantStore(
        environment.client,
        managed_prefix=MANAGED_PREFIX,
        configuration=configuration,
        embedding_dimension=DIMENSION,
    ).scroll_all_points(source_collection)
    before_counts = database_counts(environment)

    result = service.execute(
        service.plan(1),
        restoration_reason="return to validated historical policy",
    )

    assert result.new_active_generation == 3
    assert result.dense_vectors_copied == 1
    assert environment.client.collection_exists(source_collection)
    assert environment.client.collection_exists(
        managed_generation_collection_name(MANAGED_PREFIX, 2)
    )
    assert environment.client.collection_exists(result.collection)
    with environment.sessions() as session:
        states = LineageRepository(session).list_index_states(environment.corpus_id)
        assert [(state.generation, state.status) for state in states] == [
            (1, "RETIRED"),
            (2, "RETIRED"),
            (3, "ACTIVE"),
        ]
        restored = states[2]
        assert restored.restored_from_index_state_id == source_id
        assert restored.restoration_reason == "return to validated historical policy"
        assert restored.ingestion_run_id == states[0].ingestion_run_id
        assert restored.bm25_state == states[0].bm25_state
        assert restored.bm25_state_hash == states[0].bm25_state_hash
        assert restored.bm25_corpus_hash == states[0].bm25_corpus_hash
        memberships = repository_memberships = LineageRepository(
            session
        ).list_index_state_chunks(restored.index_state_id)
        assert len(repository_memberships) == 1
        assert memberships[0].chunk_pk == environment.chunk_pk
        assert memberships[0].dense_action == "REUSED"
        assert memberships[0].dense_source_chunk_pk == environment.chunk_pk
        assert sum(state.status == "ACTIVE" for state in states) == 1
        assert session.get(IndexState, previous_active_id).status == "RETIRED"  # type: ignore[union-attr]
    after_counts = database_counts(environment)
    assert after_counts[:6] == before_counts[:6]
    assert after_counts[6:] == (before_counts[6] + 1, before_counts[7] + 1)
    store = ManagedQdrantStore(
        environment.client,
        managed_prefix=MANAGED_PREFIX,
        configuration=configuration,
        embedding_dimension=DIMENSION,
    )
    source_points_after = store.scroll_all_points(source_collection)
    restored_points = store.scroll_all_points(result.collection)
    assert source_points_after == source_points_before
    assert restored_points[0].dense_vector == pytest.approx(
        source_points_before[0].dense_vector
    )
    assert restored_points[0].sparse_vector == source_points_before[0].sparse_vector
    expected_payload = dict(source_points_before[0].payload)
    expected_payload["index_generation"] = 3
    expected_payload["index_state_id"] = str(result.new_active_index_state_id)
    assert dict(restored_points[0].payload) == expected_payload


def test_physical_copy_failure_keeps_active_and_records_failed_candidate(
    environment: OperationsEnvironment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, configuration = restoration_service(environment)
    add_generation(
        environment,
        1,
        "RETIRED",
        retired_at=NOW - timedelta(days=2),
        configuration=configuration,
    )
    active_id = add_generation(environment, 2, "ACTIVE", configuration=configuration)
    plan = service.plan(1)

    def fail_copy(*_args: object, **_kwargs: object) -> int:
        raise RuntimeError("injected copy failure")

    monkeypatch.setattr(ManagedQdrantStore, "copy_generation_points", fail_copy)
    with pytest.raises(RestorationBuildError, match="failed before READY"):
        service.execute(plan, restoration_reason="failure test")

    with environment.sessions() as session:
        states = LineageRepository(session).list_index_states(environment.corpus_id)
        assert [(state.generation, state.status) for state in states] == [
            (1, "RETIRED"),
            (2, "ACTIVE"),
            (3, "FAILED"),
        ]
        assert session.get(IndexState, active_id).status == "ACTIVE"  # type: ignore[union-attr]


def test_cleanup_protects_source_of_in_progress_restoration(
    environment: OperationsEnvironment,
) -> None:
    _, configuration = restoration_service(environment)
    source_id = add_generation(
        environment,
        1,
        "RETIRED",
        retired_at=NOW - timedelta(days=10),
        configuration=configuration,
    )
    add_generation(environment, 2, "ACTIVE", configuration=configuration)
    with environment.sessions() as session, session.begin():
        repository = LineageRepository(session)
        repository.create_building_index_state(
            environment.corpus_id,
            3,
            environment.run_id,
            managed_generation_collection_name(MANAGED_PREFIX, 3),
            "s" * 64,
            configuration.embedding_fingerprint,
            configuration.chunking_fingerprint,
            restored_from_index_state_id=source_id,
            restoration_reason="in progress",
        )
    settings = environment.settings.model_copy(
        update={"retained_retired_generations": 0}
    )
    service = ManagedIndexOperationsService(
        session_factory=environment.sessions,
        qdrant_client=environment.client,
        corpus_key=CORPUS_KEY,
        settings=settings,
        clock=lambda: NOW,
    )

    source = next(
        item for item in service.cleanup_plan().candidates if item.generation == 1
    )

    assert source.classification is CleanupClassification.RESTORATION_SOURCE_PROTECTED


def test_stale_active_prevents_final_restoration_activation(
    environment: OperationsEnvironment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, configuration = restoration_service(environment)
    add_generation(
        environment,
        1,
        "RETIRED",
        retired_at=NOW - timedelta(days=2),
        configuration=configuration,
    )
    old_active_id = add_generation(environment, 2, "ACTIVE", configuration=configuration)
    plan = service.plan(1)
    original_activate = service._activate

    def publish_concurrently(candidate_id: UUID, stale_plan: Any) -> None:
        with environment.sessions() as session, session.begin():
            repository = LineageRepository(session)
            repository.transition_index_state(
                old_active_id,
                "ACTIVE",
                "RETIRED",
                occurred_at=NOW,
            )
            concurrent = repository.create_building_index_state(
                environment.corpus_id,
                4,
                environment.run_id,
                managed_generation_collection_name(MANAGED_PREFIX, 4),
                "s" * 64,
                configuration.embedding_fingerprint,
                configuration.chunking_fingerprint,
            )
            concurrent.bm25_state = {"placeholder": True}
            concurrent.status = "ACTIVE"
            concurrent.embedding_dimension = DIMENSION
            concurrent.activated_at = NOW
        original_activate(candidate_id, stale_plan)

    monkeypatch.setattr(service, "_activate", publish_concurrently)
    with pytest.raises(
        RestorationActivationConflictError,
        match=r"atomically activated|different ACTIVE",
    ):
        service.execute(plan, restoration_reason="stale active test")

    with environment.sessions() as session:
        states = LineageRepository(session).list_index_states(environment.corpus_id)
        assert [(state.generation, state.status) for state in states] == [
            (1, "RETIRED"),
            (2, "RETIRED"),
            (3, "FAILED"),
            (4, "ACTIVE"),
        ]
