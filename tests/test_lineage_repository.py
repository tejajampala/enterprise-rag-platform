from __future__ import annotations

from collections.abc import Generator
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

import pytest
from sqlalchemy import Engine, create_engine, event, select
from sqlalchemy.orm import Session

from consumer_finance_rag.lineage.models import (
    Base,
    Corpus,
    DocumentVersion,
    IndexState,
    IngestionRun,
    LineageDocument,
)
from consumer_finance_rag.lineage.repository import (
    InvalidLineageStateError,
    LineageConflictError,
    LineageNotFoundError,
    LineageRepository,
)
from consumer_finance_rag.lineage.types import ChunkPersistenceInput


@pytest.fixture
def engine() -> Generator[Engine]:
    sqlite_engine = create_engine("sqlite+pysqlite:///:memory:")

    @event.listens_for(sqlite_engine, "connect")
    def enable_foreign_keys(dbapi_connection: Any, _connection_record: Any) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(sqlite_engine)
    yield sqlite_engine
    sqlite_engine.dispose()


@pytest.fixture
def session(engine: Engine) -> Generator[Session]:
    with Session(engine) as database_session:
        yield database_session


@pytest.fixture
def repository(session: Session) -> LineageRepository:
    return LineageRepository(session)


def hash_value(value: int) -> str:
    return f"{value:064x}"


def create_corpus_and_run(
    repository: LineageRepository, suffix: str = "managed"
) -> tuple[Corpus, IngestionRun]:
    corpus = repository.create_corpus(
        f"consumer-finance-{suffix}", f"consumer-finance-{suffix}"
    )
    run = repository.start_ingestion_run(corpus.corpus_id, authoritative_snapshot=True)
    return corpus, run


def create_document_and_version(
    repository: LineageRepository,
    *,
    suffix: str = "managed",
    version_number: int = 1,
    hash_seed: int = 1,
) -> tuple[Corpus, IngestionRun, LineageDocument, DocumentVersion]:
    corpus, run = create_corpus_and_run(repository, suffix)
    document = repository.create_document(
        corpus.corpus_id,
        source_key=f"cfpb:{suffix}",
        source_id=f"source-{suffix}",
        source_url=f"https://www.consumerfinance.gov/{suffix}",
        first_seen_run_id=run.run_id,
    )
    version = repository.create_document_version(
        document.document_pk,
        version_number=version_number,
        content_hash=hash_value(hash_seed),
        metadata_hash=hash_value(hash_seed + 100),
        version_hash=hash_value(hash_seed + 200),
        compatibility_document_id=f"document-{hash_seed}",
        title="Consumer-finance guidance",
        category="credit-cards",
        source_url=document.source_url,
        source_version="2026-09-18",
        normalized_text="Normalized public guidance.",
        created_run_id=run.run_id,
    )
    return corpus, run, document, version


def chunk_input(
    ordinal: int,
    *,
    chunk_hash: str | None = None,
    fingerprint: str | None = None,
) -> ChunkPersistenceInput:
    return ChunkPersistenceInput(
        chunking_fingerprint=fingerprint or hash_value(400),
        compatibility_chunk_id=f"chunk-{ordinal}",
        ordinal=ordinal,
        chunk_hash=chunk_hash or hash_value(500 + ordinal),
        text=f"Chunk {ordinal}",
        metadata={"ordinal": ordinal},
    )


def create_index_state(
    repository: LineageRepository,
    corpus: Corpus,
    run_id: UUID,
    *,
    generation: int,
    status: str,
) -> IndexState:
    state = repository.create_building_index_state(
        corpus.corpus_id,
        generation=generation,
        ingestion_run_id=run_id,
        qdrant_collection=f"{corpus.qdrant_collection_prefix}-g{generation:04d}",
        snapshot_hash=hash_value(700 + generation),
        embedding_fingerprint=hash_value(800),
        chunking_fingerprint=hash_value(400),
        document_count=1,
        chunk_count=2,
    )
    if status == "BUILDING":
        return state
    repository.update_index_build_metadata(
        state.index_state_id,
        bm25_state={"vocabulary": {"finance": 0}},
        bm25_corpus_hash=hash_value(900),
        bm25_state_hash=hash_value(901),
        index_configuration={"format_version": "test-v1"},
        index_configuration_fingerprint=hash_value(902),
        embedding_dimension=1024,
        document_count=1,
        chunk_count=2,
    )
    repository.transition_index_state(state.index_state_id, "BUILDING", "READY")
    if status == "ACTIVE":
        repository.transition_index_state(
            state.index_state_id,
            "READY",
            "ACTIVE",
            occurred_at=datetime.now(UTC),
        )
    return state


def test_create_get_and_idempotently_get_corpus(repository: LineageRepository) -> None:
    created = repository.create_corpus("consumer-finance-managed", "managed-prefix")

    fetched = repository.get_corpus_by_key("consumer-finance-managed")
    idempotent = repository.get_or_create_corpus(
        "consumer-finance-managed", "managed-prefix"
    )

    assert fetched is created
    assert idempotent is created


def test_get_or_create_corpus_rejects_conflicting_prefix(
    repository: LineageRepository,
) -> None:
    repository.create_corpus("consumer-finance-managed", "managed-prefix")

    with pytest.raises(LineageConflictError, match="already uses Qdrant prefix"):
        repository.get_or_create_corpus("consumer-finance-managed", "different-prefix")


def test_start_and_finish_ingestion_run_with_counts(
    repository: LineageRepository,
) -> None:
    corpus = repository.create_corpus("consumer-finance-managed", "managed-prefix")
    run = repository.start_ingestion_run(
        corpus.corpus_id,
        authoritative_snapshot=True,
        snapshot_hash=hash_value(1),
    )
    completed_at = datetime.now(UTC)

    finished = repository.finish_ingestion_run(
        run.run_id,
        "SUCCEEDED",
        completed_at,
        documents_new=2,
        documents_changed=3,
        documents_unchanged=4,
        documents_deleted=1,
    )

    assert repository.get_ingestion_run(run.run_id) is run
    assert finished.status == "SUCCEEDED"
    assert finished.completed_at == completed_at
    assert (
        finished.documents_new,
        finished.documents_changed,
        finished.documents_unchanged,
        finished.documents_deleted,
    ) == (2, 3, 4, 1)


def test_fail_ingestion_run_and_reject_second_transition(
    repository: LineageRepository,
) -> None:
    _corpus, run = create_corpus_and_run(repository)
    completed_at = datetime.now(UTC)

    failed = repository.fail_ingestion_run(run.run_id, "source unavailable", completed_at)

    assert failed.status == "FAILED"
    assert failed.error_message == "source unavailable"
    assert failed.completed_at == completed_at
    with pytest.raises(InvalidLineageStateError, match="not RUNNING"):
        repository.fail_ingestion_run(run.run_id, "again", completed_at)
    with pytest.raises(LineageNotFoundError, match="Ingestion run not found"):
        repository.fail_ingestion_run(uuid4(), "missing", completed_at)


def test_create_get_mark_seen_and_soft_delete_document(
    repository: LineageRepository,
) -> None:
    corpus, first_run = create_corpus_and_run(repository)
    document = repository.create_document(
        corpus.corpus_id,
        source_key="cfpb:cards",
        source_id="source-v1",
        source_url="https://example.test/v1",
        first_seen_run_id=first_run.run_id,
    )
    second_run = repository.start_ingestion_run(
        corpus.corpus_id, authoritative_snapshot=True
    )

    seen = repository.mark_document_seen(
        document.document_pk,
        second_run.run_id,
        source_id="source-v2",
        source_url="https://example.test/v2",
    )

    assert repository.get_document_by_source_key(corpus.corpus_id, "cfpb:cards") is document
    assert seen.last_seen_run_id == second_run.run_id
    assert seen.source_id == "source-v2"
    assert repository.list_active_documents(corpus.corpus_id) == [document]

    deleted = repository.mark_document_deleted(document.document_pk, second_run.run_id)

    assert deleted.state == "DELETED"
    assert deleted.deleted_run_id == second_run.run_id
    assert repository.list_active_documents(corpus.corpus_id) == []


def test_document_versions_are_looked_up_deterministically(
    repository: LineageRepository,
) -> None:
    corpus, run = create_corpus_and_run(repository)
    document = repository.create_document(
        corpus.corpus_id,
        "cfpb:cards",
        "source",
        None,
        run.run_id,
    )
    second = repository.create_document_version(
        document.document_pk,
        2,
        hash_value(2),
        hash_value(102),
        hash_value(202),
        "document-2",
        "Second version",
        "credit-cards",
        None,
        "2",
        "Second normalized version.",
        run.run_id,
    )
    first = repository.create_document_version(
        document.document_pk,
        1,
        hash_value(1),
        hash_value(101),
        hash_value(201),
        "document-1",
        "First version",
        "credit-cards",
        None,
        "1",
        "First normalized version.",
        run.run_id,
    )

    assert repository.get_latest_document_version(document.document_pk) is second
    assert (
        repository.get_document_version_by_hash(document.document_pk, first.version_hash)
        is first
    )


def test_active_documents_load_with_latest_version_without_dropping_unversioned(
    repository: LineageRepository,
) -> None:
    corpus, run = create_corpus_and_run(repository)
    versioned = repository.create_document(
        corpus.corpus_id,
        "cfpb:versioned",
        "source-versioned",
        None,
        run.run_id,
    )
    first = repository.create_document_version(
        versioned.document_pk,
        1,
        hash_value(1),
        hash_value(101),
        hash_value(201),
        "document-1",
        "First",
        "credit-cards",
        None,
        "1",
        "First.",
        run.run_id,
    )
    latest = repository.create_document_version(
        versioned.document_pk,
        2,
        hash_value(2),
        hash_value(102),
        hash_value(202),
        "document-2",
        "Second",
        "credit-cards",
        None,
        "2",
        "Second.",
        run.run_id,
    )
    repository.create_document(
        corpus.corpus_id,
        "cfpb:unversioned",
        "source-unversioned",
        None,
        run.run_id,
    )

    rows = repository.list_active_documents_with_latest_versions(corpus.corpus_id)
    by_source_key = {document.source_key: version for document, version in rows}

    assert by_source_key == {
        "cfpb:unversioned": None,
        "cfpb:versioned": latest,
    }
    assert first is not latest


def test_document_version_conflicts_are_domain_errors(
    repository: LineageRepository,
) -> None:
    _corpus, run, document, version = create_document_and_version(repository)

    with pytest.raises(LineageConflictError, match="version number"):
        repository.create_document_version(
            document.document_pk,
            1,
            hash_value(2),
            hash_value(102),
            hash_value(202),
            "document-2",
            "Duplicate",
            "credit-cards",
            None,
            "2",
            "Duplicate.",
            run.run_id,
        )
    assert version.version_number == 1


def test_chunk_batch_creation_and_listing_are_deterministic(
    repository: LineageRepository,
) -> None:
    _corpus, _run, _document, version = create_document_and_version(repository)

    created = repository.create_chunks(
        version.document_version_id,
        [chunk_input(1), chunk_input(0)],
    )
    listed = repository.list_chunks_for_version(version.document_version_id)

    assert [chunk.ordinal for chunk in created] == [1, 0]
    assert [chunk.ordinal for chunk in listed] == [0, 1]
    assert listed[0].compatibility_chunk_id == "chunk-0"
    assert listed[0].chunk_metadata == {"ordinal": 0}


def test_record_and_get_run_documents(repository: LineageRepository) -> None:
    _corpus, run, document, version = create_document_and_version(repository)

    record = repository.record_run_document(
        run.run_id,
        document.document_pk,
        "NEW",
        observed_version_id=version.document_version_id,
        content_changed=True,
        metadata_changed=True,
    )

    assert repository.get_run_documents(run.run_id) == [record]
    assert record.observed_version_id == version.document_version_id
    with pytest.raises(LineageConflictError, match="already records document"):
        repository.record_run_document(run.run_id, document.document_pk, "UNCHANGED")


def test_create_latest_and_active_index_state(repository: LineageRepository) -> None:
    corpus, first_run = create_corpus_and_run(repository)
    first = create_index_state(repository, corpus, first_run.run_id, generation=1, status="READY")
    second_run = repository.start_ingestion_run(
        corpus.corpus_id, authoritative_snapshot=True
    )
    second = create_index_state(
        repository, corpus, second_run.run_id, generation=2, status="BUILDING"
    )

    repository.update_index_build_metadata(
        second.index_state_id,
        bm25_state={"vocabulary": {"finance": 0}},
        bm25_corpus_hash=hash_value(900),
        bm25_state_hash=hash_value(901),
        index_configuration={"format_version": "test-v1"},
        index_configuration_fingerprint=hash_value(902),
        embedding_dimension=1024,
        document_count=1,
        chunk_count=2,
    )
    repository.transition_index_state(second.index_state_id, "BUILDING", "READY")
    activated = repository.transition_index_state(
        second.index_state_id,
        "READY",
        "ACTIVE",
        occurred_at=datetime.now(UTC),
    )

    assert repository.get_latest_index_state(corpus.corpus_id) is second
    assert repository.get_active_index_state(corpus.corpus_id) is second
    assert activated.status == "ACTIVE"
    assert first.status == "READY"

    bm25_projection = repository.get_active_bm25_generation_state(corpus.corpus_id)
    assert bm25_projection is not None
    assert bm25_projection.index_state_id == second.index_state_id
    assert bm25_projection.bm25_state == second.bm25_state
    assert bm25_projection.bm25_state is not second.bm25_state
    assert bm25_projection.bm25_corpus_hash == second.bm25_corpus_hash
    assert bm25_projection.index_configuration == second.index_configuration
    assert bm25_projection.chunk_count == second.chunk_count


def test_building_index_state_can_be_incomplete_and_one_run_can_have_multiple(
    repository: LineageRepository,
) -> None:
    corpus, run = create_corpus_and_run(repository)

    first = repository.create_building_index_state(
        corpus.corpus_id,
        generation=1,
        ingestion_run_id=run.run_id,
        qdrant_collection="consumer-finance-managed-g000001",
        snapshot_hash=hash_value(700),
        embedding_fingerprint=hash_value(800),
        chunking_fingerprint=hash_value(400),
    )
    second = repository.create_building_index_state(
        corpus.corpus_id,
        generation=2,
        ingestion_run_id=run.run_id,
        qdrant_collection="consumer-finance-managed-g000002",
        snapshot_hash=hash_value(700),
        embedding_fingerprint=hash_value(800),
        chunking_fingerprint=hash_value(400),
    )

    assert first.embedding_dimension is None
    assert first.bm25_state is None
    assert first.index_configuration is None
    assert repository.get_index_state(first.index_state_id) is first
    assert repository.list_index_states(corpus.corpus_id) == [first, second]
    assert repository.get_latest_index_state(corpus.corpus_id) is second
    assert repository.get_latest_generation_number(corpus.corpus_id) == 2
    assert run.index_states == [first, second]


def test_build_metadata_and_lifecycle_transitions(repository: LineageRepository) -> None:
    corpus, run = create_corpus_and_run(repository)
    state = repository.create_building_index_state(
        corpus.corpus_id,
        generation=1,
        ingestion_run_id=run.run_id,
        qdrant_collection="consumer-finance-managed-g000001",
        snapshot_hash=hash_value(700),
        embedding_fingerprint=hash_value(800),
        chunking_fingerprint=hash_value(400),
    )

    with pytest.raises(InvalidLineageStateError, match="not ready"):
        repository.transition_index_state(state.index_state_id, "BUILDING", "READY")

    updated = repository.update_index_build_metadata(
        state.index_state_id,
        bm25_state={"vocabulary": {"credit": 0}, "document_count": 1},
        bm25_state_hash=hash_value(901),
        bm25_corpus_hash=hash_value(900),
        index_configuration={"format_version": "managed-hybrid-index-v1"},
        index_configuration_fingerprint=hash_value(902),
        embedding_dimension=1024,
        document_count=6,
        chunk_count=12,
    )
    ready = repository.transition_index_state(
        state.index_state_id, "BUILDING", "READY"
    )
    assert ready.status == "READY"
    activated_at = datetime.now(UTC)
    active = repository.transition_index_state(
        state.index_state_id,
        "READY",
        "ACTIVE",
        occurred_at=activated_at,
    )
    assert active.status == "ACTIVE"
    retired_at = datetime.now(UTC)
    retired = repository.transition_index_state(
        state.index_state_id,
        "ACTIVE",
        "RETIRED",
        occurred_at=retired_at,
    )

    assert updated.embedding_dimension == 1024
    assert updated.document_count == 6
    assert updated.chunk_count == 12
    assert active.activated_at == activated_at
    assert retired.status == "RETIRED"
    assert retired.retired_at == retired_at


def test_failed_index_state_cannot_be_activated(repository: LineageRepository) -> None:
    corpus, run = create_corpus_and_run(repository)
    state = repository.create_building_index_state(
        corpus.corpus_id,
        generation=1,
        ingestion_run_id=run.run_id,
        qdrant_collection="consumer-finance-managed-g000001",
        snapshot_hash=hash_value(700),
        embedding_fingerprint=hash_value(800),
        chunking_fingerprint=hash_value(400),
    )
    failed_at = datetime.now(UTC)

    failed = repository.mark_index_state_failed(
        state.index_state_id,
        "BUILDING",
        "validation failed",
        failed_at,
    )

    assert failed.status == "FAILED"
    assert failed.error_message == "validation failed"
    assert failed.failed_at == failed_at
    with pytest.raises(InvalidLineageStateError, match="not allowed"):
        repository.transition_index_state(
            state.index_state_id,
            "FAILED",
            "ACTIVE",
            occurred_at=datetime.now(UTC),
        )


def test_multiple_active_index_states_are_rejected(repository: LineageRepository) -> None:
    corpus, first_run = create_corpus_and_run(repository)
    first = create_index_state(
        repository, corpus, first_run.run_id, generation=1, status="ACTIVE"
    )
    second_run = repository.start_ingestion_run(
        corpus.corpus_id, authoritative_snapshot=True
    )
    with pytest.raises(InvalidLineageStateError, match="already has ACTIVE"):
        create_index_state(
            repository, corpus, second_run.run_id, generation=2, status="ACTIVE"
        )
    assert first.status == "ACTIVE"


def test_index_membership_and_exact_hash_reuse_lookup(
    repository: LineageRepository,
) -> None:
    corpus, run, _document, version = create_document_and_version(repository)
    exact_hash = hash_value(550)
    exact, different = repository.create_chunks(
        version.document_version_id,
        [chunk_input(0, chunk_hash=exact_hash), chunk_input(1, chunk_hash=hash_value(551))],
    )
    lookalike_document = repository.create_document(
        corpus.corpus_id,
        "cfpb:lookalike",
        "source-lookalike",
        None,
        run.run_id,
    )
    lookalike_version = repository.create_document_version(
        lookalike_document.document_pk,
        1,
        hash_value(10),
        hash_value(110),
        hash_value(210),
        "document-lookalike",
        "Lookalike",
        "credit-cards",
        None,
        "1",
        "Different content with the same chunk position.",
        run.run_id,
    )
    repository.create_chunks(
        lookalike_version.document_version_id,
        [chunk_input(0, chunk_hash=hash_value(552))],
    )
    index_state = create_index_state(
        repository, corpus, run.run_id, generation=1, status="READY"
    )
    membership = repository.record_index_state_chunk(
        index_state.index_state_id,
        exact.chunk_pk,
        uuid4(),
        "COMPUTED",
    )
    repository.record_index_state_chunk(
        index_state.index_state_id,
        different.chunk_pk,
        uuid4(),
        "COMPUTED",
    )

    listed = repository.list_index_state_chunks(index_state.index_state_id)
    matches = repository.find_reusable_chunk(corpus.corpus_id, exact_hash)

    assert listed[0] is membership
    assert [match.chunk_pk for match in matches] == [exact.chunk_pk]
    assert matches[0].index_memberships == [membership]
    assert matches[0].index_memberships[0].index_state.embedding_fingerprint == hash_value(800)


@pytest.mark.parametrize(
    ("dense_action", "with_source", "message"),
    [
        ("COMPUTED", True, "cannot reference"),
        ("REUSED", False, "requires a source"),
    ],
)
def test_dense_provenance_pairing_is_validated(
    repository: LineageRepository,
    dense_action: str,
    with_source: bool,
    message: str,
) -> None:
    corpus, run, _document, version = create_document_and_version(repository)
    chunk = repository.create_chunks(version.document_version_id, [chunk_input(0)])[0]
    state = create_index_state(
        repository, corpus, run.run_id, generation=1, status="BUILDING"
    )

    with pytest.raises(InvalidLineageStateError, match=message):
        repository.record_index_state_chunk(
            state.index_state_id,
            chunk.chunk_pk,
            uuid4(),
            dense_action,
            dense_source_chunk_pk=chunk.chunk_pk if with_source else None,
        )


def test_repository_methods_do_not_commit(session: Session) -> None:
    commits: list[bool] = []

    @event.listens_for(session, "after_commit")
    def observe_commit(_session: Session) -> None:
        commits.append(True)

    repository = LineageRepository(session)
    corpus, run = create_corpus_and_run(repository)
    document = repository.create_document(
        corpus.corpus_id, "cfpb:cards", "source", None, run.run_id
    )

    assert document.document_pk is not None
    assert session.in_transaction()
    assert commits == []


def test_outer_transaction_rollback_removes_repository_writes(engine: Engine) -> None:
    class RollbackRequested(Exception):
        pass

    with Session(engine) as transaction_session:
        repository = LineageRepository(transaction_session)
        with pytest.raises(RollbackRequested), transaction_session.begin():
            corpus, run = create_corpus_and_run(repository)
            document = repository.create_document(
                corpus.corpus_id, "cfpb:cards", "source", None, run.run_id
            )
            version = repository.create_document_version(
                document.document_pk,
                1,
                hash_value(1),
                hash_value(101),
                hash_value(201),
                "document-1",
                "Cards",
                "credit-cards",
                None,
                "1",
                "Normalized.",
                run.run_id,
            )
            repository.create_chunks(
                version.document_version_id,
                [chunk_input(0)],
            )
            repository.record_run_document(
                run.run_id,
                document.document_pk,
                "NEW",
                observed_version_id=version.document_version_id,
                content_changed=True,
                metadata_changed=True,
            )
            raise RollbackRequested

    with Session(engine) as verification_session:
        assert verification_session.scalar(select(Corpus)) is None
