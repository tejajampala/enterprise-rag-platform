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
    return repository.create_index_state(
        corpus.corpus_id,
        generation=generation,
        ingestion_run_id=run_id,
        status=status,
        qdrant_collection=f"{corpus.qdrant_collection_prefix}-g{generation:04d}",
        snapshot_hash=hash_value(700 + generation),
        embedding_fingerprint=hash_value(800),
        embedding_dimension=1024,
        chunking_fingerprint=hash_value(400),
        document_count=1,
        chunk_count=2,
        bm25_corpus_hash=hash_value(900),
        bm25_state_hash=hash_value(901),
    )


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

    activated = repository.set_index_state_status(second.index_state_id, "ACTIVE")

    assert repository.get_latest_index_state(corpus.corpus_id) is second
    assert repository.get_active_index_state(corpus.corpus_id) is second
    assert activated.status == "ACTIVE"
    assert first.status == "READY"


def test_multiple_active_index_states_are_rejected(repository: LineageRepository) -> None:
    corpus, first_run = create_corpus_and_run(repository)
    first = create_index_state(
        repository, corpus, first_run.run_id, generation=1, status="ACTIVE"
    )
    second_run = repository.start_ingestion_run(
        corpus.corpus_id, authoritative_snapshot=True
    )
    create_index_state(repository, corpus, second_run.run_id, generation=2, status="ACTIVE")

    with pytest.raises(InvalidLineageStateError, match="more than one ACTIVE"):
        repository.get_active_index_state(corpus.corpus_id)
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
