from __future__ import annotations

from collections.abc import Generator
from typing import Any

import pytest
from sqlalchemy import Engine, create_engine, event, func, select
from sqlalchemy.orm import Session

from consumer_finance_rag.acquisition import SourceRecord
from consumer_finance_rag.chunking import ParagraphChunker
from consumer_finance_rag.corpus import CorpusSnapshot
from consumer_finance_rag.lineage.diff import ChangeType
from consumer_finance_rag.lineage.fingerprints import metadata_hash, version_hash
from consumer_finance_rag.lineage.models import (
    Base,
    Corpus,
    DocumentVersion,
    IngestionRun,
    LineageDocument,
)
from consumer_finance_rag.lineage.repository import (
    InvalidLineageStateError,
    LineageRepository,
)
from consumer_finance_rag.lineage.service import (
    LineagePlanningService,
    current_states_from_snapshot,
    persisted_document_state,
)
from consumer_finance_rag.normalization import normalize_record


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


def snapshot_for(
    source_id: str = "cards",
    *,
    title: str = "Credit card guidance",
    text: str = "Public credit card guidance.",
) -> CorpusSnapshot:
    document = normalize_record(
        SourceRecord(
            source_id=source_id,
            title=title,
            source_url=f"https://example.test/{source_id}",
            category="credit-cards",
            source_version="2026-09-18",
            text=text,
        )
    )
    chunks = ParagraphChunker(chunk_size=100, overlap=10).chunk(document)
    return CorpusSnapshot(
        documents=[document],
        chunks=chunks,
        fixture_document_count=1,
        expanded_document_count=0,
        category_distribution={"credit-cards": 1},
    )


def empty_snapshot() -> CorpusSnapshot:
    return CorpusSnapshot(
        documents=[],
        chunks=[],
        fixture_document_count=0,
        expanded_document_count=0,
        category_distribution={},
    )


def persist_snapshot_state(
    repository: LineageRepository,
    snapshot: CorpusSnapshot,
    *,
    stored_content_hash: str | None = None,
    stored_metadata_hash: str | None = None,
) -> tuple[LineageDocument, DocumentVersion, IngestionRun]:
    current = current_states_from_snapshot(snapshot)[0]
    corpus = repository.get_or_create_corpus(
        "consumer-finance-managed", "consumer_finance_managed"
    )
    run = repository.start_ingestion_run(corpus.corpus_id, authoritative_snapshot=True)
    document = repository.create_document(
        corpus.corpus_id,
        current.source_key,
        current.document.source_id,
        current.document.source_url,
        run.run_id,
    )
    content = stored_content_hash or current.document.content_hash
    metadata = stored_metadata_hash or current.metadata_hash
    version = repository.create_document_version(
        document.document_pk,
        1,
        content,
        metadata,
        version_hash(content_hash=content, metadata_hash=metadata),
        current.document.document_id,
        current.document.title,
        current.document.category,
        current.document.source_url,
        current.document.source_version,
        current.document.text,
        run.run_id,
    )
    return document, version, run


def test_current_snapshot_conversion_preserves_ids_and_populates_fingerprints() -> None:
    snapshot = snapshot_for()

    state = current_states_from_snapshot(snapshot)[0]
    document = snapshot.documents[0]
    expected_metadata = metadata_hash(
        title=document.title,
        source_url=document.source_url,
        category=document.category,
        source_version=document.source_version,
    )

    assert state.document is document
    assert state.document.document_id == document.document_id
    assert [chunk.chunk_id for chunk in state.chunks] == [
        chunk.chunk_id for chunk in snapshot.chunks
    ]
    assert state.metadata_hash == expected_metadata
    assert state.version_hash == version_hash(
        content_hash=document.content_hash,
        metadata_hash=expected_metadata,
    )


def test_persisted_orm_rows_convert_to_immutable_planner_state(
    repository: LineageRepository,
) -> None:
    document, version, _run = persist_snapshot_state(repository, snapshot_for())

    state = persisted_document_state(document, version)

    assert state.document_pk == document.document_pk
    assert state.latest_document_version_id == version.document_version_id
    assert state.compatibility_document_id == version.compatibility_document_id
    assert state.content_hash == version.content_hash
    assert state.metadata_hash == version.metadata_hash


def test_uninitialized_corpus_plans_all_current_documents_as_new_without_writes(
    session: Session, repository: LineageRepository
) -> None:
    service = LineagePlanningService(repository)

    result = service.plan_snapshot(
        snapshot_for(),
        corpus_key="consumer-finance-managed",
        authoritative_snapshot=False,
    )

    assert not result.corpus_initialized
    assert result.previous_active_document_count == 0
    assert result.plan.new_count == 1
    assert repository.get_corpus_by_key("consumer-finance-managed") is None
    assert session.scalar(select(func.count()).select_from(Corpus)) == 0
    assert session.scalar(select(func.count()).select_from(IngestionRun)) == 0


def test_existing_identical_state_is_unchanged(repository: LineageRepository) -> None:
    snapshot = snapshot_for()
    persist_snapshot_state(repository, snapshot)

    result = LineagePlanningService(repository).plan_snapshot(
        snapshot,
        corpus_key="consumer-finance-managed",
        authoritative_snapshot=True,
    )

    assert result.corpus_initialized
    assert result.plan.items[0].change_type is ChangeType.UNCHANGED


def test_changed_stored_content_is_changed(repository: LineageRepository) -> None:
    snapshot = snapshot_for()
    persist_snapshot_state(repository, snapshot, stored_content_hash="f" * 64)

    item = LineagePlanningService(repository).plan_snapshot(
        snapshot,
        corpus_key="consumer-finance-managed",
        authoritative_snapshot=True,
    ).plan.items[0]

    assert item.change_type is ChangeType.CHANGED
    assert item.content_changed
    assert not item.metadata_changed


def test_changed_stored_metadata_is_changed(repository: LineageRepository) -> None:
    snapshot = snapshot_for()
    persist_snapshot_state(repository, snapshot, stored_metadata_hash="e" * 64)

    item = LineagePlanningService(repository).plan_snapshot(
        snapshot,
        corpus_key="consumer-finance-managed",
        authoritative_snapshot=True,
    ).plan.items[0]

    assert item.change_type is ChangeType.CHANGED
    assert not item.content_changed
    assert item.metadata_changed


@pytest.mark.parametrize(
    ("authoritative", "expected_deleted"),
    [(True, 1), (False, 0)],
)
def test_missing_active_document_respects_authoritative_deletion_semantics(
    repository: LineageRepository,
    authoritative: bool,
    expected_deleted: int,
) -> None:
    persist_snapshot_state(repository, snapshot_for())

    result = LineagePlanningService(repository).plan_snapshot(
        empty_snapshot(),
        corpus_key="consumer-finance-managed",
        authoritative_snapshot=authoritative,
    )

    assert result.plan.deleted_count == expected_deleted
    assert result.plan.deletion_detection_performed is authoritative


def test_active_document_without_version_is_invalid(repository: LineageRepository) -> None:
    corpus = repository.create_corpus(
        "consumer-finance-managed", "consumer_finance_managed"
    )
    run = repository.start_ingestion_run(corpus.corpus_id, authoritative_snapshot=True)
    repository.create_document(
        corpus.corpus_id,
        "https://example.test/cards",
        "cards",
        "https://example.test/cards",
        run.run_id,
    )

    with pytest.raises(InvalidLineageStateError, match="has no document version"):
        LineagePlanningService(repository).plan_snapshot(
            snapshot_for(),
            corpus_key="consumer-finance-managed",
            authoritative_snapshot=False,
        )


def test_planning_existing_state_does_not_flush_or_create_runs(
    session: Session, repository: LineageRepository
) -> None:
    snapshot = snapshot_for()
    persist_snapshot_state(repository, snapshot)
    initial_run_count = session.scalar(select(func.count()).select_from(IngestionRun))
    flushes: list[bool] = []

    @event.listens_for(session, "before_flush")
    def observe_flush(_session: Session, _context: Any, _instances: Any) -> None:
        flushes.append(True)

    LineagePlanningService(repository).plan_snapshot(
        snapshot,
        corpus_key="consumer-finance-managed",
        authoritative_snapshot=True,
    )

    assert flushes == []
    assert session.scalar(select(func.count()).select_from(IngestionRun)) == initial_run_count
