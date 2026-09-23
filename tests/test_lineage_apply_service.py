from __future__ import annotations

from collections.abc import Generator, Sequence
from typing import Any

import pytest
from sqlalchemy import Engine, create_engine, event, func, select
from sqlalchemy.orm import Session

from consumer_finance_rag.acquisition import SourceRecord
from consumer_finance_rag.chunking import ParagraphChunker
from consumer_finance_rag.corpus import CorpusSnapshot
from consumer_finance_rag.lineage.diff import ChangePlan, ChangeType
from consumer_finance_rag.lineage.fingerprints import chunking_fingerprint
from consumer_finance_rag.lineage.models import (
    Base,
    Corpus,
    DocumentVersion,
    IndexState,
    IndexStateChunk,
    IngestionRun,
    IngestionRunDocument,
    LineageChunk,
    LineageDocument,
)
from consumer_finance_rag.lineage.repository import LineageRepository
from consumer_finance_rag.lineage.service import (
    LineageApplyResult,
    LineageApplyService,
    LineageDryRunResult,
    LineagePlanningService,
    UnsupportedResurrectionError,
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


def snapshot_for(
    records: Sequence[tuple[str, str, str]] = (
        ("cards", "Credit card guidance", "Public credit card guidance."),
    ),
) -> CorpusSnapshot:
    documents = [
        normalize_record(
            SourceRecord(
                source_id=source_id,
                title=title,
                source_url=f"https://example.test/{source_id}",
                category="credit-cards",
                source_version="2026-09-18",
                text=text,
            )
        )
        for source_id, title, text in records
    ]
    chunker = ParagraphChunker(chunk_size=100, overlap=10)
    chunks = [chunk for document in documents for chunk in chunker.chunk(document)]
    return CorpusSnapshot(
        documents=documents,
        chunks=chunks,
        fixture_document_count=len(documents),
        expanded_document_count=0,
        category_distribution={"credit-cards": len(documents)} if documents else {},
    )


def plan_snapshot(
    engine: Engine,
    snapshot: CorpusSnapshot,
    *,
    authoritative: bool = False,
) -> LineageDryRunResult:
    with Session(engine) as session, session.begin():
        return LineagePlanningService(LineageRepository(session)).plan_snapshot(
            snapshot,
            corpus_key="consumer-finance-managed",
            authoritative_snapshot=authoritative,
        )


def apply_plan(
    engine: Engine,
    result: LineageDryRunResult,
) -> LineageApplyResult:
    with Session(engine) as session:
        return LineageApplyService(session, chunk_size=100, overlap=10).apply_plan(
            result.plan,
            corpus_key="consumer-finance-managed",
            qdrant_collection_prefix="consumer_finance_managed",
            snapshot_hash=result.snapshot_hash,
        )


def apply_snapshot(
    engine: Engine,
    snapshot: CorpusSnapshot,
    *,
    authoritative: bool = False,
) -> tuple[LineageDryRunResult, LineageApplyResult]:
    plan = plan_snapshot(engine, snapshot, authoritative=authoritative)
    return plan, apply_plan(engine, plan)


def count_rows(session: Session, model: type[Any]) -> int:
    return session.scalar(select(func.count()).select_from(model)) or 0


def test_first_apply_persists_new_lineage_without_index_state(engine: Engine) -> None:
    snapshot = snapshot_for(
        (
            ("cards", "Credit card guidance", "Public credit card guidance."),
            ("reports", "Credit report guidance", "Public credit report guidance."),
        )
    )

    plan, applied = apply_snapshot(engine, snapshot)

    assert plan.plan.new_count == 2
    assert applied.status == "SUCCEEDED"
    with Session(engine) as session:
        corpus = session.scalar(select(Corpus))
        run = session.get(IngestionRun, applied.run_id)
        documents = list(
            session.scalars(select(LineageDocument).order_by(LineageDocument.source_key))
        )
        versions = list(
            session.scalars(select(DocumentVersion).order_by(DocumentVersion.title))
        )
        chunks = list(session.scalars(select(LineageChunk).order_by(LineageChunk.text)))
        classifications = list(session.scalars(select(IngestionRunDocument)))

        assert corpus is not None
        assert corpus.corpus_key == "consumer-finance-managed"
        assert corpus.qdrant_collection_prefix == "consumer_finance_managed"
        assert run is not None
        assert run.status == "SUCCEEDED"
        assert run.snapshot_hash == plan.snapshot_hash
        assert run.documents_new == 2
        assert len(documents) == 2
        assert all(document.first_seen_run_id == applied.run_id for document in documents)
        assert all(document.last_seen_run_id == applied.run_id for document in documents)
        assert [version.version_number for version in versions] == [1, 1]
        assert {version.compatibility_document_id for version in versions} == {
            document.document_id for document in snapshot.documents
        }
        assert len(chunks) == len(snapshot.chunks)
        assert {chunk.compatibility_chunk_id for chunk in chunks} == {
            chunk.chunk_id for chunk in snapshot.chunks
        }
        assert {chunk.chunk_hash for chunk in chunks} == {
            chunk.chunk_hash for chunk in snapshot.chunks
        }
        assert {chunk.chunking_fingerprint for chunk in chunks} == {
            chunking_fingerprint(chunk_size=100, overlap=10)
        }
        assert {row.change_type for row in classifications} == {ChangeType.NEW.value}
        assert all(row.previous_version_id is None for row in classifications)
        assert all(row.observed_version_id is not None for row in classifications)
        assert count_rows(session, IndexState) == 0
        assert count_rows(session, IndexStateChunk) == 0


def test_identical_second_apply_is_noop_without_new_versions_or_chunks(
    engine: Engine,
) -> None:
    snapshot = snapshot_for()
    _first_plan, first = apply_snapshot(engine, snapshot)
    _second_plan, second = apply_snapshot(engine, snapshot)

    assert second.status == "NOOP"
    assert second.unchanged_count == 1
    with Session(engine) as session:
        document = session.scalar(select(LineageDocument))
        run = session.get(IngestionRun, second.run_id)
        row = session.scalar(
            select(IngestionRunDocument).where(
                IngestionRunDocument.run_id == second.run_id
            )
        )

        assert document is not None
        assert run is not None
        assert row is not None
        assert count_rows(session, DocumentVersion) == 1
        assert count_rows(session, LineageChunk) == len(snapshot.chunks)
        assert document.first_seen_run_id == first.run_id
        assert document.last_seen_run_id == second.run_id
        assert run.documents_unchanged == 1
        assert row.change_type == ChangeType.UNCHANGED.value
        assert row.previous_version_id == row.observed_version_id


def test_changed_content_creates_version_two_and_preserves_history(engine: Engine) -> None:
    original = snapshot_for()
    changed = snapshot_for(
        (("cards", "Credit card guidance", "Updated public credit card guidance."),)
    )
    _original_plan, _original_apply = apply_snapshot(engine, original)
    _changed_plan, changed_apply = apply_snapshot(engine, changed)

    with Session(engine) as session:
        versions = list(
            session.scalars(
                select(DocumentVersion).order_by(DocumentVersion.version_number)
            )
        )
        classification = session.scalar(
            select(IngestionRunDocument).where(
                IngestionRunDocument.run_id == changed_apply.run_id
            )
        )

        assert changed_apply.status == "SUCCEEDED"
        assert [version.version_number for version in versions] == [1, 2]
        assert versions[0].content_hash == original.documents[0].content_hash
        assert versions[0].normalized_text == original.documents[0].text
        assert versions[1].content_hash == changed.documents[0].content_hash
        assert versions[1].normalized_text == changed.documents[0].text
        assert len(versions[0].chunks) == len(original.chunks)
        assert len(versions[1].chunks) == len(changed.chunks)
        assert classification is not None
        assert classification.change_type == ChangeType.CHANGED.value
        assert classification.previous_version_id == versions[0].document_version_id
        assert classification.observed_version_id == versions[1].document_version_id
        assert classification.content_changed
        assert not classification.metadata_changed


def test_metadata_only_change_creates_new_version_with_correct_flags(
    engine: Engine,
) -> None:
    original = snapshot_for()
    renamed = snapshot_for(
        (("cards", "Renamed credit card guidance", "Public credit card guidance."),)
    )
    _original_plan, _original_apply = apply_snapshot(engine, original)
    _renamed_plan, renamed_apply = apply_snapshot(engine, renamed)

    with Session(engine) as session:
        versions = list(
            session.scalars(
                select(DocumentVersion).order_by(DocumentVersion.version_number)
            )
        )
        classification = session.scalar(
            select(IngestionRunDocument).where(
                IngestionRunDocument.run_id == renamed_apply.run_id
            )
        )

        assert [version.version_number for version in versions] == [1, 2]
        assert versions[0].title == "Credit card guidance"
        assert versions[1].title == "Renamed credit card guidance"
        assert versions[0].content_hash == versions[1].content_hash
        assert classification is not None
        assert not classification.content_changed
        assert classification.metadata_changed


def test_authoritative_missing_document_is_soft_deleted_with_history_intact(
    engine: Engine,
) -> None:
    complete = snapshot_for(
        (
            ("cards", "Credit card guidance", "Public credit card guidance."),
            ("reports", "Credit report guidance", "Public credit report guidance."),
        )
    )
    remaining = snapshot_for(
        (("cards", "Credit card guidance", "Public credit card guidance."),)
    )
    _first_plan, first = apply_snapshot(engine, complete, authoritative=True)
    _delete_plan, deletion = apply_snapshot(engine, remaining, authoritative=True)

    with Session(engine) as session:
        deleted = session.scalar(
            select(LineageDocument).where(LineageDocument.source_id == "reports")
        )
        classification = session.scalar(
            select(IngestionRunDocument).where(
                IngestionRunDocument.run_id == deletion.run_id,
                IngestionRunDocument.change_type == ChangeType.DELETED.value,
            )
        )

        assert deletion.status == "SUCCEEDED"
        assert deletion.deleted_count == 1
        assert deleted is not None
        assert deleted.state == "DELETED"
        assert deleted.deleted_run_id == deletion.run_id
        assert deleted.last_seen_run_id == first.run_id
        assert classification is not None
        assert classification.previous_version_id is not None
        assert classification.observed_version_id is None
        assert count_rows(session, DocumentVersion) == 2
        assert count_rows(session, LineageChunk) == len(complete.chunks)


def test_non_authoritative_missing_document_is_not_deleted(engine: Engine) -> None:
    complete = snapshot_for(
        (
            ("cards", "Credit card guidance", "Public credit card guidance."),
            ("reports", "Credit report guidance", "Public credit report guidance."),
        )
    )
    remaining = snapshot_for(
        (("cards", "Credit card guidance", "Public credit card guidance."),)
    )
    _first_plan, first = apply_snapshot(engine, complete, authoritative=True)
    _second_plan, second = apply_snapshot(engine, remaining, authoritative=False)

    with Session(engine) as session:
        omitted = session.scalar(
            select(LineageDocument).where(LineageDocument.source_id == "reports")
        )

        assert second.status == "NOOP"
        assert second.deleted_count == 0
        assert omitted is not None
        assert omitted.state == "ACTIVE"
        assert omitted.last_seen_run_id == first.run_id
        assert omitted.deleted_run_id is None


def test_deleted_source_resurrection_fails_and_records_failed_run(engine: Engine) -> None:
    snapshot = snapshot_for()
    _first_plan, _first = apply_snapshot(engine, snapshot, authoritative=True)
    _delete_plan, _delete = apply_snapshot(
        engine,
        snapshot_for(()),
        authoritative=True,
    )
    resurrection_plan = plan_snapshot(engine, snapshot, authoritative=True)

    with pytest.raises(UnsupportedResurrectionError, match="not supported"):
        apply_plan(engine, resurrection_plan)

    with Session(engine) as session:
        document = session.scalar(select(LineageDocument))
        runs = list(session.scalars(select(IngestionRun).order_by(IngestionRun.started_at)))

        assert document is not None
        assert document.state == "DELETED"
        assert count_rows(session, LineageDocument) == 1
        assert count_rows(session, DocumentVersion) == 1
        assert count_rows(session, LineageChunk) == len(snapshot.chunks)
        assert runs[-1].status == "FAILED"
        assert "UnsupportedResurrectionError" in (runs[-1].error_message or "")


def test_apply_failure_rolls_back_lineage_mutations_and_marks_run_failed(
    engine: Engine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = plan_snapshot(engine, snapshot_for())

    def fail_chunks(*_args: Any, **_kwargs: Any) -> list[LineageChunk]:
        raise RuntimeError("forced chunk persistence failure")

    monkeypatch.setattr(LineageRepository, "create_chunks", fail_chunks)

    with pytest.raises(RuntimeError, match="forced chunk persistence failure"):
        apply_plan(engine, result)

    with Session(engine) as session:
        run = session.scalar(select(IngestionRun))

        assert count_rows(session, Corpus) == 1
        assert count_rows(session, IngestionRun) == 1
        assert count_rows(session, LineageDocument) == 0
        assert count_rows(session, DocumentVersion) == 0
        assert count_rows(session, LineageChunk) == 0
        assert count_rows(session, IngestionRunDocument) == 0
        assert run is not None
        assert run.status == "FAILED"
        assert run.documents_new == 1
        assert run.documents_changed == 0
        assert run.documents_unchanged == 0
        assert run.documents_deleted == 0
        assert run.completed_at is not None
        assert "forced chunk persistence failure" in (run.error_message or "")


def test_apply_consumes_existing_plan_without_reclassifying(
    engine: Engine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = plan_snapshot(engine, snapshot_for())

    def unexpected_planning(*_args: Any, **_kwargs: Any) -> ChangePlan:
        raise AssertionError("apply must consume the supplied ChangePlan")

    monkeypatch.setattr(
        "consumer_finance_rag.lineage.diff.plan_document_changes",
        unexpected_planning,
    )

    applied = apply_plan(engine, result)

    assert applied.new_count == 1
