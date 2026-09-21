from __future__ import annotations

import ast
from collections.abc import Generator, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest
from sqlalchemy import Engine, create_engine, event, func, select
from sqlalchemy.orm import Session

from consumer_finance_rag.acquisition import SourceRecord
from consumer_finance_rag.chunking import ParagraphChunker
from consumer_finance_rag.corpus import CorpusSnapshot
from consumer_finance_rag.lineage.chunk_materialization import (
    ActiveSnapshotResolutionError,
    ChunkDerivationMismatchError,
    ChunkMaterializationService,
    DocumentContentMismatchError,
    PIIClassificationMismatchError,
    active_chunk_manifest_hash,
)
from consumer_finance_rag.lineage.fingerprints import chunking_fingerprint
from consumer_finance_rag.lineage.index_types import ActiveChunkManifest
from consumer_finance_rag.lineage.models import (
    Base,
    DocumentVersion,
    DocumentVersionPII,
    IndexState,
    IndexStateChunk,
    IngestionRun,
    LineageChunk,
)
from consumer_finance_rag.lineage.repository import (
    InvalidLineageStateError,
    LineageRepository,
    MissingActiveDocumentVersionError,
)
from consumer_finance_rag.lineage.service import (
    LineageApplyService,
    LineagePlanningService,
)
from consumer_finance_rag.lineage.types import ChunkPersistenceInput
from consumer_finance_rag.normalization import normalize_record
from consumer_finance_rag.pii import DeterministicPIIDetector, PIIType

CORPUS_KEY = "consumer-finance-managed"
COLLECTION_PREFIX = "consumer_finance_managed"
INITIAL_CHUNK_SIZE = 100
INITIAL_OVERLAP = 10


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


def make_snapshot(
    records: Sequence[tuple[str, str, str]] = (
        (
            "cards",
            "Credit card guidance",
            "First paragraph about cards.\n\nSecond paragraph about billing disputes.",
        ),
    ),
) -> CorpusSnapshot:
    documents = [
        normalize_record(
            SourceRecord(
                source_id=source_id,
                title=title,
                source_url=f"https://example.test/{source_id}",
                category="credit-cards",
                source_version="2026-09-19",
                text=text,
            )
        )
        for source_id, title, text in records
    ]
    chunker = ParagraphChunker(INITIAL_CHUNK_SIZE, INITIAL_OVERLAP)
    chunks = [chunk for document in documents for chunk in chunker.chunk(document)]
    return CorpusSnapshot(
        documents=documents,
        chunks=chunks,
        fixture_document_count=len(documents),
        expanded_document_count=0,
        category_distribution={"credit-cards": len(documents)} if documents else {},
    )


def apply_snapshot(
    engine: Engine,
    snapshot: CorpusSnapshot,
    *,
    authoritative: bool = True,
) -> tuple[str, UUID]:
    with Session(engine) as session, session.begin():
        planned = LineagePlanningService(LineageRepository(session)).plan_snapshot(
            snapshot,
            corpus_key=CORPUS_KEY,
            authoritative_snapshot=authoritative,
        )
    with Session(engine) as session:
        result = LineageApplyService(
            session,
            chunk_size=INITIAL_CHUNK_SIZE,
            overlap=INITIAL_OVERLAP,
        ).apply_plan(
            planned.plan,
            corpus_key=CORPUS_KEY,
            qdrant_collection_prefix=COLLECTION_PREFIX,
            snapshot_hash=planned.snapshot_hash,
        )
    return planned.snapshot_hash, result.run_id


def materialize(
    engine: Engine,
    *,
    chunk_size: int = INITIAL_CHUNK_SIZE,
    overlap: int = INITIAL_OVERLAP,
    pii_detector: DeterministicPIIDetector | None = None,
) -> ActiveChunkManifest:
    with Session(engine) as session, session.begin():
        return ChunkMaterializationService(
            session,
            chunk_size=chunk_size,
            overlap=overlap,
            pii_detector=pii_detector,
        ).materialize_active_manifest(CORPUS_KEY)


def row_count(session: Session, model: type[Any]) -> int:
    return session.scalar(select(func.count()).select_from(model)) or 0


def test_active_latest_version_resolves_and_preserves_compatibility_id(
    engine: Engine,
) -> None:
    original = make_snapshot()
    changed = make_snapshot(
        (("cards", "Updated card guidance", "New normalized card guidance."),)
    )
    apply_snapshot(engine, original)
    apply_snapshot(engine, changed)

    manifest = materialize(engine)

    assert manifest.document_count == 1
    assert manifest.documents[0].version_number == 2
    assert manifest.documents[0].compatibility_document_id == changed.documents[0].document_id
    assert {chunk.document_version_id for chunk in manifest.chunks} == {
        manifest.documents[0].document_version_id
    }


def test_deleted_document_is_excluded_but_history_remains(engine: Engine) -> None:
    complete = make_snapshot(
        (
            ("cards", "Cards", "Card guidance."),
            ("reports", "Reports", "Credit report guidance."),
        )
    )
    remaining = make_snapshot((("cards", "Cards", "Card guidance."),))
    apply_snapshot(engine, complete)
    apply_snapshot(engine, remaining)

    manifest = materialize(engine)

    assert [document.source_id for document in manifest.documents] == ["cards"]
    with Session(engine) as session:
        assert row_count(session, DocumentVersion) == 2
        assert row_count(session, LineageChunk) == len(complete.chunks)


def test_later_failed_run_is_not_selected_as_active_snapshot(engine: Engine) -> None:
    expected_snapshot_hash, _run_id = apply_snapshot(engine, make_snapshot())
    with Session(engine) as session, session.begin():
        repository = LineageRepository(session)
        corpus = repository.get_corpus_by_key(CORPUS_KEY)
        assert corpus is not None
        failed = repository.start_ingestion_run(
            corpus.corpus_id,
            authoritative_snapshot=True,
            snapshot_hash="f" * 64,
        )
        repository.fail_ingestion_run(
            failed.run_id,
            "forced failure",
            datetime.now(UTC),
        )

    manifest = materialize(engine)

    assert manifest.source_snapshot_hash == expected_snapshot_hash


def test_non_authoritative_run_hash_does_not_drop_still_active_documents(
    engine: Engine,
) -> None:
    complete = make_snapshot(
        (
            ("cards", "Cards", "Card guidance."),
            ("reports", "Reports", "Credit report guidance."),
        )
    )
    partial = make_snapshot((("cards", "Cards", "Card guidance."),))
    complete_hash, _first_run = apply_snapshot(engine, complete)
    partial_hash, _second_run = apply_snapshot(engine, partial, authoritative=False)

    manifest = materialize(engine)

    assert manifest.document_count == 2
    assert manifest.source_snapshot_hash == complete_hash
    assert manifest.source_snapshot_hash != partial_hash


def test_active_document_without_version_fails_clearly(engine: Engine) -> None:
    with Session(engine) as session, session.begin():
        repository = LineageRepository(session)
        corpus = repository.create_corpus(CORPUS_KEY, COLLECTION_PREFIX)
        run = repository.start_ingestion_run(
            corpus.corpus_id,
            authoritative_snapshot=True,
            snapshot_hash="0" * 64,
        )
        repository.create_document(
            corpus.corpus_id,
            "https://example.test/orphan",
            "orphan",
            "https://example.test/orphan",
            run.run_id,
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

    with pytest.raises(MissingActiveDocumentVersionError, match="no document version"):
        materialize(engine)


def test_authoritative_snapshot_inconsistency_fails(engine: Engine) -> None:
    apply_snapshot(engine, make_snapshot())
    with Session(engine) as session, session.begin():
        run = session.scalar(select(IngestionRun).where(IngestionRun.status == "SUCCEEDED"))
        assert run is not None
        run.snapshot_hash = "f" * 64

    with pytest.raises(ActiveSnapshotResolutionError, match="does not match"):
        materialize(engine)


def test_materialization_requires_caller_owned_transaction(engine: Engine) -> None:
    apply_snapshot(engine, make_snapshot())
    with Session(engine) as session:
        service = ChunkMaterializationService(
            session,
            chunk_size=INITIAL_CHUNK_SIZE,
            overlap=INITIAL_OVERLAP,
        )
        with pytest.raises(InvalidLineageStateError, match="caller-owned transaction"):
            service.materialize_active_manifest(CORPUS_KEY)


def test_reconstructed_content_hash_is_validated(engine: Engine) -> None:
    apply_snapshot(engine, make_snapshot())
    with Session(engine) as session, session.begin():
        version = session.scalar(select(DocumentVersion))
        assert version is not None
        version.normalized_text = "corrupted text"

    with pytest.raises(DocumentContentMismatchError, match="content hash mismatch"):
        materialize(engine)


def test_requested_fingerprint_is_deterministic_and_missing_derivation_is_created(
    engine: Engine,
) -> None:
    apply_snapshot(engine, make_snapshot())
    expected = chunking_fingerprint(chunk_size=45, overlap=5)

    manifest = materialize(engine, chunk_size=45, overlap=5)

    assert manifest.chunking_fingerprint == expected
    assert manifest.chunk_count > 0
    assert {chunk.chunking_fingerprint for chunk in manifest.chunks} == {expected}
    with Session(engine) as session:
        persisted = list(
            session.scalars(
                select(LineageChunk).where(
                    LineageChunk.chunking_fingerprint == expected
                )
            )
        )
        assert len(persisted) == manifest.chunk_count


def test_complete_derivation_is_reused_without_duplicates(engine: Engine) -> None:
    apply_snapshot(engine, make_snapshot())
    first = materialize(engine, chunk_size=45, overlap=5)
    with Session(engine) as session:
        before = row_count(session, LineageChunk)

    second = materialize(engine, chunk_size=45, overlap=5)

    with Session(engine) as session:
        assert row_count(session, LineageChunk) == before
    assert second.manifest_hash == first.manifest_hash
    assert [chunk.chunk_pk for chunk in second.chunks] == [
        chunk.chunk_pk for chunk in first.chunks
    ]


def test_pii_classification_is_derived_persisted_and_versioned_separately(
    engine: Engine,
) -> None:
    snapshot = make_snapshot(
        (("cards", "Cards", "Contact alice@example.com about TFN 123456782."),)
    )
    apply_snapshot(engine, snapshot)

    first = materialize(engine)
    second = materialize(
        engine,
        pii_detector=DeterministicPIIDetector("deterministic-pii-v2-test"),
    )

    assert first.documents[0].pii_summary.pii_types == (
        PIIType.AU_TFN,
        PIIType.EMAIL,
    )
    assert first.chunks[0].pii_summary.contains_pii is True
    assert first.manifest_hash != second.manifest_hash
    assert first.source_snapshot_hash == second.source_snapshot_hash
    assert first.documents[0].content_hash == second.documents[0].content_hash
    assert first.documents[0].version_hash == second.documents[0].version_hash
    with Session(engine) as session:
        assert row_count(session, DocumentVersion) == 1
        classifications = list(
            session.scalars(
                select(DocumentVersionPII).order_by(
                    DocumentVersionPII.detector_version
                )
            )
        )
        assert len(classifications) == 2
        assert classifications[0].pii_types == ["AU_TFN", "EMAIL"]
        assert classifications[0].match_count == 2
        persisted = str(
            {
                "hash": classifications[0].classification_hash,
                "types": classifications[0].pii_types,
                "count": classifications[0].match_count,
            }
        )
        assert "alice@example.com" not in persisted
        assert "123456782" not in persisted


def test_changed_detector_output_requires_a_new_detector_version(engine: Engine) -> None:
    apply_snapshot(engine, make_snapshot())
    materialize(engine)
    with Session(engine) as session, session.begin():
        classification = session.scalar(select(DocumentVersionPII))
        assert classification is not None
        classification.classification_hash = "f" * 64

    with pytest.raises(PIIClassificationMismatchError, match="bump detector version"):
        materialize(engine)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("text", "persisted text drift"),
        ("chunk_hash", "f" * 64),
        ("compatibility_chunk_id", "wrong-chunk-id"),
    ],
)
def test_existing_derivation_mismatch_fails(
    engine: Engine,
    field: str,
    value: str,
) -> None:
    apply_snapshot(engine, make_snapshot())
    fingerprint = chunking_fingerprint(
        chunk_size=INITIAL_CHUNK_SIZE,
        overlap=INITIAL_OVERLAP,
    )
    with Session(engine) as session, session.begin():
        chunk = session.scalar(
            select(LineageChunk).where(
                LineageChunk.chunking_fingerprint == fingerprint
            )
        )
        assert chunk is not None
        setattr(chunk, field, value)

    with pytest.raises(ChunkDerivationMismatchError, match="derivation mismatch"):
        materialize(engine)


def test_partial_derivation_fails_instead_of_appending(engine: Engine) -> None:
    long_text = " ".join(f"sentence-{number}." for number in range(80))
    apply_snapshot(engine, make_snapshot((("cards", "Cards", long_text),)))
    fingerprint = chunking_fingerprint(
        chunk_size=INITIAL_CHUNK_SIZE,
        overlap=INITIAL_OVERLAP,
    )
    with Session(engine) as session, session.begin():
        chunks = list(
            session.scalars(
                select(LineageChunk)
                .where(LineageChunk.chunking_fingerprint == fingerprint)
                .order_by(LineageChunk.ordinal)
            )
        )
        assert len(chunks) > 1
        session.delete(chunks[-1])

    with pytest.raises(ChunkDerivationMismatchError, match=r"expected .* found"):
        materialize(engine)


def test_materialization_failure_rolls_back_new_derivations_for_whole_pass(
    engine: Engine,
) -> None:
    snapshot = make_snapshot(
        (
            ("a", "A", "A paragraph. " * 15),
            ("z", "Z", "Z paragraph. " * 15),
        )
    )
    apply_snapshot(engine, snapshot)
    alternate_fingerprint = chunking_fingerprint(chunk_size=40, overlap=5)
    with Session(engine) as session, session.begin():
        repository = LineageRepository(session)
        corpus = repository.get_corpus_by_key(CORPUS_KEY)
        assert corpus is not None
        active = repository.list_active_latest_document_versions(corpus.corpus_id)
        z_document = next(item for item in active if item.source_id == "z")
        domain_document = next(
            document for document in snapshot.documents if document.source_id == "z"
        )
        expected = ParagraphChunker(40, 5).chunk(domain_document)
        assert len(expected) > 1
        repository.create_chunks(
            z_document.document_version_id,
            [
                ChunkPersistenceInput(
                    chunking_fingerprint=alternate_fingerprint,
                    compatibility_chunk_id=expected[0].chunk_id,
                    ordinal=expected[0].ordinal,
                    chunk_hash=expected[0].chunk_hash,
                    text=expected[0].text,
                    metadata=expected[0].metadata,
                )
            ],
        )

    with pytest.raises(ChunkDerivationMismatchError):
        materialize(engine, chunk_size=40, overlap=5)

    with Session(engine) as session:
        alternate = list(
            session.scalars(
                select(LineageChunk).where(
                    LineageChunk.chunking_fingerprint == alternate_fingerprint
                )
            )
        )
        assert len(alternate) == 1
        assert alternate[0].document_version_id == z_document.document_version_id


def test_alternate_chunking_creates_derivation_not_document_version(
    engine: Engine,
) -> None:
    apply_snapshot(engine, make_snapshot())
    old_fingerprint = chunking_fingerprint(
        chunk_size=INITIAL_CHUNK_SIZE,
        overlap=INITIAL_OVERLAP,
    )
    new_manifest = materialize(engine, chunk_size=45, overlap=5)

    with Session(engine) as session:
        assert row_count(session, DocumentVersion) == 1
        fingerprints = set(session.scalars(select(LineageChunk.chunking_fingerprint)))
        assert fingerprints == {old_fingerprint, new_manifest.chunking_fingerprint}


def test_metadata_only_latest_version_supplies_current_citation_metadata(
    engine: Engine,
) -> None:
    original = make_snapshot()
    renamed = make_snapshot(
        (("cards", "Current citation title", original.documents[0].text),)
    )
    apply_snapshot(engine, original)
    apply_snapshot(engine, renamed)

    manifest = materialize(engine)

    assert {chunk.title for chunk in manifest.chunks} == {"Current citation title"}
    assert {chunk.version_number for chunk in manifest.chunks} == {2}
    with Session(engine) as session:
        assert row_count(session, DocumentVersion) == 2


def test_manifest_order_counts_and_index_independence(engine: Engine) -> None:
    snapshot = make_snapshot(
        (
            ("z", "Z", "Z evidence. " * 20),
            ("a", "A", "A evidence. " * 20),
        )
    )
    apply_snapshot(engine, snapshot)

    manifest = materialize(engine, chunk_size=45, overlap=5)

    assert [document.source_id for document in manifest.documents] == ["a", "z"]
    assert list(manifest.chunks) == sorted(
        manifest.chunks,
        key=lambda chunk: (
            chunk.source_key,
            chunk.ordinal,
            chunk.compatibility_chunk_id,
        ),
    )
    assert manifest.document_count == len(manifest.documents) == 2
    assert manifest.chunk_count == len(manifest.chunks)
    with Session(engine) as session:
        assert row_count(session, IndexState) == 0
        assert row_count(session, IndexStateChunk) == 0


def test_manifest_hash_is_deterministic_and_independent_of_database_uuids(
    engine: Engine,
) -> None:
    apply_snapshot(engine, make_snapshot())
    manifest = materialize(engine)
    remapped_documents = tuple(
        replace(
            document,
            document_pk=uuid4(),
            document_version_id=uuid4(),
        )
        for document in reversed(manifest.documents)
    )
    remapped_chunks = tuple(
        replace(
            chunk,
            document_pk=uuid4(),
            document_version_id=uuid4(),
            chunk_pk=uuid4(),
        )
        for chunk in reversed(manifest.chunks)
    )

    remapped_hash = active_chunk_manifest_hash(
        corpus_key=manifest.corpus_key,
        source_snapshot_hash=manifest.source_snapshot_hash,
        chunking_fingerprint=manifest.chunking_fingerprint,
        documents=remapped_documents,
        chunks=remapped_chunks,
    )

    assert remapped_hash == manifest.manifest_hash


def test_manifest_hash_changes_with_text_membership_and_chunking(engine: Engine) -> None:
    apply_snapshot(engine, make_snapshot())
    manifest = materialize(engine)
    chunk = manifest.chunks[0]
    changed_chunk = replace(chunk, text=f"{chunk.text} changed", chunk_hash="f" * 64)

    changed_text_hash = active_chunk_manifest_hash(
        corpus_key=manifest.corpus_key,
        source_snapshot_hash=manifest.source_snapshot_hash,
        chunking_fingerprint=manifest.chunking_fingerprint,
        documents=manifest.documents,
        chunks=(changed_chunk, *manifest.chunks[1:]),
    )
    changed_membership_hash = active_chunk_manifest_hash(
        corpus_key=manifest.corpus_key,
        source_snapshot_hash=manifest.source_snapshot_hash,
        chunking_fingerprint=manifest.chunking_fingerprint,
        documents=(),
        chunks=(),
    )
    changed_chunking_hash = active_chunk_manifest_hash(
        corpus_key=manifest.corpus_key,
        source_snapshot_hash=manifest.source_snapshot_hash,
        chunking_fingerprint="f" * 64,
        documents=manifest.documents,
        chunks=manifest.chunks,
    )

    assert changed_text_hash != manifest.manifest_hash
    assert changed_membership_hash != manifest.manifest_hash
    assert changed_chunking_hash != manifest.manifest_hash


def test_materialization_module_has_no_index_runtime_dependencies() -> None:
    path = Path("src/consumer_finance_rag/lineage/chunk_materialization.py")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    forbidden = {
        "qdrant_client",
        "consumer_finance_rag.providers",
        "consumer_finance_rag.sparse",
        "consumer_finance_rag.vector_store",
    }
    assert not {
        name
        for name in imported
        if any(name == root or name.startswith(f"{root}.") for root in forbidden)
    }
