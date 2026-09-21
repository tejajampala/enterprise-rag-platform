from __future__ import annotations

from collections.abc import Generator
from datetime import UTC
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import Engine, create_engine, event, inspect, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from consumer_finance_rag.lineage.models import (
    Base,
    Corpus,
    DocumentVersion,
    DocumentVersionPII,
    IndexState,
    IndexStateChunk,
    IngestionRun,
    IngestionRunDocument,
    LineageChunk,
    LineageDocument,
)

APPLICATION_TABLES = {
    "chunks",
    "corpora",
    "document_versions",
    "document_version_pii",
    "documents",
    "index_state_chunks",
    "index_states",
    "ingestion_run_documents",
    "ingestion_runs",
}


def test_alembic_history_has_one_head() -> None:
    """Reject migration branches before a database connection is needed."""

    scripts = ScriptDirectory.from_config(Config("alembic.ini"))
    heads = scripts.get_heads()

    assert len(heads) == 1, f"expected one Alembic head, found {heads}"
    assert scripts.get_current_head() == heads[0]
    assert list(scripts.walk_revisions()), "expected at least one application migration"


def test_alembic_revision_ids_and_graph_are_version_table_compatible() -> None:
    """Keep the linear graph compatible with Alembic's VARCHAR(32) version column."""

    scripts = ScriptDirectory.from_config(Config("alembic.ini"))
    revisions = list(scripts.walk_revisions())
    heads = scripts.get_heads()

    assert len(heads) == 1, f"expected one Alembic head, found {heads}"
    assert revisions, "expected at least one application migration"

    revision_ids = {migration.revision for migration in revisions}
    assert len(revision_ids) == len(revisions), "Alembic revision IDs must be unique"

    children_by_parent: dict[str, list[str]] = {
        revision_id: [] for revision_id in revision_ids
    }
    bases: list[str] = []
    parents_by_revision: dict[str, tuple[str, ...]] = {}

    for migration in revisions:
        revision_id = migration.revision
        assert revision_id, "Alembic revision ID must be non-empty"
        assert len(revision_id) <= 32, (
            f"Alembic revision ID {revision_id!r} exceeds the default "
            "alembic_version.version_num VARCHAR(32) limit"
        )

        down_revision = migration.down_revision
        parents = (
            ()
            if down_revision is None
            else (down_revision,)
            if isinstance(down_revision, str)
            else tuple(down_revision)
        )
        parents_by_revision[revision_id] = parents
        assert len(parents) <= 1, (
            f"Alembic migration {revision_id!r} has multiple parents: {parents}"
        )

        if not parents:
            bases.append(revision_id)
        for parent in parents:
            assert parent in revision_ids, (
                f"Alembic migration {revision_id!r} references missing "
                f"down_revision {parent!r}"
            )
            children_by_parent[parent].append(revision_id)

    assert len(bases) == 1, f"expected one Alembic base, found {bases}"
    for parent, children in children_by_parent.items():
        assert len(children) <= 1, (
            f"Alembic history branches after {parent!r}: {children}"
        )

    visited: set[str] = set()
    current_revision: str | None = heads[0]
    while current_revision is not None:
        assert current_revision not in visited, (
            f"Alembic history contains a cycle at {current_revision!r}"
        )
        visited.add(current_revision)
        parents = parents_by_revision[current_revision]
        current_revision = parents[0] if parents else None

    assert visited == revision_ids, (
        "Alembic history is disconnected; unreachable revisions: "
        f"{sorted(revision_ids - visited)}"
    )


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


def hash_value(value: int) -> str:
    return f"{value:064x}"


def make_corpus(session: Session, suffix: str = "primary") -> Corpus:
    corpus = Corpus(
        corpus_key=f"consumer-finance-{suffix}",
        qdrant_collection_prefix=f"consumer-finance-{suffix}",
    )
    session.add(corpus)
    session.flush()
    return corpus


def make_run(session: Session, corpus: Corpus, status: str = "RUNNING") -> IngestionRun:
    run = IngestionRun(corpus=corpus, status=status, authoritative_snapshot=True)
    session.add(run)
    session.flush()
    return run


def make_document(
    session: Session,
    corpus: Corpus,
    run: IngestionRun,
    *,
    source_key: str = "cfpb:credit-cards",
) -> LineageDocument:
    document = LineageDocument(
        corpus=corpus,
        source_key=source_key,
        source_id=source_key,
        source_url="https://www.consumerfinance.gov/example",
        state="ACTIVE",
        first_seen_run=run,
        last_seen_run=run,
    )
    session.add(document)
    session.flush()
    return document


def make_version(
    session: Session,
    document: LineageDocument,
    run: IngestionRun,
    *,
    version_number: int = 1,
    hash_seed: int = 1,
) -> DocumentVersion:
    version = DocumentVersion(
        document=document,
        version_number=version_number,
        content_hash=hash_value(hash_seed),
        metadata_hash=hash_value(hash_seed + 100),
        version_hash=hash_value(hash_seed + 200),
        compatibility_document_id=f"document-{hash_seed}",
        title="Credit card guidance",
        category="credit-cards",
        source_url=document.source_url,
        source_version="2026-09-18",
        normalized_text="Normalized public consumer-finance guidance.",
        created_run=run,
    )
    session.add(version)
    session.flush()
    return version


def make_chunk(
    session: Session,
    version: DocumentVersion,
    *,
    ordinal: int = 0,
    compatibility_id: str = "chunk-0",
    hash_seed: int = 500,
) -> LineageChunk:
    chunk = LineageChunk(
        document_version=version,
        chunking_fingerprint=hash_value(400),
        compatibility_chunk_id=compatibility_id,
        ordinal=ordinal,
        chunk_hash=hash_value(hash_seed),
        text="A citation-ready consumer-finance passage.",
        chunk_metadata={"title": "Credit card guidance"},
    )
    session.add(chunk)
    session.flush()
    return chunk


def make_index_state(
    session: Session,
    corpus: Corpus,
    run: IngestionRun,
    *,
    generation: int = 1,
    collection: str = "consumer_finance_managed_g0001",
) -> IndexState:
    state = IndexState(
        corpus=corpus,
        generation=generation,
        ingestion_run=run,
        status="BUILDING",
        qdrant_collection=collection,
        snapshot_hash=hash_value(700),
        embedding_fingerprint=hash_value(701),
        embedding_dimension=1024,
        chunking_fingerprint=hash_value(702),
        document_count=1,
        chunk_count=1,
    )
    session.add(state)
    session.flush()
    return state


def test_metadata_creates_all_application_tables(engine: Engine) -> None:
    assert set(inspect(engine).get_table_names()) == APPLICATION_TABLES
    version_columns = {
        column["name"] for column in inspect(engine).get_columns("document_versions")
    }
    assert {"access_visibility", "allowed_groups"}.issubset(version_columns)
    columns = {column["name"] for column in inspect(engine).get_columns("index_states")}
    assert {
        "bm25_state",
        "index_configuration",
        "index_configuration_fingerprint",
        "error_message",
        "failed_at",
    }.issubset(columns)


def test_document_version_defaults_to_public_access(session: Session) -> None:
    corpus = make_corpus(session)
    run = make_run(session, corpus)
    document = make_document(session, corpus, run)

    version = make_version(session, document, run)

    assert version.access_visibility == "PUBLIC"
    assert version.allowed_groups == []


def test_document_version_accepts_multiple_immutable_pii_detector_results(
    session: Session,
) -> None:
    corpus = make_corpus(session)
    run = make_run(session, corpus)
    document = make_document(session, corpus, run)
    version = make_version(session, document, run)
    first = DocumentVersionPII(
        document_version=version,
        detector_version="deterministic-pii-v1",
        classification_hash="d" * 64,
        pii_types=["EMAIL"],
        match_count=1,
    )
    second = DocumentVersionPII(
        document_version=version,
        detector_version="deterministic-pii-v2",
        classification_hash="e" * 64,
        pii_types=[],
        match_count=0,
    )
    session.add_all([first, second])
    session.flush()

    assert version.pii_classifications == [first, second]


def test_duplicate_pii_detector_version_is_rejected(session: Session) -> None:
    corpus = make_corpus(session)
    run = make_run(session, corpus)
    document = make_document(session, corpus, run)
    version = make_version(session, document, run)
    session.add_all(
        [
            DocumentVersionPII(
                document_version=version,
                detector_version="deterministic-pii-v1",
                classification_hash="d" * 64,
                pii_types=[],
                match_count=0,
            ),
            DocumentVersionPII(
                document_version=version,
                detector_version="deterministic-pii-v1",
                classification_hash="e" * 64,
                pii_types=["EMAIL"],
                match_count=1,
            ),
        ]
    )

    with pytest.raises(IntegrityError):
        session.flush()


def test_domain_statuses_reject_unknown_values() -> None:
    with pytest.raises(ValueError, match="status must be one of"):
        IngestionRun(status="UNKNOWN", authoritative_snapshot=False)
    with pytest.raises(ValueError, match="change_type must be one of"):
        IngestionRunDocument(
            change_type="UNKNOWN", content_changed=False, metadata_changed=False
        )
    with pytest.raises(ValueError, match="dense_action must be one of"):
        IndexStateChunk(qdrant_point_id=uuid4(), dense_action="UNKNOWN")


def test_corpus_key_is_unique(session: Session) -> None:
    make_corpus(session)
    session.add(
        Corpus(
            corpus_key="consumer-finance-primary",
            qdrant_collection_prefix="different-prefix",
        )
    )
    with pytest.raises(IntegrityError):
        session.flush()


def test_document_source_key_is_unique_within_a_corpus(session: Session) -> None:
    corpus = make_corpus(session)
    run = make_run(session, corpus)
    make_document(session, corpus, run)
    duplicate = LineageDocument(
        corpus=corpus,
        source_key="cfpb:credit-cards",
        source_id="another-source-id",
        state="ACTIVE",
        first_seen_run=run,
        last_seen_run=run,
    )
    session.add(duplicate)
    with pytest.raises(IntegrityError):
        session.flush()


def test_logical_document_accepts_multiple_versions(session: Session) -> None:
    corpus = make_corpus(session)
    run = make_run(session, corpus)
    document = make_document(session, corpus, run)
    first = make_version(session, document, run)
    second = make_version(session, document, run, version_number=2, hash_seed=2)

    session.refresh(document)

    assert document.versions == [first, second]


def test_duplicate_document_version_number_is_rejected(session: Session) -> None:
    corpus = make_corpus(session)
    run = make_run(session, corpus)
    document = make_document(session, corpus, run)
    make_version(session, document, run)
    session.add(
        DocumentVersion(
            document=document,
            version_number=1,
            content_hash=hash_value(2),
            metadata_hash=hash_value(102),
            version_hash=hash_value(202),
            compatibility_document_id="different-document-id",
            title="Updated guidance",
            category="credit-cards",
            source_version="2",
            normalized_text="Updated normalized text.",
            created_run=run,
        )
    )
    with pytest.raises(IntegrityError):
        session.flush()


def test_duplicate_version_hash_for_same_document_is_rejected(session: Session) -> None:
    corpus = make_corpus(session)
    run = make_run(session, corpus)
    document = make_document(session, corpus, run)
    first = make_version(session, document, run)
    session.add(
        DocumentVersion(
            document=document,
            version_number=2,
            content_hash=hash_value(2),
            metadata_hash=hash_value(102),
            version_hash=first.version_hash,
            compatibility_document_id="different-document-id",
            title="Updated guidance",
            category="credit-cards",
            source_version="2",
            normalized_text="Updated normalized text.",
            created_run=run,
        )
    )
    with pytest.raises(IntegrityError):
        session.flush()


@pytest.mark.parametrize("collision", ["ordinal", "compatibility_id"])
def test_chunk_identity_is_unique_within_version_and_fingerprint(
    session: Session, collision: str
) -> None:
    corpus = make_corpus(session)
    run = make_run(session, corpus)
    document = make_document(session, corpus, run)
    version = make_version(session, document, run)
    make_chunk(session, version)
    duplicate = LineageChunk(
        document_version=version,
        chunking_fingerprint=hash_value(400),
        compatibility_chunk_id="chunk-0" if collision == "compatibility_id" else "chunk-1",
        ordinal=0 if collision == "ordinal" else 1,
        chunk_hash=hash_value(501),
        text="Another passage.",
        chunk_metadata={},
    )
    session.add(duplicate)
    with pytest.raises(IntegrityError):
        session.flush()


def test_ingestion_run_document_has_composite_primary_key(session: Session) -> None:
    corpus = make_corpus(session)
    run = make_run(session, corpus)
    document = make_document(session, corpus, run)
    session.add_all(
        [
            IngestionRunDocument(
                run=run,
                document=document,
                change_type="NEW",
                content_changed=True,
                metadata_changed=True,
            ),
            IngestionRunDocument(
                run=run,
                document=document,
                change_type="UNCHANGED",
                content_changed=False,
                metadata_changed=False,
            ),
        ]
    )
    with pytest.raises(IntegrityError):
        session.flush()


def test_index_generation_is_unique_per_corpus(session: Session) -> None:
    corpus = make_corpus(session)
    first_run = make_run(session, corpus)
    second_run = make_run(session, corpus)
    make_index_state(session, corpus, first_run)
    session.add(
        IndexState(
            corpus=corpus,
            generation=1,
            ingestion_run=second_run,
            status="BUILDING",
            qdrant_collection="another-collection",
            snapshot_hash=hash_value(710),
            embedding_fingerprint=hash_value(711),
            embedding_dimension=1024,
            chunking_fingerprint=hash_value(712),
            document_count=1,
            chunk_count=1,
        )
    )
    with pytest.raises(IntegrityError):
        session.flush()


def test_one_ingestion_run_may_reference_multiple_index_states(session: Session) -> None:
    corpus = make_corpus(session)
    run = make_run(session, corpus)
    first = make_index_state(session, corpus, run)
    second = make_index_state(
        session,
        corpus,
        run,
        generation=2,
        collection="consumer_finance_managed_g0002",
    )

    assert run.index_states == [first, second]


def test_partial_unique_index_rejects_multiple_active_generations(
    session: Session,
) -> None:
    corpus = make_corpus(session)
    first_run = make_run(session, corpus)
    second_run = make_run(session, corpus)
    first = make_index_state(session, corpus, first_run)
    second = make_index_state(
        session,
        corpus,
        second_run,
        generation=2,
        collection="consumer_finance_managed_g0002",
    )
    first.status = "ACTIVE"
    session.flush()
    second.status = "ACTIVE"

    with pytest.raises(IntegrityError):
        session.flush()


@pytest.mark.parametrize("collision", ["composite_key", "point_id"])
def test_index_state_chunk_identity_constraints(session: Session, collision: str) -> None:
    corpus = make_corpus(session)
    run = make_run(session, corpus)
    document = make_document(session, corpus, run)
    version = make_version(session, document, run)
    first_chunk = make_chunk(session, version)
    second_chunk = make_chunk(
        session, version, ordinal=1, compatibility_id="chunk-1", hash_seed=501
    )
    index_state = make_index_state(session, corpus, run)
    point_id = uuid4()
    session.add(
        IndexStateChunk(
            index_state=index_state,
            chunk=first_chunk,
            qdrant_point_id=point_id,
            dense_action="COMPUTED",
        )
    )
    session.flush()
    session.add(
        IndexStateChunk(
            index_state=index_state,
            chunk=first_chunk if collision == "composite_key" else second_chunk,
            qdrant_point_id=uuid4() if collision == "composite_key" else point_id,
            dense_action="REUSED",
            dense_source_chunk=first_chunk,
        )
    )
    with pytest.raises(IntegrityError):
        session.flush()


@pytest.mark.parametrize(
    ("dense_action", "use_source"),
    [("COMPUTED", True), ("REUSED", False)],
)
def test_dense_provenance_check_constraint(
    session: Session, dense_action: str, use_source: bool
) -> None:
    corpus = make_corpus(session)
    run = make_run(session, corpus)
    document = make_document(session, corpus, run)
    version = make_version(session, document, run)
    chunk = make_chunk(session, version)
    index_state = make_index_state(session, corpus, run)
    session.add(
        IndexStateChunk(
            index_state=index_state,
            chunk=chunk,
            qdrant_point_id=uuid4(),
            dense_action=dense_action,
            dense_source_chunk=chunk if use_source else None,
        )
    )

    with pytest.raises(IntegrityError):
        session.flush()


def test_relationships_uuid_defaults_and_timestamps_are_populated(session: Session) -> None:
    corpus = make_corpus(session)
    run = make_run(session, corpus)
    document = make_document(session, corpus, run)
    version = make_version(session, document, run)
    chunk = make_chunk(session, version)
    classification = IngestionRunDocument(
        run=run,
        document=document,
        change_type="NEW",
        observed_version=version,
        content_changed=True,
        metadata_changed=True,
    )
    session.add(classification)
    index_state = make_index_state(session, corpus, run)
    membership = IndexStateChunk(
        index_state=index_state,
        chunk=chunk,
        qdrant_point_id=uuid4(),
        dense_action="COMPUTED",
    )
    session.add(membership)
    session.flush()

    assert isinstance(corpus.corpus_id, UUID)
    assert isinstance(run.run_id, UUID)
    assert isinstance(document.document_pk, UUID)
    assert isinstance(version.document_version_id, UUID)
    assert isinstance(chunk.chunk_pk, UUID)
    assert isinstance(index_state.index_state_id, UUID)
    assert corpus.created_at.utcoffset() == UTC.utcoffset(corpus.created_at)
    assert run.started_at.utcoffset() == UTC.utcoffset(run.started_at)
    assert document.versions == [version]
    assert version.document is document
    assert version.chunks == [chunk]
    assert classification in run.classifications
    assert membership in index_state.chunks
    assert membership.chunk is chunk


def test_alembic_revision_preserves_unrelated_tables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database_path = tmp_path / "lineage.sqlite"
    database_url = f"sqlite+pysqlite:///{database_path.as_posix()}"
    migration_engine = create_engine(database_url)
    with migration_engine.begin() as connection:
        connection.execute(text("CREATE TABLE unrelated_mlflow_table (id INTEGER PRIMARY KEY)"))
    migration_engine.dispose()

    monkeypatch.setenv("RAG__DATABASE__URL", database_url)
    alembic_config = Config("alembic.ini")
    command.upgrade(alembic_config, "head")

    inspection_engine = create_engine(database_url)
    inspector = inspect(inspection_engine)
    assert APPLICATION_TABLES.issubset(inspector.get_table_names())
    index_columns = {column["name"] for column in inspector.get_columns("index_states")}
    assert {
        "bm25_state",
        "index_configuration",
        "index_configuration_fingerprint",
        "error_message",
        "failed_at",
    }.issubset(index_columns)
    ingestion_uniques = inspector.get_unique_constraints("index_states")
    assert not any(
        constraint.get("column_names") == ["ingestion_run_id"]
        for constraint in ingestion_uniques
    )
    inspection_engine.dispose()

    command.downgrade(alembic_config, "base")

    inspection_engine = create_engine(database_url)
    remaining_tables = set(inspect(inspection_engine).get_table_names())
    inspection_engine.dispose()
    assert not APPLICATION_TABLES.intersection(remaining_tables)
    assert "unrelated_mlflow_table" in remaining_tables


def test_acl_migration_preserves_existing_versions_with_public_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "acl-migration.sqlite"
    database_url = f"sqlite+pysqlite:///{database_path.as_posix()}"
    monkeypatch.setenv("RAG__DATABASE__URL", database_url)
    alembic_config = Config("alembic.ini")
    command.upgrade(alembic_config, "0002_managed_index_generations")
    migration_engine = create_engine(database_url)
    corpus_id = "00000000000000000000000000000001"
    run_id = "00000000000000000000000000000002"
    document_pk = "00000000000000000000000000000003"
    version_id = "00000000000000000000000000000004"
    with migration_engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO corpora "
                "(corpus_id, corpus_key, qdrant_collection_prefix) "
                "VALUES (:id, 'acl-test', 'acl_test')"
            ),
            {"id": corpus_id},
        )
        connection.execute(
            text(
                "INSERT INTO ingestion_runs "
                "(run_id, corpus_id, status, authoritative_snapshot) "
                "VALUES (:run, :corpus, 'SUCCEEDED', 1)"
            ),
            {"run": run_id, "corpus": corpus_id},
        )
        connection.execute(
            text(
                "INSERT INTO documents "
                "(document_pk, corpus_id, source_key, source_id, state, "
                "first_seen_run_id, last_seen_run_id) "
                "VALUES (:document, :corpus, 'source', 'source', 'ACTIVE', :run, :run)"
            ),
            {"document": document_pk, "corpus": corpus_id, "run": run_id},
        )
        connection.execute(
            text(
                "INSERT INTO document_versions "
                "(document_version_id, document_pk, version_number, content_hash, "
                "metadata_hash, version_hash, compatibility_document_id, title, "
                "category, source_version, normalized_text, created_run_id) "
                "VALUES (:version, :document, 1, :content, :metadata, :combined, "
                "'legacy-document', 'Legacy title', 'credit-cards', 'v1', "
                "'Legacy text', :run)"
            ),
            {
                "version": version_id,
                "document": document_pk,
                "content": "a" * 64,
                "metadata": "b" * 64,
                "combined": "c" * 64,
                "run": run_id,
            },
        )
    migration_engine.dispose()

    command.upgrade(alembic_config, "head")

    inspection_engine = create_engine(database_url)
    with inspection_engine.connect() as connection:
        row = connection.execute(
            text(
                "SELECT access_visibility, allowed_groups FROM document_versions "
                "WHERE document_version_id = :version"
            ),
            {"version": version_id},
        ).one()
        pii_count = connection.execute(
            text("SELECT count(*) FROM document_version_pii")
        ).scalar_one()
    inspection_engine.dispose()
    assert row.access_visibility == "PUBLIC"
    assert row.allowed_groups == "[]"
    assert pii_count == 0

    command.downgrade(alembic_config, "0002_managed_index_generations")
    inspection_engine = create_engine(database_url)
    columns = {
        column["name"]
        for column in inspect(inspection_engine).get_columns("document_versions")
    }
    table_names = set(inspect(inspection_engine).get_table_names())
    with inspection_engine.connect() as connection:
        assert connection.execute(
            text("SELECT count(*) FROM document_versions")
        ).scalar_one() == 1
    inspection_engine.dispose()
    assert "access_visibility" not in columns
    assert "allowed_groups" not in columns
    assert "document_version_pii" not in table_names
