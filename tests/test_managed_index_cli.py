from __future__ import annotations

import ast
import importlib.util
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from consumer_finance_rag.config import DatabaseSettings, load_settings
from consumer_finance_rag.lineage.index_types import (
    ActiveChunkManifest,
    BM25PlanAction,
)
from consumer_finance_rag.lineage.indexing import (
    ManagedIndexBuildError,
    ManagedIndexPublishOutcome,
    ManagedIndexPublishResult,
)
from consumer_finance_rag.lineage.models import Base, IndexState
from consumer_finance_rag.lineage.repository import LineageRepository

SCRIPT_PATH = Path("scripts/index_managed.py")
SCRIPT_SPEC = importlib.util.spec_from_file_location("phase3_managed_index_cli", SCRIPT_PATH)
if SCRIPT_SPEC is None or SCRIPT_SPEC.loader is None:  # pragma: no cover - repository invariant
    raise RuntimeError(f"Unable to load {SCRIPT_PATH}")
index_managed: Any = importlib.util.module_from_spec(SCRIPT_SPEC)
sys.modules[SCRIPT_SPEC.name] = index_managed
SCRIPT_SPEC.loader.exec_module(index_managed)

CORPUS_KEY = "consumer-finance-managed"
PREFIX = "consumer_finance_managed"


class FakeClient:
    def __init__(self) -> None:
        self.closed = False

    def close(self, *_args: object, **_kwargs: object) -> None:
        self.closed = True


class FakePublisher:
    def __init__(
        self,
        result: ManagedIndexPublishResult | None = None,
        error: Exception | None = None,
    ) -> None:
        self.result = result
        self.error = error
        self.calls: list[tuple[ActiveChunkManifest, UUID]] = []

    def publish(
        self,
        manifest: ActiveChunkManifest,
        target_ingestion_run_id: UUID,
    ) -> ManagedIndexPublishResult:
        self.calls.append((manifest, target_ingestion_run_id))
        if self.error is not None:
            raise self.error
        assert self.result is not None
        return self.result


def configured_settings(database_url: str) -> Any:
    return load_settings("config/config.yaml").model_copy(
        update={"database": DatabaseSettings(url=database_url)}
    )


def sqlite_database(tmp_path: Path) -> tuple[str, UUID]:
    database_url = f"sqlite+pysqlite:///{(tmp_path / 'lineage.sqlite').as_posix()}"
    engine = create_engine(database_url)
    Base.metadata.create_all(engine)
    with Session(engine) as session, session.begin():
        repository = LineageRepository(session)
        corpus = repository.create_corpus(CORPUS_KEY, PREFIX)
        run = repository.start_ingestion_run(
            corpus.corpus_id,
            authoritative_snapshot=False,
            snapshot_hash="a" * 64,
        )
        repository.finish_ingestion_run(
            run.run_id,
            "NOOP",
            datetime.now(UTC),
            0,
            0,
            0,
            0,
        )
        run_id = run.run_id
    engine.dispose()
    return database_url, run_id


def manifest_for(corpus_id: UUID) -> ActiveChunkManifest:
    return ActiveChunkManifest(
        corpus_id=corpus_id,
        corpus_key=CORPUS_KEY,
        source_snapshot_hash="a" * 64,
        chunking_fingerprint="b" * 64,
        documents=(),
        chunks=(),
        document_count=0,
        chunk_count=0,
        manifest_hash="c" * 64,
    )


def publish_result(outcome: ManagedIndexPublishOutcome) -> ManagedIndexPublishResult:
    return ManagedIndexPublishResult(
        outcome=outcome,
        index_state_id=uuid4(),
        generation=4,
        collection=f"{PREFIX}_g000004",
        snapshot_hash="a" * 64,
        bm25_action=(
            BM25PlanAction.REUSE
            if outcome is ManagedIndexPublishOutcome.NO_INDEX_CHANGE
            else BM25PlanAction.REFIT
        ),
        dense_reused_count=8,
        dense_computed_count=2,
        document_count=3,
        chunk_count=10,
    )


def test_default_invocation_is_read_only_and_avoids_all_publish_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database_url, run_id = sqlite_database(tmp_path)
    settings = configured_settings(database_url)
    monkeypatch.setattr(index_managed, "load_settings", lambda _path: settings)

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("read-only status must not perform publication work")

    monkeypatch.setattr(index_managed, "materialize_for_publication", forbidden)
    monkeypatch.setattr(index_managed, "build_managed_index_runtime", forbidden)

    assert index_managed.main([]) == 0

    output = capsys.readouterr().out
    assert "Phase 3 managed index" in output
    assert f"Latest lineage run: {run_id}" in output
    assert "Publishing: NOT REQUESTED" in output
    assert "Active index generation: none" in output

    engine = create_engine(database_url)
    with Session(engine) as session:
        assert session.scalar(select(func.count()).select_from(IndexState)) == 0
    engine.dispose()


def test_details_remains_database_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database_url, _run_id = sqlite_database(tmp_path)
    settings = configured_settings(database_url)
    monkeypatch.setattr(index_managed, "load_settings", lambda _path: settings)
    monkeypatch.setattr(
        index_managed,
        "build_managed_index_runtime",
        lambda *_args: (_ for _ in ()).throw(AssertionError("must not create Qdrant client")),
    )

    assert index_managed.main(["--details"]) == 0
    output = capsys.readouterr().out
    assert "Latest snapshot hash:" in output
    assert "Physical Qdrant check: NOT REQUESTED" in output


def test_publish_uses_exact_committed_manifest_and_latest_successful_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database_url, successful_run_id = sqlite_database(tmp_path)
    settings = configured_settings(database_url)
    engine = create_engine(database_url)
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
            "expected test failure",
            datetime.now(UTC) + timedelta(seconds=1),
        )
        expected_manifest = manifest_for(corpus.corpus_id)
    engine.dispose()

    observed_session: list[Session] = []

    class FakeMaterializer:
        def __init__(self, session: Session, **_kwargs: object) -> None:
            observed_session.append(session)

        def materialize_active_manifest(self, corpus_key: str) -> ActiveChunkManifest:
            assert corpus_key == CORPUS_KEY
            assert observed_session[0].in_transaction()
            return expected_manifest

    publisher = FakePublisher(publish_result(ManagedIndexPublishOutcome.PUBLISHED))
    client = FakeClient()

    def runtime_factory(_settings: Any, _sessions: Any) -> Any:
        assert not observed_session[0].in_transaction()
        return index_managed.ManagedIndexRuntime(publisher=publisher, client=client)

    monkeypatch.setattr(index_managed, "load_settings", lambda _path: settings)
    monkeypatch.setattr(index_managed, "ChunkMaterializationService", FakeMaterializer)
    monkeypatch.setattr(index_managed, "build_managed_index_runtime", runtime_factory)

    assert index_managed.main(["--publish"]) == 0
    assert publisher.calls == [(expected_manifest, successful_run_id)]
    assert client.closed
    output = capsys.readouterr().out
    assert "Result: PUBLISHED" in output
    assert "Generation: g000004" in output
    assert "BM25: REFIT" in output
    assert "Dense vectors reused: 8" in output
    assert "Dense vectors computed: 2" in output
    assert "Previous ACTIVE generation was retained." in output


def test_no_index_change_output_is_explicit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database_url, _run_id = sqlite_database(tmp_path)
    settings = configured_settings(database_url)
    engine = create_engine(database_url)
    with Session(engine) as session:
        corpus = LineageRepository(session).get_corpus_by_key(CORPUS_KEY)
        assert corpus is not None
        expected_manifest = manifest_for(corpus.corpus_id)
    engine.dispose()
    publisher = FakePublisher(publish_result(ManagedIndexPublishOutcome.NO_INDEX_CHANGE))
    client = FakeClient()
    monkeypatch.setattr(index_managed, "load_settings", lambda _path: settings)
    monkeypatch.setattr(
        index_managed,
        "materialize_for_publication",
        lambda _sessions, _settings: (expected_manifest, uuid4()),
    )
    monkeypatch.setattr(
        index_managed,
        "build_managed_index_runtime",
        lambda _settings, _sessions: index_managed.ManagedIndexRuntime(
            publisher=publisher,
            client=client,
        ),
    )

    assert index_managed.main(["--publish"]) == 0
    output = capsys.readouterr().out
    assert "Result: NO_INDEX_CHANGE" in output
    assert "No Qdrant or index-state changes were required." in output
    assert "Dense vectors computed" not in output


def test_missing_database_url_is_actionable_and_stops_before_database_use(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = load_settings("config/config.yaml").model_copy(
        update={"database": DatabaseSettings()}
    )
    monkeypatch.setattr(index_managed, "load_settings", lambda _path: settings)
    monkeypatch.setattr(
        index_managed,
        "create_database_engine",
        lambda *_args: (_ for _ in ()).throw(AssertionError("must not create engine")),
    )

    assert index_managed.main([]) == 2
    assert "RAG__DATABASE__URL" in capsys.readouterr().err


def test_missing_managed_corpus_is_actionable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database_url = f"sqlite+pysqlite:///{(tmp_path / 'empty-corpus.sqlite').as_posix()}"
    engine = create_engine(database_url)
    Base.metadata.create_all(engine)
    engine.dispose()
    settings = configured_settings(database_url)
    monkeypatch.setattr(index_managed, "load_settings", lambda _path: settings)

    assert index_managed.main([]) == 2
    error = capsys.readouterr().err
    assert "not initialized" in error
    assert "ingest_incremental.py" in error


def test_missing_successful_lineage_state_ignores_running_and_failed_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database_url = f"sqlite+pysqlite:///{(tmp_path / 'no-success.sqlite').as_posix()}"
    engine = create_engine(database_url)
    Base.metadata.create_all(engine)
    with Session(engine) as session, session.begin():
        repository = LineageRepository(session)
        corpus = repository.create_corpus(CORPUS_KEY, PREFIX)
        failed = repository.start_ingestion_run(corpus.corpus_id, False)
        repository.fail_ingestion_run(failed.run_id, "failed", datetime.now(UTC))
        repository.start_ingestion_run(corpus.corpus_id, False)
    engine.dispose()
    settings = configured_settings(database_url)
    monkeypatch.setattr(index_managed, "load_settings", lambda _path: settings)

    assert index_managed.main(["--publish"]) == 2
    assert "no successful/no-op lineage run" in capsys.readouterr().err


def test_missing_schema_instructs_operator_to_run_migration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database_url = f"sqlite+pysqlite:///{(tmp_path / 'no-schema.sqlite').as_posix()}"
    monkeypatch.setattr(
        index_managed,
        "load_settings",
        lambda _path: configured_settings(database_url),
    )

    assert index_managed.main([]) == 2
    assert "uv run alembic upgrade head" in capsys.readouterr().err


def test_publisher_failure_is_nonzero_and_closes_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database_url, _run_id = sqlite_database(tmp_path)
    settings = configured_settings(database_url)
    engine = create_engine(database_url)
    with Session(engine) as session:
        corpus = LineageRepository(session).get_corpus_by_key(CORPUS_KEY)
        assert corpus is not None
        expected_manifest = manifest_for(corpus.corpus_id)
    engine.dispose()
    publisher = FakePublisher(error=ManagedIndexBuildError("build failed"))
    client = FakeClient()
    monkeypatch.setattr(index_managed, "load_settings", lambda _path: settings)
    monkeypatch.setattr(
        index_managed,
        "materialize_for_publication",
        lambda _sessions, _settings: (expected_manifest, uuid4()),
    )
    monkeypatch.setattr(
        index_managed,
        "build_managed_index_runtime",
        lambda _settings, _sessions: index_managed.ManagedIndexRuntime(
            publisher=publisher,
            client=client,
        ),
    )

    assert index_managed.main(["--publish"]) == 3
    assert "Qdrant or embedding/model work" in capsys.readouterr().err
    assert client.closed


def test_script_delegates_index_algorithms_and_has_no_protected_collection_names() -> None:
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(SCRIPT_PATH))
    called_attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    imported_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_names.update(alias.name for alias in node.names)

    assert "upsert" not in called_attributes
    assert "upsert_points" not in called_attributes
    assert "plan_bm25_generation" not in imported_names
    assert "DenseGenerationPlanner" not in imported_names
    assert "consumer_finance_phase1" not in source
    assert "consumer_finance_evaluation" not in source
