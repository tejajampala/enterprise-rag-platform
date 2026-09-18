from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from consumer_finance_rag.acquisition import SourceRecord
from consumer_finance_rag.chunking import ParagraphChunker
from consumer_finance_rag.config import DatabaseSettings, load_settings
from consumer_finance_rag.corpus import CorpusSnapshot
from consumer_finance_rag.lineage.diff import current_document_state, plan_document_changes
from consumer_finance_rag.lineage.models import Base, Corpus, IngestionRun
from consumer_finance_rag.lineage.service import LineageDryRunResult
from consumer_finance_rag.normalization import normalize_record

SCRIPT_PATH = Path("scripts/ingest_incremental.py")
SCRIPT_SPEC = importlib.util.spec_from_file_location("phase3_incremental_cli", SCRIPT_PATH)
if SCRIPT_SPEC is None or SCRIPT_SPEC.loader is None:  # pragma: no cover - repository invariant
    raise RuntimeError(f"Unable to load {SCRIPT_PATH}")
ingest_incremental: Any = importlib.util.module_from_spec(SCRIPT_SPEC)
SCRIPT_SPEC.loader.exec_module(ingest_incremental)


def snapshot_for(*source_ids: str) -> CorpusSnapshot:
    documents = [
        normalize_record(
            SourceRecord(
                source_id=source_id,
                title=f"Guidance for {source_id}",
                source_url=None,
                category="credit-cards",
                source_version="2026-09-18",
                text=f"Public guidance for {source_id}.",
            )
        )
        for source_id in source_ids
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


@pytest.mark.parametrize(
    ("mode", "include_expanded", "require_expanded"),
    [("fixture", False, False), ("evaluation", True, True)],
)
def test_snapshot_modes_reuse_existing_corpus_requirements(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    include_expanded: bool,
    require_expanded: bool,
) -> None:
    settings = load_settings("config/config.yaml")
    expected = snapshot_for("cards")
    calls: list[dict[str, Any]] = []

    def fake_load(_settings: Any, **kwargs: Any) -> CorpusSnapshot:
        calls.append(kwargs)
        return expected

    monkeypatch.setattr(ingest_incremental, "load_corpus_snapshot", fake_load)

    actual = ingest_incremental.assemble_snapshot(settings, mode)

    assert actual is expected
    assert calls == [
        {
            "include_expanded": include_expanded,
            "require_expanded": require_expanded,
        }
    ]


def test_summary_and_details_are_deterministic_and_ordered() -> None:
    snapshot = snapshot_for("b", "a")
    chunks_by_document = {
        document.document_id: [
            chunk for chunk in snapshot.chunks if chunk.document_id == document.document_id
        ]
        for document in snapshot.documents
    }
    current = [
        current_document_state(document, chunks_by_document[document.document_id])
        for document in snapshot.documents
    ]
    result = LineageDryRunResult(
        plan=plan_document_changes(current, [], authoritative_snapshot=False),
        corpus_initialized=False,
        current_document_count=2,
        previous_active_document_count=0,
    )

    first = ingest_incremental.format_dry_run_summary(
        result,
        corpus_key="consumer-finance-managed",
        qdrant_collection_prefix="consumer_finance_managed",
        mode="fixture",
        details=True,
    )
    second = ingest_incremental.format_dry_run_summary(
        result,
        corpus_key="consumer-finance-managed",
        qdrant_collection_prefix="consumer_finance_managed",
        mode="fixture",
        details=True,
    )

    assert first == second
    assert "NEW:        2" in first
    assert "Deletion detection: SKIPPED (non-authoritative snapshot)" in first
    assert first.index("  source_key: a") < first.index("  source_key: b")
    assert "Public guidance" not in first


def test_missing_database_configuration_is_actionable_and_stops_before_assembly(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    settings = load_settings("config/config.yaml").model_copy(
        update={"database": DatabaseSettings()}
    )

    def unexpected_assembly(*args: Any, **kwargs: Any) -> CorpusSnapshot:
        del args, kwargs
        raise AssertionError("corpus should not be assembled without database configuration")

    monkeypatch.setattr(ingest_incremental, "load_settings", lambda _path: settings)
    monkeypatch.setattr(ingest_incremental, "assemble_snapshot", unexpected_assembly)

    exit_code = ingest_incremental.main([])

    assert exit_code == 2
    assert "RAG__DATABASE__URL" in capsys.readouterr().err


def test_missing_schema_instructs_developer_to_run_alembic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database_url = f"sqlite+pysqlite:///{(tmp_path / 'empty.sqlite').as_posix()}"
    settings = load_settings("config/config.yaml").model_copy(
        update={"database": DatabaseSettings(url=database_url)}
    )
    monkeypatch.setattr(ingest_incremental, "load_settings", lambda _path: settings)
    monkeypatch.setattr(
        ingest_incremental, "assemble_snapshot", lambda _settings, _mode: snapshot_for("cards")
    )

    exit_code = ingest_incremental.main([])

    assert exit_code == 2
    assert "uv run alembic upgrade head" in capsys.readouterr().err


def test_cli_first_use_is_read_only_and_creates_no_corpus_or_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database_path = tmp_path / "lineage.sqlite"
    database_url = f"sqlite+pysqlite:///{database_path.as_posix()}"
    engine = create_engine(database_url)
    Base.metadata.create_all(engine)
    engine.dispose()
    settings = load_settings("config/config.yaml").model_copy(
        update={"database": DatabaseSettings(url=database_url)}
    )
    monkeypatch.setattr(ingest_incremental, "load_settings", lambda _path: settings)
    monkeypatch.setattr(
        ingest_incremental,
        "assemble_snapshot",
        lambda _settings, _mode: snapshot_for("cards"),
    )

    exit_code = ingest_incremental.main(["--mode", "fixture", "--details"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "UNINITIALIZED" in output
    assert "NEW:        1" in output
    assert "No PostgreSQL, Qdrant, BM25, or model changes were made." in output

    engine = create_engine(database_url)
    with Session(engine) as session:
        assert session.scalar(select(func.count()).select_from(Corpus)) == 0
        assert session.scalar(select(func.count()).select_from(IngestionRun)) == 0
    engine.dispose()


def test_cli_and_service_have_no_index_or_model_side_effect_imports() -> None:
    paths = [
        Path(ingest_incremental.__file__),
        Path("src/consumer_finance_rag/lineage/service.py"),
    ]
    forbidden_modules = {
        "consumer_finance_rag.ingestion",
        "consumer_finance_rag.providers",
        "consumer_finance_rag.sparse",
        "consumer_finance_rag.vector_store",
    }
    imported: set[str] = set()
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported.add(node.module)

    assert not imported.intersection(forbidden_modules)
