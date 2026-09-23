from __future__ import annotations

import importlib.util
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import UUID

import pytest

from consumer_finance_rag.config import MLflowSettings
from consumer_finance_rag.observability import TraceContentPolicy

SCRIPT_PATH = Path("scripts/evaluate_managed_rag.py")
SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "phase3_managed_rag_evaluation_cli",
    SCRIPT_PATH,
)
if SCRIPT_SPEC is None or SCRIPT_SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"Unable to load {SCRIPT_PATH}")
evaluate_managed_rag_cli: Any = importlib.util.module_from_spec(SCRIPT_SPEC)
sys.modules[SCRIPT_SPEC.name] = evaluate_managed_rag_cli
SCRIPT_SPEC.loader.exec_module(evaluate_managed_rag_cli)


class FakeDataset:
    corpus_key = "consumer-finance-managed"

    def limited_cases(self, limit: int | None) -> tuple[str, ...]:
        cases = ("case-1", "case-2")
        return cases if limit is None else cases[:limit]


class FakeRun:
    active = True
    run_id = "combined-run"


def _settings(*, mlflow_enabled: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        evaluation=SimpleNamespace(managed_dataset_path=Path("managed.json")),
        managed_ingestion=SimpleNamespace(corpus_key="consumer-finance-managed"),
        models=SimpleNamespace(generation=SimpleNamespace(model="bedrock-generation")),
        mlflow=MLflowSettings(
            enabled=mlflow_enabled,
            judge_model="bedrock:/bedrock-judge",
            report_directory=Path("reports/evaluation"),
        ),
    )


def _runtime() -> SimpleNamespace:
    return SimpleNamespace(
        generation=11,
        index_state_id=UUID(int=11),
        qdrant_collection="consumer_finance_managed_g000011",
        retrieval_service=object(),
    )


def _report(*, judge_run_id: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        mode="generated_answers" if judge_run_id else "retrieval_only",
        generation=11,
        index_state_id=UUID(int=11),
        qdrant_collection="consumer_finance_managed_g000011",
        dataset_name="managed-public",
        dataset_version="1.0",
        case_count=2,
        retrieval_quality=SimpleNamespace(
            k=5,
            recall_at_k=1.0,
            mrr=1.0,
            hit_rate=1.0,
        ),
        answer_quality=SimpleNamespace(
            generation=SimpleNamespace(status="COMPLETED"),
            mlflow_genai=SimpleNamespace(status="COMPLETED", run_id=judge_run_id),
            ragas=SimpleNamespace(status="NOT_REQUESTED"),
        ),
    )


def _patch_common(
    monkeypatch: pytest.MonkeyPatch,
    *,
    mlflow_enabled: bool = False,
) -> tuple[SimpleNamespace, SimpleNamespace]:
    settings = _settings(mlflow_enabled=mlflow_enabled)
    runtime = _runtime()
    monkeypatch.setattr(evaluate_managed_rag_cli, "load_settings", lambda _path: settings)
    monkeypatch.setattr(
        evaluate_managed_rag_cli,
        "load_managed_rag_dataset",
        lambda _path: FakeDataset(),
    )
    monkeypatch.setattr(
        evaluate_managed_rag_cli,
        "validate_managed_dataset_compatibility",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        evaluate_managed_rag_cli,
        "write_managed_rag_report",
        lambda _report, path: Path(path),
    )
    return settings, runtime


def test_default_cli_pins_active_once_and_does_not_build_generation_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    _settings_value, runtime = _patch_common(monkeypatch)
    builds: list[object] = []
    evaluated: list[object] = []

    def build_current(_settings: object, _telemetry: object) -> SimpleNamespace:
        builds.append(_settings)
        return runtime

    def evaluate(pinned: object, *_args: object, **kwargs: object) -> SimpleNamespace:
        evaluated.append(pinned)
        assert kwargs["pipeline"] is None
        return _report()

    monkeypatch.setattr(evaluate_managed_rag_cli, "_build_current_runtime", build_current)
    monkeypatch.setattr(evaluate_managed_rag_cli, "evaluate_managed_rag", evaluate)
    monkeypatch.setattr(
        evaluate_managed_rag_cli,
        "build_application_runtime",
        lambda *_args, **_kwargs: pytest.fail("retrieval-only mode built generation models"),
    )

    result = evaluate_managed_rag_cli.main(
        ["--current", "--output", str(tmp_path / "report.json")]
    )

    assert result == 0
    assert len(builds) == 1
    assert evaluated == [runtime]
    assert "Generation: g000011" in capsys.readouterr().out


def test_judges_without_generation_fail_before_runtime_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        evaluate_managed_rag_cli,
        "load_settings",
        lambda _path: _settings(mlflow_enabled=True),
    )
    monkeypatch.setattr(
        evaluate_managed_rag_cli,
        "_build_current_runtime",
        lambda *_args: pytest.fail("invalid flags reached runtime construction"),
    )

    with pytest.raises(SystemExit, match="require --generate"):
        evaluate_managed_rag_cli.main(["--current", "--with-mlflow-judges"])


def test_combined_mode_reuses_pinned_runtime_and_one_mlflow_judge_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    settings, runtime = _patch_common(monkeypatch, mlflow_enabled=True)
    application_pipeline = object()
    wrappers: list[object] = []
    run_arguments: list[tuple[str, str | None]] = []
    logged: list[object] = []

    monkeypatch.setattr(
        evaluate_managed_rag_cli,
        "_build_current_runtime",
        lambda _settings, _telemetry: runtime,
    )

    def build_application(
        application_settings: object,
        *,
        retrieval_runtime: Any,
        telemetry_sink: object,
        trace_content_policy: TraceContentPolicy,
    ) -> SimpleNamespace:
        del telemetry_sink
        assert application_settings is settings
        assert trace_content_policy is TraceContentPolicy.APPROVED_PUBLIC_EVALUATION
        wrappers.append(retrieval_runtime)
        assert retrieval_runtime.generation == runtime.generation
        assert retrieval_runtime.index_state_id == runtime.index_state_id
        assert retrieval_runtime.qdrant_collection == runtime.qdrant_collection
        assert retrieval_runtime.retrieval_service is runtime.retrieval_service
        return SimpleNamespace(pipeline=application_pipeline)

    def evaluate(pinned: object, *_args: object, **kwargs: object) -> SimpleNamespace:
        assert pinned is runtime
        assert kwargs["pipeline"] is application_pipeline
        assert kwargs["with_mlflow_judges"] is True
        return _report(judge_run_id="judge-run")

    @contextmanager
    def fake_run(
        _settings: MLflowSettings,
        *,
        run_name: str,
        existing_run_id: str | None = None,
    ) -> Iterator[FakeRun]:
        run_arguments.append((run_name, existing_run_id))
        yield FakeRun()

    monkeypatch.setattr(evaluate_managed_rag_cli, "build_application_runtime", build_application)
    monkeypatch.setattr(evaluate_managed_rag_cli, "evaluate_managed_rag", evaluate)
    monkeypatch.setattr(evaluate_managed_rag_cli, "evaluation_run", fake_run)
    monkeypatch.setattr(
        evaluate_managed_rag_cli,
        "log_managed_rag_report",
        lambda report, _path, _run: logged.append(report),
    )

    result = evaluate_managed_rag_cli.main(
        [
            "--current",
            "--generate",
            "--with-mlflow-judges",
            "--log-mlflow",
            "--output",
            str(tmp_path / "report.json"),
        ]
    )

    assert result == 0
    assert len(wrappers) == 1
    assert run_arguments == [("managed-rag-g000011", "judge-run")]
    assert len(logged) == 1
