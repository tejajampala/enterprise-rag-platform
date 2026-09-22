from __future__ import annotations

import importlib.util
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from consumer_finance_rag.config import MLflowSettings

SCRIPT_PATH = Path("scripts/evaluate_managed_retrieval.py")
SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "phase3_managed_retrieval_quality_cli",
    SCRIPT_PATH,
)
if SCRIPT_SPEC is None or SCRIPT_SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"Unable to load {SCRIPT_PATH}")
evaluate_managed_retrieval_cli: Any = importlib.util.module_from_spec(SCRIPT_SPEC)
sys.modules[SCRIPT_SPEC.name] = evaluate_managed_retrieval_cli
SCRIPT_SPEC.loader.exec_module(evaluate_managed_retrieval_cli)


class FakeRun:
    active = True
    run_id = "mlflow-run-456"


def _report() -> SimpleNamespace:
    return SimpleNamespace(
        generation=4,
        qdrant_collection="consumer_finance_managed_g000004",
        case_count=2,
        metrics=SimpleNamespace(k=5),
        recall_at_k=1.0,
        mrr=0.75,
        hit_rate=1.0,
        missed_case_count=0,
    )


def _patch_offline_evaluation(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = SimpleNamespace(
        evaluation=SimpleNamespace(dataset_path=Path("unused.json")),
        mlflow=MLflowSettings(enabled=True),
    )
    monkeypatch.setattr(
        evaluate_managed_retrieval_cli,
        "load_settings",
        lambda _path: settings,
    )
    monkeypatch.setattr(
        evaluate_managed_retrieval_cli,
        "_build_current_runtime",
        lambda _settings: object(),
    )
    monkeypatch.setattr(
        evaluate_managed_retrieval_cli,
        "load_evaluation_set",
        lambda _path: object(),
    )
    monkeypatch.setattr(
        evaluate_managed_retrieval_cli,
        "evaluation_cases_from_seed",
        lambda _dataset: (),
    )
    monkeypatch.setattr(
        evaluate_managed_retrieval_cli,
        "evaluate_managed_retrieval",
        lambda *_args, **_kwargs: _report(),
    )


def test_default_cli_does_not_initialize_or_require_mlflow(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _patch_offline_evaluation(monkeypatch)

    def unexpected_run(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("default CLI must not initialize MLflow")

    monkeypatch.setattr(
        evaluate_managed_retrieval_cli,
        "evaluation_run",
        unexpected_run,
    )

    assert evaluate_managed_retrieval_cli.main(["--current"]) == 0
    output = capsys.readouterr().out
    assert "Generation: g000004" in output
    assert "MLflow run ID" not in output


def test_log_mlflow_flag_logs_report_and_prints_run_id(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _patch_offline_evaluation(monkeypatch)
    started: list[str] = []
    logged: list[object] = []

    @contextmanager
    def fake_run(
        _settings: MLflowSettings,
        *,
        run_name: str,
    ) -> Iterator[FakeRun]:
        started.append(run_name)
        yield FakeRun()

    monkeypatch.setattr(
        evaluate_managed_retrieval_cli,
        "evaluation_run",
        fake_run,
    )
    monkeypatch.setattr(
        evaluate_managed_retrieval_cli,
        "log_retrieval_quality_report",
        lambda report, _run: logged.append(report),
    )

    assert (
        evaluate_managed_retrieval_cli.main(["--current", "--log-mlflow"])
        == 0
    )
    output = capsys.readouterr().out
    assert started == ["managed-retrieval-quality-g000004"]
    assert len(logged) == 1
    assert "MLflow run ID: mlflow-run-456" in output

