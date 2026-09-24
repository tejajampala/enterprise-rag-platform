from __future__ import annotations

import importlib.util
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from consumer_finance_rag.config import load_settings

SCRIPT_PATH = Path("scripts/assess_managed_drift.py")
SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "phase4_managed_drift_cli",
    SCRIPT_PATH,
)
if SCRIPT_SPEC is None or SCRIPT_SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"Unable to load {SCRIPT_PATH}")
assess_managed_drift_cli: Any = importlib.util.module_from_spec(SCRIPT_SPEC)
sys.modules[SCRIPT_SPEC.name] = assess_managed_drift_cli
SCRIPT_SPEC.loader.exec_module(assess_managed_drift_cli)


def _managed_settings() -> Any:
    settings = load_settings("config/config.yaml")
    return settings.model_copy(
        update={
            "runtime": settings.runtime.model_copy(update={"profile": "managed"}),
        }
    )


def test_default_and_metadata_only_modes_do_not_initialize_judges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _managed_settings()
    args = assess_managed_drift_cli.build_parser().parse_args(
        ["--baseline-generation", "3", "--skip-retrieval-evaluation"]
    )
    monkeypatch.setattr(
        assess_managed_drift_cli,
        "require_judge_model",
        lambda *_args, **_kwargs: pytest.fail("basic assessment initialized a judge"),
    )
    monkeypatch.setattr(
        assess_managed_drift_cli,
        "build_ragas_evaluator",
        lambda *_args, **_kwargs: pytest.fail("basic assessment initialized RAGAS"),
    )

    assert assess_managed_drift_cli._validate_modes(args, settings) is None
    assert args.generate_answers is False
    assert args.with_mlflow_judges is False
    assert args.with_ragas is False


def test_paid_modes_require_explicit_generation() -> None:
    settings = _managed_settings()
    args = assess_managed_drift_cli.build_parser().parse_args(
        ["--baseline-generation", "3", "--with-mlflow-judges"]
    )

    with pytest.raises(SystemExit, match="require --generate-answers"):
        assess_managed_drift_cli._validate_modes(args, settings)


def test_fixture_profile_and_conflicting_modes_fail_before_service_access() -> None:
    settings = load_settings("config/config.yaml")
    fixture_args = assess_managed_drift_cli.build_parser().parse_args(
        ["--baseline-generation", "3"]
    )
    conflicting_args = assess_managed_drift_cli.build_parser().parse_args(
        [
            "--baseline-generation",
            "3",
            "--skip-retrieval-evaluation",
            "--generate-answers",
        ]
    )

    with pytest.raises(SystemExit, match="PROFILE=managed"):
        assess_managed_drift_cli._validate_modes(fixture_args, settings)
    with pytest.raises(SystemExit, match="cannot be combined"):
        assess_managed_drift_cli._validate_modes(
            conflicting_args, _managed_settings()
        )


def test_unavailable_mlflow_is_reported_without_backend_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @contextmanager
    def unavailable(*_args: object, **_kwargs: object) -> Iterator[object]:
        raise ConnectionError("postgresql://operator:secret@internal/traces")
        yield object()  # pragma: no cover

    monkeypatch.setattr(assess_managed_drift_cli, "evaluation_run", unavailable)
    report = SimpleNamespace(
        baseline=SimpleNamespace(generation=3),
        candidate=SimpleNamespace(generation=4),
    )

    status = assess_managed_drift_cli._log_assessment(
        _managed_settings(),
        report,
        Path("unused.json"),
    )

    assert status == "MLflow assessment logging: UNAVAILABLE (ConnectionError)"
    assert "postgresql://" not in status
    assert "secret" not in status
