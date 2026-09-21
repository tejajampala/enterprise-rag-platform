from __future__ import annotations

import importlib.util
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

from sqlalchemy import create_engine

from consumer_finance_rag.config import DatabaseSettings, load_settings
from consumer_finance_rag.lineage.index_operations import (
    GenerationCleanupExecutionResult,
    GenerationCleanupPlan,
    ManagedGenerationValidationReport,
)

SCRIPT_PATH = Path("scripts/manage_index_generations.py")
SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "phase3_index_operations_cli",
    SCRIPT_PATH,
)
if SCRIPT_SPEC is None or SCRIPT_SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"Unable to load {SCRIPT_PATH}")
manage_index_generations: Any = importlib.util.module_from_spec(SCRIPT_SPEC)
sys.modules[SCRIPT_SPEC.name] = manage_index_generations
SCRIPT_SPEC.loader.exec_module(manage_index_generations)


class FakeClient:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class FakeOperationsService:
    confirmations: ClassVar[list[bool]] = []
    status_calls = 0
    validate_calls: ClassVar[list[bool]] = []

    def __init__(self, **_kwargs: object) -> None:
        pass

    def status(self) -> tuple[object, ...]:
        type(self).status_calls += 1
        return ()

    def validate(self, *, deep: bool = False) -> ManagedGenerationValidationReport:
        type(self).validate_calls.append(deep)
        return ManagedGenerationValidationReport(
            corpus_key="test",
            active_count=1,
            valid=True,
            issues=(),
            generations=(),
        )

    def cleanup_plan(self) -> GenerationCleanupPlan:
        return GenerationCleanupPlan(
            corpus_key="consumer-finance-managed",
            managed_prefix="consumer_finance_managed",
            planned_at=datetime(2026, 9, 21, tzinfo=UTC),
            retained_retired_generations=2,
            retired_cleanup_min_age_hours=24,
            candidates=(),
        )

    def execute_cleanup(
        self,
        _plan: GenerationCleanupPlan,
        *,
        confirmed: bool,
    ) -> GenerationCleanupExecutionResult:
        type(self).confirmations.append(confirmed)
        return GenerationCleanupExecutionResult(confirmed=confirmed, items=())


def configure_cli(monkeypatch: Any) -> None:
    settings = load_settings("config/config.yaml").model_copy(
        update={"database": DatabaseSettings(url="sqlite+pysqlite:///:memory:")}
    )
    monkeypatch.setattr(manage_index_generations, "load_settings", lambda _path: settings)
    monkeypatch.setattr(
        manage_index_generations,
        "create_database_engine",
        lambda _settings: create_engine("sqlite+pysqlite:///:memory:"),
    )
    monkeypatch.setattr(manage_index_generations, "QdrantClient", FakeClient)
    monkeypatch.setattr(
        manage_index_generations,
        "ManagedIndexOperationsService",
        FakeOperationsService,
    )
    FakeOperationsService.confirmations = []
    FakeOperationsService.status_calls = 0
    FakeOperationsService.validate_calls = []


def test_cli_defaults_to_read_only_status(monkeypatch: Any) -> None:
    configure_cli(monkeypatch)

    assert manage_index_generations.main([]) == 0

    assert FakeOperationsService.status_calls == 1
    assert FakeOperationsService.confirmations == []


def test_cli_cleanup_requires_explicit_confirm(monkeypatch: Any) -> None:
    configure_cli(monkeypatch)

    assert manage_index_generations.main(["--cleanup"]) == 0
    assert FakeOperationsService.confirmations == [False]

    assert manage_index_generations.main(["--cleanup", "--confirm"]) == 0
    assert FakeOperationsService.confirmations == [False, True]


def test_cli_deep_validation_is_explicit(monkeypatch: Any) -> None:
    configure_cli(monkeypatch)

    assert manage_index_generations.main(["--validate", "--deep"]) == 0

    assert FakeOperationsService.validate_calls == [True]
