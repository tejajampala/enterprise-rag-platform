from __future__ import annotations

import importlib.util
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar
from uuid import UUID

from sqlalchemy import create_engine

from consumer_finance_rag.config import DatabaseSettings, load_settings
from consumer_finance_rag.lineage.index_operations import (
    GenerationCleanupExecutionResult,
    GenerationCleanupPlan,
    ManagedGenerationValidationReport,
)
from consumer_finance_rag.lineage.restoration import (
    GenerationRestorationPlan,
    GenerationRestorationResult,
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


class FakeRestorationService:
    plan_calls: ClassVar[list[int]] = []
    execution_reasons: ClassVar[list[str]] = []

    def __init__(self, **_kwargs: object) -> None:
        pass

    def plan(self, source_generation: int) -> GenerationRestorationPlan:
        type(self).plan_calls.append(source_generation)
        return GenerationRestorationPlan(
            corpus_key="consumer-finance-managed",
            source_generation=source_generation,
            source_index_state_id=UUID("00000000-0000-0000-0000-000000000001"),
            source_collection="consumer_finance_managed_g000001",
            current_active_generation=2,
            current_active_index_state_id=UUID(
                "00000000-0000-0000-0000-000000000002"
            ),
            proposed_generation=3,
            proposed_target_collection="consumer_finance_managed_g000003",
            document_count=1,
            chunk_count=2,
            bm25_state_hash="b" * 64,
            embedding_fingerprint="e" * 64,
            configuration_fingerprint="c" * 64,
            payload_schema_version="managed-qdrant-payload-v3",
            warnings=(),
        )

    def execute(
        self,
        plan: GenerationRestorationPlan,
        *,
        restoration_reason: str,
    ) -> GenerationRestorationResult:
        type(self).execution_reasons.append(restoration_reason)
        return GenerationRestorationResult(
            source_generation=plan.source_generation,
            source_index_state_id=plan.source_index_state_id,
            previous_active_generation=plan.current_active_generation,
            previous_active_index_state_id=plan.current_active_index_state_id,
            new_active_generation=plan.proposed_generation,
            new_active_index_state_id=UUID(
                "00000000-0000-0000-0000-000000000003"
            ),
            collection=plan.proposed_target_collection,
            dense_vectors_copied=plan.chunk_count,
            chunk_count=plan.chunk_count,
        )


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
    monkeypatch.setattr(
        manage_index_generations,
        "ManagedGenerationRestorationService",
        FakeRestorationService,
    )
    FakeOperationsService.confirmations = []
    FakeOperationsService.status_calls = 0
    FakeOperationsService.validate_calls = []
    FakeRestorationService.plan_calls = []
    FakeRestorationService.execution_reasons = []


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


def test_cli_restore_is_plan_only_without_confirm(monkeypatch: Any) -> None:
    configure_cli(monkeypatch)

    assert manage_index_generations.main(["--restore-generation", "1"]) == 0

    assert FakeRestorationService.plan_calls == [1]
    assert FakeRestorationService.execution_reasons == []


def test_cli_restore_executes_only_with_confirm(monkeypatch: Any) -> None:
    configure_cli(monkeypatch)

    assert (
        manage_index_generations.main(
            ["--restore-generation", "1", "--confirm"]
        )
        == 0
    )

    assert FakeRestorationService.plan_calls == [1]
    assert FakeRestorationService.execution_reasons == [
        "Operator-confirmed restoration of g000001"
    ]
