from __future__ import annotations

import ast
import importlib.util
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast
from uuid import UUID

import pytest
from sqlalchemy import Engine, create_engine

from consumer_finance_rag.config import (
    DatabaseSettings,
    MLflowSettings,
    QdrantSettings,
    RuntimeSettings,
    load_settings,
)
from consumer_finance_rag.environment_check import (
    CheckStatus,
    EngineFactory,
    EnvironmentReport,
    ManagedReadinessObservation,
    ManagedReadinessReaderFactory,
    MigrationCompatibility,
    MigrationReader,
    QdrantClientFactory,
    ReadinessStatus,
    format_environment_report,
    run_environment_check,
)
from consumer_finance_rag.lineage.index_operations import (
    CleanupClassification,
    GenerationValidationClassification,
    ManagedGenerationStatus,
    ManagedGenerationValidationReport,
    ManagedGenerationValidationResult,
)

NOW = datetime(2026, 9, 23, tzinfo=UTC)
SCRIPT_PATH = Path("scripts/check_environment.py")


class FakeQdrant:
    def __init__(self) -> None:
        self.closed = False
        self.write_calls = 0

    def get_collections(self) -> object:
        return object()

    def close(self, *_args: object, **_kwargs: object) -> None:
        self.closed = True

    def delete_collection(self, *_args: object, **_kwargs: object) -> None:
        self.write_calls += 1
        raise AssertionError("preflight must not delete collections")

    def upsert(self, *_args: object, **_kwargs: object) -> None:
        self.write_calls += 1
        raise AssertionError("preflight must not upsert points")


class FakeManagedReader:
    def __init__(self, observation: ManagedReadinessObservation) -> None:
        self.observation = observation
        self.deep_calls: list[bool] = []
        self.write_calls = 0

    def observe(self, *, deep: bool) -> ManagedReadinessObservation:
        self.deep_calls.append(deep)
        return self.observation

    def publish(self) -> None:
        self.write_calls += 1
        raise AssertionError("preflight must not publish")


def configured_settings(
    profile: Literal["fixture", "evaluation", "managed"],
    *,
    database_url: str | None = "postgresql+psycopg://app:secret@db/lineage",
) -> Any:
    settings = load_settings("config/config.yaml")
    models = settings.models.model_copy(
        update={
            "generation": settings.models.generation.model_copy(
                update={"model": "test.bedrock-model-v1"}
            )
        }
    )
    return settings.model_copy(
        update={
            "models": models,
            "runtime": RuntimeSettings(profile=profile),
            "database": DatabaseSettings(url=database_url),
            "qdrant": QdrantSettings(url="http://127.0.0.1:6333"),
            "mlflow": MLflowSettings(enabled=False),
        }
    )


def active_status(*, physical: bool = True) -> ManagedGenerationStatus:
    return ManagedGenerationStatus(
        index_state_id=UUID(int=4),
        generation=4,
        status="ACTIVE",
        qdrant_collection="consumer_finance_managed_g000004",
        physical_present=physical,
        ingestion_run_id=UUID(int=40),
        document_count=3,
        chunk_count=6,
        embedding_dimension=384,
        created_at=NOW,
        activated_at=NOW,
        retired_at=None,
        failed_at=None,
        error_message=None,
        restored_from_index_state_id=None,
        restoration_reason=None,
        cleanup_classification=CleanupClassification.ACTIVE_PROTECTED,
        cleanup_reason="protected",
    )


def validation_report(
    *,
    active_valid: bool = True,
    historical_warning: bool = False,
    deep: bool = True,
) -> ManagedGenerationValidationReport:
    generations: list[ManagedGenerationValidationResult] = []
    if historical_warning:
        generations.append(
            ManagedGenerationValidationResult(
                index_state_id=UUID(int=1),
                generation=1,
                status="RETIRED",
                qdrant_collection="consumer_finance_managed_g000001",
                physical_present=True,
                deep=deep,
                valid=False,
                issues=(),
                notes=("HISTORICAL_SCHEMA_INCOMPATIBLE",),
                classification=(
                    GenerationValidationClassification.HISTORICAL_SCHEMA_INCOMPATIBLE
                ),
            )
        )
    generations.append(
        ManagedGenerationValidationResult(
            index_state_id=UUID(int=4),
            generation=4,
            status="ACTIVE",
            qdrant_collection="consumer_finance_managed_g000004",
            physical_present=active_valid,
            deep=deep,
            valid=active_valid,
            issues=() if active_valid else ("ACTIVE Qdrant collection is missing",),
            notes=(),
            classification=(
                GenerationValidationClassification.PASS
                if active_valid
                else GenerationValidationClassification.FAIL
            ),
        )
    )
    return ManagedGenerationValidationReport(
        corpus_key="consumer-finance-managed",
        active_count=1,
        valid=active_valid,
        issues=(),
        generations=tuple(generations),
    )


def observation(
    *,
    active_valid: bool = True,
    historical_warning: bool = False,
) -> ManagedReadinessObservation:
    return ManagedReadinessObservation(
        statuses=(active_status(physical=active_valid),),
        validation=validation_report(
            active_valid=active_valid,
            historical_warning=historical_warning,
        ),
        corpus_identity_matches=True,
        active_membership_count=6,
        active_point_count=6 if active_valid else None,
    )


def compatible_migrations(
    _connection: object,
    _path: Path,
) -> MigrationCompatibility:
    return MigrationCompatibility(("0005_restore_provenance",), "0005_restore_provenance")


def run_managed(
    reader: FakeManagedReader,
    *,
    deep: bool = True,
) -> tuple[EnvironmentReport, FakeQdrant]:
    qdrant = FakeQdrant()

    def reader_factory(*_args: object, **_kwargs: object) -> FakeManagedReader:
        return reader

    report = run_environment_check(
        configured_settings("managed"),
        deep_managed=deep,
        engine_factory=lambda _settings: create_engine("sqlite+pysqlite:///:memory:"),
        qdrant_factory=cast(QdrantClientFactory, lambda _url: qdrant),
        migration_reader=cast(MigrationReader, compatible_migrations),
        managed_reader_factory=cast(ManagedReadinessReaderFactory, reader_factory),
    )
    return report, qdrant


def test_missing_managed_configuration_fails_without_connecting() -> None:
    settings = configured_settings("managed", database_url=None).model_copy(
        update={"qdrant": QdrantSettings(url="")}
    )
    calls: list[str] = []

    report = run_environment_check(
        settings,
        engine_factory=cast(
            EngineFactory,
            lambda _settings: calls.append("postgres"),
        ),
        qdrant_factory=cast(
            QdrantClientFactory,
            lambda _url: calls.append("qdrant"),
        ),
    )

    assert report.status is ReadinessStatus.NOT_READY
    assert calls == []
    assert {check.key for check in report.checks if check.status is CheckStatus.FAIL} >= {
        "qdrant_configuration",
        "lineage_database_configuration",
        "managed_readiness",
    }


def test_missing_required_model_configuration_fails_without_initializing_models() -> None:
    settings = configured_settings("fixture")
    settings = settings.model_copy(
        update={
            "models": settings.models.model_copy(
                update={
                    "embedding": settings.models.embedding.model_copy(
                        update={"model": ""}
                    )
                }
            )
        }
    )

    report = run_environment_check(
        settings,
        qdrant_factory=cast(QdrantClientFactory, lambda _url: FakeQdrant()),
    )

    required = next(check for check in report.checks if check.key == "required_settings")
    assert required.status is CheckStatus.FAIL
    assert report.status is ReadinessStatus.NOT_READY


@pytest.mark.parametrize("profile", ["fixture", "evaluation"])
def test_fixture_and_evaluation_are_postgresql_independent(
    profile: Literal["fixture", "evaluation"],
) -> None:
    qdrant = FakeQdrant()

    def database_must_not_be_used(_settings: object) -> Engine:
        raise AssertionError("fixture/evaluation must not initialize PostgreSQL")

    report = run_environment_check(
        configured_settings(profile, database_url=None),
        engine_factory=cast(EngineFactory, database_must_not_be_used),
        qdrant_factory=cast(QdrantClientFactory, lambda _url: qdrant),
    )

    assert report.status is ReadinessStatus.READY
    assert report.managed is None
    assert qdrant.closed


def test_healthy_managed_smoke_is_ready_and_read_only() -> None:
    reader = FakeManagedReader(observation())

    report, qdrant = run_managed(reader)

    assert report.status is ReadinessStatus.READY
    assert report.ready
    assert reader.deep_calls == [True]
    assert reader.write_calls == qdrant.write_calls == 0
    output = format_environment_report(report)
    assert "ACTIVE validation: PASS" in output
    assert "historical audit: PASS" in output
    assert "PostgreSQL memberships: 6" in output
    assert "Qdrant points: 6" in output


def test_historical_warnings_do_not_block_healthy_active_readiness() -> None:
    report, _ = run_managed(
        FakeManagedReader(observation(historical_warning=True))
    )

    assert report.ready
    assert report.status is ReadinessStatus.READY_WITH_WARNINGS
    assert report.managed is not None
    assert report.managed.validation.overall_status.value == "PASS_WITH_WARNINGS"
    rendered = format_environment_report(report)
    assert "ACTIVE validation: PASS" in rendered
    assert "historical audit: WARNINGS" in rendered


def test_unhealthy_active_blocks_managed_readiness() -> None:
    report, _ = run_managed(FakeManagedReader(observation(active_valid=False)))

    assert not report.ready
    assert report.status is ReadinessStatus.NOT_READY
    active_check = next(
        check for check in report.checks if check.key == "active_validation"
    )
    assert active_check.status is CheckStatus.FAIL


def test_missing_services_and_exception_secrets_are_safely_reported() -> None:
    secret = "postgresql://admin:password@internal/private?api_key=secret"

    def fail_postgres(_settings: object) -> Engine:
        raise ConnectionError(secret)

    def fail_qdrant(_url: str) -> Any:
        raise TimeoutError(secret)

    report = run_environment_check(
        configured_settings("managed"),
        engine_factory=cast(EngineFactory, fail_postgres),
        qdrant_factory=cast(QdrantClientFactory, fail_qdrant),
    )
    rendered = format_environment_report(report)

    assert report.status is ReadinessStatus.NOT_READY
    assert "ConnectionError" in rendered
    assert "TimeoutError" in rendered
    assert "postgresql://" not in rendered
    assert "password" not in rendered
    assert "api_key" not in rendered


def test_basic_preflight_does_not_import_models_or_mutation_paths() -> None:
    source_paths = [Path("src/consumer_finance_rag/environment_check.py"), SCRIPT_PATH]
    forbidden_imports = {
        "consumer_finance_rag.bedrock",
        "consumer_finance_rag.factory",
        "consumer_finance_rag.pipeline",
        "consumer_finance_rag.providers",
    }
    for path in source_paths:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        assert imported.isdisjoint(forbidden_imports)
        for forbidden in (
            ".commit(",
            ".begin(",
            "execute_cleanup",
            "delete_collection",
            "upsert_points",
            "start_run",
            "command.upgrade",
        ):
            assert forbidden not in source


def test_normal_application_startup_has_no_operator_mutations() -> None:
    source = Path("app/streamlit_app.py").read_text(encoding="utf-8")

    for forbidden in (
        "alembic.command",
        "command.upgrade",
        "index_managed",
        "publish(",
        "execute_cleanup",
        "restore_generation",
        "delete_collection",
        "with_mlflow_judges",
        "evaluate_rag",
    ):
        assert forbidden not in source


def test_cli_invalid_configuration_withholds_details(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec = importlib.util.spec_from_file_location("environment_check_cli", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module: Any = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        module,
        "load_settings",
        lambda _path: (_ for _ in ()).throw(
            ValueError("password=secret raw configuration")
        ),
    )

    assert module.main([]) == 2
    output = capsys.readouterr().out
    assert "ValueError" in output
    assert "password" not in output
    assert "secret" not in output
