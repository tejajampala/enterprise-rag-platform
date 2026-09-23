"""Read-only configuration preflight and selected-profile service readiness."""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol
from urllib.parse import urlsplit

from alembic.config import Config as AlembicConfig
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from qdrant_client import QdrantClient
from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine, make_url
from sqlalchemy.orm import Session, sessionmaker

from consumer_finance_rag.config import AppSettings, DatabaseSettings
from consumer_finance_rag.lineage.database import create_database_engine
from consumer_finance_rag.lineage.index_operations import (
    ManagedGenerationStatus,
    ManagedGenerationValidationReport,
    ManagedIndexOperationsService,
    ManagedValidationStatus,
)
from consumer_finance_rag.lineage.managed_vector_store import (
    ManagedQdrantError,
    managed_generation_collection_name,
)
from consumer_finance_rag.lineage.repository import LineageRepository


class CheckStatus(StrEnum):
    """Severity of one safe operator check."""

    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"


class ReadinessStatus(StrEnum):
    """Selected-profile readiness without conflating warnings with failure."""

    READY = "READY"
    READY_WITH_WARNINGS = "READY_WITH_WARNINGS"
    NOT_READY = "NOT_READY"


@dataclass(frozen=True, slots=True)
class EnvironmentCheck:
    """One allowlisted preflight result with no raw exception or connection detail."""

    key: str
    status: CheckStatus
    message: str


@dataclass(frozen=True, slots=True)
class MigrationCompatibility:
    """Read-only comparison of database revisions with the repository head."""

    current_heads: tuple[str, ...]
    expected_head: str | None

    @property
    def compatible(self) -> bool:
        return self.expected_head is not None and self.current_heads == (
            self.expected_head,
        )


@dataclass(frozen=True, slots=True)
class ManagedReadinessObservation:
    """Existing operations results plus explicit ACTIVE plane counts."""

    statuses: tuple[ManagedGenerationStatus, ...]
    validation: ManagedGenerationValidationReport
    corpus_identity_matches: bool
    active_membership_count: int | None
    active_point_count: int | None

    @property
    def active(self) -> ManagedGenerationStatus | None:
        active = tuple(item for item in self.statuses if item.status == "ACTIVE")
        return active[0] if len(active) == 1 else None


@dataclass(frozen=True, slots=True)
class EnvironmentReport:
    """Immutable preflight/readiness result safe for terminal rendering."""

    profile: str
    checks: tuple[EnvironmentCheck, ...]
    managed: ManagedReadinessObservation | None = None

    @property
    def ready(self) -> bool:
        return all(check.status is not CheckStatus.FAIL for check in self.checks)

    @property
    def status(self) -> ReadinessStatus:
        if not self.ready:
            return ReadinessStatus.NOT_READY
        if any(check.status is CheckStatus.WARNING for check in self.checks):
            return ReadinessStatus.READY_WITH_WARNINGS
        return ReadinessStatus.READY


class ManagedReadinessReader(Protocol):
    """Read-only managed inspection used by the deployment preflight."""

    def observe(self, *, deep: bool) -> ManagedReadinessObservation: ...


class ManagedReadinessReaderFactory(Protocol):
    def __call__(
        self,
        sessions: sessionmaker[Session],
        client: QdrantClient,
        settings: AppSettings,
    ) -> ManagedReadinessReader: ...


class EngineFactory(Protocol):
    def __call__(self, settings: DatabaseSettings) -> Engine: ...


class QdrantClientFactory(Protocol):
    def __call__(self, url: str) -> QdrantClient: ...


class MigrationReader(Protocol):
    def __call__(
        self,
        connection: Connection,
        alembic_config_path: Path,
    ) -> MigrationCompatibility: ...


class _ManagedReadinessAdapter:
    """Adapter over existing operations validation; contains no second validator."""

    def __init__(
        self,
        sessions: sessionmaker[Session],
        client: QdrantClient,
        settings: AppSettings,
    ) -> None:
        self._sessions = sessions
        self._client = client
        self._corpus_key = settings.managed_ingestion.corpus_key
        self._collection_prefix = (
            settings.managed_ingestion.qdrant_collection_prefix
        )
        self._service = ManagedIndexOperationsService(
            session_factory=sessions,
            qdrant_client=client,
            corpus_key=settings.managed_ingestion.corpus_key,
            settings=settings.managed_index,
        )

    def observe(self, *, deep: bool) -> ManagedReadinessObservation:
        statuses = self._service.status()
        validation = self._service.validate(deep=deep)
        active = tuple(item for item in statuses if item.status == "ACTIVE")
        with self._sessions() as session:
            repository = LineageRepository(session)
            corpus = repository.get_corpus_by_key(self._corpus_key)
            corpus_identity_matches = (
                corpus is not None
                and corpus.qdrant_collection_prefix == self._collection_prefix
            )
        if len(active) != 1:
            return ManagedReadinessObservation(
                statuses,
                validation,
                corpus_identity_matches,
                None,
                None,
            )
        selected = active[0]
        with self._sessions() as session:
            membership_count = LineageRepository(session).count_index_state_chunks(
                selected.index_state_id
            )
        point_count = (
            int(
                self._client.count(
                    collection_name=selected.qdrant_collection,
                    exact=True,
                ).count
            )
            if selected.physical_present
            else None
        )
        return ManagedReadinessObservation(
            statuses=statuses,
            validation=validation,
            corpus_identity_matches=corpus_identity_matches,
            active_membership_count=membership_count,
            active_point_count=point_count,
        )


def read_migration_compatibility(
    connection: Connection,
    alembic_config_path: Path,
) -> MigrationCompatibility:
    """Read database revisions and repository head without running migrations."""

    alembic_config = AlembicConfig(str(alembic_config_path))
    expected_head = ScriptDirectory.from_config(alembic_config).get_current_head()
    current_heads = tuple(MigrationContext.configure(connection).get_current_heads())
    return MigrationCompatibility(current_heads, expected_head)


def create_qdrant_read_client(url: str) -> QdrantClient:
    """Construct the normal REST client lazily for preflight reads."""

    return QdrantClient(url=url)


def run_environment_check(
    settings: AppSettings,
    *,
    deep_managed: bool = False,
    alembic_config_path: Path = Path("alembic.ini"),
    engine_factory: EngineFactory = create_database_engine,
    qdrant_factory: QdrantClientFactory = create_qdrant_read_client,
    migration_reader: MigrationReader = read_migration_compatibility,
    managed_reader_factory: ManagedReadinessReaderFactory = _ManagedReadinessAdapter,
) -> EnvironmentReport:
    """Validate configuration and selected-profile readiness using reads only."""

    checks: list[EnvironmentCheck] = [
        EnvironmentCheck(
            "runtime_profile",
            CheckStatus.PASS,
            f"Selected runtime profile: {settings.runtime.profile}.",
        )
    ]
    required_settings_valid = bool(
        settings.models.embedding.model.strip()
        and settings.models.reranker.model.strip()
        and settings.aws.region.strip()
        and _selected_collection_identity(settings)
    )
    checks.append(
        EnvironmentCheck(
            "required_settings",
            CheckStatus.PASS if required_settings_valid else CheckStatus.FAIL,
            (
                "Required model, region, and selected collection settings are present."
                if required_settings_valid
                else "Required model, region, or selected collection settings are missing."
            ),
        )
    )
    generation_configured = bool(
        settings.models.generation.model
        and settings.models.generation.model.strip()
    )
    checks.append(
        EnvironmentCheck(
            "bedrock_generation_configuration",
            CheckStatus.PASS if generation_configured else CheckStatus.WARNING,
            (
                "Optional live-query Bedrock model is configured."
                if generation_configured
                else "Live-query Bedrock model is not configured; read-only smoke is available."
            ),
        )
    )
    qdrant_configured = _valid_http_endpoint(settings.qdrant.url)
    checks.append(
        EnvironmentCheck(
            "qdrant_configuration",
            CheckStatus.PASS if qdrant_configured else CheckStatus.FAIL,
            (
                "Qdrant endpoint configuration is valid."
                if qdrant_configured
                else "Qdrant endpoint configuration is missing or invalid."
            ),
        )
    )
    checks.append(_mlflow_configuration_check(settings))

    managed_profile = settings.runtime.profile == "managed"
    database_configured = False
    corpus_configured = False
    if managed_profile:
        database_configured = _is_postgresql_configuration(settings.database)
        checks.append(
            EnvironmentCheck(
                "lineage_database_configuration",
                CheckStatus.PASS if database_configured else CheckStatus.FAIL,
                (
                    "Managed lineage PostgreSQL configuration is present."
                    if database_configured
                    else "Managed mode requires a PostgreSQL RAG__DATABASE__URL."
                ),
            )
        )
        corpus_configured = _valid_managed_corpus_identity(settings)
        checks.append(
            EnvironmentCheck(
                "managed_corpus_identity",
                CheckStatus.PASS if corpus_configured else CheckStatus.FAIL,
                (
                    "Managed corpus identity is valid."
                    if corpus_configured
                    else "Managed corpus key or collection prefix is invalid."
                ),
            )
        )
    else:
        checks.append(
            EnvironmentCheck(
                "lineage_database_configuration",
                CheckStatus.PASS,
                "Lineage PostgreSQL is not required for this runtime profile.",
            )
        )

    client: QdrantClient | None = None
    qdrant_available = False
    if qdrant_configured:
        try:
            client = qdrant_factory(settings.qdrant.url)
            client.get_collections()
            qdrant_available = True
            checks.append(
                EnvironmentCheck(
                    "qdrant_connectivity",
                    CheckStatus.PASS,
                    "Qdrant is reachable.",
                )
            )
        except Exception as exc:
            checks.append(_safe_failure("qdrant_connectivity", "Qdrant", exc))

    engine: Engine | None = None
    postgres_available = False
    migrations_compatible = False
    managed: ManagedReadinessObservation | None = None
    try:
        if managed_profile and database_configured:
            try:
                engine = engine_factory(settings.database)
                with engine.connect() as connection:
                    connection.execute(text("SELECT 1"))
                    postgres_available = True
                    checks.append(
                        EnvironmentCheck(
                            "postgresql_connectivity",
                            CheckStatus.PASS,
                            "Lineage PostgreSQL is reachable.",
                        )
                    )
                    migration = migration_reader(connection, alembic_config_path)
                migrations_compatible = migration.compatible
                checks.append(
                    EnvironmentCheck(
                        "alembic_compatibility",
                        (
                            CheckStatus.PASS
                            if migrations_compatible
                            else CheckStatus.FAIL
                        ),
                        (
                            "Database migration revision matches repository head."
                            if migrations_compatible
                            else "Database migration revision does not match repository head."
                        ),
                    )
                )
            except Exception as exc:
                if not postgres_available:
                    checks.append(
                        _safe_failure(
                            "postgresql_connectivity",
                            "Lineage PostgreSQL",
                            exc,
                        )
                    )
                else:
                    checks.append(
                        _safe_failure(
                            "alembic_compatibility",
                            "Alembic compatibility",
                            exc,
                        )
                    )

        if (
            managed_profile
            and corpus_configured
            and postgres_available
            and migrations_compatible
            and qdrant_available
            and engine is not None
            and client is not None
        ):
            sessions = sessionmaker(bind=engine, class_=Session, expire_on_commit=False)
            try:
                reader = managed_reader_factory(sessions, client, settings)
                managed = reader.observe(deep=deep_managed)
            except Exception as exc:
                checks.append(
                    _safe_failure("managed_readiness", "Managed readiness", exc)
                )
            else:
                checks.extend(_managed_checks(managed, deep=deep_managed))
        elif managed_profile:
            checks.append(
                EnvironmentCheck(
                    "managed_readiness",
                    CheckStatus.FAIL,
                    "Managed readiness requires PostgreSQL, migrations, and Qdrant.",
                )
            )
    finally:
        if client is not None:
            with suppress(Exception):
                client.close()
        if engine is not None:
            with suppress(Exception):
                engine.dispose()

    return EnvironmentReport(
        profile=settings.runtime.profile,
        checks=tuple(_deduplicate_checks(checks)),
        managed=managed,
    )


def format_environment_report(report: EnvironmentReport) -> str:
    """Render only allowlisted fields; never render URLs or exception messages."""

    lines = ["Environment preflight (READ ONLY)", "", f"Profile: {report.profile}"]
    lines.extend(
        f"[{check.status.value}] {check.message}" for check in report.checks
    )
    if report.managed is not None and report.managed.active is not None:
        active = report.managed.active
        lines.extend(
            [
                "",
                "Managed ACTIVE:",
                f"  generation: g{active.generation:06d}",
                f"  collection: {_safe_identifier(active.qdrant_collection)}",
                f"  intended chunks: {active.chunk_count}",
                f"  PostgreSQL memberships: {report.managed.active_membership_count}",
                f"  Qdrant points: {report.managed.active_point_count}",
                (
                    "  ACTIVE validation: "
                    f"{report.managed.validation.active_health.value}"
                ),
                f"  historical audit: {report.managed.validation.historical_audit}",
                f"  validation overall: {report.managed.validation.overall_status.value}",
            ]
        )
    lines.extend(["", f"Readiness: {report.status.value}"])
    return "\n".join(lines)


def _managed_checks(
    observation: ManagedReadinessObservation,
    *,
    deep: bool,
) -> tuple[EnvironmentCheck, ...]:
    validation = observation.validation
    active = observation.active
    active_result = next(
        (item for item in validation.generations if item.status == "ACTIVE"),
        None,
    )
    checks: list[EnvironmentCheck] = []
    checks.append(
        EnvironmentCheck(
            "managed_corpus_registration",
            (
                CheckStatus.PASS
                if observation.corpus_identity_matches
                else CheckStatus.FAIL
            ),
            (
                "Configured managed corpus identity matches PostgreSQL."
                if observation.corpus_identity_matches
                else "Configured managed corpus identity does not match PostgreSQL."
            ),
        )
    )
    if active is None:
        checks.append(
            EnvironmentCheck(
                "active_generation",
                CheckStatus.FAIL,
                "Exactly one ACTIVE managed generation is required.",
            )
        )
    else:
        checks.extend(
            [
                EnvironmentCheck(
                    "active_generation",
                    CheckStatus.PASS,
                    f"ACTIVE generation is g{active.generation:06d}.",
                ),
                EnvironmentCheck(
                    "active_collection",
                    CheckStatus.PASS if active.physical_present else CheckStatus.FAIL,
                    (
                        "ACTIVE Qdrant collection exists."
                        if active.physical_present
                        else "ACTIVE Qdrant collection is missing."
                    ),
                ),
            ]
        )
    checks.append(
        EnvironmentCheck(
            "active_validation",
            (
                CheckStatus.PASS
                if validation.active_health is ManagedValidationStatus.PASS
                else CheckStatus.FAIL
            ),
            f"ACTIVE validation is {validation.active_health.value}.",
        )
    )
    checks.append(
        EnvironmentCheck(
            "historical_audit",
            (
                CheckStatus.WARNING
                if validation.historical_findings
                else CheckStatus.PASS
            ),
            f"Historical audit is {validation.historical_audit}.",
        )
    )
    if active is not None:
        membership_matches = observation.active_membership_count == active.chunk_count
        point_count_matches = observation.active_point_count == active.chunk_count
        checks.extend(
            [
                EnvironmentCheck(
                    "active_membership_count",
                    CheckStatus.PASS if membership_matches else CheckStatus.FAIL,
                    (
                        "ACTIVE PostgreSQL membership count matches intended chunks."
                        if membership_matches
                        else "ACTIVE PostgreSQL membership count is inconsistent."
                    ),
                ),
                EnvironmentCheck(
                    "active_point_count",
                    CheckStatus.PASS if point_count_matches else CheckStatus.FAIL,
                    (
                        "ACTIVE Qdrant point count matches intended chunks."
                        if point_count_matches
                        else "ACTIVE Qdrant point count is inconsistent."
                    ),
                ),
            ]
        )
    if deep:
        payload_compatible = active_result is not None and active_result.valid
        checks.append(
            EnvironmentCheck(
                "active_payload_compatibility",
                CheckStatus.PASS if payload_compatible else CheckStatus.FAIL,
                (
                    "ACTIVE payload compatibility passed deep validation."
                    if payload_compatible
                    else "ACTIVE payload compatibility failed or could not be established."
                ),
            )
        )
    else:
        checks.append(
            EnvironmentCheck(
                "active_payload_compatibility",
                CheckStatus.WARNING,
                "ACTIVE payload scan was not requested; use --managed-smoke.",
            )
        )
    return tuple(checks)


def _mlflow_configuration_check(settings: AppSettings) -> EnvironmentCheck:
    if not settings.mlflow.enabled:
        return EnvironmentCheck(
            "mlflow_configuration",
            CheckStatus.PASS,
            "Optional MLflow tracking is disabled.",
        )
    configured = bool(
        settings.mlflow.tracking_uri.strip()
        and settings.mlflow.experiment_name.strip()
    )
    return EnvironmentCheck(
        "mlflow_configuration",
        CheckStatus.PASS if configured else CheckStatus.WARNING,
        (
            "Optional MLflow configuration is present."
            if configured
            else "Optional MLflow configuration is incomplete."
        ),
    )


def _is_postgresql_configuration(settings: DatabaseSettings) -> bool:
    if settings.url is None or not settings.url.strip():
        return False
    try:
        return make_url(settings.url).get_backend_name() == "postgresql"
    except Exception:
        return False


def _valid_http_endpoint(value: str) -> bool:
    try:
        parsed = urlsplit(value)
    except ValueError:
        return False
    return parsed.scheme in {"http", "https"} and bool(parsed.hostname)


def _valid_managed_corpus_identity(settings: AppSettings) -> bool:
    if not settings.managed_ingestion.corpus_key.strip():
        return False
    try:
        managed_generation_collection_name(
            settings.managed_ingestion.qdrant_collection_prefix,
            1,
        )
    except (ManagedQdrantError, ValueError):
        return False
    return True


def _selected_collection_identity(settings: AppSettings) -> bool:
    if settings.runtime.profile == "fixture":
        return bool(settings.qdrant.collection_name.strip())
    if settings.runtime.profile == "evaluation":
        return bool(settings.evaluation.qdrant_collection_name.strip())
    return bool(settings.managed_ingestion.qdrant_collection_prefix.strip())


def _safe_failure(key: str, component: str, exc: Exception) -> EnvironmentCheck:
    return EnvironmentCheck(
        key,
        CheckStatus.FAIL,
        f"{component} is unavailable ({type(exc).__name__}); details withheld.",
    )


def _safe_identifier(value: str) -> str:
    if not value or len(value) > 255:
        return "<withheld-invalid-identifier>"
    if not all(character.isalnum() or character in "_.:-" for character in value):
        return "<withheld-invalid-identifier>"
    return value


def _deduplicate_checks(checks: list[EnvironmentCheck]) -> list[EnvironmentCheck]:
    """Keep the latest result when a failed probe refines an earlier phase."""

    positions: dict[str, int] = {}
    deduplicated: list[EnvironmentCheck] = []
    for check in checks:
        position = positions.get(check.key)
        if position is None:
            positions[check.key] = len(deduplicated)
            deduplicated.append(check)
        else:
            deduplicated[position] = check
    return deduplicated
