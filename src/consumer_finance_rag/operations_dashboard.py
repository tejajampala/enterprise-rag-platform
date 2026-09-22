"""Read-only view models and data adapters for the Phase 3 operations dashboard."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol
from urllib.parse import urlsplit, urlunsplit
from uuid import UUID

from consumer_finance_rag.config import MLflowSettings
from consumer_finance_rag.lineage.index_operations import (
    CleanupClassification,
    GenerationCleanupPlan,
    ManagedGenerationStatus,
    ManagedGenerationValidationReport,
)
from consumer_finance_rag.retrieval_quality import (
    RetrievalQualityComparison,
    RetrievalQualityGateResult,
    RetrievalQualityReport,
    RetrievalQualityThresholds,
    compare_quality_reports,
    evaluate_quality_thresholds,
)

TableValue = str | int | float | bool | None
TableRow = dict[str, TableValue]


class ReadOnlyOperationsService(Protocol):
    """Existing read methods used by the dashboard."""

    def status(self) -> tuple[ManagedGenerationStatus, ...]: ...

    def validate(self, *, deep: bool = False) -> ManagedGenerationValidationReport: ...

    def cleanup_plan(self) -> GenerationCleanupPlan: ...


class MLflowExperimentClient(Protocol):
    """Read-only subset of the MLflow client used for availability checks."""

    def get_experiment_by_name(self, name: str) -> Any: ...


@dataclass(frozen=True, slots=True)
class ComponentHealth:
    component: str
    available: bool
    message: str


@dataclass(frozen=True, slots=True)
class OperationsSnapshot:
    """Independent read results so one failed section does not hide another."""

    corpus_key: str
    statuses: tuple[ManagedGenerationStatus, ...]
    validation: ManagedGenerationValidationReport | None
    cleanup_plan: GenerationCleanupPlan | None
    active_point_count: int | None
    status_error: str | None = None
    validation_error: str | None = None
    cleanup_error: str | None = None
    point_count_error: str | None = None

    @property
    def active(self) -> ManagedGenerationStatus | None:
        return next((item for item in self.statuses if item.status == "ACTIVE"), None)


@dataclass(frozen=True, slots=True)
class MLflowDashboardStatus:
    configured: bool
    available: bool
    message: str
    experiment_name: str
    link: str | None = None


@dataclass(frozen=True, slots=True)
class QualityComparisonView:
    baseline: RetrievalQualityReport
    candidate: RetrievalQualityReport
    comparison: RetrievalQualityComparison
    threshold_result: RetrievalQualityGateResult


_CLEANUP_EXPLANATIONS: dict[CleanupClassification, str] = {
    CleanupClassification.ACTIVE_PROTECTED: "Current ACTIVE generation is always protected.",
    CleanupClassification.READY_PROTECTED: "READY generations are protected from cleanup.",
    CleanupClassification.BUILDING_PROTECTED: "BUILDING generations are protected from cleanup.",
    CleanupClassification.RESTORATION_SOURCE_PROTECTED: (
        "Referenced restoration source is protected."
    ),
    CleanupClassification.RETAINED: "Within the configured retired-generation retention set.",
    CleanupClassification.RECENT: "Younger than the configured cleanup minimum age.",
    CleanupClassification.ELIGIBLE_RETIRED: (
        "Retired, outside retention, and old enough for operator-confirmed CLI cleanup."
    ),
    CleanupClassification.ELIGIBLE_FAILED: (
        "Failed generation is eligible for operator-confirmed CLI cleanup."
    ),
    CleanupClassification.PHYSICAL_MISSING: "Physical collection is already absent.",
    CleanupClassification.ORPHAN: "Managed-looking physical collection has no lineage row.",
    CleanupClassification.OTHER_PROTECTED: "Current state is not cleanup-eligible.",
}
_SAFE_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,254}\Z")


def collect_operations_snapshot(
    service: ReadOnlyOperationsService,
    *,
    corpus_key: str,
    point_counter: Callable[[str], int],
    deep_validation: bool = False,
) -> OperationsSnapshot:
    """Call only existing inspection/planning APIs, isolating section failures."""

    statuses: tuple[ManagedGenerationStatus, ...] = ()
    validation: ManagedGenerationValidationReport | None = None
    cleanup: GenerationCleanupPlan | None = None
    point_count: int | None = None
    status_error: str | None = None
    validation_error: str | None = None
    cleanup_error: str | None = None
    point_count_error: str | None = None

    try:
        statuses = service.status()
    except Exception as exc:
        status_error = safe_failure("Generation history", exc)

    try:
        validation = service.validate(deep=deep_validation)
    except Exception as exc:
        validation_error = safe_failure("Generation validation", exc)

    try:
        cleanup = service.cleanup_plan()
    except Exception as exc:
        cleanup_error = safe_failure("Cleanup plan", exc)

    active = next((item for item in statuses if item.status == "ACTIVE"), None)
    if active is not None and active.physical_present:
        try:
            point_count = point_counter(active.qdrant_collection)
        except Exception as exc:
            point_count_error = safe_failure("Physical point count", exc)

    return OperationsSnapshot(
        corpus_key=corpus_key,
        statuses=statuses,
        validation=validation,
        cleanup_plan=cleanup,
        active_point_count=point_count,
        status_error=status_error,
        validation_error=validation_error,
        cleanup_error=cleanup_error,
        point_count_error=point_count_error,
    )


def probe_component(component: str, probe: Callable[[], Any]) -> ComponentHealth:
    """Run one lazy read probe without reflecting connection details on failure."""

    try:
        probe()
    except Exception as exc:
        return ComponentHealth(component, False, safe_failure(component, exc))
    return ComponentHealth(component, True, f"{component} is available.")


def generation_history_rows(
    statuses: Sequence[ManagedGenerationStatus],
) -> tuple[TableRow, ...]:
    """Allowlist generation metadata; never expose stored error/restoration text."""

    return tuple(
        {
            "generation": status.generation,
            "status": "★ ACTIVE" if status.status == "ACTIVE" else status.status,
            "collection": safe_identifier(status.qdrant_collection),
            "document_count": status.document_count,
            "chunk_count": status.chunk_count,
            "physical_present": status.physical_present,
            "created_at": status.created_at.isoformat(),
            "activated_at": _timestamp(status.activated_at),
            "retired_at": _timestamp(status.retired_at),
            "restored_from_index_state_id": _optional_uuid(
                status.restored_from_index_state_id
            ),
            "cleanup_classification": status.cleanup_classification.value,
            "failure": (
                "Recorded (details withheld)" if status.error_message else None
            ),
        }
        for status in statuses
    )


def validation_rows(report: ManagedGenerationValidationReport) -> tuple[TableRow, ...]:
    """Render invariant outcomes without forwarding raw backend exception text."""

    return tuple(
        {
            "generation": result.generation,
            "status": result.status,
            "mode": "deep" if result.deep else "bounded",
            "result": "PASS" if result.valid else "FAIL",
            "physical_present": result.physical_present,
            "issues": _safe_diagnostics(result.issues),
            "notes": _safe_diagnostics(result.notes),
        }
        for result in report.generations
    )


def cleanup_plan_rows(plan: GenerationCleanupPlan) -> tuple[TableRow, ...]:
    """Render planner classifications using fixed explanations, never execution controls."""

    return tuple(
        {
            "generation": candidate.generation,
            "status": candidate.status,
            "collection": safe_identifier(candidate.qdrant_collection),
            "physical_present": candidate.physical_present,
            "classification": candidate.classification.value,
            "protected": not candidate.eligible,
            "reason": _CLEANUP_EXPLANATIONS[candidate.classification],
        }
        for candidate in plan.candidates
    )


def restoration_provenance_rows(
    statuses: Sequence[ManagedGenerationStatus],
) -> tuple[TableRow, ...]:
    """Expose only immutable restoration linkage, not free-form operator reasons."""

    return tuple(
        {
            "generation": status.generation,
            "index_state_id": str(status.index_state_id),
            "restored_from_index_state_id": str(status.restored_from_index_state_id),
            "collection": safe_identifier(status.qdrant_collection),
        }
        for status in statuses
        if status.restored_from_index_state_id is not None
    )


def load_quality_report_json(value: str | bytes) -> RetrievalQualityReport:
    """Validate a versioned safe report; unknown fields are never rendered."""

    return RetrievalQualityReport.model_validate_json(value)


def quality_summary(report: RetrievalQualityReport) -> TableRow:
    return {
        "generation": report.generation,
        "collection": safe_identifier(report.qdrant_collection),
        "case_count": report.case_count,
        "k": report.metrics.k,
        "recall_at_k": report.recall_at_k,
        "mrr": report.mrr,
        "hit_rate": report.hit_rate,
        "missed_case_count": report.missed_case_count,
    }


def compare_quality_for_dashboard(
    baseline: RetrievalQualityReport,
    candidate: RetrievalQualityReport,
    thresholds: RetrievalQualityThresholds,
) -> QualityComparisonView:
    """Delegate all comparison and threshold calculations to retrieval_quality.py."""

    comparison = compare_quality_reports(baseline, candidate)
    return QualityComparisonView(
        baseline=baseline,
        candidate=candidate,
        comparison=comparison,
        threshold_result=evaluate_quality_thresholds(comparison, thresholds),
    )


def comparison_summary(view: QualityComparisonView) -> TableRow:
    comparison = view.comparison
    return {
        "baseline_generation": comparison.baseline_generation,
        "candidate_generation": comparison.candidate_generation,
        "recall_delta": comparison.recall_delta,
        "mrr_delta": comparison.mrr_delta,
        "hit_rate_delta": comparison.hit_rate_delta,
        "improved_case_count": len(comparison.cases_improved),
        "regressed_case_count": len(comparison.cases_regressed),
        "newly_missed_case_count": len(comparison.newly_missed_cases),
        "newly_recovered_case_count": len(comparison.newly_recovered_cases),
        "threshold_result": view.threshold_result.status,
    }


def inspect_mlflow(
    settings: MLflowSettings,
    *,
    client_factory: Callable[[str], MLflowExperimentClient] | None = None,
) -> MLflowDashboardStatus:
    """Check one configured experiment without creating a run or logging data."""

    if not settings.enabled:
        return MLflowDashboardStatus(
            configured=False,
            available=False,
            message="MLflow is disabled by configuration.",
            experiment_name=settings.evaluation_experiment_name,
        )
    try:
        factory = client_factory or _mlflow_client
        experiment = factory(settings.tracking_uri).get_experiment_by_name(
            settings.evaluation_experiment_name
        )
    except Exception as exc:
        return MLflowDashboardStatus(
            configured=True,
            available=False,
            message=safe_failure("MLflow", exc),
            experiment_name=settings.evaluation_experiment_name,
        )
    if experiment is None:
        return MLflowDashboardStatus(
            configured=True,
            available=True,
            message="MLflow is available; the retrieval-quality experiment does not exist yet.",
            experiment_name=settings.evaluation_experiment_name,
        )
    experiment_id = str(getattr(experiment, "experiment_id", ""))
    return MLflowDashboardStatus(
        configured=True,
        available=True,
        message="MLflow and the configured retrieval-quality experiment are available.",
        experiment_name=settings.evaluation_experiment_name,
        link=_mlflow_experiment_link(settings.tracking_uri, experiment_id),
    )


def safe_failure(component: str, error: Exception) -> str:
    """Return a useful status without reflecting exception text or connection details."""

    return f"{component} unavailable ({type(error).__name__})."


def safe_identifier(value: str) -> str:
    """Display only bounded operational identifiers, never arbitrary free-form text."""

    return value if _SAFE_IDENTIFIER.fullmatch(value) else "<withheld-invalid-identifier>"


def _safe_diagnostics(values: Sequence[str]) -> str:
    return " | ".join(_safe_diagnostic(value) for value in values) or "None"


def _safe_diagnostic(value: str) -> str:
    lowered = value.lower()
    if lowered.startswith("collection is not readable:"):
        return "collection is not readable"
    if lowered.startswith("deep collection scan failed:"):
        return "deep collection scan failed"
    if "acl payload is invalid:" in lowered:
        return value.split(":", maxsplit=1)[0] + " is invalid"
    if lowered.startswith("historical configuration is incomplete:"):
        return "historical configuration is incomplete"
    redacted = re.sub(r"[a-z][a-z0-9+.-]*://\S+", "<redacted-url>", value)
    redacted = re.sub(
        r"(?i)\b(password|api[_-]?key|secret|token)=\S+",
        r"\1=***",
        redacted,
    )
    return redacted[:500]


def _mlflow_experiment_link(tracking_uri: str, experiment_id: str) -> str | None:
    parsed = urlsplit(tracking_uri)
    if parsed.scheme not in {"http", "https"} or parsed.username or parsed.password:
        return None
    base = urlunsplit((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", ""))
    if not experiment_id:
        return base
    return f"{base}/#/experiments/{experiment_id}"


def _mlflow_client(tracking_uri: str) -> MLflowExperimentClient:
    from mlflow.tracking import MlflowClient

    return MlflowClient(tracking_uri=tracking_uri)


def _timestamp(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _optional_uuid(value: UUID | None) -> str | None:
    return str(value) if value is not None else None
