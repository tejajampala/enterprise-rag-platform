"""Read-only, baseline-driven managed RAG drift and regression assessment."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from enum import StrEnum
from pathlib import Path
from statistics import fmean
from typing import Any, Final, Literal, cast
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from consumer_finance_rag.config import DriftMonitoringSettings
from consumer_finance_rag.evaluation.managed_rag import ManagedRAGEvaluationReport
from consumer_finance_rag.evaluation.tracking import EvaluationRun
from consumer_finance_rag.lineage.index_operations import ManagedGenerationValidationResult
from consumer_finance_rag.lineage.managed_retrieval import ManagedRetrievalRuntime
from consumer_finance_rag.lineage.models import (
    Corpus,
    DocumentVersion,
    IndexState,
    IndexStateChunk,
    IngestionRun,
    LineageChunk,
    LineageDocument,
)
from consumer_finance_rag.retrieval_quality import (
    RetrievalQualityComparison,
    RetrievalQualityGateResult,
    RetrievalQualityReport,
    RetrievalQualityThresholds,
    RetrievalRuntimeConfiguration,
    compare_quality_reports,
    evaluate_quality_thresholds,
)

REPORT_FORMAT_VERSION: Final = "managed-drift-assessment-v1"
TELEMETRY_WINDOW_FORMAT_VERSION: Final = "operational-telemetry-window-v1"


class DriftFindingCategory(StrEnum):
    STRUCTURAL_FAILURE = "STRUCTURAL_FAILURE"
    GOVERNANCE_FAILURE = "GOVERNANCE_FAILURE"
    QUALITY_REGRESSION = "QUALITY_REGRESSION"
    DISTRIBUTION_WARNING = "DISTRIBUTION_WARNING"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    NO_REGRESSION_DETECTED = "NO_REGRESSION_DETECTED"


class AssessmentStatus(StrEnum):
    AVAILABLE = "AVAILABLE"
    NOT_REQUESTED = "NOT_REQUESTED"
    UNAVAILABLE = "UNAVAILABLE"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    INCOMPATIBLE = "INCOMPATIBLE"
    FAILED = "FAILED"


class _ImmutableModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class GenerationIdentity(_ImmutableModel):
    generation: int = Field(ge=1)
    index_state_id: UUID
    status: str
    qdrant_collection: str
    snapshot_hash: str
    index_configuration_fingerprint: str | None
    embedding_fingerprint: str
    chunking_fingerprint: str
    bm25_state_hash: str | None
    bm25_corpus_hash: str | None
    document_count: int = Field(ge=0)
    chunk_count: int = Field(ge=0)
    ingestion_run_id: UUID
    authoritative_snapshot: bool


class EvaluationIdentity(_ImmutableModel):
    dataset_name: str
    dataset_version: str
    dataset_format_version: str
    dataset_privacy_classification: str
    embedding_model: str
    reranker_model: str
    generation_model: str | None = None
    judge_model: str | None = None


class SourceFreshnessSummary(_ImmutableModel):
    status: Literal["AVAILABLE", "UNAVAILABLE"]
    timestamped_document_count: int = Field(ge=0)
    document_count: int = Field(ge=0)
    newest_source_timestamp: datetime | None = None
    oldest_source_timestamp: datetime | None = None
    oldest_source_age_days: float | None = Field(default=None, ge=0)
    reason: str | None = None


class SourceChangeSummary(_ImmutableModel):
    new_count: int = Field(ge=0)
    changed_count: int = Field(ge=0)
    unchanged_count: int = Field(ge=0)
    deleted_count: int = Field(ge=0)
    content_changed_count: int = Field(ge=0)
    metadata_only_changed_count: int = Field(ge=0)
    governance_metadata_changed_count: int = Field(ge=0)
    unassessed_missing_count: int = Field(ge=0)
    deletion_detection_performed: bool
    baseline_category_distribution: dict[str, int]
    candidate_category_distribution: dict[str, int]
    category_count_deltas: dict[str, int]
    baseline_freshness: SourceFreshnessSummary
    candidate_freshness: SourceFreshnessSummary


class IndexCompatibilitySummary(_ImmutableModel):
    embedding_fingerprint_match: bool
    chunking_fingerprint_match: bool
    bm25_configuration_match: bool | None
    bm25_state_reused: bool
    pii_detector_version_match: bool | None
    payload_schema_version_match: bool | None
    point_id_format_version_match: bool | None
    configuration_fingerprint_match: bool
    baseline_schema_supported: bool
    candidate_schema_supported: bool
    comparison_supported: bool
    incompatibilities: tuple[str, ...]


class PhysicalValidationSummary(_ImmutableModel):
    generation: int = Field(ge=1)
    status: str
    validation_outcome: Literal["PASS", "WARNING", "FAIL"]
    classification: str
    physical_present: bool
    deep_validation: bool
    issue_count: int = Field(ge=0)
    note_count: int = Field(ge=0)
    membership_count: int = Field(ge=0)
    physical_point_count: int | None = Field(default=None, ge=0)
    governance_failure_detected: bool


class StructuralAssessment(_ImmutableModel):
    source_changes: SourceChangeSummary
    index_compatibility: IndexCompatibilitySummary
    baseline_validation: PhysicalValidationSummary
    candidate_validation: PhysicalValidationSummary
    document_count_delta: int
    intended_chunk_count_delta: int
    membership_count_delta: int
    physical_point_count_delta: int | None


class QualityMetricSnapshot(_ImmutableModel):
    recall_at_k: float = Field(ge=0, le=1)
    mrr: float = Field(ge=0, le=1)
    hit_rate: float = Field(ge=0, le=1)
    missed_case_count: int = Field(ge=0)


class RetrievalRegressionAssessment(_ImmutableModel):
    status: AssessmentStatus
    reason: str | None = None
    baseline: QualityMetricSnapshot | None = None
    candidate: QualityMetricSnapshot | None = None
    comparison: RetrievalQualityComparison | None = None
    threshold_result: RetrievalQualityGateResult | None = None
    improved_case_ids: tuple[str, ...] = ()
    regressed_case_ids: tuple[str, ...] = ()
    newly_missed_case_ids: tuple[str, ...] = ()
    newly_missed_expected_identities: tuple[str, ...] = ()
    acl_negative_case_count: int = Field(default=0, ge=0)
    acl_negative_passed_count: int = Field(default=0, ge=0)
    acl_negative_failed_case_ids: tuple[str, ...] = ()


class AnswerMetricDelta(_ImmutableModel):
    backend: Literal["mlflow_genai", "ragas"]
    metric: str
    baseline: float
    candidate: float
    delta: float


class AnswerRegressionAssessment(_ImmutableModel):
    status: AssessmentStatus
    reason: str | None = None
    baseline_generation_status: str | None = None
    candidate_generation_status: str | None = None
    baseline_mlflow_status: str | None = None
    candidate_mlflow_status: str | None = None
    baseline_ragas_status: str | None = None
    candidate_ragas_status: str | None = None
    metric_deltas: tuple[AnswerMetricDelta, ...] = ()
    violations: tuple[str, ...] = ()


class OperationalTelemetrySample(_ImmutableModel):
    timestamp: datetime
    generation: int | None = Field(default=None, ge=1)
    success: bool
    total_latency_ms: float = Field(ge=0)
    dense_latency_ms: float | None = Field(default=None, ge=0)
    sparse_latency_ms: float | None = Field(default=None, ge=0)
    fusion_latency_ms: float | None = Field(default=None, ge=0)
    reranker_latency_ms: float | None = Field(default=None, ge=0)
    pii_latency_ms: float | None = Field(default=None, ge=0)
    dense_candidate_count: int = Field(ge=0)
    sparse_candidate_count: int = Field(ge=0)
    fused_candidate_count: int = Field(ge=0)
    final_context_count: int = Field(ge=0)
    redacted_context_count: int = Field(ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)


class OperationalTelemetryWindow(_ImmutableModel):
    format_version: Literal["operational-telemetry-window-v1"] = (
        TELEMETRY_WINDOW_FORMAT_VERSION
    )
    window_start: datetime
    window_end: datetime
    samples: tuple[OperationalTelemetrySample, ...]

    @model_validator(mode="after")
    def valid_window(self) -> OperationalTelemetryWindow:
        if self.window_start.tzinfo is None or self.window_end.tzinfo is None:
            raise ValueError("telemetry window timestamps must be timezone-aware")
        if self.window_end <= self.window_start:
            raise ValueError("telemetry window end must follow start")
        if any(
            sample.timestamp.tzinfo is None
            or sample.timestamp < self.window_start
            or sample.timestamp >= self.window_end
            for sample in self.samples
        ):
            raise ValueError("telemetry samples must fall inside the declared window")
        return self


class OperationalAggregate(_ImmutableModel):
    request_count: int = Field(ge=0)
    failure_count: int = Field(ge=0)
    failure_rate: float = Field(ge=0, le=1)
    mean_total_latency_ms: float = Field(ge=0)
    mean_stage_latencies_ms: dict[str, float | None]
    mean_candidate_counts: dict[str, float]
    mean_redacted_context_count: float = Field(ge=0)
    mean_token_usage: dict[str, float | None]
    generation_distribution: dict[str, int]


class OperationalTrendAssessment(_ImmutableModel):
    status: AssessmentStatus
    reason: str | None = None
    baseline: OperationalAggregate | None = None
    candidate: OperationalAggregate | None = None
    failure_rate_delta: float | None = None
    mean_total_latency_delta_ms: float | None = None
    mean_total_latency_increase_ratio: float | None = None
    warnings: tuple[str, ...] = ()


class DriftFinding(_ImmutableModel):
    category: DriftFindingCategory
    code: str
    summary: str
    operator_response: str


class DriftThresholds(_ImmutableModel):
    max_recall_drop: float = Field(ge=0, le=1)
    max_mrr_drop: float = Field(ge=0, le=1)
    max_answer_metric_drop: float = Field(ge=0)
    minimum_operational_samples: int = Field(ge=2)
    max_failure_rate_increase: float = Field(ge=0, le=1)
    max_mean_latency_increase_ratio: float = Field(ge=0)

    @classmethod
    def from_settings(cls, settings: DriftMonitoringSettings) -> DriftThresholds:
        return cls.model_validate(settings.model_dump())


class ManagedDriftAssessmentReport(_ImmutableModel):
    format_version: Literal["managed-drift-assessment-v1"] = REPORT_FORMAT_VERSION
    generated_at: datetime
    corpus_key: str
    baseline: GenerationIdentity
    candidate: GenerationIdentity
    evaluation: EvaluationIdentity
    thresholds: DriftThresholds
    structural: StructuralAssessment
    retrieval: RetrievalRegressionAssessment
    answer_quality: AnswerRegressionAssessment
    operational_trends: OperationalTrendAssessment
    findings: tuple[DriftFinding, ...]

    @model_validator(mode="after")
    def identities_are_distinct_or_explicitly_identical(
        self,
    ) -> ManagedDriftAssessmentReport:
        if self.generated_at.tzinfo is None:
            raise ValueError("generated_at must be timezone-aware")
        return self


@dataclass(frozen=True, slots=True)
class SourceDocumentSnapshot:
    source_key: str
    content_hash: str
    metadata_hash: str
    category: str
    source_version: str
    access_policy_fingerprint: str


@dataclass(frozen=True, slots=True)
class GenerationSnapshot:
    identity: GenerationIdentity
    index_configuration: Mapping[str, Any] | None
    documents: tuple[SourceDocumentSnapshot, ...]
    membership_count: int


def load_generation_snapshot(
    session: Session,
    *,
    corpus_key: str,
    generation: int,
) -> GenerationSnapshot:
    """Detach one immutable generation snapshot without changing lineage state."""

    corpus = session.scalar(select(Corpus).where(Corpus.corpus_key == corpus_key))
    if corpus is None:
        raise ValueError(f"Managed corpus {corpus_key!r} does not exist")
    state = session.scalar(
        select(IndexState).where(
            IndexState.corpus_id == corpus.corpus_id,
            IndexState.generation == generation,
        )
    )
    if state is None:
        raise ValueError(f"Managed generation g{generation:06d} does not exist")
    run = session.get(IngestionRun, state.ingestion_run_id)
    if run is None:
        raise ValueError("Generation ingestion run is missing")
    rows = session.execute(
        select(
            LineageDocument.source_key,
            DocumentVersion.content_hash,
            DocumentVersion.metadata_hash,
            DocumentVersion.category,
            DocumentVersion.source_version,
            DocumentVersion.access_visibility,
            DocumentVersion.allowed_groups,
        )
        .select_from(IndexStateChunk)
        .join(LineageChunk, IndexStateChunk.chunk_pk == LineageChunk.chunk_pk)
        .join(
            DocumentVersion,
            LineageChunk.document_version_id == DocumentVersion.document_version_id,
        )
        .join(
            LineageDocument,
            DocumentVersion.document_pk == LineageDocument.document_pk,
        )
        .where(IndexStateChunk.index_state_id == state.index_state_id)
        .distinct()
        .order_by(LineageDocument.source_key)
    ).all()
    documents = tuple(
        SourceDocumentSnapshot(
            source_key=str(row.source_key),
            content_hash=str(row.content_hash),
            metadata_hash=str(row.metadata_hash),
            category=str(row.category),
            source_version=str(row.source_version),
            access_policy_fingerprint=_access_policy_fingerprint(
                str(row.access_visibility),
                tuple(str(item) for item in row.allowed_groups),
            ),
        )
        for row in rows
    )
    if len({item.source_key for item in documents}) != len(documents):
        raise ValueError("Generation membership contains multiple versions of one source")
    membership_count = int(
        session.scalar(
            select(func.count())
            .select_from(IndexStateChunk)
            .where(IndexStateChunk.index_state_id == state.index_state_id)
        )
        or 0
    )
    return GenerationSnapshot(
        identity=GenerationIdentity(
            generation=state.generation,
            index_state_id=state.index_state_id,
            status=state.status,
            qdrant_collection=state.qdrant_collection,
            snapshot_hash=state.snapshot_hash,
            index_configuration_fingerprint=state.index_configuration_fingerprint,
            embedding_fingerprint=state.embedding_fingerprint,
            chunking_fingerprint=state.chunking_fingerprint,
            bm25_state_hash=state.bm25_state_hash,
            bm25_corpus_hash=state.bm25_corpus_hash,
            document_count=state.document_count,
            chunk_count=state.chunk_count,
            ingestion_run_id=state.ingestion_run_id,
            authoritative_snapshot=run.authoritative_snapshot,
        ),
        index_configuration=(
            dict(state.index_configuration)
            if isinstance(state.index_configuration, Mapping)
            else None
        ),
        documents=documents,
        membership_count=membership_count,
    )


def summarize_physical_validation(
    validation: ManagedGenerationValidationResult,
    *,
    membership_count: int,
    physical_point_count: int | None,
) -> PhysicalValidationSummary:
    diagnostics = tuple(value.lower() for value in (*validation.issues, *validation.notes))
    governance_failure = any("acl" in value or "pii" in value for value in diagnostics)
    return PhysicalValidationSummary(
        generation=validation.generation,
        status=validation.status,
        validation_outcome=cast(
            Literal["PASS", "WARNING", "FAIL"], validation.outcome
        ),
        classification=validation.classification.value,
        physical_present=validation.physical_present,
        deep_validation=validation.deep,
        issue_count=len(validation.issues),
        note_count=len(validation.notes),
        membership_count=membership_count,
        physical_point_count=physical_point_count,
        governance_failure_detected=governance_failure,
    )


def assess_structural_drift(
    baseline: GenerationSnapshot,
    candidate: GenerationSnapshot,
    *,
    baseline_validation: PhysicalValidationSummary,
    candidate_validation: PhysicalValidationSummary,
    assessed_at: datetime | None = None,
) -> StructuralAssessment:
    timestamp = assessed_at or datetime.now(UTC)
    if timestamp.tzinfo is None:
        raise ValueError("assessed_at must be timezone-aware")
    source_changes = _source_changes(baseline, candidate, assessed_at=timestamp)
    compatibility = _index_compatibility(baseline, candidate)
    physical_delta = (
        candidate_validation.physical_point_count
        - baseline_validation.physical_point_count
        if baseline_validation.physical_point_count is not None
        and candidate_validation.physical_point_count is not None
        else None
    )
    return StructuralAssessment(
        source_changes=source_changes,
        index_compatibility=compatibility,
        baseline_validation=baseline_validation,
        candidate_validation=candidate_validation,
        document_count_delta=(
            candidate.identity.document_count - baseline.identity.document_count
        ),
        intended_chunk_count_delta=(
            candidate.identity.chunk_count - baseline.identity.chunk_count
        ),
        membership_count_delta=(
            candidate.membership_count - baseline.membership_count
        ),
        physical_point_count_delta=physical_delta,
    )


def assess_retrieval_regression(
    baseline: RetrievalQualityReport | None,
    candidate: RetrievalQualityReport | None,
    *,
    thresholds: DriftThresholds,
    comparison_supported: bool,
) -> RetrievalRegressionAssessment:
    if baseline is None or candidate is None:
        return RetrievalRegressionAssessment(
            status=AssessmentStatus.NOT_REQUESTED,
            reason="Comparable retrieval evaluation reports were not supplied.",
        )
    if not comparison_supported:
        return RetrievalRegressionAssessment(
            status=AssessmentStatus.INCOMPATIBLE,
            reason="Generation schemas are not compatible for quality comparison.",
        )
    try:
        comparison = compare_quality_reports(baseline, candidate)
        gate = evaluate_quality_thresholds(
            comparison,
            RetrievalQualityThresholds(
                max_recall_drop=thresholds.max_recall_drop,
                max_mrr_drop=thresholds.max_mrr_drop,
            ),
        )
    except ValueError as exc:
        return RetrievalRegressionAssessment(
            status=AssessmentStatus.INCOMPATIBLE,
            reason=f"Retrieval reports are incompatible ({type(exc).__name__}).",
        )
    candidate_cases = {item.case_id: item for item in candidate.cases}
    newly_missed_identities = sorted(
        {
            expected_id
            for case_id in comparison.newly_missed_cases
            for expected_id in candidate_cases[case_id].expected_ids
        }
    )
    negative = tuple(item for item in candidate.cases if not item.expected_access)
    negative_failures = tuple(
        sorted(item.case_id for item in negative if not item.access_expectation_met)
    )
    return RetrievalRegressionAssessment(
        status=AssessmentStatus.AVAILABLE,
        baseline=_quality_snapshot(baseline),
        candidate=_quality_snapshot(candidate),
        comparison=comparison,
        threshold_result=gate,
        improved_case_ids=comparison.cases_improved,
        regressed_case_ids=comparison.cases_regressed,
        newly_missed_case_ids=comparison.newly_missed_cases,
        newly_missed_expected_identities=tuple(newly_missed_identities),
        acl_negative_case_count=len(negative),
        acl_negative_passed_count=len(negative) - len(negative_failures),
        acl_negative_failed_case_ids=negative_failures,
    )


def retrieval_report_from_managed_evaluation(
    report: ManagedRAGEvaluationReport,
    runtime: ManagedRetrievalRuntime,
) -> RetrievalQualityReport:
    """Project the existing managed evaluation result into the shared quality report type."""

    if report.generation != runtime.generation or report.index_state_id != runtime.index_state_id:
        raise ValueError("Managed evaluation report does not match its pinned runtime")
    service = runtime.retrieval_service
    settings = service.settings
    return RetrievalQualityReport(
        generated_at=report.generated_at,
        generation=report.generation,
        index_state_id=report.index_state_id,
        qdrant_collection=report.qdrant_collection,
        snapshot_hash=report.snapshot_hash,
        embedding_fingerprint=report.embedding_fingerprint,
        chunking_fingerprint=report.chunking_fingerprint,
        index_configuration_fingerprint=report.index_configuration_fingerprint,
        retrieval_settings=RetrievalRuntimeConfiguration(
            dense_candidates=settings.dense_candidates,
            sparse_candidates=settings.sparse_candidates,
            fused_candidates=settings.fused_candidates,
            final_contexts=settings.final_contexts,
            rrf_k=settings.rrf_k,
            hnsw_ef=settings.hnsw_ef,
            reranker_model=service.reranker.model_name,
            acl_filtering_enabled=service.access_control_enabled,
        ),
        case_count=report.case_count,
        metrics=report.retrieval_quality,
        cases=report.retrieval_cases,
    )


def assess_answer_regression(
    baseline: ManagedRAGEvaluationReport | None,
    candidate: ManagedRAGEvaluationReport | None,
    *,
    thresholds: DriftThresholds,
) -> AnswerRegressionAssessment:
    if baseline is None or candidate is None:
        return AnswerRegressionAssessment(
            status=AssessmentStatus.NOT_REQUESTED,
            reason="Generated-answer evaluation was not explicitly requested.",
        )
    identity = (
        baseline.dataset_name,
        baseline.dataset_version,
        baseline.dataset_format_version,
        baseline.case_count,
    )
    candidate_identity = (
        candidate.dataset_name,
        candidate.dataset_version,
        candidate.dataset_format_version,
        candidate.case_count,
    )
    model_identity = (
        baseline.models.embedding_model,
        baseline.models.reranker_model,
        baseline.models.generation_model,
        baseline.models.judge_model,
    )
    candidate_models = (
        candidate.models.embedding_model,
        candidate.models.reranker_model,
        candidate.models.generation_model,
        candidate.models.judge_model,
    )
    if identity != candidate_identity or model_identity != candidate_models:
        return _answer_assessment(
            status=AssessmentStatus.INCOMPATIBLE,
            reason="Answer reports use different dataset or evaluation model identities.",
            baseline=baseline,
            candidate=candidate,
        )
    if (
        baseline.answer_quality.generation.status != "COMPLETED"
        or candidate.answer_quality.generation.status != "COMPLETED"
    ):
        return _answer_assessment(
            status=AssessmentStatus.FAILED,
            reason="Answer generation did not complete for both generations.",
            baseline=baseline,
            candidate=candidate,
        )
    deltas: list[AnswerMetricDelta] = []
    for backend in ("mlflow_genai", "ragas"):
        before = getattr(baseline.answer_quality, backend)
        after = getattr(candidate.answer_quality, backend)
        if before.status != "COMPLETED" or after.status != "COMPLETED":
            continue
        for metric in sorted(set(before.aggregate_metrics) & set(after.aggregate_metrics)):
            baseline_value = before.aggregate_metrics[metric]
            candidate_value = after.aggregate_metrics[metric]
            deltas.append(
                AnswerMetricDelta(
                    backend=backend,
                    metric=metric,
                    baseline=baseline_value,
                    candidate=candidate_value,
                    delta=candidate_value - baseline_value,
                )
            )
    if not deltas:
        statuses = {
            baseline.answer_quality.mlflow_genai.status,
            candidate.answer_quality.mlflow_genai.status,
            baseline.answer_quality.ragas.status,
            candidate.answer_quality.ragas.status,
        }
        status = (
            AssessmentStatus.FAILED
            if "FAILED" in statuses
            else AssessmentStatus.INSUFFICIENT_DATA
        )
        return _answer_assessment(
            status=status,
            reason="No common completed answer-quality judge metrics are available.",
            baseline=baseline,
            candidate=candidate,
        )
    violations = tuple(
        f"{item.backend}.{item.metric} drop {-item.delta:.6f} exceeds "
        f"{thresholds.max_answer_metric_drop:.6f}"
        for item in deltas
        if item.delta < -thresholds.max_answer_metric_drop
    )
    return _answer_assessment(
        status=AssessmentStatus.AVAILABLE,
        reason=None,
        baseline=baseline,
        candidate=candidate,
        metric_deltas=tuple(deltas),
        violations=violations,
    )


def assess_operational_trends(
    baseline: OperationalTelemetryWindow | None,
    candidate: OperationalTelemetryWindow | None,
    *,
    thresholds: DriftThresholds,
) -> OperationalTrendAssessment:
    if baseline is None or candidate is None:
        return OperationalTrendAssessment(
            status=AssessmentStatus.UNAVAILABLE,
            reason="Equivalent safe telemetry windows were not supplied.",
        )
    baseline_duration = (baseline.window_end - baseline.window_start).total_seconds()
    candidate_duration = (candidate.window_end - candidate.window_start).total_seconds()
    if abs(baseline_duration - candidate_duration) > 1.0:
        return OperationalTrendAssessment(
            status=AssessmentStatus.INCOMPATIBLE,
            reason="Operational telemetry windows do not have equivalent durations.",
        )
    if (
        len(baseline.samples) < thresholds.minimum_operational_samples
        or len(candidate.samples) < thresholds.minimum_operational_samples
    ):
        return OperationalTrendAssessment(
            status=AssessmentStatus.INSUFFICIENT_DATA,
            reason=(
                "Both telemetry windows must meet the configured minimum sample count "
                f"of {thresholds.minimum_operational_samples}."
            ),
        )
    before = _operational_aggregate(baseline.samples)
    after = _operational_aggregate(candidate.samples)
    failure_delta = after.failure_rate - before.failure_rate
    latency_delta = after.mean_total_latency_ms - before.mean_total_latency_ms
    latency_ratio = (
        latency_delta / before.mean_total_latency_ms
        if before.mean_total_latency_ms > 0
        else (0.0 if after.mean_total_latency_ms == 0 else None)
    )
    warnings: list[str] = []
    if failure_delta > thresholds.max_failure_rate_increase:
        warnings.append(
            f"failure-rate increase {failure_delta:.6f} exceeds "
            f"{thresholds.max_failure_rate_increase:.6f}"
        )
    if (
        latency_ratio is not None
        and latency_ratio > thresholds.max_mean_latency_increase_ratio
    ):
        warnings.append(
            f"mean-latency increase ratio {latency_ratio:.6f} exceeds "
            f"{thresholds.max_mean_latency_increase_ratio:.6f}"
        )
    return OperationalTrendAssessment(
        status=AssessmentStatus.AVAILABLE,
        baseline=before,
        candidate=after,
        failure_rate_delta=failure_delta,
        mean_total_latency_delta_ms=latency_delta,
        mean_total_latency_increase_ratio=latency_ratio,
        warnings=tuple(warnings),
    )


def _answer_assessment(
    *,
    status: AssessmentStatus,
    reason: str | None,
    baseline: ManagedRAGEvaluationReport,
    candidate: ManagedRAGEvaluationReport,
    metric_deltas: tuple[AnswerMetricDelta, ...] = (),
    violations: tuple[str, ...] = (),
) -> AnswerRegressionAssessment:
    return AnswerRegressionAssessment(
        status=status,
        reason=reason,
        baseline_generation_status=baseline.answer_quality.generation.status,
        candidate_generation_status=candidate.answer_quality.generation.status,
        baseline_mlflow_status=baseline.answer_quality.mlflow_genai.status,
        candidate_mlflow_status=candidate.answer_quality.mlflow_genai.status,
        baseline_ragas_status=baseline.answer_quality.ragas.status,
        candidate_ragas_status=candidate.answer_quality.ragas.status,
        metric_deltas=metric_deltas,
        violations=violations,
    )


def build_drift_report(
    *,
    corpus_key: str,
    baseline: GenerationSnapshot,
    candidate: GenerationSnapshot,
    evaluation: EvaluationIdentity,
    thresholds: DriftThresholds,
    structural: StructuralAssessment,
    retrieval: RetrievalRegressionAssessment,
    answer_quality: AnswerRegressionAssessment,
    operational_trends: OperationalTrendAssessment,
    generated_at: datetime | None = None,
) -> ManagedDriftAssessmentReport:
    timestamp = generated_at or datetime.now(UTC)
    findings = _classify_findings(
        structural,
        retrieval,
        answer_quality,
        operational_trends,
    )
    return ManagedDriftAssessmentReport(
        generated_at=timestamp,
        corpus_key=corpus_key,
        baseline=baseline.identity,
        candidate=candidate.identity,
        evaluation=evaluation,
        thresholds=thresholds,
        structural=structural,
        retrieval=retrieval,
        answer_quality=answer_quality,
        operational_trends=operational_trends,
        findings=findings,
    )


def write_drift_report(report: ManagedDriftAssessmentReport, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return output


def load_drift_report(value: str | bytes) -> ManagedDriftAssessmentReport:
    return ManagedDriftAssessmentReport.model_validate_json(value)


def load_operational_window(path: str | Path) -> OperationalTelemetryWindow:
    return OperationalTelemetryWindow.model_validate_json(
        Path(path).read_text(encoding="utf-8")
    )


def log_drift_report(
    report: ManagedDriftAssessmentReport,
    report_path: Path,
    run: EvaluationRun,
) -> None:
    """Log one metadata-only assessment artifact to one caller-owned run."""

    if not run.active:
        return
    run.log_parameters(
        {
            "drift.format_version": report.format_version,
            "drift.corpus_key": report.corpus_key,
            "drift.baseline_generation": report.baseline.generation,
            "drift.candidate_generation": report.candidate.generation,
            "drift.dataset_name": report.evaluation.dataset_name,
            "drift.dataset_version": report.evaluation.dataset_version,
            "drift.finding_categories": ",".join(
                sorted({item.category.value for item in report.findings})
            ),
        }
    )
    metrics: dict[str, float] = {
        "drift.sources.new": float(report.structural.source_changes.new_count),
        "drift.sources.changed": float(report.structural.source_changes.changed_count),
        "drift.sources.deleted": float(report.structural.source_changes.deleted_count),
    }
    if report.retrieval.comparison is not None:
        metrics.update(
            {
                "drift.retrieval.recall_delta": report.retrieval.comparison.recall_delta,
                "drift.retrieval.mrr_delta": report.retrieval.comparison.mrr_delta,
                "drift.retrieval.hit_rate_delta": report.retrieval.comparison.hit_rate_delta,
            }
        )
    if report.operational_trends.failure_rate_delta is not None:
        metrics["drift.operations.failure_rate_delta"] = (
            report.operational_trends.failure_rate_delta
        )
    run.log_metrics(metrics)
    run.log_tags(
        {
            "drift.read_only": True,
            "drift.automatic_lifecycle_action": False,
        }
    )
    run.log_artifacts([report_path])


def _source_changes(
    baseline: GenerationSnapshot,
    candidate: GenerationSnapshot,
    *,
    assessed_at: datetime,
) -> SourceChangeSummary:
    before = {item.source_key: item for item in baseline.documents}
    after = {item.source_key: item for item in candidate.documents}
    new_count = len(set(after) - set(before))
    missing = set(before) - set(after)
    deletion_assessed = candidate.identity.authoritative_snapshot
    deleted_count = len(missing) if deletion_assessed else 0
    unassessed_missing = 0 if deletion_assessed else len(missing)
    content_changed = 0
    metadata_only = 0
    governance_changed = 0
    unchanged = 0
    for source_key in set(before) & set(after):
        previous = before[source_key]
        current = after[source_key]
        policy_changed = (
            previous.access_policy_fingerprint
            != current.access_policy_fingerprint
        )
        if previous.content_hash != current.content_hash:
            content_changed += 1
        elif previous.metadata_hash != current.metadata_hash or policy_changed:
            metadata_only += 1
        else:
            unchanged += 1
        if policy_changed:
            governance_changed += 1
    baseline_categories = Counter(item.category for item in baseline.documents)
    candidate_categories = Counter(item.category for item in candidate.documents)
    category_keys = sorted(set(baseline_categories) | set(candidate_categories))
    return SourceChangeSummary(
        new_count=new_count,
        changed_count=content_changed + metadata_only,
        unchanged_count=unchanged,
        deleted_count=deleted_count,
        content_changed_count=content_changed,
        metadata_only_changed_count=metadata_only,
        governance_metadata_changed_count=governance_changed,
        unassessed_missing_count=unassessed_missing,
        deletion_detection_performed=deletion_assessed,
        baseline_category_distribution=dict(sorted(baseline_categories.items())),
        candidate_category_distribution=dict(sorted(candidate_categories.items())),
        category_count_deltas={
            key: candidate_categories[key] - baseline_categories[key]
            for key in category_keys
        },
        baseline_freshness=_source_freshness(baseline.documents, assessed_at),
        candidate_freshness=_source_freshness(candidate.documents, assessed_at),
    )


def _source_freshness(
    documents: Sequence[SourceDocumentSnapshot], assessed_at: datetime
) -> SourceFreshnessSummary:
    timestamps = tuple(
        value
        for item in documents
        if (value := _strict_source_timestamp(item.source_version)) is not None
    )
    if not timestamps:
        return SourceFreshnessSummary(
            status="UNAVAILABLE",
            timestamped_document_count=0,
            document_count=len(documents),
            reason="No source version is a strict ISO date or timestamp.",
        )
    oldest = min(timestamps)
    newest = max(timestamps)
    return SourceFreshnessSummary(
        status="AVAILABLE",
        timestamped_document_count=len(timestamps),
        document_count=len(documents),
        newest_source_timestamp=newest,
        oldest_source_timestamp=oldest,
        oldest_source_age_days=max(
            0.0,
            (assessed_at.astimezone(UTC) - oldest).total_seconds() / 86400.0,
        ),
    )


def _strict_source_timestamp(value: str) -> datetime | None:
    if len(value) == 10:
        try:
            parsed_date = date.fromisoformat(value)
        except ValueError:
            pass
        else:
            return datetime.combine(parsed_date, datetime.min.time(), tzinfo=UTC)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(UTC)


def _index_compatibility(
    baseline: GenerationSnapshot,
    candidate: GenerationSnapshot,
) -> IndexCompatibilitySummary:
    before = _configuration_components(baseline.index_configuration)
    after = _configuration_components(candidate.index_configuration)
    supported_before = before is not None
    supported_after = after is not None
    incompatibilities: list[str] = []
    if not supported_before:
        incompatibilities.append("baseline index configuration schema is unsupported")
    if not supported_after:
        incompatibilities.append("candidate index configuration schema is unsupported")
    bm25_match = _component_match(before, after, "bm25")
    pii_match = _component_match(before, after, "pii_detector_version")
    payload_match = _component_match(before, after, "payload_schema_version")
    point_match = _component_match(before, after, "point_id_format_version")
    embedding_match = (
        baseline.identity.embedding_fingerprint
        == candidate.identity.embedding_fingerprint
    )
    chunking_match = (
        baseline.identity.chunking_fingerprint
        == candidate.identity.chunking_fingerprint
    )
    for compatible, label in (
        (embedding_match, "embedding fingerprint differs"),
        (chunking_match, "chunking fingerprint differs"),
        (bm25_match, "BM25 configuration differs"),
        (pii_match, "PII detector version differs"),
        (payload_match, "payload schema version differs"),
        (point_match, "point ID format version differs"),
    ):
        if compatible is False:
            incompatibilities.append(label)
    comparison_supported = (
        supported_before
        and supported_after
        and embedding_match
        and chunking_match
        and bm25_match is True
        and pii_match is True
        and payload_match is True
        and point_match is True
    )
    return IndexCompatibilitySummary(
        embedding_fingerprint_match=embedding_match,
        chunking_fingerprint_match=chunking_match,
        bm25_configuration_match=bm25_match,
        bm25_state_reused=(
            baseline.identity.bm25_state_hash is not None
            and baseline.identity.bm25_state_hash == candidate.identity.bm25_state_hash
        ),
        pii_detector_version_match=pii_match,
        payload_schema_version_match=payload_match,
        point_id_format_version_match=point_match,
        configuration_fingerprint_match=(
            baseline.identity.index_configuration_fingerprint is not None
            and baseline.identity.index_configuration_fingerprint
            == candidate.identity.index_configuration_fingerprint
        ),
        baseline_schema_supported=supported_before,
        candidate_schema_supported=supported_after,
        comparison_supported=comparison_supported,
        incompatibilities=tuple(incompatibilities),
    )


def _configuration_components(
    payload: Mapping[str, Any] | None,
) -> dict[str, object] | None:
    if payload is None:
        return None
    try:
        bm25 = payload["bm25"]
        if not isinstance(bm25, Mapping):
            return None
        return {
            "bm25": (
                bm25["implementation_version"],
                bm25["tokenizer_version"],
                bm25["k1"],
                bm25["b"],
            ),
            "pii_detector_version": payload["pii_detector_version"],
            "payload_schema_version": payload["payload_schema_version"],
            "point_id_format_version": payload["point_id_format_version"],
        }
    except (KeyError, TypeError):
        return None


def _component_match(
    before: Mapping[str, object] | None,
    after: Mapping[str, object] | None,
    key: str,
) -> bool | None:
    if before is None or after is None:
        return None
    return before[key] == after[key]


def _quality_snapshot(report: RetrievalQualityReport) -> QualityMetricSnapshot:
    return QualityMetricSnapshot(
        recall_at_k=report.recall_at_k,
        mrr=report.mrr,
        hit_rate=report.hit_rate,
        missed_case_count=report.missed_case_count,
    )


def _operational_aggregate(
    samples: Sequence[OperationalTelemetrySample],
) -> OperationalAggregate:
    failures = sum(not item.success for item in samples)
    return OperationalAggregate(
        request_count=len(samples),
        failure_count=failures,
        failure_rate=failures / len(samples),
        mean_total_latency_ms=fmean(item.total_latency_ms for item in samples),
        mean_stage_latencies_ms={
            "dense": _optional_mean(item.dense_latency_ms for item in samples),
            "sparse": _optional_mean(item.sparse_latency_ms for item in samples),
            "fusion": _optional_mean(item.fusion_latency_ms for item in samples),
            "reranker": _optional_mean(item.reranker_latency_ms for item in samples),
            "pii": _optional_mean(item.pii_latency_ms for item in samples),
        },
        mean_candidate_counts={
            "dense": fmean(item.dense_candidate_count for item in samples),
            "sparse": fmean(item.sparse_candidate_count for item in samples),
            "fused": fmean(item.fused_candidate_count for item in samples),
            "final": fmean(item.final_context_count for item in samples),
        },
        mean_redacted_context_count=fmean(
            item.redacted_context_count for item in samples
        ),
        mean_token_usage={
            "input": _optional_mean(item.input_tokens for item in samples),
            "output": _optional_mean(item.output_tokens for item in samples),
            "total": _optional_mean(item.total_tokens for item in samples),
        },
        generation_distribution=dict(
            sorted(
                Counter(
                    str(item.generation) if item.generation is not None else "unavailable"
                    for item in samples
                ).items()
            )
        ),
    )


def _optional_mean(values: Iterable[float | int | None]) -> float | None:
    available = [float(value) for value in values if value is not None]
    return fmean(available) if available else None


def _classify_findings(
    structural: StructuralAssessment,
    retrieval: RetrievalRegressionAssessment,
    answer: AnswerRegressionAssessment,
    operational: OperationalTrendAssessment,
) -> tuple[DriftFinding, ...]:
    findings: list[DriftFinding] = []
    candidate_validation = structural.candidate_validation
    if candidate_validation.validation_outcome == "FAIL":
        findings.append(
            _finding(
                DriftFindingCategory.STRUCTURAL_FAILURE,
                "candidate_integrity_failed",
                "Candidate generation failed managed physical or lineage validation.",
            )
        )
    if candidate_validation.governance_failure_detected:
        findings.append(
            _finding(
                DriftFindingCategory.GOVERNANCE_FAILURE,
                "candidate_governance_validation_failed",
                "Candidate validation found an ACL or PII payload failure.",
            )
        )
    if retrieval.acl_negative_failed_case_ids:
        findings.append(
            _finding(
                DriftFindingCategory.GOVERNANCE_FAILURE,
                "acl_negative_case_failed",
                "One or more ACL-negative evaluation cases exposed a protected identity.",
            )
        )
    if retrieval.threshold_result is not None and retrieval.threshold_result.status == "FAIL":
        findings.append(
            _finding(
                DriftFindingCategory.QUALITY_REGRESSION,
                "retrieval_threshold_failed",
                "Candidate retrieval quality exceeded an approved regression threshold.",
            )
        )
    if answer.violations:
        findings.append(
            _finding(
                DriftFindingCategory.QUALITY_REGRESSION,
                "answer_quality_threshold_failed",
                "Candidate answer-quality judge metrics exceeded a regression threshold.",
            )
        )
    changes = structural.source_changes
    if (
        changes.new_count
        or changes.changed_count
        or changes.deleted_count
        or changes.unassessed_missing_count
        or any(changes.category_count_deltas.values())
    ):
        findings.append(
            _finding(
                DriftFindingCategory.DISTRIBUTION_WARNING,
                "source_distribution_changed",
                "Source identity, metadata, content, or category distribution changed.",
            )
        )
    if operational.warnings:
        findings.append(
            _finding(
                DriftFindingCategory.DISTRIBUTION_WARNING,
                "operational_threshold_exceeded",
                "Safe aggregate operational behavior exceeded a configured warning threshold.",
            )
        )
    unavailable_sections = tuple(
        label
        for label, status in (
            ("retrieval", retrieval.status),
            ("answer quality", answer.status),
            ("operational trends", operational.status),
        )
        if status is not AssessmentStatus.AVAILABLE
    )
    if unavailable_sections or not structural.index_compatibility.comparison_supported:
        findings.append(
            _finding(
                DriftFindingCategory.INSUFFICIENT_DATA,
                "assessment_evidence_incomplete",
                "One or more assessment dimensions are unavailable, incompatible, or incomplete.",
            )
        )
    critical = {
        DriftFindingCategory.STRUCTURAL_FAILURE,
        DriftFindingCategory.GOVERNANCE_FAILURE,
        DriftFindingCategory.QUALITY_REGRESSION,
        DriftFindingCategory.INSUFFICIENT_DATA,
    }
    if not any(item.category in critical for item in findings):
        findings.append(
            _finding(
                DriftFindingCategory.NO_REGRESSION_DETECTED,
                "thresholds_passed",
                "No assessed structural, governance, or quality regression was detected.",
            )
        )
    return tuple(findings)


def _finding(
    category: DriftFindingCategory,
    code: str,
    summary: str,
) -> DriftFinding:
    responses = {
        DriftFindingCategory.STRUCTURAL_FAILURE: (
            "Stop candidate promotion or serving changes; inspect managed validation and repair "
            "infrastructure or lineage through an explicit operator workflow."
        ),
        DriftFindingCategory.GOVERNANCE_FAILURE: (
            "Treat as a security incident candidate; inspect ACL/PII derivation and payloads, "
            "then use existing explicit publication or restoration procedures if approved."
        ),
        DriftFindingCategory.QUALITY_REGRESSION: (
            "Review regressed cases, dataset expectations, retrieval settings, and judge output; "
            "do not change ACTIVE automatically."
        ),
        DriftFindingCategory.DISTRIBUTION_WARNING: (
            "Confirm the source or traffic change is expected and expand reviewed coverage where "
            "the changed distribution is operationally important."
        ),
        DriftFindingCategory.INSUFFICIENT_DATA: (
            "Collect compatible reviewed reports or equivalent safe telemetry windows before "
            "making a lifecycle decision."
        ),
        DriftFindingCategory.NO_REGRESSION_DETECTED: (
            "Record the assessment and continue normal operator review; this is not an automatic "
            "publication or deployment approval."
        ),
    }
    return DriftFinding(
        category=category,
        code=code,
        summary=summary,
        operator_response=responses[category],
    )


def _access_policy_fingerprint(visibility: str, groups: tuple[str, ...]) -> str:
    payload = json.dumps(
        {"visibility": visibility, "groups": sorted(groups)},
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
