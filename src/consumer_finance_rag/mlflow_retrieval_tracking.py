"""Optional MLflow adapters for safe retrieval telemetry and quality reports."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Protocol

from consumer_finance_rag.config import MLflowSettings
from consumer_finance_rag.evaluation.tracking import EvaluationRun
from consumer_finance_rag.observability import RetrievalMetrics
from consumer_finance_rag.retrieval_quality import (
    RetrievalQualityComparison,
    RetrievalQualityGateResult,
    RetrievalQualityReport,
    write_retrieval_quality_report,
)

COMPARISON_FORMAT_VERSION = "retrieval-quality-comparison-v1"


class _MLflowTelemetryClient(Protocol):
    """Small surface needed to append telemetry to a caller-owned active run."""

    def active_run(self) -> object | None: ...

    def log_metrics(self, metrics: dict[str, float]) -> Any: ...

    def set_tags(self, tags: dict[str, Any]) -> Any: ...


def log_retrieval_quality_report(
    report: RetrievalQualityReport,
    run: EvaluationRun,
    *,
    comparison: RetrievalQualityComparison | None = None,
    threshold_result: RetrievalQualityGateResult | None = None,
) -> None:
    """Log one safe report and optional comparison to a caller-owned run."""

    if not run.active:
        return
    if threshold_result is not None and comparison is None:
        raise ValueError("threshold_result requires a quality comparison")

    run.log_parameters(_quality_parameters(report))
    run.log_metrics(_quality_metrics(report))
    run.log_tags(
        {
            "retrieval_quality.format_version": report.format_version,
            "retrieval_quality.generation": report.generation,
        }
    )

    with TemporaryDirectory(prefix="retrieval-quality-") as directory:
        artifact_directory = Path(directory)
        report_path = write_retrieval_quality_report(
            report,
            artifact_directory
            / f"retrieval_quality_g{report.generation:06d}.json",
        )
        artifacts = [report_path]
        if comparison is not None:
            run.log_parameters(_comparison_parameters(comparison, threshold_result))
            run.log_metrics(_comparison_metrics(comparison))
            if threshold_result is not None:
                run.log_tags(
                    {"retrieval_quality.threshold_result": threshold_result.status}
                )
            artifacts.append(
                _write_comparison_artifact(
                    comparison,
                    threshold_result,
                    artifact_directory / "retrieval_quality_comparison.json",
                )
            )
        run.log_artifacts(artifacts)


class MLflowRetrievalTelemetrySink:
    """Append aggregate-safe request metrics to an existing MLflow run.

    The sink never creates a run. Disabled settings, a missing active run, import failures, and
    logging failures all degrade to a no-op so normal retrieval remains independent of MLflow.
    """

    def __init__(
        self,
        settings: MLflowSettings,
        *,
        client: _MLflowTelemetryClient | None = None,
    ) -> None:
        self._enabled = settings.enabled
        self._client = client

    def emit(self, metrics: RetrievalMetrics) -> None:
        if not self._enabled:
            return
        try:
            client = self._client or _import_mlflow()
            if client.active_run() is None:
                return
            client.log_metrics(_operational_metrics(metrics))
            client.set_tags(_operational_tags(metrics))
        except Exception:
            return


def _quality_parameters(report: RetrievalQualityReport) -> dict[str, str | int]:
    return {
        "generation": report.generation,
        "index_state_id": str(report.index_state_id),
        "qdrant_collection": report.qdrant_collection,
        "snapshot_hash": report.snapshot_hash,
        "embedding_fingerprint": report.embedding_fingerprint,
        "chunking_fingerprint": report.chunking_fingerprint,
        "index_configuration_fingerprint": report.index_configuration_fingerprint,
        "report_format_version": report.format_version,
        "case_count": report.case_count,
        "retrieval_k": report.metrics.k,
    }


def _quality_metrics(report: RetrievalQualityReport) -> dict[str, float]:
    return {
        "recall_at_k": report.recall_at_k,
        "mrr": report.mrr,
        "hit_rate": report.hit_rate,
        "missed_case_count": float(report.missed_case_count),
    }


def _comparison_parameters(
    comparison: RetrievalQualityComparison,
    threshold_result: RetrievalQualityGateResult | None,
) -> dict[str, str | int]:
    parameters: dict[str, str | int] = {
        "baseline_generation": comparison.baseline_generation,
        "candidate_generation": comparison.candidate_generation,
        "baseline_collection": comparison.baseline_collection,
        "candidate_collection": comparison.candidate_collection,
    }
    if threshold_result is not None:
        parameters["threshold_result"] = threshold_result.status
        parameters["max_recall_drop"] = str(
            threshold_result.thresholds.max_recall_drop
        )
        parameters["max_mrr_drop"] = str(threshold_result.thresholds.max_mrr_drop)
    return parameters


def _comparison_metrics(
    comparison: RetrievalQualityComparison,
) -> dict[str, float]:
    return {
        "recall_delta": comparison.recall_delta,
        "mrr_delta": comparison.mrr_delta,
        "hit_rate_delta": comparison.hit_rate_delta,
        "improved_case_count": float(len(comparison.cases_improved)),
        "regressed_case_count": float(len(comparison.cases_regressed)),
        "newly_missed_case_count": float(len(comparison.newly_missed_cases)),
        "newly_recovered_case_count": float(
            len(comparison.newly_recovered_cases)
        ),
    }


def _write_comparison_artifact(
    comparison: RetrievalQualityComparison,
    threshold_result: RetrievalQualityGateResult | None,
    path: Path,
) -> Path:
    payload = {
        "format_version": COMPARISON_FORMAT_VERSION,
        "comparison": comparison.model_dump(mode="json"),
        "threshold_result": (
            threshold_result.model_dump(mode="json")
            if threshold_result is not None
            else None
        ),
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def _operational_metrics(metrics: RetrievalMetrics) -> dict[str, float]:
    values: dict[str, float] = {
        "retrieval.total_latency_ms": metrics.total_retrieval_latency_ms,
        "retrieval.dense_candidate_count": float(metrics.dense_candidate_count),
        "retrieval.sparse_candidate_count": float(metrics.sparse_candidate_count),
        "retrieval.fused_candidate_count": float(metrics.fused_candidate_count),
        "retrieval.final_context_count": float(metrics.final_context_count),
        "retrieval.acl_filter_applied": float(metrics.acl_filter_applied),
        "retrieval.request_group_count": float(metrics.request_group_count),
        "retrieval.rejected_candidate_count": float(
            metrics.rejected_candidate_count
        ),
        "retrieval.redacted_context_count": float(metrics.redacted_context_count),
        "retrieval.success": float(metrics.status.value == "SUCCESS"),
        "retrieval.failure": float(metrics.status.value == "ERROR"),
    }
    optional_latencies = {
        "retrieval.dense_latency_ms": metrics.dense_search_latency_ms,
        "retrieval.sparse_latency_ms": metrics.sparse_search_latency_ms,
        "retrieval.fusion_latency_ms": metrics.fusion_latency_ms,
        "retrieval.reranker_latency_ms": metrics.reranker_latency_ms,
        "retrieval.pii_latency_ms": metrics.pii_processing_latency_ms,
    }
    values.update(
        {key: value for key, value in optional_latencies.items() if value is not None}
    )
    if metrics.error_stage is not None:
        values[f"retrieval.failure_stage.{metrics.error_stage.value.lower()}"] = 1.0
    return values


def _operational_tags(metrics: RetrievalMetrics) -> dict[str, str]:
    return {
        "retrieval.profile": metrics.profile,
        "retrieval.generation": (
            str(metrics.generation) if metrics.generation is not None else "unavailable"
        ),
        "retrieval.index_state_id": (
            str(metrics.index_state_id)
            if metrics.index_state_id is not None
            else "unavailable"
        ),
        "retrieval.qdrant_collection": metrics.qdrant_collection or "unavailable",
        "retrieval.status": metrics.status.value,
        "retrieval.failure_stage": (
            metrics.error_stage.value if metrics.error_stage is not None else "none"
        ),
        "retrieval.pii_policy": metrics.pii_policy,
    }


def _import_mlflow() -> _MLflowTelemetryClient:
    import mlflow

    return mlflow
