from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from consumer_finance_rag.config import MLflowSettings
from consumer_finance_rag.evaluation.tracking import ParameterValue
from consumer_finance_rag.mlflow_retrieval_tracking import (
    MLflowRetrievalTelemetrySink,
    log_retrieval_quality_report,
)
from consumer_finance_rag.observability import (
    PIITypeCount,
    RetrievalACLMode,
    RetrievalFailureStage,
    RetrievalMetrics,
    RetrievalStatus,
)
from consumer_finance_rag.retrieval_quality import (
    RetrievalCaseResult,
    RetrievalQualityComparison,
    RetrievalQualityGateResult,
    RetrievalQualityMetrics,
    RetrievalQualityReport,
    RetrievalQualityThresholds,
    RetrievalRuntimeConfiguration,
    RetrievedIdentity,
)

NOW = datetime(2026, 9, 22, 3, 4, 5, tzinfo=UTC)


class RecordingRun:
    def __init__(self, *, active: bool = True) -> None:
        self.active = active
        self.run_id = "run-123" if active else None
        self.parameters: dict[str, ParameterValue] = {}
        self.metrics: dict[str, float] = {}
        self.tags: dict[str, ParameterValue] = {}
        self.artifacts: dict[str, str] = {}

    def log_parameters(self, parameters: Mapping[str, ParameterValue]) -> None:
        self.parameters.update(parameters)

    def log_metrics(self, metrics: Mapping[str, float]) -> None:
        self.metrics.update(metrics)

    def log_artifacts(self, paths: list[Path]) -> None:
        for path in paths:
            self.artifacts[path.name] = path.read_text(encoding="utf-8")

    def log_tags(self, tags: Mapping[str, ParameterValue]) -> None:
        self.tags.update(tags)


class RecordingMLflowClient:
    def __init__(
        self,
        *,
        active: bool = True,
        failure: Exception | None = None,
    ) -> None:
        self.active = active
        self.failure = failure
        self.active_run_calls = 0
        self.metrics: list[dict[str, float]] = []
        self.tags: list[dict[str, Any]] = []

    def active_run(self) -> object | None:
        self.active_run_calls += 1
        if self.failure is not None:
            raise self.failure
        return object() if self.active else None

    def log_metrics(self, metrics: dict[str, float]) -> None:
        if self.failure is not None:
            raise self.failure
        self.metrics.append(dict(metrics))

    def set_tags(self, tags: dict[str, Any]) -> None:
        if self.failure is not None:
            raise self.failure
        self.tags.append(dict(tags))


def quality_report(
    generation: int = 4,
    *,
    recall: float = 0.75,
    mrr: float = 0.625,
    hit_rate: float = 0.75,
) -> RetrievalQualityReport:
    collection = f"consumer_finance_managed_g{generation:06d}"
    case = RetrievalCaseResult(
        case_id="safe-case-id",
        generation=generation,
        qdrant_collection=collection,
        identity_kind="source_id",
        expected_ids=("safe-source-id",),
        expected_access=True,
        access_expectation_met=True,
        retrieved_identities=(
            RetrievedIdentity(
                source_id="safe-source-id",
                document_id="safe-document-id",
                chunk_id="safe-chunk-id",
            ),
        ),
        hit_at_k=True,
        recall_at_k=1.0,
        reciprocal_rank=1.0,
        expected_rank=1,
        final_context_count=1,
    )
    return RetrievalQualityReport(
        generated_at=NOW,
        generation=generation,
        index_state_id=UUID(int=generation),
        qdrant_collection=collection,
        snapshot_hash=f"snapshot-{generation}",
        embedding_fingerprint=f"embedding-{generation}",
        chunking_fingerprint=f"chunking-{generation}",
        index_configuration_fingerprint=f"index-{generation}",
        retrieval_settings=RetrievalRuntimeConfiguration(
            dense_candidates=20,
            sparse_candidates=20,
            fused_candidates=12,
            final_contexts=5,
            rrf_k=60,
            reranker_model="safe-reranker",
            acl_filtering_enabled=True,
        ),
        case_count=1,
        metrics=RetrievalQualityMetrics(
            k=5,
            recall_at_k=recall,
            mrr=mrr,
            hit_rate=hit_rate,
            missed_case_count=1 if hit_rate < 1.0 else 0,
        ),
        cases=(case,),
    )


def comparison() -> RetrievalQualityComparison:
    return RetrievalQualityComparison(
        baseline_generation=3,
        candidate_generation=4,
        baseline_collection="consumer_finance_managed_g000003",
        candidate_collection="consumer_finance_managed_g000004",
        recall_delta=-0.1,
        mrr_delta=0.05,
        hit_rate_delta=-0.2,
        cases_improved=("improved",),
        cases_regressed=("regressed-a", "regressed-b"),
        cases_unchanged=("unchanged",),
        newly_missed_cases=("newly-missed",),
        newly_recovered_cases=("recovered-a", "recovered-b"),
    )


def retrieval_metrics() -> RetrievalMetrics:
    return RetrievalMetrics(
        request_id="request-id-not-logged",
        profile="managed",
        index_state_id=UUID(int=4),
        generation=4,
        qdrant_collection="consumer_finance_managed_g000004",
        query_timestamp=NOW,
        total_retrieval_latency_ms=12.5,
        query_embedding_latency_ms=1.0,
        dense_search_latency_ms=2.0,
        sparse_search_latency_ms=3.0,
        fusion_latency_ms=1.0,
        reranker_latency_ms=4.0,
        pii_processing_latency_ms=0.5,
        dense_candidate_count=20,
        sparse_candidate_count=19,
        fused_candidate_count=12,
        final_context_count=5,
        acl_mode=RetrievalACLMode.FILTER_AND_VALIDATE,
        acl_filter_applied=True,
        request_group_count=2,
        rejected_candidate_count=1,
        pii_policy="REDACT",
        redacted_context_count=2,
        detected_pii_type_counts=(PIITypeCount("EMAIL", 2),),
        redacted_pii_type_counts=(PIITypeCount("EMAIL", 2),),
        reranker_enabled=True,
        status=RetrievalStatus.ERROR,
        error_stage=RetrievalFailureStage.PII,
        error_type="PIIRedactionError",
    )


def test_quality_metrics_parameters_and_safe_report_artifact_are_logged() -> None:
    run = RecordingRun()
    report = quality_report()

    log_retrieval_quality_report(report, run)

    assert run.parameters == {
        "generation": 4,
        "index_state_id": str(UUID(int=4)),
        "qdrant_collection": "consumer_finance_managed_g000004",
        "snapshot_hash": "snapshot-4",
        "embedding_fingerprint": "embedding-4",
        "chunking_fingerprint": "chunking-4",
        "index_configuration_fingerprint": "index-4",
        "report_format_version": "retrieval-quality-report-v1",
        "case_count": 1,
        "retrieval_k": 5,
    }
    assert run.metrics == {
        "recall_at_k": 0.75,
        "mrr": 0.625,
        "hit_rate": 0.75,
        "missed_case_count": 1.0,
    }
    assert run.tags["retrieval_quality.generation"] == 4
    artifact = run.artifacts["retrieval_quality_g000004.json"]
    assert "safe-case-id" in artifact
    assert "raw query" not in artifact
    assert "raw chunk text" not in artifact
    assert "alice@example.test" not in artifact
    assert "fraud-analysts" not in artifact
    assert "principal-123" not in artifact
    assert "credential-secret" not in artifact


def test_comparison_metrics_and_failed_threshold_are_logged() -> None:
    run = RecordingRun()
    compared = comparison()
    gate = RetrievalQualityGateResult(
        status="FAIL",
        thresholds=RetrievalQualityThresholds(
            max_recall_drop=0.02,
            max_mrr_drop=0.02,
        ),
        violations=("recall drop 0.100000 exceeds 0.020000",),
    )

    log_retrieval_quality_report(
        quality_report(),
        run,
        comparison=compared,
        threshold_result=gate,
    )

    assert run.parameters["baseline_generation"] == 3
    assert run.parameters["candidate_generation"] == 4
    assert run.parameters["threshold_result"] == "FAIL"
    assert run.metrics["recall_delta"] == -0.1
    assert run.metrics["mrr_delta"] == 0.05
    assert run.metrics["hit_rate_delta"] == -0.2
    assert run.metrics["improved_case_count"] == 1.0
    assert run.metrics["regressed_case_count"] == 2.0
    assert run.metrics["newly_missed_case_count"] == 1.0
    assert run.metrics["newly_recovered_case_count"] == 2.0
    assert run.tags["retrieval_quality.threshold_result"] == "FAIL"
    artifact = run.artifacts["retrieval_quality_comparison.json"]
    assert '"status": "FAIL"' in artifact
    assert '"baseline_generation": 3' in artifact

    passed_run = RecordingRun()
    passed_gate = gate.model_copy(update={"status": "PASS", "violations": ()})
    log_retrieval_quality_report(
        quality_report(),
        passed_run,
        comparison=compared,
        threshold_result=passed_gate,
    )
    assert passed_run.parameters["threshold_result"] == "PASS"
    assert passed_run.tags["retrieval_quality.threshold_result"] == "PASS"


def test_disabled_quality_logging_is_a_complete_noop() -> None:
    run = RecordingRun(active=False)

    log_retrieval_quality_report(quality_report(), run)

    assert run.parameters == {}
    assert run.metrics == {}
    assert run.tags == {}
    assert run.artifacts == {}


def test_operational_sink_maps_only_aggregate_safe_fields() -> None:
    client = RecordingMLflowClient()
    sink = MLflowRetrievalTelemetrySink(MLflowSettings(enabled=True), client=client)

    sink.emit(retrieval_metrics())

    assert client.active_run_calls == 1
    assert client.metrics == [
        {
            "retrieval.total_latency_ms": 12.5,
            "retrieval.dense_candidate_count": 20.0,
            "retrieval.sparse_candidate_count": 19.0,
            "retrieval.fused_candidate_count": 12.0,
            "retrieval.final_context_count": 5.0,
            "retrieval.acl_filter_applied": 1.0,
            "retrieval.request_group_count": 2.0,
            "retrieval.rejected_candidate_count": 1.0,
            "retrieval.redacted_context_count": 2.0,
            "retrieval.success": 0.0,
            "retrieval.failure": 1.0,
            "retrieval.query_embedding_latency_ms": 1.0,
            "retrieval.dense_latency_ms": 2.0,
            "retrieval.sparse_latency_ms": 3.0,
            "retrieval.fusion_latency_ms": 1.0,
            "retrieval.reranker_latency_ms": 4.0,
            "retrieval.pii_latency_ms": 0.5,
            "retrieval.failure_stage.pii": 1.0,
        }
    ]
    assert client.tags == [
        {
            "retrieval.profile": "managed",
            "retrieval.generation": "4",
            "retrieval.index_state_id": str(UUID(int=4)),
            "retrieval.qdrant_collection": "consumer_finance_managed_g000004",
            "retrieval.status": "ERROR",
            "retrieval.failure_stage": "PII",
            "retrieval.pii_policy": "REDACT",
        }
    ]
    recorded = repr((client.metrics, client.tags))
    assert "request-id-not-logged" not in recorded
    assert "alice@example.test" not in recorded
    assert "fraud-analysts" not in recorded


def test_operational_sink_never_starts_a_run_and_inactive_run_is_noop() -> None:
    client = RecordingMLflowClient(active=False)
    sink = MLflowRetrievalTelemetrySink(MLflowSettings(enabled=True), client=client)

    sink.emit(retrieval_metrics())

    assert client.active_run_calls == 1
    assert client.metrics == []
    assert client.tags == []


def test_disabled_or_unavailable_operational_mlflow_never_breaks_retrieval() -> None:
    disabled_client = RecordingMLflowClient(failure=AssertionError("must not contact"))
    MLflowRetrievalTelemetrySink(
        MLflowSettings(enabled=False),
        client=disabled_client,
    ).emit(retrieval_metrics())
    assert disabled_client.active_run_calls == 0

    unavailable = RecordingMLflowClient(failure=ConnectionError("unavailable"))
    MLflowRetrievalTelemetrySink(
        MLflowSettings(enabled=True),
        client=unavailable,
    ).emit(retrieval_metrics())
    assert unavailable.active_run_calls == 1
