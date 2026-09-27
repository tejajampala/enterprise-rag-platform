"""Generation-pinned managed RAG answer evaluation and privacy-safe reporting."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean
from time import perf_counter
from typing import Final, Literal, Protocol
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from consumer_finance_rag.access import AccessContext
from consumer_finance_rag.config import MLflowSettings
from consumer_finance_rag.costs import ModelUsageSummary
from consumer_finance_rag.evaluation.judge_call_debug import (
    ApprovedPublicDebugCase,
    JudgeCallDebugCapture,
    JudgeCallDebugConfig,
)
from consumer_finance_rag.evaluation.judge_preflight import (
    JudgeBackendPreflight,
    JudgePreflightResult,
    JudgePreflightStatus,
)
from consumer_finance_rag.evaluation.mlflow_genai import (
    MLflowJudgeResults,
    MLflowMetricStatus,
    run_mlflow_judges,
)
from consumer_finance_rag.evaluation.models import (
    Difficulty,
    EvaluationCase,
    EvaluationCategory,
    JudgeMetricValue,
    RAGCaseResult,
)
from consumer_finance_rag.evaluation.ragas_adapter import (
    RagasEvaluator,
    build_ragas_evaluator,
)
from consumer_finance_rag.evaluation.ragas_execution_trace import (
    RagasExecutionTraceContext,
    RagasExecutionTraceReference,
)
from consumer_finance_rag.evaluation.tracking import EvaluationRun
from consumer_finance_rag.lineage.managed_retrieval import ManagedRetrievalRuntime
from consumer_finance_rag.observability import (
    RetrievalMetrics,
    RetrievalStatus,
    RetrievalTelemetrySink,
    Tracer,
    bedrock_token_usage,
    build_tracer,
)
from consumer_finance_rag.pii import PIIHandlingPolicy
from consumer_finance_rag.pipeline import RAGPipeline
from consumer_finance_rag.retrieval_quality import (
    RetrievalCaseResult,
    RetrievalEvaluationCase,
    RetrievalQualityMetrics,
    evaluate_managed_retrieval,
)

DATASET_FORMAT_VERSION: Final = "managed-rag-evaluation-dataset-v1"
REPORT_FORMAT_VERSION: Final = "managed-rag-evaluation-report-v3"
EvaluationMode = Literal["retrieval_only", "generated_answers"]
StageStatus = Literal[
    "NOT_REQUESTED", "COMPLETED", "PARTIAL", "FAILED", "UNAVAILABLE"
]
CaseScoringStatus = Literal[
    "NOT_REQUESTED", "COMPLETED", "PARTIAL", "FAILED", "UNAVAILABLE"
]


class ManagedEvaluationDatasetError(ValueError):
    """Raised when the managed dataset is unreviewed, unsafe, or incompatible."""


class ManagedEvaluationAuthorization(BaseModel):
    """Explicit public-only authorization for the initial content-bearing evaluation."""

    model_config = ConfigDict(frozen=True)

    visibility: Literal["PUBLIC"]
    principal_id: None = None
    groups: tuple[str, ...] = ()

    @model_validator(mode="after")
    def public_context_contains_no_identity(self) -> ManagedEvaluationAuthorization:
        if self.groups:
            raise ValueError("managed PUBLIC evaluation cases cannot contain access groups")
        return self

    def access_context(self) -> AccessContext:
        return AccessContext()


class ManagedRAGEvaluationCase(BaseModel):
    """Reviewed synthetic question grounded in immutable PUBLIC source identities."""

    model_config = ConfigDict(frozen=True)

    case_id: str = Field(min_length=1)
    question: str = Field(min_length=1, repr=False)
    category: EvaluationCategory
    difficulty: Difficulty
    reference_answer: str = Field(min_length=1, repr=False)
    expected_facts: tuple[str, ...] = Field(min_length=1, repr=False)
    expected_source_ids: tuple[str, ...] = Field(min_length=1)
    expected_document_ids: tuple[str, ...] = Field(min_length=1)
    expected_chunk_ids: tuple[str, ...] = Field(min_length=1)
    review_status: Literal["reviewed"]
    authorization: ManagedEvaluationAuthorization

    @model_validator(mode="after")
    def identities_are_unique_and_coherent(self) -> ManagedRAGEvaluationCase:
        identity_groups = (
            self.expected_source_ids,
            self.expected_document_ids,
            self.expected_chunk_ids,
        )
        for values in identity_groups:
            if len(values) != len(set(values)):
                raise ValueError("expected identities must be unique")
            if any(not value.strip() for value in values):
                raise ValueError("expected identities cannot be blank")
        if any(not fact.strip() for fact in self.expected_facts):
            raise ValueError("expected facts cannot be blank")
        if any(
            not any(
                document_id.startswith(f"{source_id}:")
                for source_id in self.expected_source_ids
            )
            for document_id in self.expected_document_ids
        ):
            raise ValueError("expected document identities must belong to an expected source")
        if any(
            not any(
                chunk_id.startswith(f"{document_id}:")
                for document_id in self.expected_document_ids
            )
            for chunk_id in self.expected_chunk_ids
        ):
            raise ValueError("expected chunk identities must belong to an expected document")
        return self

    def retrieval_case(self) -> RetrievalEvaluationCase:
        return RetrievalEvaluationCase(
            case_id=self.case_id,
            query=self.question,
            expected_source_ids=self.expected_source_ids,
            access_context=self.authorization.access_context(),
            scenario="reviewed_public_managed_answer",
            category=self.category,
        )

    def phase2_case(self) -> EvaluationCase:
        """Adapt to the existing MLflow scorer contract without changing the adapter."""

        return EvaluationCase(
            case_id=self.case_id,
            question=self.question,
            category=self.category,
            difficulty=self.difficulty,
            expected_facts=list(self.expected_facts),
            reference_answer=self.reference_answer,
            relevant_document_ids=list(self.expected_document_ids),
            relevant_chunk_ids=list(self.expected_chunk_ids),
            review_status="reviewed",
        )


class ManagedRAGEvaluationDataset(BaseModel):
    """Versioned, reviewed PUBLIC dataset permitted at content-bearing boundaries."""

    model_config = ConfigDict(frozen=True)

    format_version: Literal["managed-rag-evaluation-dataset-v1"]
    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    status: Literal["reviewed"]
    corpus_key: str = Field(min_length=1)
    privacy_classification: Literal["PUBLIC_SYNTHETIC_REVIEWED"]
    cases: tuple[ManagedRAGEvaluationCase, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def case_ids_are_unique(self) -> ManagedRAGEvaluationDataset:
        case_ids = [case.case_id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("managed evaluation case IDs must be unique")
        return self

    def limited_cases(self, limit: int | None) -> tuple[ManagedRAGEvaluationCase, ...]:
        if limit is None:
            return self.cases
        if limit <= 0:
            raise ValueError("limit must be positive")
        return self.cases[:limit]


class SafeMetricValue(BaseModel):
    """Judge value without free-form rationale that could repeat evaluation content."""

    model_config = ConfigDict(frozen=True)

    value: float | str | bool | None


class JudgeStageSummary(BaseModel):
    """One optional judge backend, including explicit non-success status."""

    model_config = ConfigDict(frozen=True)

    status: StageStatus
    aggregate_metrics: dict[str, float] = Field(default_factory=dict)
    reason: str | None = None
    skipped_metrics: dict[str, str] = Field(default_factory=dict)
    metric_statuses: dict[str, str] = Field(default_factory=dict)
    failed_metrics: dict[str, str] = Field(default_factory=dict)
    run_id: str | None = None
    usage: ModelUsageSummary | None = None
    execution_trace: RagasExecutionTraceReference | None = None


class ManagedModelIdentity(BaseModel):
    model_config = ConfigDict(frozen=True)

    embedding_model: str
    reranker_model: str
    generation_model: str | None
    generation_region: str | None = None
    judge_model: str | None
    judge_region: str | None = None


class RetrievalLatencySummary(BaseModel):
    """Aggregate request latency; no query or context content is retained."""

    model_config = ConfigDict(frozen=True)

    request_count: int = Field(ge=0)
    mean_total_ms: float | None = Field(default=None, ge=0)
    mean_dense_ms: float | None = Field(default=None, ge=0)
    mean_sparse_ms: float | None = Field(default=None, ge=0)
    mean_fusion_ms: float | None = Field(default=None, ge=0)
    mean_reranker_ms: float | None = Field(default=None, ge=0)
    mean_pii_ms: float | None = Field(default=None, ge=0)


class TokenUsageSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    available_case_count: int = Field(ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)


class GenerationSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: StageStatus
    generated_case_count: int = Field(ge=0)
    mean_latency_ms: float | None = Field(default=None, ge=0)
    token_usage: TokenUsageSummary
    reason: str | None = None


class CaseScoringState(BaseModel):
    model_config = ConfigDict(frozen=True)

    mlflow_genai: CaseScoringStatus
    ragas: CaseScoringStatus


class ManagedAnswerCaseReport(BaseModel):
    """Content-free per-case answer result suitable for a report artifact."""

    model_config = ConfigDict(frozen=True)

    case_id: str
    request_id: str | None = None
    trace_id: str | None = None
    generation_status: Literal["COMPLETED"]
    answer_character_count: int = Field(ge=0)
    citation_count: int = Field(ge=0)
    cited_chunk_ids: tuple[str, ...]
    latency_ms: float = Field(ge=0)
    token_usage: dict[str, int] | None
    scoring: CaseScoringState
    mlflow_judges: dict[str, SafeMetricValue] = Field(default_factory=dict)
    ragas: dict[str, SafeMetricValue] = Field(default_factory=dict)


class AnswerQualitySummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    generation: GenerationSummary
    mlflow_genai: JudgeStageSummary
    ragas: JudgeStageSummary


class ApprovedAnswerEvaluation(BaseModel):
    """Judge results for one existing answer from the reviewed PUBLIC contract."""

    model_config = ConfigDict(frozen=True)

    case_id: str
    generation_model: str
    judge_model: str | None
    judge_region: str | None = None
    mlflow_genai: JudgeStageSummary
    ragas: JudgeStageSummary


class ManagedRAGEvaluationReport(BaseModel):
    """Versioned report that separates deterministic retrieval from answer quality."""

    model_config = ConfigDict(frozen=True)

    format_version: Literal[
        "managed-rag-evaluation-report-v1",
        "managed-rag-evaluation-report-v2",
        "managed-rag-evaluation-report-v3",
    ] = REPORT_FORMAT_VERSION
    generated_at: datetime
    mode: EvaluationMode
    corpus_key: str
    snapshot_hash: str
    corpus_document_count: int = Field(ge=0)
    corpus_chunk_count: int = Field(ge=0)
    dataset_name: str
    dataset_version: str
    dataset_format_version: str
    dataset_privacy_classification: str
    case_count: int = Field(ge=1)
    generation: int = Field(ge=1)
    index_state_id: UUID
    qdrant_collection: str
    embedding_fingerprint: str
    chunking_fingerprint: str
    index_configuration_fingerprint: str
    models: ManagedModelIdentity
    retrieval_quality: RetrievalQualityMetrics
    retrieval_cases: tuple[RetrievalCaseResult, ...]
    retrieval_latency: RetrievalLatencySummary
    answer_quality: AnswerQualitySummary
    answer_cases: tuple[ManagedAnswerCaseReport, ...]

    @model_validator(mode="after")
    def identities_and_counts_are_consistent(self) -> ManagedRAGEvaluationReport:
        if self.case_count != len(self.retrieval_cases):
            raise ValueError("case_count must equal retrieval case count")
        if self.mode == "retrieval_only" and self.answer_cases:
            raise ValueError("retrieval-only reports cannot contain generated answers")
        if self.mode == "generated_answers" and self.case_count != len(self.answer_cases):
            raise ValueError("generated reports require one answer result per case")
        if any(case.generation != self.generation for case in self.retrieval_cases):
            raise ValueError("retrieval cases must use the report generation")
        return self


class CollectingRetrievalTelemetrySink(RetrievalTelemetrySink):
    """Evaluation-local in-memory sink; reports aggregate safe latency only."""

    def __init__(self) -> None:
        self._events: list[RetrievalMetrics] = []

    def emit(self, metrics: RetrievalMetrics) -> None:
        self._events.append(metrics)

    def checkpoint(self) -> int:
        return len(self._events)

    def since(self, checkpoint: int) -> tuple[RetrievalMetrics, ...]:
        return tuple(self._events[checkpoint:])


class MLflowJudgeRunner(Protocol):
    def __call__(
        self,
        cases: Sequence[EvaluationCase],
        results: Sequence[RAGCaseResult],
        settings: MLflowSettings,
    ) -> MLflowJudgeResults: ...


class RagasBuilder(Protocol):
    def __call__(
        self,
        *,
        enabled: bool,
        judge_model: str | None,
        judge_region: str | None,
        references: dict[str, str],
        execution_context: RagasExecutionTraceContext | None = None,
        execution_tracer: Tracer | None = None,
        judge_max_tokens: int | None = None,
        context_precision_max_tokens: int | None = None,
        debug_capture: JudgeCallDebugCapture | None = None,
    ) -> RagasEvaluator: ...


TraceIdProvider = Callable[[], str | None]


def load_managed_rag_dataset(path: str | Path) -> ManagedRAGEvaluationDataset:
    """Load only the reviewed PUBLIC managed dataset contract."""

    dataset_path = Path(path)
    try:
        payload = json.loads(dataset_path.read_text(encoding="utf-8"))
        return ManagedRAGEvaluationDataset.model_validate(payload)
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise ManagedEvaluationDatasetError(
            f"Managed evaluation dataset is invalid: {type(exc).__name__}"
        ) from exc


def validate_managed_dataset_compatibility(
    dataset: ManagedRAGEvaluationDataset,
    runtime: ManagedRetrievalRuntime,
    *,
    configured_corpus_key: str,
    cases: Sequence[ManagedRAGEvaluationCase] | None = None,
) -> None:
    """Reject a dataset that does not describe the pinned ACTIVE collection."""

    if dataset.corpus_key != configured_corpus_key:
        raise ManagedEvaluationDatasetError(
            "Managed evaluation dataset corpus_key does not match configured managed corpus"
        )
    selected = tuple(dataset.cases if cases is None else cases)
    if not selected:
        raise ManagedEvaluationDatasetError(
            "At least one managed evaluation case is required"
        )
    inventory_reader = getattr(runtime.retrieval_service.store, "indexed_chunk_hashes", None)
    if not callable(inventory_reader):
        raise ManagedEvaluationDatasetError(
            "Managed retrieval store cannot expose a read-only chunk identity inventory"
        )
    inventory = inventory_reader()
    if not isinstance(inventory, Mapping):
        raise ManagedEvaluationDatasetError("Managed chunk identity inventory is invalid")
    expected = {chunk_id for case in selected for chunk_id in case.expected_chunk_ids}
    missing = sorted(expected - set(inventory))
    if missing:
        raise ManagedEvaluationDatasetError(
            f"Managed evaluation dataset references {len(missing)} chunks absent from the "
            "pinned ACTIVE generation"
        )


def evaluate_managed_rag(
    runtime: ManagedRetrievalRuntime,
    dataset: ManagedRAGEvaluationDataset,
    *,
    configured_corpus_key: str,
    selected_cases: Sequence[ManagedRAGEvaluationCase] | None = None,
    pipeline: RAGPipeline | None = None,
    generation_model: str | None = None,
    generation_region: str | None = None,
    judge_model: str | None = None,
    with_mlflow_judges: bool = False,
    with_ragas: bool = False,
    mlflow_settings: MLflowSettings | None = None,
    telemetry_sink: CollectingRetrievalTelemetrySink | None = None,
    trace_id_provider: TraceIdProvider | None = None,
    mlflow_judge_runner: MLflowJudgeRunner = run_mlflow_judges,
    ragas_builder: RagasBuilder = build_ragas_evaluator,
    judge_preflight: JudgePreflightResult | None = None,
    evaluation_configuration_id: str | None = None,
    evaluation_run_id: str | None = None,
    ragas_debug_capture: JudgeCallDebugCapture | None = None,
    ragas_execution_tracer: Tracer | None = None,
    generated_at: datetime | None = None,
) -> ManagedRAGEvaluationReport:
    """Evaluate retrieval and optionally answers through one already-pinned runtime."""

    cases = tuple(dataset.cases if selected_cases is None else selected_cases)
    if not cases:
        raise ValueError("at least one managed evaluation case is required")
    generate = pipeline is not None
    if not generate and (with_mlflow_judges or with_ragas):
        raise ValueError("answer judges require generated-answer mode")
    if ragas_debug_capture is not None and not with_ragas:
        raise ValueError("judge-call debug capture requires explicit RAGAS evaluation")
    if ragas_debug_capture is not None and (
        mlflow_settings is None or not mlflow_settings.enabled
    ):
        raise ValueError("judge-call debug capture requires active MLflow span correlation")
    if (
        (with_mlflow_judges or with_ragas)
        and mlflow_settings is not None
        and judge_model != mlflow_settings.judge_model
    ):
        raise ValueError("MLflow and RAGAS must use the configured judge model identity")
    actual_generation_model = pipeline.llm.model_name if pipeline is not None else None
    if (
        generation_model is not None
        and actual_generation_model is not None
        and generation_model != actual_generation_model
    ):
        raise ValueError("generation model identity does not match the supplied pipeline")
    validate_managed_dataset_compatibility(
        dataset,
        runtime,
        configured_corpus_key=configured_corpus_key,
        cases=cases,
    )

    checkpoint = telemetry_sink.checkpoint() if telemetry_sink is not None else 0
    retrieval_report = evaluate_managed_retrieval(
        runtime,
        tuple(case.retrieval_case() for case in cases),
    )
    retrieval_events = telemetry_sink.since(checkpoint) if telemetry_sink is not None else ()

    generated_results: list[RAGCaseResult] = []
    generation_latencies: list[float] = []
    if pipeline is not None:
        for case in cases:
            started = perf_counter()
            response = pipeline.answer(
                case.question,
                access_context=case.authorization.access_context(),
                pii_policy=PIIHandlingPolicy.REDACT,
            )
            generation_latencies.append((perf_counter() - started) * 1000.0)
            generated_results.append(
                RAGCaseResult(
                    case_id=case.case_id,
                    question=case.question,
                    answer=response.answer,
                    contexts=[item.chunk.text for item in response.retrieved_chunks],
                    chunk_ids=[item.chunk.chunk_id for item in response.retrieved_chunks],
                    token_usage=bedrock_token_usage(response.generation_metadata),
                    request_id=response.request_id,
                    trace_id=(
                        response.trace_id
                        or (
                            trace_id_provider()
                            if trace_id_provider is not None
                            else None
                        )
                    ),
                )
            )

    mlflow_summary, mlflow_by_case = _run_mlflow_stage(
        cases,
        generated_results,
        requested=with_mlflow_judges,
        settings=mlflow_settings,
        runner=mlflow_judge_runner,
        preflight=(judge_preflight.mlflow_genai if judge_preflight else None),
    )
    ragas_summary, ragas_by_case = _run_ragas_stage(
        cases,
        generated_results,
        requested=with_ragas,
        judge_model=judge_model,
        judge_region=(mlflow_settings.judge_region if mlflow_settings else None),
        builder=ragas_builder,
        preflight=(judge_preflight.ragas if judge_preflight else None),
        evaluation_configuration_id=(
            evaluation_configuration_id
            or _managed_evaluation_configuration_id(
                dataset=dataset,
                cases=cases,
                generation=runtime.generation,
                judge_model=judge_model,
                judge_region=(mlflow_settings.judge_region if mlflow_settings else None),
            )
        ),
        evaluation_run_id=evaluation_run_id or mlflow_summary.run_id,
        mlflow_settings=mlflow_settings,
        debug_capture=ragas_debug_capture,
        execution_tracer=ragas_execution_tracer,
    )
    answer_cases = _safe_answer_cases(
        generated_results,
        generation_latencies,
        mlflow_summary=mlflow_summary,
        mlflow_by_case=mlflow_by_case,
        ragas_summary=ragas_summary,
        ragas_by_case=ragas_by_case,
    )
    timestamp = generated_at or datetime.now(UTC)
    if timestamp.tzinfo is None:
        raise ValueError("generated_at must be timezone-aware")
    service = runtime.retrieval_service
    return ManagedRAGEvaluationReport(
        generated_at=timestamp,
        mode="generated_answers" if pipeline is not None else "retrieval_only",
        corpus_key=configured_corpus_key,
        snapshot_hash=runtime.snapshot_hash,
        corpus_document_count=runtime.document_count,
        corpus_chunk_count=runtime.chunk_count,
        dataset_name=dataset.name,
        dataset_version=dataset.version,
        dataset_format_version=dataset.format_version,
        dataset_privacy_classification=dataset.privacy_classification,
        case_count=len(cases),
        generation=runtime.generation,
        index_state_id=runtime.index_state_id,
        qdrant_collection=runtime.qdrant_collection,
        embedding_fingerprint=runtime.embedding_fingerprint,
        chunking_fingerprint=runtime.chunking_fingerprint,
        index_configuration_fingerprint=runtime.index_configuration_fingerprint,
        models=ManagedModelIdentity(
            embedding_model=service.embedding_provider.model_name,
            reranker_model=service.reranker.model_name,
            generation_model=actual_generation_model,
            generation_region=generation_region,
            judge_model=judge_model if with_mlflow_judges or with_ragas else None,
            judge_region=(
                mlflow_settings.judge_region
                if mlflow_settings is not None and (with_mlflow_judges or with_ragas)
                else None
            ),
        ),
        retrieval_quality=retrieval_report.metrics,
        retrieval_cases=retrieval_report.cases,
        retrieval_latency=_retrieval_latency(retrieval_events, runtime),
        answer_quality=AnswerQualitySummary(
            generation=_generation_summary(generated_results, generation_latencies),
            mlflow_genai=mlflow_summary,
            ragas=ragas_summary,
        ),
        answer_cases=answer_cases,
    )


def evaluate_approved_answer(
    case: ManagedRAGEvaluationCase,
    result: RAGCaseResult,
    *,
    generation_model: str,
    judge_model: str | None,
    with_mlflow_judges: bool,
    with_ragas: bool,
    mlflow_settings: MLflowSettings,
    mlflow_judge_runner: MLflowJudgeRunner = run_mlflow_judges,
    ragas_builder: RagasBuilder = build_ragas_evaluator,
    judge_preflight: JudgePreflightResult | None = None,
    evaluation_configuration_id: str | None = None,
    evaluation_run_id: str | None = None,
    ragas_debug_capture: JudgeCallDebugCapture | None = None,
    ragas_execution_tracer: Tracer | None = None,
) -> ApprovedAnswerEvaluation:
    """Judge one existing answer only when it is bound to one reviewed PUBLIC case."""

    if result.case_id != case.case_id or result.question != case.question:
        raise ManagedEvaluationDatasetError(
            "Existing answer does not match the approved evaluation case"
        )
    if case.authorization.visibility != "PUBLIC" or case.authorization.groups:
        raise ManagedEvaluationDatasetError(
            "Existing-answer evaluation requires reviewed PUBLIC authorization"
        )
    if ragas_debug_capture is not None and not with_ragas:
        raise ValueError("judge-call debug capture requires explicit RAGAS evaluation")
    if ragas_debug_capture is not None and not mlflow_settings.enabled:
        raise ValueError("judge-call debug capture requires active MLflow span correlation")
    if (
        with_mlflow_judges or with_ragas
    ) and judge_model != mlflow_settings.judge_model:
        raise ValueError("MLflow and RAGAS must use the configured judge model identity")
    mlflow_summary, _ = _run_mlflow_stage(
        [case],
        [result],
        requested=with_mlflow_judges,
        settings=mlflow_settings,
        runner=mlflow_judge_runner,
        preflight=(judge_preflight.mlflow_genai if judge_preflight else None),
    )
    ragas_summary, _ = _run_ragas_stage(
        [case],
        [result],
        requested=with_ragas,
        judge_model=judge_model,
        judge_region=mlflow_settings.judge_region,
        builder=ragas_builder,
        preflight=(judge_preflight.ragas if judge_preflight else None),
        evaluation_configuration_id=(
            evaluation_configuration_id
            or _approved_answer_configuration_id(
                case=case,
                generation_model=generation_model,
                judge_model=judge_model,
                judge_region=mlflow_settings.judge_region,
            )
        ),
        evaluation_run_id=evaluation_run_id or mlflow_summary.run_id,
        mlflow_settings=mlflow_settings,
        debug_capture=ragas_debug_capture,
        execution_tracer=ragas_execution_tracer,
    )
    return ApprovedAnswerEvaluation(
        case_id=case.case_id,
        generation_model=generation_model,
        judge_model=judge_model if with_mlflow_judges or with_ragas else None,
        judge_region=(
            mlflow_settings.judge_region if with_mlflow_judges or with_ragas else None
        ),
        mlflow_genai=mlflow_summary,
        ragas=ragas_summary,
    )


def write_managed_rag_report(
    report: ManagedRAGEvaluationReport,
    path: str | Path,
) -> Path:
    """Write the safe report; questions, answers, contexts, and judge rationales are absent."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return output


def log_managed_rag_report(
    report: ManagedRAGEvaluationReport,
    report_path: Path,
    run: EvaluationRun,
) -> None:
    """Log safe aggregates and the content-free report to one caller-owned run."""

    if not run.active:
        return
    run.log_parameters(
        {
            "managed_rag.format_version": report.format_version,
            "managed_rag.mode": report.mode,
            "managed_rag.corpus_key": report.corpus_key,
            "managed_rag.dataset_name": report.dataset_name,
            "managed_rag.dataset_version": report.dataset_version,
            "managed_rag.dataset_privacy": report.dataset_privacy_classification,
            "managed_rag.generation": report.generation,
            "managed_rag.index_state_id": str(report.index_state_id),
            "managed_rag.qdrant_collection": report.qdrant_collection,
            "managed_rag.embedding_fingerprint": report.embedding_fingerprint,
            "managed_rag.chunking_fingerprint": report.chunking_fingerprint,
            "managed_rag.index_configuration_fingerprint": (
                report.index_configuration_fingerprint
            ),
            "managed_rag.embedding_model": report.models.embedding_model,
            "managed_rag.reranker_model": report.models.reranker_model,
            "managed_rag.generation_model": report.models.generation_model,
            "managed_rag.judge_model": report.models.judge_model,
            "managed_rag.case_count": report.case_count,
        }
    )
    metrics = {
        "managed_rag.retrieval.recall_at_k": report.retrieval_quality.recall_at_k,
        "managed_rag.retrieval.mrr": report.retrieval_quality.mrr,
        "managed_rag.retrieval.hit_rate": report.retrieval_quality.hit_rate,
        "managed_rag.retrieval.missed_cases": float(
            report.retrieval_quality.missed_case_count
        ),
    }
    if report.retrieval_latency.mean_total_ms is not None:
        metrics["managed_rag.retrieval.mean_latency_ms"] = (
            report.retrieval_latency.mean_total_ms
        )
    if report.answer_quality.generation.mean_latency_ms is not None:
        metrics["managed_rag.generation.mean_latency_ms"] = (
            report.answer_quality.generation.mean_latency_ms
        )
    for backend, summary in (
        ("mlflow_genai", report.answer_quality.mlflow_genai),
        ("ragas", report.answer_quality.ragas),
    ):
        metrics.update(
            {
                f"managed_rag.{backend}.{name}": value
                for name, value in summary.aggregate_metrics.items()
            }
        )
    run.log_metrics(metrics)
    run.log_tags(
        {
            "managed_rag.mlflow_genai_status": report.answer_quality.mlflow_genai.status,
            "managed_rag.ragas_status": report.answer_quality.ragas.status,
        }
    )
    run.log_artifacts([report_path])


def _run_mlflow_stage(
    cases: Sequence[ManagedRAGEvaluationCase],
    results: Sequence[RAGCaseResult],
    *,
    requested: bool,
    settings: MLflowSettings | None,
    runner: MLflowJudgeRunner,
    preflight: JudgeBackendPreflight | None = None,
) -> tuple[JudgeStageSummary, dict[str, dict[str, JudgeMetricValue]]]:
    if not requested:
        return (
            JudgeStageSummary(
                status="NOT_REQUESTED",
                reason="MLflow GenAI judges were not explicitly requested.",
            ),
            {},
        )
    if settings is None:
        raise ValueError("MLflow settings are required for MLflow judges")
    if not settings.judge_model:
        return (
            JudgeStageSummary(
                status="UNAVAILABLE",
                reason="MLflow GenAI judge model is not configured.",
            ),
            {},
        )
    if preflight is not None and preflight.status is not JudgePreflightStatus.AVAILABLE:
        return (
            JudgeStageSummary(
                status="UNAVAILABLE",
                reason=preflight.reason,
                metric_statuses={
                    name: "UNAVAILABLE" for name in preflight.unavailable_metrics
                },
                failed_metrics=preflight.unavailable_metrics,
            ),
            {},
        )
    try:
        result = runner([case.phase2_case() for case in cases], results, settings)
    except Exception as exc:
        status, category = _judge_exception_status(exc)
        return (
            JudgeStageSummary(
                status=status,
                reason=(
                    "Explicitly requested MLflow GenAI judging did not complete "
                    f"({category})."
                ),
            ),
            {},
        )
    skipped: dict[str, str] = {
        "retrieval_groundedness": (
            "Built-in MLflow RetrievalGroundedness is unavailable for the configured "
            "Bedrock provider; RAGAS groundedness metrics remain separate."
        )
    }
    metric_statuses = {
        name: diagnostic.status.value
        for name, diagnostic in result.metric_diagnostics.items()
    }
    failed_metrics = {
        name: diagnostic.error_category or "SCORER_ERROR"
        for name, diagnostic in result.metric_diagnostics.items()
        if diagnostic.status is not MLflowMetricStatus.COMPLETED
    }
    if result.metric_diagnostics:
        completed_count = sum(
            diagnostic.status is MLflowMetricStatus.COMPLETED
            for diagnostic in result.metric_diagnostics.values()
        )
        if completed_count == len(result.metric_diagnostics):
            stage_status: StageStatus = "COMPLETED"
            reason = "All requested MLflow GenAI metrics produced valid results."
        elif completed_count:
            stage_status = "PARTIAL"
            reason = (
                f"{completed_count} of {len(result.metric_diagnostics)} requested MLflow "
                "GenAI metrics produced valid results."
            )
        else:
            stage_status = "FAILED"
            reason = "No requested MLflow GenAI metric produced a valid result."
    else:
        # Compatibility for injected runners predating metric diagnostics.
        stage_status = "COMPLETED" if result.by_case else "FAILED"
        reason = (
            "The injected MLflow GenAI runner returned case results."
            if result.by_case
            else "The MLflow GenAI runner returned no metric results."
        )
    return (
        JudgeStageSummary(
            status=stage_status,
            reason=reason,
            aggregate_metrics=_numeric_metrics(result.aggregate),
            skipped_metrics=skipped,
            metric_statuses=metric_statuses,
            failed_metrics=failed_metrics,
            run_id=result.run_id,
        ),
        result.by_case,
    )


def _configuration_id(payload: Mapping[str, object]) -> str:
    """Return a stable content-free configuration identity for trace correlation."""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _managed_evaluation_configuration_id(
    *,
    dataset: ManagedRAGEvaluationDataset,
    cases: Sequence[ManagedRAGEvaluationCase],
    generation: int,
    judge_model: str | None,
    judge_region: str | None,
) -> str:
    return _configuration_id(
        {
            "kind": "managed_ragas_evaluation",
            "dataset": dataset.name,
            "dataset_version": dataset.version,
            "case_ids": [case.case_id for case in cases],
            "generation": generation,
            "judge_model": judge_model,
            "judge_region": judge_region,
            "metrics": [
                "faithfulness",
                "context_precision",
                "context_recall",
                "response_groundedness",
            ],
        }
    )


def managed_evaluation_configuration_id(
    *,
    dataset: ManagedRAGEvaluationDataset,
    cases: Sequence[ManagedRAGEvaluationCase],
    generation: int,
    judge_model: str | None,
    judge_region: str | None,
) -> str:
    """Public content-free identity used by explicit debug and execution paths."""

    return _managed_evaluation_configuration_id(
        dataset=dataset,
        cases=cases,
        generation=generation,
        judge_model=judge_model,
        judge_region=judge_region,
    )


def create_approved_public_judge_debug_capture(
    config: JudgeCallDebugConfig,
    *,
    dataset: ManagedRAGEvaluationDataset,
    cases: Sequence[ManagedRAGEvaluationCase],
    evaluation_configuration_id: str,
    evaluation_run_id: str | None,
    judge_model: str,
    judge_region: str,
) -> JudgeCallDebugCapture:
    """Authorize a separate local content artifact for exact reviewed PUBLIC cases."""

    selected = tuple(cases)
    if dataset.status != "reviewed" or (
        dataset.privacy_classification != "PUBLIC_SYNTHETIC_REVIEWED"
    ):
        raise ManagedEvaluationDatasetError(
            "Judge-call debug capture requires the reviewed PUBLIC dataset"
        )
    by_id = {case.case_id: case for case in dataset.cases}
    if not selected or any(by_id.get(case.case_id) != case for case in selected):
        raise ManagedEvaluationDatasetError(
            "Judge-call debug capture requires exact reviewed dataset cases"
        )
    approved = tuple(
        ApprovedPublicDebugCase(
            case_id=case.case_id,
            question=case.question,
            review_status=case.review_status,
            visibility=case.authorization.visibility,
            groups=case.authorization.groups,
        )
        for case in selected
    )
    return JudgeCallDebugCapture(
        config,
        cases=approved,
        evaluation_configuration_id=evaluation_configuration_id,
        evaluation_run_id=evaluation_run_id,
        model_id=judge_model.removeprefix("bedrock:/"),
        region=judge_region,
    )


def _approved_answer_configuration_id(
    *,
    case: ManagedRAGEvaluationCase,
    generation_model: str,
    judge_model: str | None,
    judge_region: str | None,
) -> str:
    return _configuration_id(
        {
            "kind": "approved_answer_ragas_evaluation",
            "case_id": case.case_id,
            "generation_model": generation_model,
            "judge_model": judge_model,
            "judge_region": judge_region,
            "metrics": [
                "faithfulness",
                "context_precision",
                "context_recall",
                "response_groundedness",
            ],
        }
    )


def _run_ragas_stage(
    cases: Sequence[ManagedRAGEvaluationCase],
    results: Sequence[RAGCaseResult],
    *,
    requested: bool,
    judge_model: str | None,
    judge_region: str | None,
    builder: RagasBuilder,
    preflight: JudgeBackendPreflight | None = None,
    evaluation_configuration_id: str,
    evaluation_run_id: str | None,
    mlflow_settings: MLflowSettings | None,
    debug_capture: JudgeCallDebugCapture | None = None,
    execution_tracer: Tracer | None = None,
) -> tuple[JudgeStageSummary, dict[str, dict[str, JudgeMetricValue]]]:
    if not requested:
        return (
            JudgeStageSummary(
                status="NOT_REQUESTED",
                reason="RAGAS evaluation was not explicitly requested.",
            ),
            {},
        )
    if not judge_model:
        return (
            JudgeStageSummary(
                status="UNAVAILABLE",
                reason="RAGAS judge model is not configured.",
            ),
            {},
        )
    if preflight is not None and preflight.status is not JudgePreflightStatus.AVAILABLE:
        return (
            JudgeStageSummary(status="UNAVAILABLE", reason=preflight.reason),
            {},
        )
    evaluator: RagasEvaluator | None = None
    try:
        if debug_capture is not None:
            if debug_capture.evaluation_configuration_id != evaluation_configuration_id:
                raise ValueError(
                    "Judge-call debug capture configuration does not match evaluation"
                )
            debug_capture.attach_evaluation_run_id(evaluation_run_id)
            for result in results:
                debug_capture.bind_case_inputs(
                    case_id=result.case_id,
                    question=result.question,
                    contexts=result.contexts,
                    chunk_ids=result.chunk_ids,
                )
        execution_context = RagasExecutionTraceContext(
            configuration_id=evaluation_configuration_id,
            evaluation_run_id=evaluation_run_id,
            request_ids={result.case_id: result.request_id for result in results},
            rag_trace_ids={result.case_id: result.trace_id for result in results},
        )
        evaluator = builder(
            enabled=True,
            judge_model=judge_model,
            judge_region=judge_region,
            references={case.case_id: case.reference_answer for case in cases},
            execution_context=execution_context,
            execution_tracer=(
                execution_tracer
                if execution_tracer is not None
                else build_tracer(mlflow_settings)
                if mlflow_settings
                else None
            ),
            judge_max_tokens=(
                mlflow_settings.ragas_judge_max_tokens if mlflow_settings else None
            ),
            context_precision_max_tokens=(
                mlflow_settings.ragas_context_precision_max_tokens
                if mlflow_settings
                else None
            ),
            debug_capture=debug_capture,
        )
        by_case = evaluator.evaluate(results)
    except Exception as exc:
        status, category = _judge_exception_status(exc)
        usage = _evaluator_usage(evaluator)
        return (
            JudgeStageSummary(
                status=status,
                reason=(
                    "Explicitly requested RAGAS evaluation did not complete "
                    f"({category})."
                ),
                usage=usage,
                execution_trace=_evaluator_execution_trace(evaluator),
            ),
            {},
        )
    usage = _evaluator_usage(evaluator)
    return (
        JudgeStageSummary(
            status="COMPLETED",
            aggregate_metrics=_aggregate_case_metrics(by_case),
            usage=usage,
            execution_trace=_evaluator_execution_trace(evaluator),
        ),
        by_case,
    )


def _evaluator_usage(evaluator: RagasEvaluator | None) -> ModelUsageSummary | None:
    if evaluator is None:
        return None
    usage = getattr(evaluator, "usage_summary", None)
    return usage if isinstance(usage, ModelUsageSummary) else None


def _evaluator_execution_trace(
    evaluator: RagasEvaluator | None,
) -> RagasExecutionTraceReference | None:
    if evaluator is None:
        return None
    reference = getattr(evaluator, "execution_trace", None)
    return reference if isinstance(reference, RagasExecutionTraceReference) else None


def _safe_answer_cases(
    results: Sequence[RAGCaseResult],
    latencies: Sequence[float],
    *,
    mlflow_summary: JudgeStageSummary,
    mlflow_by_case: Mapping[str, Mapping[str, JudgeMetricValue]],
    ragas_summary: JudgeStageSummary,
    ragas_by_case: Mapping[str, Mapping[str, JudgeMetricValue]],
) -> tuple[ManagedAnswerCaseReport, ...]:
    safe: list[ManagedAnswerCaseReport] = []
    for result, latency in zip(results, latencies, strict=True):
        mlflow_values = mlflow_by_case.get(result.case_id, {})
        ragas_values = ragas_by_case.get(result.case_id, {})
        safe.append(
            ManagedAnswerCaseReport(
                case_id=result.case_id,
                request_id=result.request_id,
                trace_id=result.trace_id,
                generation_status="COMPLETED",
                answer_character_count=len(result.answer),
                citation_count=len(result.chunk_ids),
                cited_chunk_ids=tuple(result.chunk_ids),
                latency_ms=latency,
                token_usage=result.token_usage,
                scoring=CaseScoringState(
                    mlflow_genai=_case_stage_status(mlflow_summary, mlflow_values),
                    ragas=_case_stage_status(ragas_summary, ragas_values),
                ),
                mlflow_judges=_safe_metrics(mlflow_values),
                ragas=_safe_metrics(ragas_values),
            )
        )
    return tuple(safe)


def _case_stage_status(
    summary: JudgeStageSummary,
    values: Mapping[str, JudgeMetricValue],
) -> CaseScoringStatus:
    if summary.status == "NOT_REQUESTED":
        return "NOT_REQUESTED"
    if summary.status == "FAILED":
        return "FAILED"
    if summary.status == "PARTIAL":
        return "PARTIAL"
    return "COMPLETED" if values else "UNAVAILABLE"


def _safe_metrics(
    metrics: Mapping[str, JudgeMetricValue],
) -> dict[str, SafeMetricValue]:
    return {name: SafeMetricValue(value=metric.value) for name, metric in metrics.items()}


def _numeric_metrics(metrics: Mapping[str, object]) -> dict[str, float]:
    return {
        name: float(value)
        for name, value in metrics.items()
        if isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    }


def _judge_exception_status(exc: Exception) -> tuple[StageStatus, str]:
    """Return a content-free operational category for private diagnostics."""

    message = str(exc).casefold()
    if "not configured" in message or "missing judge model" in message:
        return "UNAVAILABLE", "JUDGE_MODEL_UNAVAILABLE"
    if (
        "credential" in message
        or "access key" in message
        or "awsidandkey" in message
        or "could not connect" in message
        or "endpoint" in message
        or "provider unavailable" in message
    ):
        return "UNAVAILABLE", "JUDGE_PROVIDER_UNAVAILABLE"
    if "cannot use agenerate" in message and "synchronous client" in message:
        return "FAILED", "CLIENT_MODE_MISMATCH"
    return "FAILED", type(exc).__name__


def _aggregate_case_metrics(
    by_case: Mapping[str, Mapping[str, JudgeMetricValue]],
) -> dict[str, float]:
    values: dict[str, list[float]] = {}
    for metrics in by_case.values():
        for name, metric in metrics.items():
            value = metric.value
            if isinstance(value, int | float) and not isinstance(value, bool):
                values.setdefault(name, []).append(float(value))
    return {name: fmean(items) for name, items in values.items() if items}


def _retrieval_latency(
    events: Sequence[RetrievalMetrics],
    runtime: ManagedRetrievalRuntime,
) -> RetrievalLatencySummary:
    successful = [event for event in events if event.status is RetrievalStatus.SUCCESS]
    if any(
        event.generation != runtime.generation
        or event.index_state_id != runtime.index_state_id
        or event.qdrant_collection != runtime.qdrant_collection
        for event in successful
    ):
        raise ValueError("retrieval telemetry does not match the pinned managed generation")
    return RetrievalLatencySummary(
        request_count=len(successful),
        mean_total_ms=_mean(event.total_retrieval_latency_ms for event in successful),
        mean_dense_ms=_mean_optional(event.dense_search_latency_ms for event in successful),
        mean_sparse_ms=_mean_optional(event.sparse_search_latency_ms for event in successful),
        mean_fusion_ms=_mean_optional(event.fusion_latency_ms for event in successful),
        mean_reranker_ms=_mean_optional(event.reranker_latency_ms for event in successful),
        mean_pii_ms=_mean_optional(event.pii_processing_latency_ms for event in successful),
    )


def _generation_summary(
    results: Sequence[RAGCaseResult],
    latencies: Sequence[float],
) -> GenerationSummary:
    if not results:
        return GenerationSummary(
            status="NOT_REQUESTED",
            generated_case_count=0,
            token_usage=TokenUsageSummary(available_case_count=0),
            reason="Bedrock answer generation was not explicitly requested.",
        )
    available = [result.token_usage for result in results if result.token_usage is not None]
    all_available = len(available) == len(results)
    return GenerationSummary(
        status="COMPLETED",
        generated_case_count=len(results),
        mean_latency_ms=_mean(latencies),
        token_usage=TokenUsageSummary(
            available_case_count=len(available),
            input_tokens=(
                sum(usage["input_tokens"] for usage in available) if all_available else None
            ),
            output_tokens=(
                sum(usage["output_tokens"] for usage in available) if all_available else None
            ),
            total_tokens=(
                sum(usage["total_tokens"] for usage in available) if all_available else None
            ),
        ),
    )


def _mean(values: Iterable[float]) -> float | None:
    materialized = list(values)
    return fmean(materialized) if materialized else None


def _mean_optional(values: Iterable[float | None]) -> float | None:
    materialized = [value for value in values if value is not None]
    return fmean(materialized) if materialized else None
