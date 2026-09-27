"""Fail-closed public showcase boundary around the managed RAG runtime."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager, suppress
from decimal import Decimal
from pathlib import Path
from threading import BoundedSemaphore, Lock
from time import monotonic
from urllib.parse import urlsplit
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from consumer_finance_rag.access import AccessContext, AccessVisibility
from consumer_finance_rag.config import AppSettings, PublicShowcaseSettings
from consumer_finance_rag.evaluation.managed_rag import ManagedRAGEvaluationReport
from consumer_finance_rag.factory import ApplicationRuntime
from consumer_finance_rag.models import RAGResponse, RetrievedChunk, StructuredContext
from consumer_finance_rag.observability import (
    RetrievalMetrics,
    RetrievalStatus,
    RetrievalTelemetrySink,
    TraceSpan,
    bedrock_token_usage,
)
from consumer_finance_rag.pii import PIIHandlingPolicy
from consumer_finance_rag.presentation import citation_references
from consumer_finance_rag.quality_policy import (
    QualityDecision,
    QualityProfile,
    QualityStatus,
    RuntimeQualityContext,
)
from consumer_finance_rag.runtime_groundedness import RuntimeQualityGate


class PublicShowcaseConfigurationError(RuntimeError):
    """Raised when the public app is not configured for safe managed serving."""


class PublicRequestError(RuntimeError):
    """Bounded error safe to render without exposing an internal exception."""

    def __init__(
        self,
        code: str,
        public_message: str,
        *,
        request_id: str | None = None,
    ) -> None:
        super().__init__(public_message)
        self.code = code
        self.public_message = public_message
        self.request_id = request_id


class PublicCitation(BaseModel):
    model_config = ConfigDict(frozen=True)

    citation_id: str
    title: str
    source_url: str | None
    chunk_id: str


class PublicRetrievalStage(BaseModel):
    model_config = ConfigDict(frozen=True)

    stage: str
    latency_ms: float | None = Field(default=None, ge=0)
    candidate_count: int | None = Field(default=None, ge=0)


class PublicTraceSummary(BaseModel):
    """Allowlisted request data safe for an anonymous visitor."""

    model_config = ConfigDict(frozen=True)

    request_id: str
    mlflow_trace_id: str | None
    telemetry_status: str
    generation: int = Field(ge=1)
    index_state_id: UUID
    qdrant_collection: str
    total_retrieval_latency_ms: float | None = Field(default=None, ge=0)
    stages: tuple[PublicRetrievalStage, ...]
    acl_filter_applied: bool | None
    request_group_count: int | None = Field(default=None, ge=0)
    rejected_candidate_count: int | None = Field(default=None, ge=0)
    pii_policy: str | None
    redacted_context_count: int | None = Field(default=None, ge=0)
    redacted_pii_type_counts: dict[str, int]


class PublicModelMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    embedding_model: str
    reranker_model: str
    generation_model: str


class PublicQualityDecision(BaseModel):
    """Allowlisted public gate result without rationale, trace, or provider identifiers."""

    model_config = ConfigDict(frozen=True)

    status: QualityStatus
    profile: QualityProfile
    policy_version: str
    blocking_judge_invoked: bool
    estimated_judge_cost_usd: Decimal | None = Field(default=None, ge=0)


class PublicAnswer(BaseModel):
    """Public response without retrieved text or unrestricted provider metadata."""

    model_config = ConfigDict(frozen=True)

    answer: str
    citations: tuple[PublicCitation, ...]
    trace: PublicTraceSummary
    models: PublicModelMetadata
    token_usage: dict[str, int] | None
    quality_decision: PublicQualityDecision


class RequestTelemetryCollector(RetrievalTelemetrySink):
    """Bounded process-local correlation store keyed by explicit request ID."""

    def __init__(self, *, max_pending: int = 256) -> None:
        if max_pending <= 0:
            raise ValueError("max_pending must be positive")
        self._max_pending = max_pending
        self._events: dict[str, RetrievalMetrics] = {}
        self._order: deque[str] = deque()
        self._lock = Lock()

    def emit(self, metrics: RetrievalMetrics) -> None:
        with self._lock:
            if metrics.request_id not in self._events:
                self._order.append(metrics.request_id)
            self._events[metrics.request_id] = metrics
            while len(self._order) > self._max_pending:
                oldest = self._order.popleft()
                self._events.pop(oldest, None)

    def pop(self, request_id: str) -> RetrievalMetrics | None:
        with self._lock:
            event = self._events.pop(request_id, None)
            with suppress(ValueError):
                self._order.remove(request_id)
            return event


Clock = Callable[[], float]


class PublicInferenceGate:
    """Process-wide bounded usage window plus non-blocking concurrency control."""

    def __init__(
        self,
        settings: PublicShowcaseSettings,
        *,
        clock: Clock = monotonic,
    ) -> None:
        self._settings = settings
        self._clock = clock
        self._requests: deque[float] = deque()
        self._lock = Lock()
        self._slots = BoundedSemaphore(settings.max_concurrent_requests)

    @contextmanager
    def admit(self) -> Iterator[None]:
        if not self._settings.inference_enabled:
            raise PublicRequestError(
                "inference_disabled",
                "Answer generation is temporarily disabled by the operator.",
            )
        if not self._slots.acquire(blocking=False):
            raise PublicRequestError(
                "capacity_reached",
                "The demo is currently serving its maximum number of requests. Try again shortly.",
            )
        try:
            now = self._clock()
            cutoff = now - self._settings.rate_window_seconds
            with self._lock:
                while self._requests and self._requests[0] <= cutoff:
                    self._requests.popleft()
                if len(self._requests) >= self._settings.max_requests_per_window:
                    raise PublicRequestError(
                        "usage_limit_reached",
                        "The demo usage limit has been reached. Try again after the "
                        "usage window resets.",
                    )
                self._requests.append(now)
            yield
        finally:
            self._slots.release()


def validate_public_showcase_runtime(
    settings: AppSettings,
    runtime: ApplicationRuntime,
) -> None:
    """Require one approved, generation-pinned, ACL/PII-managed runtime."""

    showcase = settings.public_showcase
    retrieval = runtime.retrieval
    service = retrieval.retrieval_service
    if not showcase.enabled:
        raise PublicShowcaseConfigurationError("The public showcase is disabled by configuration.")
    if settings.runtime.profile != "managed" or retrieval.profile != "managed":
        raise PublicShowcaseConfigurationError(
            "The public showcase requires the managed runtime profile."
        )
    if showcase.approved_corpus_key != settings.managed_ingestion.corpus_key:
        raise PublicShowcaseConfigurationError(
            "The configured managed corpus is not approved for the public showcase."
        )
    if retrieval.generation is None or retrieval.index_state_id is None:
        raise PublicShowcaseConfigurationError(
            "The public showcase requires a generation-pinned managed runtime."
        )
    if not service.access_control_enabled or not service.pii_handling_enabled:
        raise PublicShowcaseConfigurationError(
            "The public showcase requires managed ACL and PII enforcement."
        )
    if not settings.models.generation.model:
        raise PublicShowcaseConfigurationError(
            "The public showcase requires an explicitly configured Bedrock generation model."
        )
    if settings.quality_policy.default_profile is QualityProfile.EVALUATION:
        raise PublicShowcaseConfigurationError(
            "The public showcase runtime policy must be STANDARD or STRICT."
        )


def normalize_public_question(question: str, *, maximum_characters: int) -> str:
    normalized = question.strip()
    if not normalized:
        raise PublicRequestError("empty_question", "Enter a question first.")
    if len(normalized) > maximum_characters:
        raise PublicRequestError(
            "question_too_long",
            f"Questions are limited to {maximum_characters} characters.",
        )
    return normalized


class PublicRAGService:
    """Only callable public inference surface; identity and PII policy are fixed."""

    def __init__(
        self,
        settings: AppSettings,
        runtime: ApplicationRuntime,
        telemetry: RequestTelemetryCollector,
        gate: PublicInferenceGate,
        quality_gate: RuntimeQualityGate | None = None,
    ) -> None:
        validate_public_showcase_runtime(settings, runtime)
        self._settings = settings
        self._runtime = runtime
        self._telemetry = telemetry
        self._gate = gate
        self._quality_gate = quality_gate or RuntimeQualityGate(
            settings.quality_policy,
            runtime.pipeline.tracer,
        )

    def answer(self, question: str) -> PublicAnswer:
        normalized = normalize_public_question(
            question,
            maximum_characters=self._settings.public_showcase.max_question_characters,
        )
        request_id = str(uuid4())
        with self._gate.admit():
            try:
                detector = self._runtime.pipeline.retrieval.pii_detector
                safe_question = detector.redact(normalized)
                decisions: list[QualityDecision] = []
                relevance_decisions: list[QualityDecision] = []

                def approve_evidence(context: StructuredContext, parent_span: TraceSpan) -> None:
                    self._validate_public_result(
                        request_id,
                        request_id,
                        context.chunks,
                        None,
                    )
                    decision = self._quality_gate.evaluate_pre_generation(
                        context,
                        request_id=request_id,
                        parent_span=parent_span,
                    )
                    relevance_decisions.append(decision)
                    if not decision.deliver_answer:
                        raise PublicRequestError(
                            "insufficient_relevant_evidence",
                            "Unable to answer from available sources",
                        )

                def approve(candidate: RAGResponse, parent_span: TraceSpan) -> RAGResponse:
                    safe_candidate = candidate.model_copy(
                        update={"answer": detector.redact(candidate.answer)}
                    )
                    self._validate_public_result(
                        request_id,
                        safe_candidate.request_id,
                        safe_candidate.retrieved_chunks,
                        None,
                    )
                    references = citation_references(safe_candidate.answer)
                    available_citations = {item.citation_id for item in safe_candidate.citations}
                    relevance = relevance_decisions[-1].checks[0]
                    decision = self._quality_gate.evaluate(
                        safe_candidate,
                        RuntimeQualityContext(
                            request_id=request_id,
                            user_authorized=True,
                            corpus_authorized=True,
                            evidence_available=bool(safe_candidate.retrieved_chunks),
                            evidence_relevant=True,
                            evidence_relevance_score=relevance.score,
                            evidence_relevance_threshold=relevance.threshold,
                            citations_present=bool(references),
                            citations_valid=bool(references)
                            and references.issubset(available_citations),
                            output_schema_valid=bool(safe_candidate.answer.strip()),
                            pii_policy_satisfied=not bool(detector.detect(safe_candidate.answer)),
                        ),
                        parent_span=parent_span,
                    )
                    decisions.append(decision)
                    if not decision.deliver_answer:
                        raise PublicRequestError(
                            "quality_gate_blocked",
                            "The answer could not be approved from the available evidence.",
                        )
                    return safe_candidate

                response = self._runtime.pipeline.answer(
                    safe_question,
                    access_context=AccessContext(),
                    pii_policy=PIIHandlingPolicy.REDACT,
                    request_id=request_id,
                    pre_generation_gate=approve_evidence,
                    pre_response_gate=approve,
                )
                metrics = self._telemetry.pop(request_id)
                self._validate_public_result(
                    request_id,
                    response.request_id,
                    response.retrieved_chunks,
                    metrics,
                )
                safe_answer = response.answer
                quality_decision = decisions[-1]
                return PublicAnswer(
                    answer=safe_answer,
                    citations=tuple(
                        PublicCitation(
                            citation_id=item.citation_id,
                            title=item.document_title,
                            source_url=_safe_public_url(item.source_url),
                            chunk_id=item.chunk_id,
                        )
                        for item in response.citations
                    ),
                    trace=_public_trace_summary(
                        request_id=request_id,
                        trace_id=response.trace_id,
                        runtime=self._runtime,
                        metrics=metrics,
                    ),
                    models=PublicModelMetadata(
                        embedding_model=response.model_metadata.embedding_model,
                        reranker_model=response.model_metadata.reranker_model,
                        generation_model=response.model_metadata.generation_model,
                    ),
                    token_usage=bedrock_token_usage(response.generation_metadata),
                    quality_decision=PublicQualityDecision(
                        status=quality_decision.status,
                        profile=quality_decision.profile,
                        policy_version=quality_decision.policy_version,
                        blocking_judge_invoked=(
                            quality_decision.judge_execution is not None
                            and quality_decision.judge_execution.usage is not None
                            and quality_decision.judge_execution.usage.provider_call_count > 0
                        ),
                        estimated_judge_cost_usd=(
                            quality_decision.judge_execution.estimated_cost_usd
                            if quality_decision.judge_execution is not None
                            else None
                        ),
                    ),
                )
            except PublicRequestError as error:
                self._telemetry.pop(request_id)
                if error.request_id is None:
                    error.request_id = request_id
                raise
            except Exception as error:
                self._telemetry.pop(request_id)
                raise PublicRequestError(
                    "request_failed",
                    "The request could not be completed. Please try again later.",
                    request_id=request_id,
                ) from error

    def _validate_public_result(
        self,
        expected_request_id: str,
        response_request_id: str,
        chunks: Sequence[RetrievedChunk],
        metrics: RetrievalMetrics | None,
    ) -> None:
        if response_request_id != expected_request_id:
            raise PublicRequestError("correlation_failed", "Request validation failed.")
        if metrics is not None and metrics.request_id != expected_request_id:
            raise PublicRequestError("correlation_failed", "Request validation failed.")
        service = self._runtime.pipeline.retrieval
        for item in chunks:
            metadata = item.chunk.metadata
            text = item.chunk.text
            if metadata.get("access_visibility") != AccessVisibility.PUBLIC.value or metadata.get(
                "allowed_groups"
            ) not in ([], (), None):
                raise PublicRequestError("acl_failed", "Request validation failed.")
            if service.pii_detector.detect(text):
                raise PublicRequestError("pii_failed", "Request validation failed.")
        if metrics is None:
            return
        retrieval = self._runtime.retrieval
        if (
            metrics.status is not RetrievalStatus.SUCCESS
            or metrics.profile != "managed"
            or metrics.generation != retrieval.generation
            or metrics.index_state_id != retrieval.index_state_id
            or metrics.qdrant_collection != retrieval.qdrant_collection
            or not metrics.acl_filter_applied
            or metrics.request_group_count != 0
            or metrics.pii_policy != PIIHandlingPolicy.REDACT.value
        ):
            raise PublicRequestError("telemetry_failed", "Request validation failed.")


def load_managed_showcase_report(
    directory: str | Path,
    *,
    generation: int | None,
) -> ManagedRAGEvaluationReport | None:
    """Load only validated metadata-only managed report artifacts."""

    report_directory = Path(directory)
    candidates = (
        [report_directory / f"managed_rag_g{generation:06d}.json"]
        if generation is not None
        else sorted(report_directory.glob("managed_rag_g*.json"), reverse=True)
    )
    for path in candidates:
        try:
            return ManagedRAGEvaluationReport.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
    return None


def _public_trace_summary(
    *,
    request_id: str,
    trace_id: str | None,
    runtime: ApplicationRuntime,
    metrics: RetrievalMetrics | None,
) -> PublicTraceSummary:
    retrieval = runtime.retrieval
    if retrieval.generation is None or retrieval.index_state_id is None:
        raise PublicRequestError("runtime_identity_missing", "Request validation failed.")
    if metrics is None:
        return PublicTraceSummary(
            request_id=request_id,
            mlflow_trace_id=trace_id,
            telemetry_status="UNAVAILABLE",
            generation=retrieval.generation,
            index_state_id=retrieval.index_state_id,
            qdrant_collection=retrieval.qdrant_collection,
            stages=(),
            acl_filter_applied=None,
            request_group_count=None,
            rejected_candidate_count=None,
            pii_policy=None,
            redacted_context_count=None,
            redacted_pii_type_counts={},
        )
    return PublicTraceSummary(
        request_id=request_id,
        mlflow_trace_id=trace_id,
        telemetry_status=metrics.status.value,
        generation=retrieval.generation,
        index_state_id=retrieval.index_state_id,
        qdrant_collection=retrieval.qdrant_collection,
        total_retrieval_latency_ms=metrics.total_retrieval_latency_ms,
        stages=(
            PublicRetrievalStage(
                stage="Query embedding",
                latency_ms=metrics.query_embedding_latency_ms,
            ),
            PublicRetrievalStage(
                stage="Dense HNSW search",
                latency_ms=metrics.dense_search_latency_ms,
                candidate_count=metrics.dense_candidate_count,
            ),
            PublicRetrievalStage(
                stage="Sparse BM25 search",
                latency_ms=metrics.sparse_search_latency_ms,
                candidate_count=metrics.sparse_candidate_count,
            ),
            PublicRetrievalStage(
                stage="Reciprocal rank fusion",
                latency_ms=metrics.fusion_latency_ms,
                candidate_count=metrics.fused_candidate_count,
            ),
            PublicRetrievalStage(
                stage="Cross-encoder reranking",
                latency_ms=metrics.reranker_latency_ms,
                candidate_count=metrics.final_context_count,
            ),
            PublicRetrievalStage(
                stage="PII policy",
                latency_ms=metrics.pii_processing_latency_ms,
                candidate_count=metrics.final_context_count,
            ),
        ),
        acl_filter_applied=metrics.acl_filter_applied,
        request_group_count=0,
        rejected_candidate_count=metrics.rejected_candidate_count,
        pii_policy=metrics.pii_policy,
        redacted_context_count=metrics.redacted_context_count,
        redacted_pii_type_counts={
            item.pii_type: item.count for item in metrics.redacted_pii_type_counts
        },
    )


def _safe_public_url(value: str | None) -> str | None:
    if value is None:
        return None
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        return None
    return value
