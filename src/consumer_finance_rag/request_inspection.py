"""Bounded private request inspection and exact MLflow trace correlation."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from threading import Lock
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from consumer_finance_rag.access import AccessContext, AccessVisibility
from consumer_finance_rag.config import AppSettings
from consumer_finance_rag.factory import ApplicationRuntime
from consumer_finance_rag.models import Citation, RAGResponse, RetrievedChunk, StructuredContext
from consumer_finance_rag.observability import (
    RetrievalMetrics,
    RetrievalTelemetrySink,
    TraceSpan,
    bedrock_token_usage,
    trace_id_from_span,
)
from consumer_finance_rag.pii import DeterministicPIIDetector, PIIHandlingPolicy
from consumer_finance_rag.presentation import citation_references
from consumer_finance_rag.quality_policy import (
    QualityDecision,
    RuntimeQualityContext,
)
from consumer_finance_rag.retrieval import RetrievalInspectionSink, RetrievalTrace
from consumer_finance_rag.runtime_groundedness import RuntimeQualityGate


class RequestState(StrEnum):
    PENDING = "PENDING"
    FAILED = "FAILED"
    ABSTAINED = "ABSTAINED"
    COMPLETED = "COMPLETED"


class CandidateInspection(BaseModel):
    """Metadata-only candidate identity and score provenance."""

    model_config = ConfigDict(frozen=True)

    stage: str
    rank: int = Field(ge=1)
    chunk_id: str
    source_title: str
    dense_rank: int | None = Field(default=None, ge=1)
    sparse_rank: int | None = Field(default=None, ge=1)
    dense_score: float | None = None
    bm25_score: float | None = None
    rrf_score: float | None = None
    reranker_score: float | None = None
    included_in_context: bool
    inclusion_reason: str


class RetrievalInspection(BaseModel):
    model_config = ConfigDict(frozen=True)

    request_id: str
    stages: dict[str, tuple[CandidateInspection, ...]]


class RequestRecord(BaseModel):
    """Private, process-local record built only from the exact completed request."""

    model_config = ConfigDict(frozen=True)

    request_id: str
    trace_id: str | None = None
    state: RequestState
    created_at: datetime
    generation: int
    model_id: str
    region: str
    total_latency_ms: float | None = Field(default=None, ge=0)
    stage_timings_ms: dict[str, float | None] = Field(default_factory=dict)
    bottleneck: str | None = None
    metrics: RetrievalMetrics | None = None
    inspection: RetrievalInspection | None = None
    citations: tuple[Citation, ...] = ()
    token_usage: dict[str, int] | None = None
    failure_stage: str | None = None
    failure_type: str | None = None
    workload_kind: str = "answer_generation"
    quality_decision: QualityDecision | None = None


class RequestInspectionStore(RetrievalTelemetrySink, RetrievalInspectionSink):
    """Thread-safe bounded store; lookups are exact request/trace matches only."""

    def __init__(self, *, max_records: int = 256) -> None:
        if max_records <= 0:
            raise ValueError("max_records must be positive")
        self._max_records = max_records
        self._records: dict[str, RequestRecord] = {}
        self._trace_to_request: dict[str, str] = {}
        self._order: deque[str] = deque()
        self._pending_inspections: dict[str, RetrievalInspection] = {}
        self._pending_metrics: dict[str, RetrievalMetrics] = {}
        self._lock = Lock()

    def begin(self, *, request_id: str, generation: int, model_id: str, region: str) -> None:
        record = RequestRecord(
            request_id=request_id,
            state=RequestState.PENDING,
            created_at=datetime.now(UTC),
            generation=generation,
            model_id=model_id,
            region=region,
        )
        with self._lock:
            self._put(record)

    def emit(self, metrics: RetrievalMetrics) -> None:
        with self._lock:
            self._pending_metrics[metrics.request_id] = metrics
            current = self._records.get(metrics.request_id)
            if current is not None:
                self._records[metrics.request_id] = current.model_copy(update={"metrics": metrics})

    def emit_inspection(self, request_id: str, trace: RetrievalTrace) -> None:
        inspection = _inspection_from_trace(request_id, trace)
        with self._lock:
            self._pending_inspections[request_id] = inspection
            current = self._records.get(request_id)
            if current is not None:
                self._records[request_id] = current.model_copy(update={"inspection": inspection})

    def complete(self, response: RAGResponse) -> RequestRecord:
        with self._lock:
            current = self._require(response.request_id)
            pending_inspection = self._pending_inspections.pop(response.request_id, None)
            inspection = (
                _mark_final_context(pending_inspection, response.retrieved_chunks)
                if pending_inspection is not None
                else None
            )
            timings = _combined_stage_timings(
                response.stage_timings_ms,
                self._pending_metrics.get(response.request_id),
            )
            record = current.model_copy(
                update={
                    "trace_id": response.trace_id,
                    "state": RequestState.COMPLETED,
                    "total_latency_ms": response.stage_timings_ms.get("total"),
                    "stage_timings_ms": timings,
                    "bottleneck": measured_bottleneck(timings),
                    "metrics": self._pending_metrics.pop(response.request_id, None),
                    "inspection": inspection,
                    "citations": tuple(response.citations),
                    "token_usage": bedrock_token_usage(response.generation_metadata),
                }
            )
            self._records[response.request_id] = record
            if response.trace_id is not None:
                self._trace_to_request[response.trace_id] = response.request_id
            return record

    def record_candidate_metadata(self, response: RAGResponse) -> None:
        """Retain content-free generation accounting before a blocking quality decision."""

        with self._lock:
            current = self._require(response.request_id)
            pending_inspection = self._pending_inspections.get(response.request_id)
            inspection = (
                _mark_final_context(pending_inspection, response.retrieved_chunks)
                if pending_inspection is not None
                else None
            )
            timings = _combined_stage_timings(
                response.stage_timings_ms,
                self._pending_metrics.get(response.request_id),
            )
            self._records[response.request_id] = current.model_copy(
                update={
                    "trace_id": response.trace_id,
                    "total_latency_ms": response.stage_timings_ms.get("total"),
                    "stage_timings_ms": timings,
                    "bottleneck": measured_bottleneck(timings),
                    "inspection": inspection,
                    "citations": tuple(response.citations),
                    "token_usage": bedrock_token_usage(response.generation_metadata),
                }
            )
            if response.trace_id is not None:
                self._trace_to_request[response.trace_id] = response.request_id

    def fail(self, request_id: str, error: Exception) -> RequestRecord:
        with self._lock:
            current = self._require(request_id)
            metrics = self._pending_metrics.pop(request_id, None)
            stage = (
                metrics.error_stage.value
                if metrics is not None and metrics.error_stage is not None
                else "pipeline"
            )
            record = current.model_copy(
                update={
                    "state": RequestState.FAILED,
                    "metrics": metrics,
                    "inspection": self._pending_inspections.pop(request_id, None),
                    "failure_stage": stage,
                    "failure_type": type(error).__name__,
                }
            )
            self._records[request_id] = record
            return record

    def record_quality(self, request_id: str, decision: QualityDecision) -> None:
        with self._lock:
            current = self._require(request_id)
            self._records[request_id] = current.model_copy(update={"quality_decision": decision})

    def abstain(
        self,
        request_id: str,
        decision: QualityDecision,
        *,
        trace_id: str | None,
    ) -> RequestRecord:
        """Finish a controlled pre-generation abstention without inventing usage."""

        with self._lock:
            current = self._require(request_id)
            record = current.model_copy(
                update={
                    "trace_id": trace_id,
                    "state": RequestState.ABSTAINED,
                    "metrics": self._pending_metrics.pop(request_id, None),
                    "inspection": self._pending_inspections.pop(request_id, None),
                    "quality_decision": decision,
                    "failure_stage": "evidence_answerability",
                }
            )
            self._records[request_id] = record
            if trace_id is not None:
                self._trace_to_request[trace_id] = request_id
            return record

    def get_exact(self, identifier: str) -> RequestRecord | None:
        normalized = identifier.strip()
        with self._lock:
            request_id = self._trace_to_request.get(normalized, normalized)
            return self._records.get(request_id)

    def completed(self) -> tuple[RequestRecord, ...]:
        with self._lock:
            return tuple(
                self._records[request_id]
                for request_id in reversed(self._order)
                if self._records[request_id].state is RequestState.COMPLETED
            )

    def _require(self, request_id: str) -> RequestRecord:
        record = self._records.get(request_id)
        if record is None:
            raise KeyError(f"Unknown request ID: {request_id}")
        return record

    def _put(self, record: RequestRecord) -> None:
        if record.request_id not in self._records:
            self._order.append(record.request_id)
        self._records[record.request_id] = record
        while len(self._order) > self._max_records:
            oldest = self._order.popleft()
            removed = self._records.pop(oldest, None)
            self._pending_metrics.pop(oldest, None)
            self._pending_inspections.pop(oldest, None)
            if removed is not None and removed.trace_id is not None:
                self._trace_to_request.pop(removed.trace_id, None)


class WorkbenchRequestError(RuntimeError):
    def __init__(
        self,
        request_id: str,
        failure_stage: str,
        *,
        controlled_abstention: bool = False,
    ) -> None:
        super().__init__(f"Request {request_id} failed during {failure_stage}")
        self.request_id = request_id
        self.failure_stage = failure_stage
        self.controlled_abstention = controlled_abstention


@dataclass(frozen=True, slots=True)
class WorkbenchAnswer:
    response: RAGResponse
    record: RequestRecord
    quality_decision: QualityDecision


class WorkbenchRAGService:
    """Managed operator adapter that preserves the production PUBLIC/REDACT path."""

    def __init__(
        self,
        settings: AppSettings,
        runtime: ApplicationRuntime,
        records: RequestInspectionStore,
        quality_gate: RuntimeQualityGate | None = None,
    ) -> None:
        if settings.runtime.profile != "managed" or runtime.retrieval.profile != "managed":
            raise ValueError("The private workbench requires the managed runtime profile")
        if runtime.retrieval.generation is None:
            raise ValueError("The private workbench requires a pinned ACTIVE generation")
        service = runtime.pipeline.retrieval
        if not service.access_control_enabled or not service.pii_handling_enabled:
            raise ValueError("The private workbench requires ACL and PII enforcement")
        service.inspection_sink = records
        self._settings = settings
        self._runtime = runtime
        self._records = records
        self._quality_gate = quality_gate or RuntimeQualityGate(
            settings.quality_policy,
            runtime.pipeline.tracer,
        )

    def answer(self, question: str) -> WorkbenchAnswer:
        normalized = question.strip()
        if not normalized:
            raise ValueError("Enter a question first")
        if len(normalized) > self._settings.public_showcase.max_question_characters:
            raise ValueError(
                "Question exceeds the configured "
                f"{self._settings.public_showcase.max_question_characters}-character limit"
            )
        request_id = str(uuid4())
        generation = self._runtime.retrieval.generation
        assert generation is not None
        self._records.begin(
            request_id=request_id,
            generation=generation,
            model_id=self._runtime.pipeline.llm.model_name,
            region=self._settings.aws.region,
        )
        try:
            detector = self._runtime.pipeline.retrieval.pii_detector
            decisions: list[QualityDecision] = []
            relevance_decisions: list[QualityDecision] = []

            def approve_evidence(context: StructuredContext, parent_span: TraceSpan) -> None:
                _validate_public_only(context.chunks, detector)
                decision = self._quality_gate.evaluate_pre_generation(
                    context,
                    request_id=request_id,
                    parent_span=parent_span,
                )
                relevance_decisions.append(decision)
                if not decision.deliver_answer:
                    self._records.abstain(
                        request_id,
                        decision,
                        trace_id=trace_id_from_span(parent_span),
                    )
                    raise WorkbenchRequestError(
                        request_id,
                        "evidence_answerability",
                        controlled_abstention=True,
                    )

            def approve(candidate: RAGResponse, parent_span: TraceSpan) -> RAGResponse:
                safe_candidate = candidate.model_copy(
                    update={"answer": detector.redact(candidate.answer)}
                )
                _validate_public_only(safe_candidate.retrieved_chunks, detector)
                self._records.record_candidate_metadata(safe_candidate)
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
                self._records.record_quality(request_id, decision)
                if not decision.deliver_answer:
                    raise RuntimeError(f"Quality policy: {decision.status.value}")
                return safe_candidate

            response = self._runtime.pipeline.answer(
                detector.redact(normalized),
                access_context=AccessContext(),
                pii_policy=PIIHandlingPolicy.REDACT,
                request_id=request_id,
                pre_generation_gate=approve_evidence,
                pre_response_gate=approve,
            )
            quality_decision = decisions[-1]
            return WorkbenchAnswer(
                response,
                self._records.complete(response),
                quality_decision,
            )
        except WorkbenchRequestError:
            raise
        except Exception as error:
            failed = self._records.fail(request_id, error)
            raise WorkbenchRequestError(
                request_id,
                failed.failure_stage or "pipeline",
            ) from error


def measured_bottleneck(timings: dict[str, float | None]) -> str | None:
    """Return the slowest measured leaf stage, excluding overlapping totals."""

    excluded = {"total", "retrieval"}
    measured = {
        name: duration
        for name, duration in timings.items()
        if name not in excluded and duration is not None
    }
    return max(measured, key=measured.__getitem__) if measured else None


def aggregate_token_usage(
    records: Iterable[RequestRecord],
    *,
    workload_kind: str = "answer_generation",
) -> dict[str, int]:
    """Aggregate unique request usage without mixing judges with answer generation."""

    unique = {
        record.request_id: record
        for record in records
        if record.workload_kind == workload_kind and record.token_usage is not None
    }
    return {
        key: sum(record.token_usage[key] for record in unique.values() if record.token_usage)
        for key in ("input_tokens", "output_tokens", "total_tokens")
    }


def _inspection_from_trace(request_id: str, trace: RetrievalTrace) -> RetrievalInspection:
    final_ids = {item.chunk.chunk_id for item in trace.reranked}
    return RetrievalInspection(
        request_id=request_id,
        stages={
            "dense": _candidate_rows("dense", trace.dense, final_ids),
            "sparse": _candidate_rows("sparse", trace.sparse, final_ids),
            "rrf": _candidate_rows("rrf", trace.fused, final_ids),
            "reranker": _candidate_rows("reranker", trace.reranked, final_ids),
        },
    )


def _candidate_rows(
    stage: str,
    candidates: Sequence[RetrievedChunk],
    final_ids: set[str],
) -> tuple[CandidateInspection, ...]:
    rows: list[CandidateInspection] = []
    for rank, item in enumerate(candidates, start=1):
        included = item.chunk.chunk_id in final_ids
        rows.append(
            CandidateInspection(
                stage=stage,
                rank=rank,
                chunk_id=item.chunk.chunk_id,
                source_title=str(item.chunk.metadata.get("title") or "Untitled source"),
                dense_rank=item.dense_rank,
                sparse_rank=item.sparse_rank,
                dense_score=item.dense_score,
                bm25_score=item.sparse_score,
                rrf_score=item.rrf_score,
                reranker_score=item.rerank_score,
                included_in_context=included,
                inclusion_reason=(
                    "Selected by reranker; context construction pending"
                    if included
                    else "Not selected by final reranking"
                ),
            )
        )
    return tuple(rows)


def _mark_final_context(
    inspection: RetrievalInspection,
    final_contexts: Sequence[RetrievedChunk],
) -> RetrievalInspection:
    final_ids = {item.chunk.chunk_id for item in final_contexts}
    stages: dict[str, tuple[CandidateInspection, ...]] = {}
    for stage, candidates in inspection.stages.items():
        stages[stage] = tuple(
            candidate.model_copy(
                update={
                    "included_in_context": candidate.chunk_id in final_ids,
                    "inclusion_reason": (
                        "Selected by reranker and included within context budget"
                        if candidate.chunk_id in final_ids
                        else (
                            "Selected by reranker but excluded by context budget"
                            if stage == "reranker"
                            else "Not selected for final context"
                        )
                    ),
                }
            )
            for candidate in candidates
        )
    return inspection.model_copy(update={"stages": stages})


def _combined_stage_timings(
    pipeline: dict[str, float],
    metrics: RetrievalMetrics | None,
) -> dict[str, float | None]:
    values: dict[str, float | None] = {
        "ACTIVE generation pin": None,
        "query_embedding": None,
        "dense_hnsw": None,
        "sparse_bm25": None,
        "ACL_validation": None,
        "rrf": None,
        "reranker": None,
        "PII_handling": None,
        "context_build": pipeline.get("context_build"),
        "bedrock_generation": pipeline.get("bedrock_generation"),
        "answer_and_citations": None,
        "MLflow_sync": None,
        "retrieval": pipeline.get("retrieval"),
        "total": pipeline.get("total"),
    }
    if metrics is not None:
        values.update(
            {
                "query_embedding": metrics.query_embedding_latency_ms,
                "dense_hnsw": metrics.dense_search_latency_ms,
                "sparse_bm25": metrics.sparse_search_latency_ms,
                "rrf": metrics.fusion_latency_ms,
                "reranker": metrics.reranker_latency_ms,
                "PII_handling": metrics.pii_processing_latency_ms,
            }
        )
    return values


def _validate_public_only(
    chunks: Sequence[RetrievedChunk], detector: DeterministicPIIDetector
) -> None:
    for item in chunks:
        metadata = item.chunk.metadata
        if metadata.get("access_visibility") != AccessVisibility.PUBLIC.value or metadata.get(
            "allowed_groups"
        ) not in ([], (), None):
            raise RuntimeError("Managed request returned a non-PUBLIC context")
        if detector.detect(item.chunk.text):
            raise RuntimeError("Managed request returned unredacted PII")
