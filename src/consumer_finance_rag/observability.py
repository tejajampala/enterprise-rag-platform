"""Privacy-safe retrieval telemetry and optional MLflow tracing primitives."""

from __future__ import annotations

import json
import logging
import warnings
from collections import Counter
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal, Protocol, cast
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from consumer_finance_rag.config import MLflowSettings
from consumer_finance_rag.models import RetrievedChunk


class RetrievalStatus(StrEnum):
    """Terminal request-level retrieval outcome."""

    SUCCESS = "SUCCESS"
    ERROR = "ERROR"


class RetrievalFailureStage(StrEnum):
    """Bounded stage classification for retrieval failures."""

    DENSE = "dense"
    SPARSE = "sparse"
    FUSION = "fusion"
    RERANKER = "reranker"
    ACL = "ACL"
    PII = "PII"
    OTHER = "other"


class RetrievalACLMode(StrEnum):
    """Whether request candidates use managed pre-filtering and validation."""

    DISABLED = "disabled"
    FILTER_AND_VALIDATE = "qdrant_filter_and_defense_in_depth_validation"


class TraceContentPolicy(StrEnum):
    """Whether trace spans may contain reviewed evaluation text."""

    METADATA_ONLY = "metadata_only"
    APPROVED_PUBLIC_EVALUATION = "approved_public_evaluation"


@dataclass(frozen=True, slots=True)
class RetrievalTelemetryContext:
    """Runtime identity pinned once and copied into every request trace."""

    profile: str
    qdrant_collection: str | None = None
    index_state_id: UUID | None = None
    generation: int | None = None

    def __post_init__(self) -> None:
        if not self.profile.strip():
            raise ValueError("Retrieval telemetry profile cannot be empty")
        if self.profile == "managed":
            if (
                self.index_state_id is None
                or self.generation is None
                or not self.qdrant_collection
            ):
                raise ValueError(
                    "Managed retrieval telemetry requires pinned generation identity"
                )
            if self.generation <= 0:
                raise ValueError("Managed retrieval generation must be positive")
        elif self.index_state_id is not None or self.generation is not None:
            raise ValueError(
                "Non-managed retrieval telemetry cannot fabricate generation identity"
            )


@dataclass(frozen=True, slots=True)
class PIITypeCount:
    """Safe aggregate count for one PII type; matched values are never retained."""

    pii_type: str
    count: int

    def __post_init__(self) -> None:
        if not self.pii_type.strip():
            raise ValueError("PII telemetry type cannot be empty")
        if self.count < 0:
            raise ValueError("PII telemetry count cannot be negative")


@dataclass(frozen=True, slots=True)
class RetrievalMetrics:
    """Immutable terminal request telemetry without query or document content."""

    request_id: str
    profile: str
    index_state_id: UUID | None
    generation: int | None
    qdrant_collection: str | None
    query_timestamp: datetime
    total_retrieval_latency_ms: float
    query_embedding_latency_ms: float | None
    dense_search_latency_ms: float | None
    sparse_search_latency_ms: float | None
    fusion_latency_ms: float | None
    reranker_latency_ms: float | None
    pii_processing_latency_ms: float | None
    dense_candidate_count: int
    sparse_candidate_count: int
    fused_candidate_count: int
    final_context_count: int
    acl_mode: RetrievalACLMode
    acl_filter_applied: bool
    request_group_count: int
    rejected_candidate_count: int
    pii_policy: str
    redacted_context_count: int
    detected_pii_type_counts: tuple[PIITypeCount, ...]
    redacted_pii_type_counts: tuple[PIITypeCount, ...]
    reranker_enabled: bool
    status: RetrievalStatus
    error_stage: RetrievalFailureStage | None = None
    error_type: str | None = None

    def as_log_record(self) -> dict[str, object]:
        """Return a JSON-compatible allowlist containing no request content."""

        return {
            "event": "retrieval.completed",
            "request_id": self.request_id,
            "profile": self.profile,
            "index_state_id": (
                str(self.index_state_id) if self.index_state_id is not None else None
            ),
            "generation": self.generation,
            "qdrant_collection": self.qdrant_collection,
            "query_timestamp": self.query_timestamp.isoformat(),
            "total_retrieval_latency_ms": self.total_retrieval_latency_ms,
            "query_embedding_latency_ms": self.query_embedding_latency_ms,
            "dense_search_latency_ms": self.dense_search_latency_ms,
            "sparse_search_latency_ms": self.sparse_search_latency_ms,
            "fusion_latency_ms": self.fusion_latency_ms,
            "reranker_latency_ms": self.reranker_latency_ms,
            "pii_processing_latency_ms": self.pii_processing_latency_ms,
            "dense_candidate_count": self.dense_candidate_count,
            "sparse_candidate_count": self.sparse_candidate_count,
            "fused_candidate_count": self.fused_candidate_count,
            "final_context_count": self.final_context_count,
            "acl_mode": self.acl_mode.value,
            "acl_filter_applied": self.acl_filter_applied,
            "request_group_count": self.request_group_count,
            "rejected_candidate_count": self.rejected_candidate_count,
            "pii_policy": self.pii_policy,
            "redacted_context_count": self.redacted_context_count,
            "detected_pii_type_counts": {
                item.pii_type: item.count for item in self.detected_pii_type_counts
            },
            "redacted_pii_type_counts": {
                item.pii_type: item.count for item in self.redacted_pii_type_counts
            },
            "reranker_enabled": self.reranker_enabled,
            "status": self.status.value,
            "error_stage": (
                self.error_stage.value if self.error_stage is not None else None
            ),
            "error_type": self.error_type,
        }


class RetrievalTelemetrySink(Protocol):
    """Consumer boundary for one completed request-level retrieval trace."""

    def emit(self, metrics: RetrievalMetrics) -> None: ...


class NoOpRetrievalTelemetrySink:
    """Default sink that introduces no service, storage, or logging dependency."""

    def emit(self, metrics: RetrievalMetrics) -> None:
        del metrics


class StructuredLoggingRetrievalTelemetrySink:
    """Emit one privacy-safe JSON log record per completed retrieval request."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._logger = logger or logging.getLogger(
            "consumer_finance_rag.retrieval_telemetry"
        )

    def emit(self, metrics: RetrievalMetrics) -> None:
        level = (
            logging.INFO
            if metrics.status is RetrievalStatus.SUCCESS
            else logging.WARNING
        )
        self._logger.log(
            level,
            json.dumps(
                metrics.as_log_record(),
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )


class TraceSpan(Protocol):
    def set_inputs(self, inputs: Any) -> None: ...

    def set_outputs(self, outputs: Any) -> None: ...

    def set_attribute(self, key: str, value: Any) -> None: ...

    def set_attributes(self, attributes: dict[str, Any]) -> None: ...


class TracePersistenceResult(BaseModel):
    """Result of an exact, read-only backend verification after trace export."""

    model_config = ConfigDict(frozen=True)

    status: Literal[
        "CONFIRMED", "PENDING", "HIERARCHY_INCOMPLETE", "FAILED", "UNAVAILABLE"
    ]
    reason: str
    persisted_span_count: int | None = Field(default=None, ge=0)
    persisted_provider_call_count: int | None = Field(default=None, ge=0)


class Tracer(Protocol):
    @contextmanager
    def span(
        self,
        name: str,
        *,
        span_type: str,
        attributes: dict[str, Any] | None = None,
        parent_span: TraceSpan | None = None,
    ) -> Iterator[TraceSpan]: ...


class TracingStatus(BaseModel):
    model_config = ConfigDict(frozen=True)

    configured: bool
    active: bool
    reason: str | None = None


class NoOpSpan:
    def set_inputs(self, inputs: Any) -> None:
        return None

    def set_outputs(self, outputs: Any) -> None:
        return None

    def set_attribute(self, key: str, value: Any) -> None:
        return None

    def set_attributes(self, attributes: dict[str, Any]) -> None:
        return None


class NoOpTracer:
    """Tracing implementation used when observability is explicitly disabled."""

    status = TracingStatus(configured=False, active=False, reason="MLflow tracing is disabled")
    content_policy = TraceContentPolicy.METADATA_ONLY

    @contextmanager
    def span(
        self,
        name: str,
        *,
        span_type: str,
        attributes: dict[str, Any] | None = None,
        parent_span: TraceSpan | None = None,
    ) -> Iterator[TraceSpan]:
        del parent_span
        yield NoOpSpan()

    def verify_trace(
        self,
        trace_id: str,
        *,
        expected_client_request_id: str,
        expected_span_count: int,
        expected_metric_names: tuple[str, ...] = (),
        expected_provider_call_count: int = 0,
    ) -> TracePersistenceResult:
        del (
            trace_id,
            expected_client_request_id,
            expected_span_count,
            expected_metric_names,
            expected_provider_call_count,
        )
        return TracePersistenceResult(
            status="UNAVAILABLE",
            reason="MLflow tracing is disabled.",
        )


class MLflowTracer:
    """Small manual-span adapter; MLflow remains the tracing system of record."""

    def __init__(
        self,
        settings: MLflowSettings,
        *,
        content_policy: TraceContentPolicy = TraceContentPolicy.METADATA_ONLY,
    ) -> None:
        self._settings = settings
        self.content_policy = content_policy
        self._active = False
        self._reason: str | None = None
        try:
            import mlflow

            mlflow.set_tracking_uri(settings.tracking_uri)
            mlflow.set_experiment(settings.experiment_name)
            self._active = True
        except Exception as error:  # MLflow can fail through transport-specific exceptions.
            self._reason = f"MLflow tracing unavailable ({type(error).__name__})"
            warnings.warn(self._reason, RuntimeWarning, stacklevel=2)

    @property
    def status(self) -> TracingStatus:
        return TracingStatus(configured=True, active=self._active, reason=self._reason)

    @contextmanager
    def span(
        self,
        name: str,
        *,
        span_type: str,
        attributes: dict[str, Any] | None = None,
        parent_span: TraceSpan | None = None,
    ) -> Iterator[TraceSpan]:
        if not self._active:
            yield NoOpSpan()
            return
        import mlflow

        correlation_id: str | None = None
        if name in {"rag_request", "ragas_evaluation"} and attributes is not None:
            correlation_value = (
                attributes.get("request.id")
                if name == "rag_request"
                else attributes.get("evaluation.client_request_id")
            )
            if isinstance(correlation_value, str):
                correlation_id = validate_mlflow_client_request_id(correlation_value)

        if parent_span is not None:
            span = mlflow.start_span_no_context(
                name=name,
                span_type=span_type,
                parent_span=cast(Any, parent_span),
                attributes=attributes,
            )
            try:
                yield span
            finally:
                span.end()
            return

        with mlflow.start_span(name=name, span_type=span_type, attributes=attributes) as span:
            if correlation_id is not None:
                mlflow.update_current_trace(
                    client_request_id=correlation_id,
                    request_preview="[QUESTION OMITTED]",
                    response_preview="[ANSWER OMITTED]",
                )
            yield span

    def verify_trace(
        self,
        trace_id: str,
        *,
        expected_client_request_id: str,
        expected_span_count: int,
        expected_metric_names: tuple[str, ...] = (),
        expected_provider_call_count: int = 0,
    ) -> TracePersistenceResult:
        """Flush the async exporter, then verify this exact trace without inference calls."""

        if not self._active:
            return TracePersistenceResult(
                status="UNAVAILABLE",
                reason=self._reason or "MLflow tracing is unavailable.",
            )
        expected = validate_mlflow_client_request_id(expected_client_request_id)
        try:
            import mlflow

            mlflow.flush_trace_async_logging()
            trace = mlflow.MlflowClient(
                tracking_uri=self._settings.tracking_uri
            ).get_trace(trace_id, display=False)
        except Exception as error:
            if _is_missing_trace(error):
                return TracePersistenceResult(
                    status="PENDING",
                    reason="Trace syncing: the exact trace is not retrievable yet.",
                )
            return TracePersistenceResult(
                status="FAILED",
                reason=f"Trace export verification failed ({type(error).__name__}).",
            )
        actual = getattr(trace.info, "client_request_id", None)
        if actual != expected:
            return TracePersistenceResult(
                status="FAILED",
                reason=(
                    "Persisted trace correlation does not match the evaluation client "
                    "request ID."
                ),
            )
        spans = tuple(getattr(trace.data, "spans", ()))
        provider_spans = [
            span for span in spans if getattr(span, "name", None) == "bedrock_provider_call"
        ]
        persisted_provider_count = len(provider_spans)
        state = str(getattr(trace.info, "state", ""))
        if "IN_PROGRESS" in state or "UNSPECIFIED" in state:
            return TracePersistenceResult(
                status="PENDING",
                reason="Trace syncing: MLflow reports the exact trace is still in progress.",
                persisted_span_count=len(spans),
                persisted_provider_call_count=persisted_provider_count,
            )
        if len(spans) < expected_span_count:
            return TracePersistenceResult(
                status="HIERARCHY_INCOMPLETE",
                reason="Persisted trace is missing expected evaluation child spans.",
                persisted_span_count=len(spans),
                persisted_provider_call_count=persisted_provider_count,
            )
        root_count = sum(getattr(span, "name", None) == "ragas_evaluation" for span in spans)
        if root_count != 1:
            return TracePersistenceResult(
                status="FAILED",
                reason="Persisted evaluation trace has an invalid root-span hierarchy.",
                persisted_span_count=len(spans),
                persisted_provider_call_count=persisted_provider_count,
            )
        root_span = next(span for span in spans if span.name == "ragas_evaluation")
        root_span_id = str(root_span.span_id)
        span_ids = {
            str(span_id)
            for span in spans
            if (span_id := getattr(span, "span_id", None)) is not None
        }
        disconnected_children = [
            span
            for span in spans
            if getattr(span, "name", None) != "ragas_evaluation"
            and (
                (parent_id := getattr(span, "parent_id", None)) is None
                or str(parent_id) not in span_ids
            )
        ]
        if disconnected_children:
            return TracePersistenceResult(
                status="FAILED",
                reason="Persisted evaluation trace contains a disconnected child span.",
                persisted_span_count=len(spans),
                persisted_provider_call_count=persisted_provider_count,
            )
        metric_spans = [
            span for span in spans if str(getattr(span, "name", "")).startswith("ragas_metric:")
        ]
        actual_metric_names = tuple(
            str(getattr(span, "name", "")).removeprefix("ragas_metric:")
            for span in metric_spans
        )
        if Counter(actual_metric_names) != Counter(expected_metric_names):
            return TracePersistenceResult(
                status="HIERARCHY_INCOMPLETE",
                reason="Persisted trace does not contain every expected RAGAS metric span.",
                persisted_span_count=len(spans),
                persisted_provider_call_count=persisted_provider_count,
            )
        if any(str(getattr(span, "parent_id", None)) != root_span_id for span in metric_spans):
            return TracePersistenceResult(
                status="HIERARCHY_INCOMPLETE",
                reason="A persisted RAGAS metric span is not parented by the evaluation root.",
                persisted_span_count=len(spans),
                persisted_provider_call_count=persisted_provider_count,
            )
        metric_by_id = {str(span.span_id): span for span in metric_spans}
        if expected_metric_names and expected_provider_call_count == 0:
            return TracePersistenceResult(
                status="HIERARCHY_INCOMPLETE",
                reason=(
                    "No LiteLLM completion invocation was observed for the requested "
                    "RAGAS metrics."
                ),
                persisted_span_count=len(spans),
                persisted_provider_call_count=persisted_provider_count,
            )
        if len(provider_spans) != expected_provider_call_count:
            return TracePersistenceResult(
                status="HIERARCHY_INCOMPLETE",
                reason="Persisted trace is missing expected provider-call spans.",
                persisted_span_count=len(spans),
                persisted_provider_call_count=persisted_provider_count,
            )
        if any(
            str(getattr(span, "parent_id", None)) not in metric_by_id
            for span in provider_spans
        ):
            return TracePersistenceResult(
                status="HIERARCHY_INCOMPLETE",
                reason="A persisted provider-call span is not parented by a RAGAS metric span.",
                persisted_span_count=len(spans),
                persisted_provider_call_count=persisted_provider_count,
            )
        if "ERROR" in state:
            return TracePersistenceResult(
                status="FAILED",
                reason="MLflow persisted the exact trace with an error state.",
                persisted_span_count=len(spans),
                persisted_provider_call_count=persisted_provider_count,
            )
        return TracePersistenceResult(
            status="CONFIRMED",
            reason="The exact evaluation trace and all expected spans are persisted.",
            persisted_span_count=len(spans),
            persisted_provider_call_count=persisted_provider_count,
        )


def build_tracer(
    settings: MLflowSettings,
    *,
    content_policy: TraceContentPolicy = TraceContentPolicy.METADATA_ONLY,
) -> Tracer:
    if not settings.enabled:
        return NoOpTracer()
    return MLflowTracer(settings, content_policy=content_policy)


def trace_id_from_span(span: TraceSpan) -> str | None:
    """Read the ID from the active span itself, never from process-global state."""

    trace_id = getattr(span, "trace_id", None)
    return trace_id if isinstance(trace_id, str) and trace_id else None


def span_id_from_span(span: TraceSpan) -> str | None:
    """Read a span ID without depending on one concrete tracing implementation."""

    span_id = getattr(span, "span_id", None)
    return str(span_id) if span_id is not None and str(span_id) else None


def validate_mlflow_client_request_id(value: str) -> str:
    """Fail before export when an MLflow client request ID exceeds VARCHAR(50)."""

    if not value or len(value) > 50:
        raise ValueError("MLflow client_request_id must contain 1 to 50 characters")
    return value


def _is_missing_trace(error: Exception) -> bool:
    code = getattr(error, "error_code", None)
    rendered = f"{code} {error}".upper()
    return "RESOURCE_DOES_NOT_EXIST" in rendered or "NOT FOUND" in rendered


def chunk_summaries(chunks: Sequence[RetrievedChunk]) -> list[dict[str, Any]]:
    """Compact ranking details suitable for spans without embedding vectors or full text."""

    return [
        {
            "dense_rank": item.dense_rank,
            "sparse_rank": item.sparse_rank,
            "retrieval_score": item.score,
            "dense_score": item.dense_score,
            "sparse_score": item.sparse_score,
            "rrf_score": item.rrf_score,
            "rerank_score": item.rerank_score,
        }
        for item in chunks
    ]


def approved_retriever_documents(
    chunks: Sequence[RetrievedChunk],
    *,
    content_policy: TraceContentPolicy,
) -> list[dict[str, Any]]:
    """Authorized contexts in MLflow's retriever schema for approved evaluation only."""

    if content_policy is not TraceContentPolicy.APPROVED_PUBLIC_EVALUATION:
        raise ValueError("Retriever document text requires approved PUBLIC evaluation policy")

    return [
        {
            "id": item.chunk.chunk_id,
            "page_content": item.chunk.text,
            "metadata": {
                "rank": rank,
                "document_id": item.chunk.document_id,
                "title": str(item.chunk.metadata.get("title") or "Untitled source"),
            },
        }
        for rank, item in enumerate(chunks, start=1)
    ]


def bedrock_token_usage(metadata: dict[str, Any]) -> dict[str, int] | None:
    """Map Bedrock Converse usage keys to MLflow's supported token schema."""

    usage = metadata.get("usage")
    if not isinstance(usage, dict):
        return None
    mapping: dict[str, Any] = {
        "input_tokens": usage.get("inputTokens"),
        "output_tokens": usage.get("outputTokens"),
        "total_tokens": usage.get("totalTokens"),
    }
    result: dict[str, int] = {}
    for key, value in mapping.items():
        if not isinstance(value, int) or value < 0:
            return None
        result[key] = value
    return result
