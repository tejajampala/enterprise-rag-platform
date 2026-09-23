"""Privacy-safe retrieval telemetry and optional MLflow tracing primitives."""

from __future__ import annotations

import json
import logging
import warnings
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any, Protocol
from uuid import UUID

from pydantic import BaseModel, ConfigDict

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


class Tracer(Protocol):
    @contextmanager
    def span(
        self,
        name: str,
        *,
        span_type: str,
        attributes: dict[str, Any] | None = None,
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

    @contextmanager
    def span(
        self,
        name: str,
        *,
        span_type: str,
        attributes: dict[str, Any] | None = None,
    ) -> Iterator[TraceSpan]:
        yield NoOpSpan()


class MLflowTracer:
    """Small manual-span adapter; MLflow remains the tracing system of record."""

    def __init__(self, settings: MLflowSettings) -> None:
        self._settings = settings
        self._active = False
        self._reason: str | None = None
        try:
            import mlflow

            mlflow.set_tracking_uri(settings.tracking_uri)
            mlflow.set_experiment(settings.experiment_name)
            self._active = True
        except Exception as error:  # MLflow can fail through transport-specific exceptions.
            self._reason = f"MLflow tracing unavailable: {error}"
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
    ) -> Iterator[TraceSpan]:
        if not self._active:
            yield NoOpSpan()
            return
        import mlflow

        with mlflow.start_span(name=name, span_type=span_type, attributes=attributes) as span:
            yield span


def build_tracer(settings: MLflowSettings) -> Tracer:
    if not settings.enabled:
        return NoOpTracer()
    return MLflowTracer(settings)


def chunk_summaries(chunks: Sequence[RetrievedChunk]) -> list[dict[str, Any]]:
    """Compact ranking details suitable for spans without embedding vectors or full text."""

    return [
        {
            "dense_rank": item.dense_rank,
            "sparse_rank": item.sparse_rank,
            "retrieval_score": item.score,
            "rerank_score": item.rerank_score,
        }
        for item in chunks
    ]


def retriever_documents(chunks: Sequence[RetrievedChunk]) -> list[dict[str, Any]]:
    """Final contexts in MLflow's standard retriever-span document schema."""

    return [
        {
            "id": f"context-{rank}",
            "page_content": "[CONTENT OMITTED FROM TELEMETRY]",
            "metadata": {"rank": rank},
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
