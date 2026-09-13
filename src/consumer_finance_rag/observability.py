"""Optional MLflow tracing primitives shared by the production execution path."""

from __future__ import annotations

import warnings
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict

from consumer_finance_rag.config import MLflowSettings
from consumer_finance_rag.models import RetrievedChunk


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
            "chunk_id": item.chunk.chunk_id,
            "document_id": item.chunk.document_id,
            "title": item.chunk.metadata.get("title"),
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
            "id": item.chunk.chunk_id,
            "page_content": item.chunk.text,
            "metadata": {
                "chunk_id": item.chunk.chunk_id,
                "document_id": item.chunk.document_id,
                "doc_uri": item.chunk.metadata.get("source_url"),
                "title": item.chunk.metadata.get("title"),
                "rank": rank,
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
