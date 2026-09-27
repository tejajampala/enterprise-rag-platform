"""Supported exact-ID MLflow trace reads for the private workbench."""

from __future__ import annotations

import json
from enum import StrEnum
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from consumer_finance_rag.config import MLflowSettings


class TraceVisibility(StrEnum):
    PENDING = "PENDING"
    UNAVAILABLE = "UNAVAILABLE"
    FAILED = "FAILED"
    COMPLETED = "COMPLETED"


class TraceSpanView(BaseModel):
    model_config = ConfigDict(frozen=True)

    span_id: str
    parent_span_id: str | None
    name: str
    status: str
    duration_ms: float | None = Field(default=None, ge=0)


class TraceInspection(BaseModel):
    model_config = ConfigDict(frozen=True)

    trace_id: str | None
    request_id: str
    visibility: TraceVisibility
    message: str
    root_duration_ms: float | None = Field(default=None, ge=0)
    spans: tuple[TraceSpanView, ...] = ()
    token_usage: dict[str, int] | None = None
    cost_metadata: dict[str, str] = Field(default_factory=dict)


class TraceClient(Protocol):
    def get_trace(self, trace_id: str, display: bool = False) -> Any: ...


def inspect_exact_trace(
    settings: MLflowSettings,
    *,
    trace_id: str | None,
    expected_request_id: str,
    client: TraceClient | None = None,
) -> TraceInspection:
    """Read only the returned trace ID and fail closed on correlation mismatch."""

    if not settings.enabled or trace_id is None:
        return TraceInspection(
            trace_id=trace_id,
            request_id=expected_request_id,
            visibility=TraceVisibility.UNAVAILABLE,
            message="Tracing was disabled or this request returned no trace ID.",
        )
    try:
        active_client = client or _mlflow_client(settings)
        trace = active_client.get_trace(trace_id, display=False)
    except Exception as error:  # MLflow transports expose several exception classes.
        if _is_not_found(error):
            return TraceInspection(
                trace_id=trace_id,
                request_id=expected_request_id,
                visibility=TraceVisibility.PENDING,
                message="Trace syncing: MLflow has not made this exact trace retrievable yet.",
            )
        return TraceInspection(
            trace_id=trace_id,
            request_id=expected_request_id,
            visibility=TraceVisibility.FAILED,
            message=f"MLflow trace read failed ({type(error).__name__}).",
        )

    actual_request_id = getattr(trace.info, "client_request_id", None)
    if actual_request_id != expected_request_id:
        return TraceInspection(
            trace_id=trace_id,
            request_id=expected_request_id,
            visibility=TraceVisibility.FAILED,
            message="Trace correlation failed; no other request trace was substituted.",
        )
    spans = tuple(_span_view(span) for span in trace.data.spans)
    token_usage = _trace_token_usage(trace.data.spans)
    trace_metadata = getattr(trace.info, "trace_metadata", {})
    cost_metadata = (
        {
            str(key): str(value)
            for key, value in trace_metadata.items()
            if "cost" in str(key).lower()
        }
        if isinstance(trace_metadata, dict)
        else {}
    )
    state = str(getattr(trace.info, "state", ""))
    if "IN_PROGRESS" in state or "UNSPECIFIED" in state:
        visibility = TraceVisibility.PENDING
        message = "Trace syncing: MLflow is still persisting this exact trace."
    elif "ERROR" in state:
        visibility = TraceVisibility.FAILED
        message = "The exact MLflow trace completed with an error state."
    else:
        visibility = TraceVisibility.COMPLETED
        message = "The exact trace is available and backend persistence is confirmed."
    duration = getattr(trace.info, "execution_duration", None)
    return TraceInspection(
        trace_id=trace_id,
        request_id=expected_request_id,
        visibility=visibility,
        message=message,
        root_duration_ms=float(duration) if duration is not None else None,
        spans=spans,
        token_usage=token_usage,
        cost_metadata=cost_metadata,
    )


def _span_view(span: Any) -> TraceSpanView:
    start = getattr(span, "start_time_ns", None)
    end = getattr(span, "end_time_ns", None)
    duration = (
        max(0.0, float(end - start) / 1_000_000)
        if isinstance(start, int) and isinstance(end, int)
        else None
    )
    status = getattr(getattr(span, "status", None), "status_code", "UNSET")
    return TraceSpanView(
        span_id=str(getattr(span, "span_id", "not-recorded")),
        parent_span_id=(
            str(parent) if (parent := getattr(span, "parent_id", None)) else None
        ),
        name=str(getattr(span, "name", "unnamed")),
        status=str(getattr(status, "value", status)),
        duration_ms=duration,
    )


def _is_not_found(error: Exception) -> bool:
    error_code = getattr(error, "error_code", None)
    rendered = f"{error_code} {error}".upper()
    return "RESOURCE_DOES_NOT_EXIST" in rendered or "NOT FOUND" in rendered


def _trace_token_usage(spans: Any) -> dict[str, int] | None:
    for span in spans:
        attributes = getattr(span, "attributes", {})
        if not isinstance(attributes, dict):
            continue
        value = attributes.get("mlflow.chat.tokenUsage")
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except ValueError:
                continue
        if not isinstance(value, dict):
            continue
        expected = ("input_tokens", "output_tokens", "total_tokens")
        if all(isinstance(value.get(key), int) and value[key] >= 0 for key in expected):
            return {key: value[key] for key in expected}
    return None


def _mlflow_client(settings: MLflowSettings) -> TraceClient:
    from mlflow import MlflowClient, set_tracking_uri

    set_tracking_uri(settings.tracking_uri)
    return MlflowClient()
