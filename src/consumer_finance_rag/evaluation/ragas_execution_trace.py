"""Metadata-only MLflow execution tracing for the isolated RAGAS judge path."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from threading import Lock
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict

from consumer_finance_rag.costs import ModelInvocationUsage
from consumer_finance_rag.observability import (
    TracePersistenceResult,
    Tracer,
    TraceSpan,
    span_id_from_span,
    trace_id_from_span,
    validate_mlflow_client_request_id,
)


class RagasExecutionDiagnostics(BaseModel):
    """Bounded execution counters; no evaluated content is retained."""

    model_config = ConfigDict(frozen=True)

    metric_invocations_started: int = 0
    metric_invocations_completed: int = 0
    litellm_completion_invocations_started: int = 0
    litellm_completion_invocations_completed: int = 0
    success_callbacks_received: int = 0
    failure_callbacks_received: int = 0
    callback_correlation_failures: int = 0
    provider_spans_attempted: int = 0
    provider_spans_created: int = 0
    provider_spans_persisted: int = 0
    missing_span_reason_categories: tuple[str, ...] = ()


class RagasExecutionTraceReference(BaseModel):
    """Safe reference to one separately persisted RAGAS execution trace."""

    model_config = ConfigDict(frozen=True)

    trace_id: str | None = None
    configuration_id: str
    evaluation_run_id: str | None = None
    client_request_id: str | None = None
    status: Literal[
        "CAPTURED",
        "CONFIRMED",
        "PENDING",
        "HIERARCHY_INCOMPLETE",
        "EXPORT_FAILED",
        "UNAVAILABLE",
    ]
    reason: str | None = None
    persisted_span_count: int | None = None
    diagnostics: RagasExecutionDiagnostics = RagasExecutionDiagnostics()


class ProviderSpanCorrelation(BaseModel):
    """Content-free identity for one actual provider invocation span."""

    model_config = ConfigDict(frozen=True)

    trace_id: str | None
    metric_span_id: str | None
    provider_span_id: str | None
    sequence_within_metric: int


@dataclass(slots=True)
class ProviderInvocationSpan:
    """Live provider span opened around the actual awaited LiteLLM invocation."""

    correlation: ProviderSpanCorrelation
    _span: TraceSpan
    _completed: bool = False

    def complete(self, invocation: ModelInvocationUsage) -> None:
        """Attach returned provider metadata before ending the live span."""

        self._completed = True
        self._span.set_attributes(_provider_result_attributes(invocation))
        _safe_result(
            self._span,
            status=invocation.status,
            error_category=invocation.error_category,
        )

    def complete_without_telemetry(
        self,
        *,
        status: Literal["COMPLETED", "FAILED"],
        error_category: str | None,
    ) -> None:
        """End honestly when usage extraction fails after the provider returns."""

        self._completed = True
        self._span.set_attributes(
            {
                "provider.call_status": status,
                "provider.telemetry_complete": False,
                "provider.error_category": error_category or "not-recorded",
            }
        )
        _safe_result(
            self._span,
            status=status,
            error_category=error_category,
        )


@dataclass(frozen=True, slots=True)
class RagasExecutionTraceContext:
    """Correlation IDs only; reviewed content never belongs in the trace."""

    configuration_id: str
    evaluation_run_id: str | None
    request_ids: Mapping[str, str | None]
    rag_trace_ids: Mapping[str, str | None]
    client_request_id: str = field(default_factory=lambda: str(uuid4()))

    def __post_init__(self) -> None:
        validate_mlflow_client_request_id(self.client_request_id)
        try:
            UUID(self.client_request_id)
        except ValueError as error:
            raise ValueError(
                "RAGAS evaluation client_request_id must be a UUID string"
            ) from error


_ACTIVE_RAGAS_TRACE: ContextVar[RagasExecutionTrace | None] = ContextVar(
    "active_ragas_execution_trace", default=None
)


def active_ragas_execution_trace() -> RagasExecutionTrace | None:
    """Return the task-local trace only while an evaluation is actually executing."""

    return _ACTIVE_RAGAS_TRACE.get()


class RagasExecutionTrace:
    """Owns one root trace and task-local metric/provider child spans.

    Asyncio copies context variables into child tasks. That makes each provider callback
    inherit the metric span that initiated its actual LiteLLM call, even when RAGAS scores
    multiple reviewed cases concurrently.
    """

    def __init__(
        self,
        tracer: Tracer,
        *,
        context: RagasExecutionTraceContext,
        model_id: str,
        region: str,
        requested_metrics: tuple[str, ...],
    ) -> None:
        self._tracer = tracer
        self._context = context
        self._model_id = model_id
        self._region = region
        self._requested_metrics = requested_metrics
        self._trace_id: str | None = None
        self._persistence: TracePersistenceResult | None = None
        self._span_count = 0
        self._provider_attempt_count = 0
        self._metric_started_count = 0
        self._metric_completed_count = 0
        self._completion_started_count = 0
        self._completion_completed_count = 0
        self._success_callback_count = 0
        self._failure_callback_count = 0
        self._callback_correlation_failure_count = 0
        self._provider_created_count = 0
        self._provider_persisted_count = 0
        self._diagnostic_reasons: set[str] = set()
        self._root_span: TraceSpan | None = None
        self._metric_spans: dict[tuple[str, str], TraceSpan] = {}
        self._provider_sequences: dict[tuple[str, str], int] = {}
        self._span_lock = Lock()

    @property
    def reference(self) -> RagasExecutionTraceReference:
        if self._trace_id is None:
            status: Literal[
                "CAPTURED",
                "CONFIRMED",
                "PENDING",
                "HIERARCHY_INCOMPLETE",
                "EXPORT_FAILED",
                "UNAVAILABLE",
            ] = "UNAVAILABLE"
            reason = self._unavailable_reason()
            persisted_span_count = None
        elif self._persistence is None:
            status = "PENDING"
            reason = "Trace syncing: backend persistence has not been confirmed."
            persisted_span_count = None
        else:
            if self._persistence.status == "CONFIRMED":
                status = "CONFIRMED"
            elif self._persistence.status == "PENDING":
                status = "PENDING"
            elif self._persistence.status == "HIERARCHY_INCOMPLETE":
                status = "HIERARCHY_INCOMPLETE"
            elif self._persistence.status == "FAILED":
                status = "EXPORT_FAILED"
            else:
                status = "UNAVAILABLE"
            reason = self._persistence.reason
            persisted_span_count = self._persistence.persisted_span_count
        return RagasExecutionTraceReference(
            trace_id=self._trace_id,
            configuration_id=self._context.configuration_id,
            evaluation_run_id=self._context.evaluation_run_id,
            client_request_id=self._context.client_request_id,
            status=status,
            reason=reason,
            persisted_span_count=persisted_span_count,
            diagnostics=self.diagnostics,
        )

    @property
    def diagnostics(self) -> RagasExecutionDiagnostics:
        with self._span_lock:
            reasons = set(self._diagnostic_reasons)
            if self._metric_started_count > self._metric_completed_count:
                reasons.add("METRIC_INVOCATION_INCOMPLETE")
            if self._completion_started_count > self._completion_completed_count:
                reasons.add("LITELLM_COMPLETION_INCOMPLETE")
            if self._metric_completed_count and self._completion_started_count == 0:
                reasons.add("NO_LITELLM_COMPLETION_OBSERVED")
            if self._provider_attempt_count < self._completion_started_count:
                reasons.add("PROVIDER_SPAN_ATTEMPT_MISSING")
            if self._provider_created_count < self._provider_attempt_count:
                reasons.add("PROVIDER_SPAN_CORRELATION_FAILED")
            if (
                self._persistence is not None
                and self._persistence.status != "PENDING"
                and self._provider_persisted_count < self._provider_created_count
            ):
                reasons.add("PROVIDER_SPAN_NOT_PERSISTED")
            return RagasExecutionDiagnostics(
                metric_invocations_started=self._metric_started_count,
                metric_invocations_completed=self._metric_completed_count,
                litellm_completion_invocations_started=self._completion_started_count,
                litellm_completion_invocations_completed=self._completion_completed_count,
                success_callbacks_received=self._success_callback_count,
                failure_callbacks_received=self._failure_callback_count,
                callback_correlation_failures=self._callback_correlation_failure_count,
                provider_spans_attempted=self._provider_attempt_count,
                provider_spans_created=self._provider_created_count,
                provider_spans_persisted=self._provider_persisted_count,
                missing_span_reason_categories=tuple(sorted(reasons)),
            )

    def completion_started(self) -> None:
        with self._span_lock:
            self._completion_started_count += 1

    def completion_completed(self) -> None:
        with self._span_lock:
            self._completion_completed_count += 1

    def callback_received(self, *, success: bool) -> None:
        with self._span_lock:
            if success:
                self._success_callback_count += 1
            else:
                self._failure_callback_count += 1

    def callback_correlation_failed(self, reason: str) -> None:
        with self._span_lock:
            self._callback_correlation_failure_count += 1
            self._diagnostic_reasons.add(reason)

    def telemetry_failed(self, reason: str) -> None:
        with self._span_lock:
            self._diagnostic_reasons.add(reason)

    @contextmanager
    def root(self) -> Iterator[None]:
        attributes: dict[str, object] = {
            "evaluation.kind": "ragas",
            "evaluation.client_request_id": self._context.client_request_id,
            "evaluation.configuration_id": self._context.configuration_id,
            "evaluation.run_id": self._context.evaluation_run_id or "not-recorded",
            "evaluation.case_count": len(self._context.request_ids),
            "evaluation.request_ids": _json_ids(self._context.request_ids),
            "evaluation.rag_trace_ids": _json_ids(self._context.rag_trace_ids),
            "ragas.requested_metrics": list(self._requested_metrics),
            "judge.model_id": self._model_id,
            "judge.aws_region": self._region,
            "trace.content_policy": "metadata_only",
        }
        failure: Exception | None = None
        self._increment_span_count()
        with self._tracer.span(
            "ragas_evaluation", span_type="CHAIN", attributes=attributes
        ) as span:
            self._root_span = span
            self._trace_id = trace_id_from_span(span)
            token = _ACTIVE_RAGAS_TRACE.set(self)
            try:
                yield
            except Exception as error:
                failure = error
                _safe_result(
                    span,
                    status="FAILED",
                    error_category=type(error).__name__,
                )
            else:
                _safe_result(span, status="COMPLETED")
            finally:
                _ACTIVE_RAGAS_TRACE.reset(token)
                self._root_span = None
        if self._trace_id is not None:
            verifier = getattr(self._tracer, "verify_trace", None)
            if callable(verifier):
                diagnostics = self.diagnostics
                self._persistence = verifier(
                    self._trace_id,
                    expected_client_request_id=self._context.client_request_id,
                    expected_span_count=(
                        1
                        + len(self._requested_metrics) * len(self._context.request_ids)
                        + diagnostics.litellm_completion_invocations_started
                    ),
                    expected_metric_names=(
                        self._requested_metrics * len(self._context.request_ids)
                    ),
                    expected_provider_call_count=(
                        diagnostics.litellm_completion_invocations_started
                    ),
                )
                with self._span_lock:
                    self._provider_persisted_count = (
                        self._persistence.persisted_provider_call_count or 0
                    )
        if failure is not None:
            raise failure

    @contextmanager
    def metric(self, *, case_id: str, metric_name: str) -> Iterator[None]:
        attributes: dict[str, object] = {
            "evaluation.kind": "ragas_metric",
            "evaluation.client_request_id": self._context.client_request_id,
            "evaluation.configuration_id": self._context.configuration_id,
            "evaluation.run_id": self._context.evaluation_run_id or "not-recorded",
            "evaluation.case_id": case_id,
            "request.id": self._context.request_ids.get(case_id) or "not-recorded",
            "rag.trace_id": self._context.rag_trace_ids.get(case_id) or "not-recorded",
            "ragas.metric_name": metric_name,
            "judge.model_id": self._model_id,
            "judge.aws_region": self._region,
            "trace.content_policy": "metadata_only",
        }
        failure: Exception | None = None
        with self._span_lock:
            self._metric_started_count += 1
        self._increment_span_count()
        root_span = self._root_span
        with self._tracer.span(
            f"ragas_metric:{metric_name}",
            span_type="CHAIN",
            attributes=attributes,
            parent_span=root_span,
        ) as span:
            metric_key = (case_id, metric_name)
            with self._span_lock:
                self._metric_spans[metric_key] = span
            try:
                yield
            except Exception as error:
                failure = error
                _safe_result(
                    span,
                    status="FAILED",
                    error_category=type(error).__name__,
                )
            else:
                _safe_result(span, status="COMPLETED")
            finally:
                with self._span_lock:
                    self._metric_spans.pop(metric_key, None)
                    self._metric_completed_count += 1
        if failure is not None:
            raise failure

    @contextmanager
    def provider_call(
        self,
        *,
        case_id: str | None,
        metric_name: str,
        effective_output_token_limit: int | None,
    ) -> Iterator[ProviderInvocationSpan | None]:
        """Open a child span before awaiting one actual LiteLLM completion.

        Final response usage is attached through ``ProviderInvocationSpan.complete``.
        Consequently, the native MLflow span duration represents the real async
        invocation boundary, not the later callback/event processing interval.
        """

        attributes: dict[str, object] = {
            "evaluation.kind": "ragas_provider_call",
            "evaluation.client_request_id": self._context.client_request_id,
            "evaluation.configuration_id": self._context.configuration_id,
            "evaluation.run_id": self._context.evaluation_run_id or "not-recorded",
            "evaluation.case_id": case_id or "not-recorded",
            "request.id": (
                self._context.request_ids.get(case_id) if case_id is not None else None
            )
            or "not-recorded",
            "rag.trace_id": (
                self._context.rag_trace_ids.get(case_id)
                if case_id is not None
                else None
            )
            or "not-recorded",
            "ragas.metric_name": metric_name,
            "judge.model_id": self._model_id,
            "judge.aws_region": self._region,
            "provider.call_status": "IN_PROGRESS",
            "provider.duration_semantics": "awaited_litellm_completion",
            "provider.effective_output_token_limit": effective_output_token_limit,
            "provider.telemetry_complete": False,
            "trace.content_policy": "metadata_only",
        }
        with self._span_lock:
            self._provider_attempt_count += 1
            metric_key = (case_id or "", metric_name)
            metric_parent = self._metric_spans.get(metric_key)
            sequence = self._provider_sequences.get(metric_key, 0) + 1
            self._provider_sequences[metric_key] = sequence
        if metric_parent is None:
            with self._span_lock:
                self._diagnostic_reasons.add("ACTIVE_METRIC_PARENT_NOT_FOUND")
            yield None
            return
        self._increment_span_count()
        with self._tracer.span(
            "bedrock_provider_call",
            span_type="LLM",
            attributes=attributes,
            parent_span=metric_parent,
        ) as span:
            with self._span_lock:
                self._provider_created_count += 1
            handle = ProviderInvocationSpan(
                correlation=ProviderSpanCorrelation(
                    trace_id=self._trace_id,
                    metric_span_id=span_id_from_span(metric_parent),
                    provider_span_id=span_id_from_span(span),
                    sequence_within_metric=sequence,
                ),
                _span=span,
            )
            try:
                yield handle
            finally:
                if not handle._completed:
                    handle.complete_without_telemetry(
                        status="FAILED",
                        error_category="PROVIDER_TELEMETRY_INCOMPLETE",
                    )

    def _increment_span_count(self) -> None:
        with self._span_lock:
            self._span_count += 1

    def _unavailable_reason(self) -> str:
        status = getattr(self._tracer, "status", None)
        configured = getattr(status, "configured", None)
        active = getattr(status, "active", None)
        reason = getattr(status, "reason", None)
        if configured is False:
            return "Evaluation execution trace not captured: MLflow tracing is disabled."
        if active is False:
            return (
                "Evaluation execution trace not captured: "
                f"{reason or 'MLflow tracing is unavailable.'}"
            )
        return "Evaluation execution trace not captured."


def _json_ids(values: Mapping[str, str | None]) -> str:
    return json.dumps(
        {case_id: value for case_id, value in values.items() if value is not None},
        sort_keys=True,
        separators=(",", ":"),
    )


def _provider_result_attributes(
    invocation: ModelInvocationUsage,
) -> dict[str, object]:
    """Return content-free attributes observed from one completed provider attempt."""

    return {
        "provider.request_id": invocation.provider_request_id or "not-recorded",
        "provider.call_status": invocation.status,
        "provider.duration_ms": invocation.latency_ms,
        "provider.input_tokens": invocation.input_tokens,
        "provider.output_tokens": invocation.output_tokens,
        "provider.total_tokens": invocation.total_tokens,
        "provider.reasoning_tokens": invocation.reasoning_tokens,
        "provider.retry_count": invocation.retry_count,
        "provider.finish_reason": invocation.finish_reason or "not-recorded",
        "provider.effective_output_token_limit": (
            invocation.effective_output_token_limit
        ),
        "provider.response_type": invocation.response_type or "not-recorded",
        "provider.completion_returned_normally": (
            invocation.completion_returned_normally
        ),
        "provider.estimated_cost_usd": (
            str(invocation.estimated_cost_usd)
            if invocation.estimated_cost_usd is not None
            else "unavailable"
        ),
        "provider.cost_reason": invocation.estimated_cost_reason or "not-recorded",
        "provider.pricing_version": invocation.pricing_version or "unavailable",
        "provider.error_category": invocation.error_category or "not-recorded",
        "provider.telemetry_complete": True,
    }


def _safe_result(
    span: TraceSpan,
    *,
    status: Literal["COMPLETED", "FAILED"],
    error_category: str | None = None,
) -> None:
    """Record only bounded status metadata; never allow exception text into traces."""

    span.set_outputs({"status": status})
    span.set_attributes(
        {
            "evaluation.status": status,
            "evaluation.error_category": error_category or "not-recorded",
        }
    )
