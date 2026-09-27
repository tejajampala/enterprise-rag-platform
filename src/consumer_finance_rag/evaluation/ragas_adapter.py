# mypy: disable-error-code=import-untyped
"""Removable evaluation-only adapter for RAGAS 0.4 collections and experiments."""

from __future__ import annotations

import asyncio
import importlib.util
import os
import sys
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import nullcontext
from contextvars import ContextVar
from datetime import date, datetime
from decimal import Decimal
from threading import Lock
from types import ModuleType
from typing import Any, Literal, Protocol

from litellm.integrations.custom_logger import CustomLogger
from pydantic import BaseModel, ConfigDict

from consumer_finance_rag.costs import (
    ModelInvocationUsage,
    ModelUsageSummary,
    estimate_generation_cost,
    load_pricing_catalog,
)
from consumer_finance_rag.evaluation.judge_call_debug import JudgeCallDebugCapture
from consumer_finance_rag.evaluation.judge_preflight import (
    JudgeIdentity,
    JudgePreflightStatus,
    ragas_provider_support,
    resolve_judge_identity,
)
from consumer_finance_rag.evaluation.models import JudgeMetricValue, RAGCaseResult
from consumer_finance_rag.evaluation.ragas_execution_trace import (
    ProviderSpanCorrelation,
    RagasExecutionTrace,
    RagasExecutionTraceContext,
    RagasExecutionTraceReference,
)
from consumer_finance_rag.observability import NoOpTracer, Tracer

_ACTIVE_USAGE_SCOPE: ContextVar[tuple[str, str] | None] = ContextVar(
    "ragas_usage_scope", default=None
)
_ACTIVE_INSTRUCTOR_DEBUG_CALL_IDS: ContextVar[tuple[str, ...]] = ContextVar(
    "ragas_instructor_debug_call_ids", default=()
)


class RagasConfigurationError(RuntimeError):
    """Raised before a paid RAGAS evaluation when no Bedrock judge is configured."""


class RagasCompatibilityError(RuntimeError):
    """Raised when the isolated RAGAS integration cannot load or execute."""


class RagasEvaluator(Protocol):
    enabled: bool

    def evaluate(
        self, results: Sequence[RAGCaseResult]
    ) -> dict[str, dict[str, JudgeMetricValue]]: ...


class DisabledRagasEvaluator:
    enabled = False

    @property
    def usage_summary(self) -> ModelUsageSummary | None:
        return None

    @property
    def execution_trace(self) -> RagasExecutionTraceReference | None:
        return None

    def evaluate(self, results: Sequence[RAGCaseResult]) -> dict[str, dict[str, JudgeMetricValue]]:
        return {}


class RagasEvaluationInput(BaseModel):
    """Exact five-field input contract used by the RAGAS 0.4 adapter."""

    model_config = ConfigDict(frozen=True)

    case_id: str
    question: str
    answer: str
    contexts: list[str]
    reference: str


def _integer_field(value: object, *names: str) -> int | None:
    for name in names:
        candidate = (
            value.get(name)
            if isinstance(value, Mapping)
            else getattr(value, name, None)
        )
        if isinstance(candidate, int) and not isinstance(candidate, bool) and candidate >= 0:
            return candidate
    return None


def _response_usage(
    response: object,
) -> tuple[int | None, int | None, int | None, int | None]:
    usage = (
        response.get("usage")
        if isinstance(response, Mapping)
        else getattr(response, "usage", None)
    )
    if usage is None:
        return None, None, None, None
    input_tokens = _integer_field(usage, "prompt_tokens", "input_tokens")
    output_tokens = _integer_field(usage, "completion_tokens", "output_tokens")
    total_tokens = _integer_field(usage, "total_tokens")
    details = (
        usage.get("completion_tokens_details")
        if isinstance(usage, Mapping)
        else getattr(usage, "completion_tokens_details", None)
    )
    reasoning_tokens = (
        _integer_field(details, "reasoning_tokens") if details is not None else None
    )
    return input_tokens, output_tokens, total_tokens, reasoning_tokens


def _duration_ms(start_time: object, end_time: object) -> float | None:
    if isinstance(start_time, datetime) and isinstance(end_time, datetime):
        return max(0.0, (end_time - start_time).total_seconds() * 1000.0)
    return None


def _observable_retry_count(kwargs: Mapping[str, object]) -> int | None:
    for name in ("retry_count", "retries", "current_retry"):
        value = kwargs.get(name)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    return None


def _provider_request_id(response: object) -> str | None:
    """Extract only a bounded provider request identifier, never response content."""

    candidates: list[object] = []
    if isinstance(response, Mapping):
        candidates.append(response.get("request_id"))
        candidates.append(response.get("id"))
        hidden = response.get("_hidden_params")
    else:
        candidates.append(getattr(response, "request_id", None))
        candidates.append(getattr(response, "id", None))
        hidden = getattr(response, "_hidden_params", None)
    if isinstance(hidden, Mapping):
        headers = hidden.get("additional_headers") or hidden.get("headers")
        if isinstance(headers, Mapping):
            candidates.extend(
                headers.get(name)
                for name in ("x-amzn-requestid", "x-amzn-request-id", "request-id")
            )
    for value in candidates:
        if isinstance(value, str) and value.strip() and len(value) <= 256:
            return value.strip()
    return None


def _finish_reason(response: object) -> str | None:
    choices = (
        response.get("choices")
        if isinstance(response, Mapping)
        else getattr(response, "choices", None)
    )
    if not isinstance(choices, Sequence) or isinstance(choices, str | bytes) or not choices:
        return None
    first = choices[0]
    value = (
        first.get("finish_reason")
        if isinstance(first, Mapping)
        else getattr(first, "finish_reason", None)
    )
    return value if isinstance(value, str) and 0 < len(value) <= 64 else None


def _effective_output_token_limit(kwargs: Mapping[str, object]) -> int | None:
    for name in ("max_completion_tokens", "max_tokens"):
        value = kwargs.get(name)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return None


def _response_type(response: object) -> str:
    name = type(response).__name__
    return name[:128]


def _invocation_cost(
    *,
    model_id: str,
    region: str,
    input_tokens: int | None,
    output_tokens: int | None,
    total_tokens: int | None,
) -> tuple[Decimal | None, str | None, str | None]:
    if input_tokens is None or output_tokens is None:
        return None, "Provider token usage unavailable", None
    try:
        catalog = load_pricing_catalog()
        estimate = estimate_generation_cost(
            {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens or input_tokens + output_tokens,
            },
            catalog,
            provider="aws-bedrock",
            model_id=model_id,
            region=region,
            pricing_mode="on-demand-standard",
            as_of=date.today(),
        )
    except (OSError, ValueError):
        return None, "Verified pricing configuration unavailable", None
    return estimate.estimated_api_cost, estimate.reason, catalog.version


class PrivacySafeLiteLLMUsageCallback(CustomLogger):
    """Capture only response usage and operational metadata, never request content."""

    def __init__(
        self,
        *,
        model_id: str,
        region: str,
        execution_trace: Any | None = None,
        debug_capture: JudgeCallDebugCapture | None = None,
    ) -> None:
        super().__init__(turn_off_message_logging=True)
        self._model_id = model_id
        self._region = region
        self._records: list[ModelInvocationUsage] = []
        self._lock = Lock()
        self._execution_trace = execution_trace
        self._debug_capture = debug_capture

    async def async_log_success_event(
        self,
        kwargs: dict[str, object],
        response_obj: object,
        start_time: object,
        end_time: object,
    ) -> None:
        if self._execution_trace is not None:
            self._execution_trace.callback_received(success=True)
        self._append(
            kwargs=kwargs,
            response_obj=response_obj,
            start_time=start_time,
            end_time=end_time,
            status="COMPLETED",
            error_category=None,
        )

    async def async_log_failure_event(
        self,
        kwargs: dict[str, object],
        response_obj: object,
        start_time: object,
        end_time: object,
    ) -> None:
        if self._execution_trace is not None:
            self._execution_trace.callback_received(success=False)
        error = kwargs.get("exception")
        self._append(
            kwargs=kwargs,
            response_obj=response_obj,
            start_time=start_time,
            end_time=end_time,
            status="FAILED",
            error_category=(
                type(error).__name__
                if isinstance(error, Exception)
                else "PROVIDER_ERROR"
            ),
        )

    @property
    def execution_trace(self) -> Any | None:
        return self._execution_trace

    def record_completion_success(
        self,
        *,
        kwargs: Mapping[str, object],
        response_obj: object,
        start_time: object,
        end_time: object,
        scope: tuple[str, str] | None,
    ) -> ModelInvocationUsage:
        """Record one actual completion return without relying on callback dispatch."""

        return self._append(
            kwargs=kwargs,
            response_obj=response_obj,
            start_time=start_time,
            end_time=end_time,
            status="COMPLETED",
            error_category=None,
            scope=scope,
        )

    def record_completion_failure(
        self,
        *,
        kwargs: Mapping[str, object],
        start_time: object,
        end_time: object,
        error_category: str,
        scope: tuple[str, str] | None,
    ) -> ModelInvocationUsage:
        """Record a failed completion attempt without retaining exception content."""

        return self._append(
            kwargs=kwargs,
            response_obj={},
            start_time=start_time,
            end_time=end_time,
            status="FAILED",
            error_category=error_category,
            scope=scope,
        )

    def _append(
        self,
        *,
        kwargs: Mapping[str, object],
        response_obj: object,
        start_time: object,
        end_time: object,
        status: Literal["COMPLETED", "FAILED"],
        error_category: str | None,
        scope: tuple[str, str] | None = None,
    ) -> ModelInvocationUsage:
        active_scope = scope if scope is not None else _ACTIVE_USAGE_SCOPE.get()
        case_id, metric_name = (
            active_scope if active_scope is not None else (None, "unattributed")
        )
        if active_scope is None and self._execution_trace is not None:
            self._execution_trace.callback_correlation_failed(
                "ACTIVE_METRIC_SCOPE_NOT_FOUND"
            )
        input_tokens, output_tokens, total_tokens, reasoning_tokens = _response_usage(
            response_obj
        )
        estimated_cost, estimated_cost_reason, pricing_version = _invocation_cost(
            model_id=self._model_id,
            region=self._region,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )
        record = ModelInvocationUsage(
            backend="ragas",
            metric_name=metric_name,
            case_id=case_id,
            status=status,
            model_id=self._model_id,
            region=self._region,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            reasoning_tokens=reasoning_tokens,
            retry_count=_observable_retry_count(kwargs),
            latency_ms=_duration_ms(start_time, end_time),
            error_category=error_category,
            provider_request_id=_provider_request_id(response_obj),
            finish_reason=_finish_reason(response_obj),
            effective_output_token_limit=_effective_output_token_limit(kwargs),
            response_type=_response_type(response_obj),
            completion_returned_normally=status == "COMPLETED",
            estimated_cost_usd=estimated_cost,
            estimated_cost_reason=estimated_cost_reason,
            pricing_version=pricing_version,
        )
        with self._lock:
            self._records.append(record)
        return record

    def capture_debug_invocation(
        self,
        *,
        invocation: ModelInvocationUsage,
        request_kwargs: Mapping[str, object],
        response_obj: object,
        correlation: ProviderSpanCorrelation | None,
    ) -> None:
        """Capture content after the metadata-only provider span has ended."""

        if self._debug_capture is None:
            return
        if (
            correlation is None
            or correlation.trace_id is None
            or correlation.metric_span_id is None
            or correlation.provider_span_id is None
        ):
            raise RuntimeError("Judge-call debug capture requires provider span correlation")
        call_id = self._debug_capture.capture_invocation(
            invocation=invocation,
            request_kwargs=request_kwargs,
            response=response_obj,
            evaluation_trace_id=correlation.trace_id,
            metric_span_id=correlation.metric_span_id,
            provider_span_id=correlation.provider_span_id,
            sequence_within_metric=correlation.sequence_within_metric,
        )
        call_ids = _ACTIVE_INSTRUCTOR_DEBUG_CALL_IDS.get()
        _ACTIVE_INSTRUCTOR_DEBUG_CALL_IDS.set((*call_ids, call_id))

    def summary(self) -> ModelUsageSummary:
        with self._lock:
            records = tuple(self._records)
        complete = bool(records) and all(
            item.input_tokens is not None
            and item.output_tokens is not None
            for item in records
        )
        totals_complete = bool(records) and all(
            item.total_tokens is not None for item in records
        )
        retry_values = [item.retry_count for item in records if item.retry_count is not None]
        return ModelUsageSummary(
            model_id=self._model_id,
            region=self._region,
            provider_call_count=len(records),
            successful_call_count=sum(item.status == "COMPLETED" for item in records),
            failed_attempt_count=sum(item.status == "FAILED" for item in records),
            observable_retry_count=sum(retry_values) if retry_values else None,
            usage_complete=complete,
            input_tokens=(sum(item.input_tokens or 0 for item in records) if complete else None),
            output_tokens=(sum(item.output_tokens or 0 for item in records) if complete else None),
            total_tokens=(
                sum(item.total_tokens or 0 for item in records)
                if totals_complete
                else None
            ),
            reasoning_tokens=(
                sum(item.reasoning_tokens or 0 for item in records)
                if records and all(item.reasoning_tokens is not None for item in records)
                else None
            ),
            reason=(
                (
                    "Provider total-token or reasoning-token detail may be unavailable; "
                    "input/output usage is complete."
                    if complete and (not totals_complete)
                    else None
                )
                if complete
                else "One or more observable judge invocations omitted provider token usage."
            ),
            invocations=records,
        )


async def _within_usage_scope(
    case_id: str,
    metric_name: str,
    result: Awaitable[Any],
    execution_trace: Any | None = None,
) -> Any:
    token = _ACTIVE_USAGE_SCOPE.set((case_id, metric_name))
    try:
        if execution_trace is None:
            return await result
        with execution_trace.metric(case_id=case_id, metric_name=metric_name):
            return await result
    finally:
        _ACTIVE_USAGE_SCOPE.reset(token)


def instrumented_litellm_async_client(
    callback: PrivacySafeLiteLLMUsageCallback,
    *,
    completion: Callable[..., Awaitable[Any]] | None = None,
) -> Any:
    """Trace the exact async completion boundary used by RAGAS and Instructor."""

    from litellm import acompletion

    completion_callable = completion or acompletion

    async def invoke(*args: Any, **kwargs: Any) -> Any:
        scope = _ACTIVE_USAGE_SCOPE.get()
        execution_trace = callback.execution_trace
        if execution_trace is not None:
            execution_trace.completion_started()
        case_id, metric_name = scope if scope is not None else (None, "unattributed")
        provider_context = (
            execution_trace.provider_call(
                case_id=case_id,
                metric_name=metric_name,
                effective_output_token_limit=_effective_output_token_limit(kwargs),
            )
            if execution_trace is not None
            else nullcontext(None)
        )
        started = datetime.now()
        response_obj: object = {}
        record: ModelInvocationUsage | None = None
        correlation: ProviderSpanCorrelation | None = None
        try:
            with provider_context as provider_span:
                if provider_span is not None:
                    correlation = provider_span.correlation
                try:
                    response_obj = await completion_callable(*args, **kwargs)
                except BaseException as error:
                    try:
                        record = callback.record_completion_failure(
                            kwargs=kwargs,
                            start_time=started,
                            end_time=datetime.now(),
                            error_category=type(error).__name__,
                            scope=scope,
                        )
                    except Exception:
                        if provider_span is not None:
                            try:
                                provider_span.complete_without_telemetry(
                                    status="FAILED",
                                    error_category=type(error).__name__,
                                )
                            except Exception:
                                if execution_trace is not None:
                                    execution_trace.telemetry_failed(
                                        "PROVIDER_SPAN_FINALIZATION_ERROR"
                                    )
                        if execution_trace is not None:
                            execution_trace.telemetry_failed(
                                "FAILED_COMPLETION_TELEMETRY_ERROR"
                            )
                    else:
                        if provider_span is not None:
                            try:
                                provider_span.complete(record)
                            except Exception:
                                if execution_trace is not None:
                                    execution_trace.telemetry_failed(
                                        "PROVIDER_SPAN_FINALIZATION_ERROR"
                                    )
                    raise
                else:
                    try:
                        record = callback.record_completion_success(
                            kwargs=kwargs,
                            response_obj=response_obj,
                            start_time=started,
                            end_time=datetime.now(),
                            scope=scope,
                        )
                    except Exception:
                        if provider_span is not None:
                            try:
                                provider_span.complete_without_telemetry(
                                    status="COMPLETED",
                                    error_category=None,
                                )
                            except Exception:
                                if execution_trace is not None:
                                    execution_trace.telemetry_failed(
                                        "PROVIDER_SPAN_FINALIZATION_ERROR"
                                    )
                        if execution_trace is not None:
                            execution_trace.telemetry_failed(
                                "SUCCESSFUL_COMPLETION_TELEMETRY_ERROR"
                            )
                    else:
                        if provider_span is not None:
                            try:
                                provider_span.complete(record)
                            except Exception:
                                if execution_trace is not None:
                                    execution_trace.telemetry_failed(
                                        "PROVIDER_SPAN_FINALIZATION_ERROR"
                                    )
            return response_obj
        finally:
            if record is not None:
                try:
                    callback.capture_debug_invocation(
                        invocation=record,
                        request_kwargs=kwargs,
                        response_obj=response_obj,
                        correlation=correlation,
                    )
                except Exception:
                    if execution_trace is not None:
                        execution_trace.telemetry_failed(
                            "JUDGE_CALL_DEBUG_CAPTURE_ERROR"
                        )
            if execution_trace is not None:
                execution_trace.completion_completed()

    return invoke


def validate_ragas_structured_output(identity: JudgeIdentity) -> None:
    """Validate installed LiteLLM routing support without invoking the model."""

    support = ragas_provider_support(identity)
    if support.status is not JudgePreflightStatus.AVAILABLE:
        raise RagasConfigurationError(support.reason)


def build_ragas_instructor_llm(
    identity: JudgeIdentity,
    callback: PrivacySafeLiteLLMUsageCallback,
    *,
    max_tokens: int | None = None,
    timeout_seconds: float | None = None,
    max_retries: int | None = None,
    debug_capture: JudgeCallDebugCapture | None = None,
) -> Any:
    """Build the installed async Instructor adapter without invoking the provider."""

    validate_ragas_structured_output(identity)
    _install_vertexai_import_compatibility()
    from ragas.llms import llm_factory

    client = instrumented_litellm_async_client(callback)
    model_arguments: dict[str, Any] = {"aws_region_name": identity.region}
    if max_tokens is not None:
        model_arguments["max_tokens"] = max_tokens
    if timeout_seconds is not None:
        model_arguments["timeout"] = timeout_seconds
    if max_retries is not None:
        model_arguments["num_retries"] = max_retries
    llm = llm_factory(
        identity.litellm_model,
        provider="litellm",
        client=client,
        adapter="instructor",
        **model_arguments,
    )
    return _instrument_instructor_parsing(llm, debug_capture) if debug_capture else llm


def _instrument_instructor_parsing(
    llm: Any,
    capture: JudgeCallDebugCapture,
) -> Any:
    """Decorate one InstructorLLM instance while preserving its concrete RAGAS type."""

    def on_parse_error(error: Exception, **metadata: Any) -> None:
        call_ids = _ACTIVE_INSTRUCTOR_DEBUG_CALL_IDS.get()
        if not call_ids:
            return
        attempt = metadata.get("attempt_number")
        maximum = metadata.get("max_attempts")
        capture.record_instructor_failure(
            call_ids[-1],
            error_category=type(error).__name__,
            attempt_number=(attempt if isinstance(attempt, int) and attempt > 0 else None),
            maximum_attempts=(maximum if isinstance(maximum, int) and maximum > 0 else None),
        )

    original_agenerate = llm.agenerate

    async def debug_agenerate(prompt: str, response_model: Any) -> Any:
        token = _ACTIVE_INSTRUCTOR_DEBUG_CALL_IDS.set(())
        try:
            result = await original_agenerate(prompt, response_model)
        except BaseException as error:
            call_ids = _ACTIVE_INSTRUCTOR_DEBUG_CALL_IDS.get()
            if call_ids:
                capture.record_instructor_failure(
                    call_ids[-1], error_category=type(error).__name__
                )
            raise
        else:
            call_ids = _ACTIVE_INSTRUCTOR_DEBUG_CALL_IDS.get()
            if call_ids:
                capture.record_instructor_result(call_ids[-1], result)
            return result
        finally:
            _ACTIVE_INSTRUCTOR_DEBUG_CALL_IDS.reset(token)

    client = getattr(llm, "client", None)
    register = getattr(client, "on", None)
    if callable(register):
        register("parse:error", on_parse_error)
    llm.agenerate = debug_agenerate
    return llm


class _RagasMetric(BaseModel):
    model_config = ConfigDict(frozen=True)

    value: float | None
    reason: str | None = None


class _RagasOutput(BaseModel):
    model_config = ConfigDict(frozen=True)

    case_id: str
    metrics: dict[str, _RagasMetric]


def _install_vertexai_import_compatibility() -> None:
    """Work around RAGAS 0.4.3 importing a module removed by LangChain Community.

    RAGAS only uses these types for legacy provider detection. This application uses
    LiteLLM/Bedrock, so unavailable sentinels are sufficient and stay evaluation-local.
    """

    module_name = "langchain_community.chat_models.vertexai"
    if module_name in sys.modules or importlib.util.find_spec(module_name) is not None:
        return

    class _UnavailableVertexAI:
        pass

    compatibility_module = ModuleType(module_name)
    compatibility_module.ChatVertexAI = _UnavailableVertexAI  # type: ignore[attr-defined]
    compatibility_module.VertexAI = _UnavailableVertexAI  # type: ignore[attr-defined]
    sys.modules[module_name] = compatibility_module


def _metric_value(result: Any) -> _RagasMetric:
    value = result.value
    numeric = float(value) if isinstance(value, int | float) else None
    reason = result.reason if isinstance(result.reason, str) else None
    return _RagasMetric(value=numeric, reason=reason)


class Ragas043Evaluator:
    """RAGAS 0.4.3 collections runner using a LiteLLM-backed Bedrock judge."""

    enabled = True

    def __init__(
        self,
        *,
        judge_model: str,
        judge_region: str,
        references: dict[str, str],
        execution_context: RagasExecutionTraceContext | None = None,
        execution_tracer: Tracer | None = None,
        judge_max_tokens: int | None = None,
        context_precision_max_tokens: int | None = None,
        debug_capture: JudgeCallDebugCapture | None = None,
    ) -> None:
        if not judge_model:
            raise RagasConfigurationError(
                "RAGAS evaluation requires an explicit judge model. Set "
                "MLFLOW_GENAI_JUDGE_DEFAULT_MODEL=bedrock:/<enabled-model-id>."
            )
        if not judge_model.startswith("bedrock:/"):
            raise RagasConfigurationError(
                "The RAGAS adapter currently supports Bedrock judges only; use "
                "bedrock:/<enabled-model-id>."
            )
        if not judge_region:
            raise RagasConfigurationError("RAGAS evaluation requires an explicit judge Region.")
        try:
            identity = resolve_judge_identity(judge_model, judge_region)
        except ValueError as error:
            raise RagasConfigurationError(str(error)) from error
        self._identity = identity
        self._references = references
        self._judge_max_tokens = judge_max_tokens
        self._context_precision_max_tokens = context_precision_max_tokens
        self._debug_capture = debug_capture
        self._execution_trace = RagasExecutionTrace(
            execution_tracer or NoOpTracer(),
            context=(
                execution_context
                or RagasExecutionTraceContext(
                    configuration_id="not-recorded",
                    evaluation_run_id=None,
                    request_ids={},
                    rag_trace_ids={},
                )
            ),
            model_id=identity.bedrock_model_id,
            region=judge_region,
            requested_metrics=(
                "faithfulness",
                "context_precision",
                "context_recall",
                "response_groundedness",
            ),
        )
        self._usage_callback = PrivacySafeLiteLLMUsageCallback(
            model_id=identity.bedrock_model_id,
            region=judge_region,
            execution_trace=self._execution_trace,
            debug_capture=debug_capture,
        )

    @property
    def usage_summary(self) -> ModelUsageSummary:
        return self._usage_callback.summary()

    @property
    def execution_trace(self) -> RagasExecutionTraceReference:
        return self._execution_trace.reference

    def evaluate(self, results: Sequence[RAGCaseResult]) -> dict[str, dict[str, JudgeMetricValue]]:
        try:
            return asyncio.run(self._evaluate(results))
        except RagasConfigurationError:
            raise
        except Exception as error:
            raise RagasCompatibilityError(f"RAGAS 0.4.3 evaluation failed: {error}") from error
        finally:
            if self._debug_capture is not None:
                self._debug_capture.persist()

    async def _evaluate(
        self, results: Sequence[RAGCaseResult]
    ) -> dict[str, dict[str, JudgeMetricValue]]:
        _install_vertexai_import_compatibility()
        os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "true")
        validate_ragas_structured_output(self._identity)

        from ragas import experiment
        from ragas.backends import InMemoryBackend
        from ragas.dataset import Dataset
        from ragas.metrics.collections import (
            ContextPrecision,
            ContextRecall,
            Faithfulness,
            ResponseGroundedness,
        )

        # The provider-aware route is validated locally before this async
        # Instructor adapter can make any billable call.
        llm = build_ragas_instructor_llm(
            self._identity,
            self._usage_callback,
            max_tokens=self._judge_max_tokens,
            debug_capture=self._debug_capture,
        )
        context_precision_llm = (
            build_ragas_instructor_llm(
                self._identity,
                self._usage_callback,
                max_tokens=self._context_precision_max_tokens,
                debug_capture=self._debug_capture,
            )
            if self._context_precision_max_tokens is not None
            and self._context_precision_max_tokens != self._judge_max_tokens
            else llm
        )
        metrics = {
            "faithfulness": Faithfulness(llm=llm),
            "context_precision": ContextPrecision(llm=context_precision_llm),
            "context_recall": ContextRecall(llm=llm),
            "response_groundedness": ResponseGroundedness(llm=llm),
        }

        @experiment(_RagasOutput, name_prefix="consumer-finance")
        async def score(row: RagasEvaluationInput) -> _RagasOutput:
            values = {
                "faithfulness": _metric_value(
                    await _within_usage_scope(
                        row.case_id,
                        "faithfulness",
                        metrics["faithfulness"].ascore(
                            user_input=row.question,
                            response=row.answer,
                            retrieved_contexts=row.contexts,
                        ),
                        self._execution_trace,
                    )
                ),
                "context_precision": _metric_value(
                    await _within_usage_scope(
                        row.case_id,
                        "context_precision",
                        metrics["context_precision"].ascore(
                            user_input=row.question,
                            reference=row.reference,
                            retrieved_contexts=row.contexts,
                        ),
                        self._execution_trace,
                    )
                ),
                "context_recall": _metric_value(
                    await _within_usage_scope(
                        row.case_id,
                        "context_recall",
                        metrics["context_recall"].ascore(
                            user_input=row.question,
                            reference=row.reference,
                            retrieved_contexts=row.contexts,
                        ),
                        self._execution_trace,
                    )
                ),
                "response_groundedness": _metric_value(
                    await _within_usage_scope(
                        row.case_id,
                        "response_groundedness",
                        metrics["response_groundedness"].ascore(
                            response=row.answer,
                            retrieved_contexts=row.contexts,
                        ),
                        self._execution_trace,
                    )
                ),
            }
            return _RagasOutput(case_id=row.case_id, metrics=values)

        backend = InMemoryBackend()  # type: ignore[no-untyped-call]
        data = [
            RagasEvaluationInput(
                case_id=result.case_id,
                question=result.question,
                answer=result.answer,
                contexts=result.contexts,
                reference=self._references[result.case_id],
            )
            for result in results
        ]
        dataset = Dataset(
            name="consumer-finance-rag-evaluation",
            backend=backend,
            data_model=RagasEvaluationInput,
            data=data,
        )
        try:
            with self._execution_trace.root():
                completed = await score.arun(
                    dataset,
                    name="ragas-0.4.3",
                    backend=backend,
                )
                parsed = {
                    item.case_id: {
                        name: JudgeMetricValue(value=value.value, reason=value.reason)
                        for name, value in item.metrics.items()
                    }
                    for item in completed
                    if isinstance(item, _RagasOutput)
                }
                if len(parsed) != len(results):
                    raise RagasCompatibilityError(
                        "RAGAS did not return a result for every case; inspect its "
                        "experiment output."
                    )
        except Exception:
            # The root span has already recorded a bounded error category. Re-raise only
            # after it closes so MLflow cannot serialize the provider exception text.
            raise
        return parsed


def build_ragas_evaluator(
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
) -> RagasEvaluator:
    if not enabled:
        return DisabledRagasEvaluator()
    return Ragas043Evaluator(
        judge_model=judge_model or "",
        judge_region=judge_region or "",
        references=references,
        execution_context=execution_context,
        execution_tracer=execution_tracer,
        judge_max_tokens=judge_max_tokens,
        context_precision_max_tokens=context_precision_max_tokens,
        debug_capture=debug_capture,
    )
