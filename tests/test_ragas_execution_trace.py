"""Offline coverage for metadata-only RAGAS execution traces."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import pytest
from litellm import ModelResponse

import consumer_finance_rag.evaluation.ragas_adapter as ragas_adapter
from consumer_finance_rag.evaluation.judge_call_debug import (
    ApprovedPublicDebugCase,
    JudgeCallDebugCapture,
    JudgeCallDebugConfig,
    load_judge_call_debug_artifact,
)
from consumer_finance_rag.evaluation.models import RAGCaseResult
from consumer_finance_rag.evaluation.ragas_adapter import (
    PrivacySafeLiteLLMUsageCallback,
    Ragas043Evaluator,
    _install_vertexai_import_compatibility,
    _instrument_instructor_parsing,
    _within_usage_scope,
    instrumented_litellm_async_client,
)
from consumer_finance_rag.evaluation.ragas_execution_trace import (
    RagasExecutionTrace,
    RagasExecutionTraceContext,
)
from consumer_finance_rag.observability import (
    NoOpTracer,
    TracePersistenceResult,
    TraceSpan,
)


class _Span:
    def __init__(self, *, name: str, parent_id: str | None, trace_id: str) -> None:
        self.name = name
        self.parent_id = parent_id
        self.span_id = f"span-{name}-{len(_RECORDED_SPANS)}"
        self.trace_id = trace_id
        self.attributes: dict[str, Any] = {}
        self.inputs: Any = None
        self.outputs: Any = None
        self.started_at = perf_counter()
        self.ended_at: float | None = None

    @property
    def duration_ms(self) -> float | None:
        if self.ended_at is None:
            return None
        return (self.ended_at - self.started_at) * 1000.0

    def set_inputs(self, inputs: Any) -> None:
        self.inputs = inputs

    def set_outputs(self, outputs: Any) -> None:
        self.outputs = outputs

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def set_attributes(self, attributes: dict[str, Any]) -> None:
        self.attributes.update(attributes)


_RECORDED_SPANS: list[_Span] = []


class _RecordingTracer:
    def __init__(self) -> None:
        self._active: ContextVar[_Span | None] = ContextVar("test_trace_span", default=None)

    @contextmanager
    def span(
        self,
        name: str,
        *,
        span_type: str,
        attributes: dict[str, Any] | None = None,
        parent_span: TraceSpan | None = None,
    ) -> Iterator[TraceSpan]:
        del span_type
        parent = parent_span if isinstance(parent_span, _Span) else self._active.get()
        span = _Span(
            name=name,
            parent_id=parent.span_id if parent is not None else None,
            trace_id=parent.trace_id if parent is not None else "tr-ragas-execution",
        )
        if attributes is not None:
            span.attributes.update(attributes)
        _RECORDED_SPANS.append(span)
        token = self._active.set(span)
        try:
            yield span
        finally:
            span.ended_at = perf_counter()
            self._active.reset(token)

    def verify_trace(
        self,
        trace_id: str,
        *,
        expected_client_request_id: str,
        expected_span_count: int,
        expected_metric_names: tuple[str, ...] = (),
        expected_provider_call_count: int = 0,
    ) -> TracePersistenceResult:
        assert trace_id == "tr-ragas-execution"
        assert len(expected_client_request_id) <= 50
        if expected_span_count != len(_RECORDED_SPANS):
            return TracePersistenceResult(
                status="HIERARCHY_INCOMPLETE",
                reason="missing child spans",
                persisted_span_count=len(_RECORDED_SPANS),
            )
        actual_metric_names = tuple(
            span.name.removeprefix("ragas_metric:")
            for span in _RECORDED_SPANS
            if span.name.startswith("ragas_metric:")
        )
        actual_provider_count = sum(
            span.name == "bedrock_provider_call" for span in _RECORDED_SPANS
        )
        if sorted(actual_metric_names) != sorted(expected_metric_names):
            return TracePersistenceResult(
                status="HIERARCHY_INCOMPLETE",
                reason="missing metric spans",
                persisted_span_count=len(_RECORDED_SPANS),
                persisted_provider_call_count=actual_provider_count,
            )
        if actual_provider_count != expected_provider_call_count:
            return TracePersistenceResult(
                status="HIERARCHY_INCOMPLETE",
                reason="missing provider spans",
                persisted_span_count=len(_RECORDED_SPANS),
                persisted_provider_call_count=actual_provider_count,
            )
        if expected_metric_names and expected_provider_call_count == 0:
            return TracePersistenceResult(
                status="HIERARCHY_INCOMPLETE",
                reason="no completion invocation observed",
                persisted_span_count=len(_RECORDED_SPANS),
                persisted_provider_call_count=0,
            )
        return TracePersistenceResult(
            status="CONFIRMED",
            reason="persisted",
            persisted_span_count=expected_span_count,
            persisted_provider_call_count=actual_provider_count,
        )


def _trace() -> RagasExecutionTrace:
    _RECORDED_SPANS.clear()
    return RagasExecutionTrace(
        _RecordingTracer(),
        context=RagasExecutionTraceContext(
            configuration_id="a" * 64,
            evaluation_run_id="run-123",
            request_ids={"case-a": "request-a", "case-b": "request-b"},
            rag_trace_ids={"case-a": "tr-rag-a", "case-b": "tr-rag-b"},
        ),
        model_id="openai.gpt-oss-20b-1:0",
        region="ap-southeast-2",
        requested_metrics=("faithfulness", "response_groundedness"),
    )


def test_concurrent_metric_provider_spans_preserve_actual_metric_parent_and_privacy(
    tmp_path: Path,
) -> None:
    trace = _trace()
    debug_capture = JudgeCallDebugCapture(
        JudgeCallDebugConfig(artifact_directory=tmp_path),
        cases=tuple(
            ApprovedPublicDebugCase(
                case_id=case_id,
                question=f"Reviewed {case_id}?",
                review_status="reviewed",
                visibility="PUBLIC",
            )
            for case_id in ("case-a", "case-b")
        ),
        evaluation_configuration_id="a" * 64,
        evaluation_run_id="run-123",
        model_id="openai.gpt-oss-20b-1:0",
        region="ap-southeast-2",
    )
    for case_id in ("case-a", "case-b"):
        debug_capture.bind_case_inputs(
            case_id=case_id,
            question=f"Reviewed {case_id}?",
            contexts=(f"Approved {case_id} context",),
            chunk_ids=(f"{case_id}:chunk",),
        )
    callback = PrivacySafeLiteLLMUsageCallback(
        model_id="openai.gpt-oss-20b-1:0",
        region="ap-southeast-2",
        execution_trace=trace,
        debug_capture=debug_capture,
    )

    completion_count = 0

    async def fake_completion(*_args: object, **_kwargs: object) -> dict[str, object]:
        nonlocal completion_count
        completion_count += 1
        provider_request_id = f"offline-provider-request-{completion_count}"
        await asyncio.sleep(0)
        return {
            "id": provider_request_id,
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 7,
                "total_tokens": 18,
            },
        }

    completion = instrumented_litellm_async_client(
        callback,
        completion=fake_completion,
    )

    async def call(case_id: str, metric_name: str) -> None:
        async def provider() -> None:
            await completion(
                model="bedrock/converse/openai.gpt-oss-20b-1:0",
                messages=[{"role": "user", "content": "must not be captured"}],
            )

        await _within_usage_scope(case_id, metric_name, provider(), trace)

    async def run() -> None:
        with trace.root():
            await asyncio.gather(
                call("case-a", "faithfulness"),
                call("case-b", "response_groundedness"),
                call("case-a", "response_groundedness"),
                call("case-b", "faithfulness"),
            )

    asyncio.run(run())

    root = next(span for span in _RECORDED_SPANS if span.name == "ragas_evaluation")
    metrics = [span for span in _RECORDED_SPANS if span.name.startswith("ragas_metric:")]
    providers = [span for span in _RECORDED_SPANS if span.name == "bedrock_provider_call"]
    assert len(metrics) == 4
    assert len(providers) == 4
    assert all(metric.parent_id == root.span_id for metric in metrics)
    parents = {metric.span_id: metric.attributes["ragas.metric_name"] for metric in metrics}
    provider_parent_ids = [provider.parent_id for provider in providers]
    assert all(parent_id is not None for parent_id in provider_parent_ids)
    assert {parents[parent_id] for parent_id in provider_parent_ids if parent_id} == {
        "faithfulness",
        "response_groundedness",
    }
    assert all(
        provider.attributes["judge.model_id"] == "openai.gpt-oss-20b-1:0"
        for provider in providers
    )
    assert all(
        provider.attributes["judge.aws_region"] == "ap-southeast-2"
        for provider in providers
    )
    rendered = repr([(span.inputs, span.outputs, span.attributes) for span in _RECORDED_SPANS])
    assert "must not be captured" not in rendered
    assert "messages" not in rendered
    assert trace.reference.trace_id == "tr-ragas-execution"
    assert trace.reference.evaluation_run_id == "run-123"
    assert trace.reference.configuration_id == "a" * 64
    assert trace.reference.client_request_id is not None
    assert len(trace.reference.client_request_id) == 36
    assert trace.reference.status == "CONFIRMED"
    assert trace.reference.diagnostics.metric_invocations_started == 4
    assert trace.reference.diagnostics.metric_invocations_completed == 4
    assert trace.reference.diagnostics.litellm_completion_invocations_started == 4
    assert trace.reference.diagnostics.litellm_completion_invocations_completed == 4
    assert trace.reference.diagnostics.success_callbacks_received == 0
    assert trace.reference.diagnostics.failure_callbacks_received == 0
    assert trace.reference.diagnostics.provider_spans_attempted == 4
    assert trace.reference.diagnostics.provider_spans_created == 4
    assert trace.reference.diagnostics.provider_spans_persisted == 4
    assert root.attributes["evaluation.configuration_id"] == "a" * 64
    assert root.attributes["evaluation.run_id"] == "run-123"
    assert root.attributes["evaluation.request_ids"] == (
        '{"case-a":"request-a","case-b":"request-b"}'
    )
    assert root.attributes["evaluation.rag_trace_ids"] == (
        '{"case-a":"tr-rag-a","case-b":"tr-rag-b"}'
    )
    assert debug_capture.persist() is not None
    artifact = load_judge_call_debug_artifact(debug_capture.artifact_path)
    assert len(artifact.calls) == 4
    assert len({call.response.provider_request_id for call in artifact.calls}) == 4
    assert len({call.provider_span_id for call in artifact.calls}) == 4
    assert {(call.case_id, call.metric_name) for call in artifact.calls} == {
        ("case-a", "faithfulness"),
        ("case-a", "response_groundedness"),
        ("case-b", "faithfulness"),
        ("case-b", "response_groundedness"),
    }
    assert all(call.provider_span_id is not None for call in artifact.calls)
    assert "must not be captured" in debug_capture.artifact_path.read_text(
        encoding="utf-8"
    )


def test_provider_span_wraps_the_real_awaited_completion_duration() -> None:
    trace = _trace()
    callback = PrivacySafeLiteLLMUsageCallback(
        model_id="openai.gpt-oss-20b-1:0",
        region="ap-southeast-2",
        execution_trace=trace,
    )

    async def delayed_completion(
        *_args: object, **_kwargs: object
    ) -> dict[str, object]:
        await asyncio.sleep(0.04)
        return {
            "id": "provider-latency-proof",
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }

    completion = instrumented_litellm_async_client(
        callback,
        completion=delayed_completion,
    )

    async def run() -> None:
        with trace.root():
            await _within_usage_scope(
                "case-a",
                "faithfulness",
                completion(model="model", messages=[{"content": "private"}]),
                trace,
            )

    asyncio.run(run())

    provider = next(
        span for span in _RECORDED_SPANS if span.name == "bedrock_provider_call"
    )
    assert provider.duration_ms is not None and provider.duration_ms >= 30
    measured = provider.attributes["provider.duration_ms"]
    assert isinstance(measured, float) and measured >= 30
    assert provider.attributes["provider.duration_semantics"] == (
        "awaited_litellm_completion"
    )
    assert provider.attributes["provider.request_id"] == "provider-latency-proof"
    assert provider.attributes["provider.telemetry_complete"] is True


def test_real_ragas_experiment_path_opens_all_metric_and_callback_spans(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_vertexai_import_compatibility()
    completion_calls: list[str] = []
    completion_limits: list[tuple[str, int | None]] = []

    async def fake_bedrock_completion(
        *_args: object, **kwargs: object
    ) -> ModelResponse:
        messages = kwargs.get("messages")
        assert isinstance(messages, list)
        prompt = str(messages[-1]["content"])
        lowered = prompt.casefold()
        if "break down each sentence" in lowered:
            payload: dict[str, object] = {
                "statements": ["The reviewed answer is supported."]
            }
            category = "faithfulness"
        elif "judge the faithfulness" in lowered:
            payload = {
                "statements": [
                    {
                        "statement": "The reviewed answer is supported.",
                        "reason": "supported",
                        "verdict": 1,
                    }
                ]
            }
            category = "faithfulness"
        elif "context was useful" in lowered:
            payload = {"reason": "useful", "verdict": 1}
            category = "context_precision"
        elif "classify if the statement can be attributed" in lowered:
            payload = {
                "classifications": [
                    {
                        "statement": "Reviewed reference.",
                        "reason": "attributed",
                        "attributed": 1,
                    }
                ]
            }
            category = "context_recall"
        elif "groundedness" in lowered:
            payload = {"rating": 2}
            category = "response_groundedness"
        else:
            raise AssertionError("Unexpected installed RAGAS prompt contract")
        completion_calls.append(category)
        limit = kwargs.get("max_tokens")
        completion_limits.append(
            (category, limit if isinstance(limit, int) else None)
        )
        return ModelResponse(
            id=f"offline-provider-{len(completion_calls)}",
            model="bedrock/converse/openai.gpt-oss-20b-1:0",
            choices=[
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(payload),
                    },
                    "finish_reason": "stop",
                }
            ],
            usage={
                "prompt_tokens": 13,
                "completion_tokens": 5,
                "total_tokens": 18,
            },
        )

    original_instrumentation = ragas_adapter.instrumented_litellm_async_client

    def offline_instrumentation(
        callback: PrivacySafeLiteLLMUsageCallback,
    ) -> Any:
        return original_instrumentation(
            callback,
            completion=fake_bedrock_completion,
        )

    monkeypatch.setattr(
        ragas_adapter,
        "instrumented_litellm_async_client",
        offline_instrumentation,
    )

    _RECORDED_SPANS.clear()
    debug_capture = JudgeCallDebugCapture(
        JudgeCallDebugConfig(artifact_directory=tmp_path),
        cases=(
            ApprovedPublicDebugCase(
                case_id="case-a",
                question="Reviewed question?",
                review_status="reviewed",
                visibility="PUBLIC",
            ),
        ),
        evaluation_configuration_id="d" * 64,
        evaluation_run_id="run-real-ragas-path",
        model_id="openai.gpt-oss-20b-1:0",
        region="ap-southeast-2",
    )
    debug_capture.bind_case_inputs(
        case_id="case-a",
        question="Reviewed question?",
        contexts=("Approved public context A.", "Approved public context B."),
        chunk_ids=("chunk-a", "chunk-b"),
    )
    evaluator = Ragas043Evaluator(
        judge_model="bedrock:/openai.gpt-oss-20b-1:0",
        judge_region="ap-southeast-2",
        references={"case-a": "Reviewed reference."},
        execution_context=RagasExecutionTraceContext(
            configuration_id="d" * 64,
            evaluation_run_id="run-real-ragas-path",
            request_ids={"case-a": "request-a"},
            rag_trace_ids={"case-a": "tr-rag-a"},
        ),
        execution_tracer=_RecordingTracer(),
        judge_max_tokens=1024,
        context_precision_max_tokens=2048,
        debug_capture=debug_capture,
    )

    result = evaluator.evaluate(
        [
            RAGCaseResult(
                case_id="case-a",
                question="Reviewed question?",
                answer="Reviewed answer.",
                contexts=["Approved public context A.", "Approved public context B."],
                chunk_ids=["chunk-a", "chunk-b"],
            )
        ]
    )

    assert set(result["case-a"]) == {
        "faithfulness",
        "context_precision",
        "context_recall",
        "response_groundedness",
    }
    root = next(span for span in _RECORDED_SPANS if span.name == "ragas_evaluation")
    metrics = [span for span in _RECORDED_SPANS if span.name.startswith("ragas_metric:")]
    providers = [span for span in _RECORDED_SPANS if span.name == "bedrock_provider_call"]
    assert len(metrics) == 4
    assert completion_calls.count("faithfulness") == 2
    assert completion_calls.count("context_precision") == 2
    assert completion_calls.count("context_recall") == 1
    assert completion_calls.count("response_groundedness") == 2
    assert {
        limit for category, limit in completion_limits if category == "context_precision"
    } == {2048}
    assert {
        limit for category, limit in completion_limits if category != "context_precision"
    } == {1024}
    assert len(providers) == 7
    assert all(span.parent_id == root.span_id for span in metrics)
    metric_ids = {span.span_id for span in metrics}
    assert all(span.parent_id in metric_ids for span in providers)
    metric_by_id = {
        span.span_id: span.attributes["ragas.metric_name"] for span in metrics
    }
    provider_metrics = [metric_by_id[str(span.parent_id)] for span in providers]
    assert provider_metrics.count("faithfulness") == 2
    assert provider_metrics.count("context_precision") == 2
    assert provider_metrics.count("context_recall") == 1
    assert provider_metrics.count("response_groundedness") == 2
    diagnostics = evaluator.execution_trace.diagnostics
    assert diagnostics.litellm_completion_invocations_started == 7
    assert diagnostics.litellm_completion_invocations_completed == 7
    assert diagnostics.success_callbacks_received == 0
    assert diagnostics.failure_callbacks_received == 0
    assert diagnostics.callback_correlation_failures == 0
    assert diagnostics.provider_spans_attempted == 7
    assert diagnostics.provider_spans_created == 7
    assert diagnostics.provider_spans_persisted == 7
    assert all(
        span.attributes["provider.finish_reason"] == "stop" for span in providers
    )
    assert {
        span.attributes["provider.effective_output_token_limit"]
        for span in providers
        if span.attributes["ragas.metric_name"] == "context_precision"
    } == {2048}
    rendered = repr(
        [(span.inputs, span.outputs, span.attributes) for span in _RECORDED_SPANS]
    )
    assert "real-path-sensitive-canary" not in rendered
    assert debug_capture.persisted_path is not None
    artifact = load_judge_call_debug_artifact(debug_capture.persisted_path)
    assert len(artifact.calls) == 7
    assert artifact.evaluation_trace_id == "tr-ragas-execution"
    provider_span_ids = {span.span_id for span in providers}
    metric_span_ids = {span.span_id for span in metrics}
    assert {call.provider_span_id for call in artifact.calls} == provider_span_ids
    assert all(call.metric_span_id in metric_span_ids for call in artifact.calls)
    provider_by_id = {span.span_id: span for span in providers}
    assert all(
        provider_by_id[str(call.provider_span_id)].parent_id == call.metric_span_id
        for call in artifact.calls
    )
    assert len({call.response.provider_request_id for call in artifact.calls}) == 7
    assert sum(
        call.metric_name == "response_groundedness" for call in artifact.calls
    ) == 2
    assert all(call.request.messages for call in artifact.calls)
    assert all(
        call.request.structured_output_schema is not None
        for call in artifact.calls
    )
    assert all(call.response.content is not None for call in artifact.calls)
    assert all(call.instructor.validation_status == "PASSED" for call in artifact.calls)
    context_calls = [
        call for call in artifact.calls if call.metric_name == "context_precision"
    ]
    assert {call.context_chunk_ids for call in context_calls} == {
        ("chunk-a",),
        ("chunk-b",),
    }
    assert all(
        call.context_correlation == "exact_outgoing_message_match"
        for call in context_calls
    )
    usage = evaluator.usage_summary
    assert usage.provider_call_count == 7
    assert usage.successful_call_count == 7
    assert usage.failed_attempt_count == 0
    assert usage.input_tokens == 91
    assert usage.output_tokens == 35
    assert usage.total_tokens == 126
    invocation_by_request = {
        invocation.provider_request_id: invocation
        for invocation in usage.invocations
    }
    assert len(invocation_by_request) == 7
    for call in artifact.calls:
        invocation = invocation_by_request[call.response.provider_request_id]
        assert invocation.input_tokens == call.response.input_tokens
        assert invocation.output_tokens == call.response.output_tokens
        assert invocation.total_tokens == call.response.total_tokens
        assert str(invocation.estimated_cost_usd) == call.estimated_cost_usd
    serialized_report_safe = usage.model_dump_json()
    assert "Reviewed question" not in serialized_report_safe
    assert "Approved public context" not in serialized_report_safe


def test_failed_provider_call_is_traced_with_safe_category_not_exception_text() -> None:
    trace = _trace()
    callback = PrivacySafeLiteLLMUsageCallback(
        model_id="openai.gpt-oss-20b-1:0",
        region="ap-southeast-2",
        execution_trace=trace,
    )

    async def failed_completion(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("secret prompt value")

    completion = instrumented_litellm_async_client(
        callback,
        completion=failed_completion,
    )

    async def run() -> None:
        with trace.root():
            async def provider() -> None:
                with pytest.raises(RuntimeError, match="secret prompt value"):
                    await completion(
                        model="bedrock/converse/openai.gpt-oss-20b-1:0",
                        messages=[{"role": "user", "content": "private prompt"}],
                    )

            await _within_usage_scope("case-a", "faithfulness", provider(), trace)

    asyncio.run(run())

    provider = next(span for span in _RECORDED_SPANS if span.name == "bedrock_provider_call")
    assert provider.attributes["provider.call_status"] == "FAILED"
    assert provider.attributes["provider.error_category"] == "RuntimeError"
    assert "secret prompt value" not in repr(provider.attributes)
    diagnostics = trace.reference.diagnostics
    assert diagnostics.litellm_completion_invocations_started == 1
    assert diagnostics.litellm_completion_invocations_completed == 1
    assert diagnostics.failure_callbacks_received == 0
    assert diagnostics.provider_spans_attempted == 1
    assert diagnostics.provider_spans_created == 1


def test_truncated_context_precision_preserves_response_usage_and_failure_status(
    tmp_path: Path,
) -> None:
    _install_vertexai_import_compatibility()
    from instructor.core import IncompleteOutputException
    from ragas.llms import llm_factory
    from ragas.metrics.collections import ContextPrecision

    _RECORDED_SPANS.clear()
    trace = RagasExecutionTrace(
        _RecordingTracer(),
        context=RagasExecutionTraceContext(
            configuration_id="truncated-context-precision",
            evaluation_run_id="run-truncated",
            request_ids={"case-a": "request-a"},
            rag_trace_ids={"case-a": "tr-rag-a"},
        ),
        model_id="openai.gpt-oss-20b-1:0",
        region="ap-southeast-2",
        requested_metrics=("context_precision",),
    )
    callback = PrivacySafeLiteLLMUsageCallback(
        model_id="openai.gpt-oss-20b-1:0",
        region="ap-southeast-2",
        execution_trace=trace,
        debug_capture=(
            debug_capture := JudgeCallDebugCapture(
                JudgeCallDebugConfig(artifact_directory=tmp_path),
                cases=(
                    ApprovedPublicDebugCase(
                        case_id="case-a",
                        question="Reviewed question?",
                        review_status="reviewed",
                        visibility="PUBLIC",
                    ),
                ),
                evaluation_configuration_id="truncated-context-precision",
                evaluation_run_id="run-truncated",
                model_id="openai.gpt-oss-20b-1:0",
                region="ap-southeast-2",
            )
        ),
    )
    debug_capture.bind_case_inputs(
        case_id="case-a",
        question="Reviewed question?",
        contexts=("Approved public context.",),
        chunk_ids=("chunk-a",),
    )
    response = ModelResponse(
        id="truncated-provider-request",
        model="bedrock/converse/openai.gpt-oss-20b-1:0",
        choices=[
            {
                "index": 0,
                "message": {"role": "assistant", "content": '{"reason":"partial"'},
                "finish_reason": "length",
            }
        ],
        usage={
            "prompt_tokens": 1303,
            "completion_tokens": 1024,
            "total_tokens": 2327,
            "completion_tokens_details": {"reasoning_tokens": 956},
        },
    )
    completion_count = 0

    async def truncated_completion(*_args: object, **kwargs: object) -> ModelResponse:
        nonlocal completion_count
        completion_count += 1
        assert kwargs["max_tokens"] == 1024
        return response

    llm = llm_factory(
        "bedrock/converse/openai.gpt-oss-20b-1:0",
        provider="litellm",
        client=instrumented_litellm_async_client(
            callback,
            completion=truncated_completion,
        ),
        adapter="instructor",
        aws_region_name="ap-southeast-2",
        max_tokens=1024,
    )
    metric = ContextPrecision(llm=_instrument_instructor_parsing(llm, debug_capture))

    async def run() -> None:
        with trace.root():
            await _within_usage_scope(
                "case-a",
                "context_precision",
                metric.ascore(
                    user_input="Reviewed question?",
                    reference="Reviewed reference.",
                    retrieved_contexts=["Approved public context."],
                ),
                trace,
            )

    with pytest.raises(IncompleteOutputException) as caught:
        asyncio.run(run())

    assert caught.value.last_completion is response
    assert completion_count == 1
    root = next(span for span in _RECORDED_SPANS if span.name == "ragas_evaluation")
    metric_span = next(
        span for span in _RECORDED_SPANS if span.name == "ragas_metric:context_precision"
    )
    provider = next(span for span in _RECORDED_SPANS if span.name == "bedrock_provider_call")
    assert root.attributes["evaluation.status"] == "FAILED"
    assert root.attributes["evaluation.error_category"] == "IncompleteOutputException"
    assert metric_span.attributes["evaluation.status"] == "FAILED"
    assert metric_span.attributes["evaluation.error_category"] == "IncompleteOutputException"
    assert provider.parent_id == metric_span.span_id
    assert provider.attributes["provider.call_status"] == "COMPLETED"
    assert provider.attributes["provider.finish_reason"] == "length"
    assert provider.attributes["provider.effective_output_token_limit"] == 1024
    assert provider.attributes["provider.output_tokens"] == 1024
    assert provider.attributes["provider.reasoning_tokens"] == 956
    assert provider.attributes["provider.response_type"] == "ModelResponse"
    assert provider.attributes["provider.completion_returned_normally"] is True
    usage = callback.summary()
    assert usage.provider_call_count == 1
    assert usage.output_tokens == 1024
    assert usage.reasoning_tokens == 956
    assert usage.invocations[0].finish_reason == "length"
    assert usage.invocations[0].estimated_cost_usd is not None
    assert debug_capture.persist() is not None
    artifact = load_judge_call_debug_artifact(debug_capture.artifact_path)
    assert len(artifact.calls) == 1
    captured = artifact.calls[0]
    assert captured.response.finish_reason == "length"
    assert captured.response.output_tokens == 1024
    assert captured.instructor.validation_status == "FAILED"
    assert captured.instructor.error_category == "IncompleteOutputException"
    assert captured.provider_span_id == provider.span_id
    assert captured.metric_span_id == metric_span.span_id
    rendered = repr(
        [(span.inputs, span.outputs, span.attributes) for span in _RECORDED_SPANS]
    )
    assert "Reviewed question" not in rendered
    assert "Approved public context" not in rendered


def test_instructor_schema_failure_is_correlated_to_actual_provider_attempts(
    tmp_path: Path,
) -> None:
    _install_vertexai_import_compatibility()
    from instructor.core import InstructorRetryException
    from ragas.llms import llm_factory
    from ragas.metrics.collections import ContextPrecision

    _RECORDED_SPANS.clear()
    trace = RagasExecutionTrace(
        _RecordingTracer(),
        context=RagasExecutionTraceContext(
            configuration_id="schema-failure",
            evaluation_run_id="run-schema-failure",
            request_ids={"case-a": "request-a"},
            rag_trace_ids={"case-a": "tr-rag-a"},
        ),
        model_id="openai.gpt-oss-20b-1:0",
        region="ap-southeast-2",
        requested_metrics=("context_precision",),
    )
    debug_capture = JudgeCallDebugCapture(
        JudgeCallDebugConfig(artifact_directory=tmp_path),
        cases=(
            ApprovedPublicDebugCase(
                case_id="case-a",
                question="Reviewed question?",
                review_status="reviewed",
                visibility="PUBLIC",
            ),
        ),
        evaluation_configuration_id="schema-failure",
        evaluation_run_id="run-schema-failure",
        model_id="openai.gpt-oss-20b-1:0",
        region="ap-southeast-2",
    )
    debug_capture.bind_case_inputs(
        case_id="case-a",
        question="Reviewed question?",
        contexts=("Approved public context.",),
        chunk_ids=("chunk-a",),
    )
    callback = PrivacySafeLiteLLMUsageCallback(
        model_id="openai.gpt-oss-20b-1:0",
        region="ap-southeast-2",
        execution_trace=trace,
        debug_capture=debug_capture,
    )
    completion_count = 0

    async def invalid_completion(*_args: object, **_kwargs: object) -> ModelResponse:
        nonlocal completion_count
        completion_count += 1
        return ModelResponse(
            id=f"invalid-{completion_count}",
            model="bedrock/converse/openai.gpt-oss-20b-1:0",
            choices=[
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": '{"unexpected":true}',
                    },
                    "finish_reason": "stop",
                }
            ],
            usage={
                "prompt_tokens": 50,
                "completion_tokens": 5,
                "total_tokens": 55,
            },
        )

    llm = llm_factory(
        "bedrock/converse/openai.gpt-oss-20b-1:0",
        provider="litellm",
        client=instrumented_litellm_async_client(
            callback,
            completion=invalid_completion,
        ),
        adapter="instructor",
        aws_region_name="ap-southeast-2",
        max_tokens=1024,
    )
    metric = ContextPrecision(llm=_instrument_instructor_parsing(llm, debug_capture))

    async def run() -> None:
        with trace.root():
            await _within_usage_scope(
                "case-a",
                "context_precision",
                metric.ascore(
                    user_input="Reviewed question?",
                    reference="Reviewed reference.",
                    retrieved_contexts=["Approved public context."],
                ),
                trace,
            )

    with pytest.raises(InstructorRetryException):
        asyncio.run(run())

    assert completion_count >= 1
    assert debug_capture.persist() is not None
    artifact = load_judge_call_debug_artifact(debug_capture.artifact_path)
    assert len(artifact.calls) == completion_count
    assert [call.sequence_within_metric for call in artifact.calls] == list(
        range(1, completion_count + 1)
    )
    assert all(call.provider_span_id is not None for call in artifact.calls)
    provider_spans = [
        span for span in _RECORDED_SPANS if span.name == "bedrock_provider_call"
    ]
    metric_span = next(
        span
        for span in _RECORDED_SPANS
        if span.name == "ragas_metric:context_precision"
    )
    assert len(provider_spans) == completion_count
    assert all(span.parent_id == metric_span.span_id for span in provider_spans)
    assert {
        span.attributes["provider.request_id"] for span in provider_spans
    } == {f"invalid-{index}" for index in range(1, completion_count + 1)}
    assert {call.response.provider_request_id for call in artifact.calls} == {
        f"invalid-{index}" for index in range(1, completion_count + 1)
    }
    usage = callback.summary()
    assert usage.provider_call_count == completion_count
    assert usage.input_tokens == completion_count * 50
    assert usage.output_tokens == completion_count * 5
    assert usage.total_tokens == completion_count * 55
    assert all(
        invocation.estimated_cost_usd is not None
        for invocation in usage.invocations
    )
    assert any(
        call.instructor.validation_status == "FAILED" for call in artifact.calls
    )
    assert artifact.calls[-1].instructor.error_category in {
        "InstructorRetryException",
        "ValidationError",
    }


def test_completion_without_metric_scope_records_safe_correlation_failure() -> None:
    trace = _trace()
    callback = PrivacySafeLiteLLMUsageCallback(
        model_id="openai.gpt-oss-20b-1:0",
        region="ap-southeast-2",
        execution_trace=trace,
    )

    async def fake_completion(*_args: object, **_kwargs: object) -> dict[str, object]:
        return {
            "id": "unattributed-provider-call",
            "usage": {
                "prompt_tokens": 2,
                "completion_tokens": 1,
                "total_tokens": 3,
            },
        }

    completion = instrumented_litellm_async_client(
        callback,
        completion=fake_completion,
    )

    async def run() -> None:
        with trace.root():
            await completion(model="model", messages=[{"content": "private"}])

    asyncio.run(run())

    diagnostics = trace.reference.diagnostics
    assert diagnostics.litellm_completion_invocations_started == 1
    assert diagnostics.litellm_completion_invocations_completed == 1
    assert diagnostics.callback_correlation_failures == 1
    assert diagnostics.provider_spans_attempted == 1
    assert diagnostics.provider_spans_created == 0
    assert diagnostics.provider_spans_persisted == 0
    assert diagnostics.missing_span_reason_categories == (
        "ACTIVE_METRIC_PARENT_NOT_FOUND",
        "ACTIVE_METRIC_SCOPE_NOT_FOUND",
        "PROVIDER_SPAN_CORRELATION_FAILED",
    )


def test_telemetry_failure_does_not_invalidate_successful_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = _trace()
    callback = PrivacySafeLiteLLMUsageCallback(
        model_id="openai.gpt-oss-20b-1:0",
        region="ap-southeast-2",
        execution_trace=trace,
    )

    async def fake_completion(*_args: object, **_kwargs: object) -> str:
        return "successful provider result"

    def fail_telemetry(**_kwargs: object) -> None:
        raise RuntimeError("private telemetry failure detail")

    monkeypatch.setattr(callback, "record_completion_success", fail_telemetry)
    completion = instrumented_litellm_async_client(
        callback,
        completion=fake_completion,
    )

    async def run() -> str:
        with trace.root():
            return cast(
                str,
                await _within_usage_scope(
                    "case-a",
                    "faithfulness",
                    completion(model="model", messages=[{"content": "private"}]),
                    trace,
                ),
            )

    assert asyncio.run(run()) == "successful provider result"
    diagnostics = trace.reference.diagnostics
    assert diagnostics.litellm_completion_invocations_started == 1
    assert diagnostics.litellm_completion_invocations_completed == 1
    assert diagnostics.provider_spans_attempted == 1
    assert diagnostics.provider_spans_created == 1
    provider = next(
        span for span in _RECORDED_SPANS if span.name == "bedrock_provider_call"
    )
    assert provider.attributes["provider.call_status"] == "COMPLETED"
    assert provider.attributes["provider.telemetry_complete"] is False
    assert provider.attributes["provider.duration_semantics"] == (
        "awaited_litellm_completion"
    )
    assert "SUCCESSFUL_COMPLETION_TELEMETRY_ERROR" in (
        diagnostics.missing_span_reason_categories
    )
    assert "private telemetry failure detail" not in repr(diagnostics)


def test_completed_metrics_without_model_invocation_are_hierarchy_incomplete() -> None:
    _RECORDED_SPANS.clear()
    trace = RagasExecutionTrace(
        _RecordingTracer(),
        context=RagasExecutionTraceContext(
            configuration_id="no-provider-configuration",
            evaluation_run_id="run-no-provider",
            request_ids={"case-a": "request-a"},
            rag_trace_ids={"case-a": "tr-rag-a"},
        ),
        model_id="openai.gpt-oss-20b-1:0",
        region="ap-southeast-2",
        requested_metrics=("faithfulness",),
    )

    with trace.root(), trace.metric(case_id="case-a", metric_name="faithfulness"):
        pass

    assert trace.reference.status == "HIERARCHY_INCOMPLETE"
    assert trace.reference.diagnostics.litellm_completion_invocations_started == 0
    assert trace.reference.diagnostics.missing_span_reason_categories == (
        "NO_LITELLM_COMPLETION_OBSERVED",
    )


def test_disabled_tracing_reports_no_historical_execution_trace() -> None:
    trace = RagasExecutionTrace(
        NoOpTracer(),
        context=RagasExecutionTraceContext(
            configuration_id="configuration-disabled",
            evaluation_run_id=None,
            request_ids={},
            rag_trace_ids={},
        ),
        model_id="openai.gpt-oss-20b-1:0",
        region="ap-southeast-2",
        requested_metrics=("faithfulness",),
    )
    with trace.root():
        pass

    assert trace.reference.trace_id is None
    assert trace.reference.status == "UNAVAILABLE"
    assert (
        trace.reference.reason
        == "Evaluation execution trace not captured: MLflow tracing is disabled."
    )


def test_multiple_evaluations_of_same_request_get_distinct_bounded_client_ids() -> None:
    first = RagasExecutionTraceContext(
        configuration_id="b" * 64,
        evaluation_run_id="run-shared",
        request_ids={"case-a": "request-shared"},
        rag_trace_ids={"case-a": "tr-rag-shared"},
    )
    second = RagasExecutionTraceContext(
        configuration_id="b" * 64,
        evaluation_run_id="run-shared",
        request_ids={"case-a": "request-shared"},
        rag_trace_ids={"case-a": "tr-rag-shared"},
    )

    assert first.client_request_id != second.client_request_id
    assert len(first.client_request_id) == 36
    assert len(second.client_request_id) == 36
    assert first.configuration_id == second.configuration_id == "b" * 64


def test_async_export_failure_does_not_remove_local_trace_correlation() -> None:
    class _FailedExporterTracer(_RecordingTracer):
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
                status="FAILED",
                reason="Trace export verification failed (StringDataRightTruncation).",
            )

    execution = RagasExecutionTrace(
        _FailedExporterTracer(),
        context=RagasExecutionTraceContext(
            configuration_id="c" * 64,
            evaluation_run_id="run-completed",
            request_ids={"case-a": "request-completed"},
            rag_trace_ids={"case-a": "tr-rag-completed"},
        ),
        model_id="openai.gpt-oss-20b-1:0",
        region="ap-southeast-2",
        requested_metrics=("faithfulness",),
    )
    with execution.root():
        pass

    assert execution.reference.trace_id == "tr-ragas-execution"
    assert execution.reference.status == "EXPORT_FAILED"
    assert execution.reference.evaluation_run_id == "run-completed"
    assert execution.reference.configuration_id == "c" * 64
