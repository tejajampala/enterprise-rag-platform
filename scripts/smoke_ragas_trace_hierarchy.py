# mypy: disable-error-code=import-untyped
"""Persist the real RAGAS/Instructor completion path without a Bedrock call."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any
from unittest.mock import patch

import mlflow
from litellm import ModelResponse

import consumer_finance_rag.evaluation.ragas_adapter as ragas_adapter
from consumer_finance_rag.config import load_settings
from consumer_finance_rag.evaluation.models import RAGCaseResult
from consumer_finance_rag.evaluation.ragas_adapter import (
    PrivacySafeLiteLLMUsageCallback,
    Ragas043Evaluator,
    _install_vertexai_import_compatibility,
)
from consumer_finance_rag.evaluation.ragas_execution_trace import (
    RagasExecutionTraceContext,
    RagasExecutionTraceReference,
)
from consumer_finance_rag.observability import MLflowTracer

_SENSITIVE_CANARY = "offline-sensitive-content-must-not-persist"
_COMPLETION_CATEGORIES: list[str] = []
_COMPLETION_ACTIVE_SPANS: list[str | None] = []


def _response_for_prompt(prompt: str) -> tuple[str, dict[str, object]]:
    lowered = prompt.casefold()
    if "break down each sentence" in lowered:
        return "faithfulness", {"statements": ["The reviewed answer is supported."]}
    if "judge the faithfulness" in lowered:
        return "faithfulness", {
            "statements": [
                {
                    "statement": "The reviewed answer is supported.",
                    "reason": "supported",
                    "verdict": 1,
                }
            ]
        }
    if "context was useful" in lowered:
        return "context_precision", {"reason": "useful", "verdict": 1}
    if "classify if the statement can be attributed" in lowered:
        return "context_recall", {
            "classifications": [
                {
                    "statement": "Reviewed reference.",
                    "reason": "attributed",
                    "attributed": 1,
                }
            ]
        }
    if "groundedness" in lowered:
        return "response_groundedness", {"rating": 2}
    raise RuntimeError("Unexpected installed RAGAS prompt contract")


async def _fake_bedrock_completion(*_args: object, **kwargs: object) -> ModelResponse:
    """Replace only the external network boundary with a provider-shaped response."""

    messages = kwargs.get("messages")
    if not isinstance(messages, list) or not messages:
        raise RuntimeError("Instructor did not supply completion messages")
    last_message = messages[-1]
    if not isinstance(last_message, dict):
        raise RuntimeError("Instructor supplied an unsupported message shape")
    category, payload = _response_for_prompt(str(last_message.get("content", "")))
    _COMPLETION_CATEGORIES.append(category)
    active = mlflow.get_current_active_span()
    _COMPLETION_ACTIVE_SPANS.append(active.name if active is not None else None)
    return ModelResponse(
        id=f"offline-provider-{len(_COMPLETION_CATEGORIES)}",
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


def _run_real_adapter() -> RagasExecutionTraceReference:
    _install_vertexai_import_compatibility()
    _COMPLETION_CATEGORIES.clear()
    _COMPLETION_ACTIVE_SPANS.clear()
    settings = load_settings()
    evaluator = Ragas043Evaluator(
        judge_model="bedrock:/openai.gpt-oss-20b-1:0",
        judge_region="ap-southeast-2",
        references={"offline-public-case": f"Reviewed reference {_SENSITIVE_CANARY}."},
        execution_context=RagasExecutionTraceContext(
            configuration_id="f" * 64,
            evaluation_run_id="offline-trace-hierarchy-smoke",
            request_ids={"offline-public-case": "offline-request-id"},
            rag_trace_ids={"offline-public-case": "tr-offline-origin"},
        ),
        execution_tracer=MLflowTracer(settings.mlflow),
        judge_max_tokens=settings.mlflow.ragas_judge_max_tokens,
        context_precision_max_tokens=(
            settings.mlflow.ragas_context_precision_max_tokens
        ),
    )
    original_instrumentation = ragas_adapter.instrumented_litellm_async_client

    def offline_instrumentation(
        callback: PrivacySafeLiteLLMUsageCallback,
    ) -> Any:
        return original_instrumentation(
            callback,
            completion=_fake_bedrock_completion,
        )

    with patch.object(
        ragas_adapter,
        "instrumented_litellm_async_client",
        offline_instrumentation,
    ):
        results = evaluator.evaluate(
            [
                RAGCaseResult(
                    case_id="offline-public-case",
                    question=f"Reviewed question {_SENSITIVE_CANARY}?",
                    answer=f"Reviewed answer {_SENSITIVE_CANARY}.",
                    contexts=[
                        f"Approved context A {_SENSITIVE_CANARY}.",
                        f"Approved context B {_SENSITIVE_CANARY}.",
                    ],
                    chunk_ids=["offline-chunk-a", "offline-chunk-b"],
                )
            ]
        )
    if set(results["offline-public-case"]) != {
        "faithfulness",
        "context_precision",
        "context_recall",
        "response_groundedness",
    }:
        raise RuntimeError("The real RAGAS adapter did not return every requested metric")
    return evaluator.execution_trace


def main(argv: Sequence[str] | None = None) -> int:
    if argv:
        raise SystemExit("This smoke test accepts no arguments")
    reference = _run_real_adapter()
    if reference.trace_id is None:
        print("Hierarchy: UNAVAILABLE (no local trace ID)")
        return 3

    persisted = mlflow.get_trace(reference.trace_id, flush=True)
    if persisted is None:
        print(f"Trace ID: {reference.trace_id}")
        print("Hierarchy: PENDING (exact trace is not retrievable)")
        return 2
    spans = tuple(persisted.data.spans)
    roots = [span for span in spans if span.name == "ragas_evaluation"]
    metrics = [span for span in spans if span.name.startswith("ragas_metric:")]
    providers = [span for span in spans if span.name == "bedrock_provider_call"]
    if len(roots) != 1 or len(metrics) != 4:
        raise RuntimeError("Persisted trace is missing its RAGAS root or metric hierarchy")
    root = roots[0]
    if any(span.parent_id != root.span_id for span in metrics):
        raise RuntimeError("A persisted metric span is not parented by the RAGAS root")
    metric_by_id = {
        span.span_id: span.name.removeprefix("ragas_metric:") for span in metrics
    }
    if any(span.parent_id not in metric_by_id for span in providers):
        raise RuntimeError("A persisted provider span is not parented by a metric span")
    provider_metrics = [metric_by_id[str(span.parent_id)] for span in providers]
    if sorted(provider_metrics) != sorted(_COMPLETION_CATEGORIES):
        raise RuntimeError("Provider spans were associated with the wrong RAGAS metric")
    if any(span.trace_id != reference.trace_id for span in spans):
        raise RuntimeError("Persisted spans do not share the exact evaluation trace ID")
    if _SENSITIVE_CANARY in persisted.to_json():
        raise RuntimeError("Sensitive evaluated content was persisted in the trace")
    if ["ragas_evaluation"] * len(_COMPLETION_CATEGORIES) != _COMPLETION_ACTIVE_SPANS:
        raise RuntimeError("The completion boundary lost the active root trace context")
    if reference.status != "CONFIRMED":
        raise RuntimeError(f"Trace persistence was not confirmed ({reference.status})")

    diagnostics = reference.diagnostics
    print(f"Trace ID: {reference.trace_id}")
    print(f"Evaluation client request ID: {reference.client_request_id}")
    print(f"Persistence: {reference.status}")
    print(f"Persisted spans: {len(spans)}")
    print(f"Completion calls: {diagnostics.litellm_completion_invocations_started}")
    print(
        "Callbacks success/failure: "
        f"{diagnostics.success_callbacks_received}/{diagnostics.failure_callbacks_received}"
    )
    print(
        "Provider spans attempted/created/persisted: "
        f"{diagnostics.provider_spans_attempted}/"
        f"{diagnostics.provider_spans_created}/"
        f"{diagnostics.provider_spans_persisted}"
    )
    for span in spans:
        print(f"{span.name} | span_id={span.span_id} | parent_id={span.parent_id}")
    print("Hierarchy: CONFIRMED by exact mlflow.get_trace() retrieval")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
