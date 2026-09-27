from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

from consumer_finance_rag.costs import (
    CostEstimate,
    ModelUsageSummary,
    combine_cost_estimates,
    estimate_generation_cost,
    estimate_model_usage_cost,
    format_estimated_usd,
    load_pricing_catalog,
)
from consumer_finance_rag.evaluation.managed_rag import ManagedRAGEvaluationReport
from consumer_finance_rag.evaluation.ragas_adapter import (
    PrivacySafeLiteLLMUsageCallback,
    _within_usage_scope,
)


def test_exact_generation_and_judge_pricing_is_model_region_and_mode_specific() -> None:
    catalog = load_pricing_catalog()
    usage = {"input_tokens": 1_000, "output_tokens": 100, "total_tokens": 1_100}

    sydney = estimate_generation_cost(
        usage,
        catalog,
        provider="aws-bedrock",
        model_id="google.gemma-3-4b-it",
        region="ap-southeast-2",
        pricing_mode="on-demand-standard",
        as_of=date(2026, 9, 25),
    )
    virginia = estimate_generation_cost(
        usage,
        catalog,
        provider="aws-bedrock",
        model_id="meta.llama3-70b-instruct-v1:0",
        region="us-east-1",
        pricing_mode="on-demand-standard",
        as_of=date(2026, 9, 25),
    )
    wrong_region = estimate_generation_cost(
        usage,
        catalog,
        provider="aws-bedrock",
        model_id="meta.llama3-70b-instruct-v1:0",
        region="ap-southeast-2",
        pricing_mode="on-demand-standard",
        as_of=date(2026, 9, 25),
    )
    gpt_oss = estimate_generation_cost(
        usage,
        catalog,
        provider="aws-bedrock",
        model_id="openai.gpt-oss-20b-1:0",
        region="ap-southeast-2",
        pricing_mode="on-demand-standard",
        as_of=date(2026, 9, 26),
    )

    assert sydney.estimated_api_cost == Decimal("0.00004944")
    assert virginia.estimated_api_cost == Decimal("0.00300")
    assert gpt_oss.estimated_api_cost == Decimal("0.00010300")
    assert wrong_region.available is False
    assert wrong_region.reason == "No exact verified pricing rule"


def test_small_cost_format_and_partial_total_never_substitute_zero() -> None:
    tiny = CostEstimate(
        available=True,
        estimated_api_cost=Decimal("0.000049"),
        currency="USD",
    )
    unknown = CostEstimate(available=False, reason="usage missing")

    assert format_estimated_usd(tiny) == "Estimated USD $0.00004900"
    partial = combine_cost_estimates({"generation": tiny, "judge": unknown})
    complete = combine_cost_estimates({"generation": tiny})
    assert partial.known_estimated_cost == Decimal("0.000049")
    assert partial.complete is False
    assert partial.unavailable_components == ("judge",)
    assert complete.complete is True


def test_ragas_callback_captures_multiple_calls_retries_and_partial_usage_safely() -> None:
    callback = PrivacySafeLiteLLMUsageCallback(
        model_id="meta.llama3-70b-instruct-v1:0",
        region="us-east-1",
    )
    started = datetime(2026, 9, 25, tzinfo=UTC)
    ended = started + timedelta(milliseconds=12)

    async def events() -> None:
        async def two_successes() -> None:
            for retry in (0, 1):
                await callback.async_log_success_event(
                    {
                        "retry_count": retry,
                        "messages": [{"content": "secret prompt"}],
                    },
                    SimpleNamespace(
                        usage=SimpleNamespace(
                            prompt_tokens=10,
                            completion_tokens=2,
                            total_tokens=12,
                            completion_tokens_details=SimpleNamespace(
                                reasoning_tokens=1
                            ),
                        )
                    ),
                    started,
                    ended,
                )

        async def failed_without_usage() -> None:
            await callback.async_log_failure_event(
                {"exception": RuntimeError("private answer text")},
                None,
                started,
                ended,
            )

        await _within_usage_scope("case-1", "faithfulness", two_successes())
        await _within_usage_scope("case-1", "context_recall", failed_without_usage())

    asyncio.run(events())
    summary = callback.summary()
    serialized = summary.model_dump_json()

    assert summary.provider_call_count == 3
    assert summary.successful_call_count == 2
    assert summary.failed_attempt_count == 1
    assert summary.observable_retry_count == 1
    assert summary.usage_complete is False
    assert summary.total_tokens is None
    assert summary.reasoning_tokens is None
    assert [item.metric_name for item in summary.invocations] == [
        "faithfulness",
        "faithfulness",
        "context_recall",
    ]
    assert "secret prompt" not in serialized
    assert "private answer text" not in serialized
    assert "RuntimeError" in serialized


def test_complete_judge_usage_can_be_priced_without_metric_count_inference() -> None:
    usage = ModelUsageSummary(
        model_id="meta.llama3-70b-instruct-v1:0",
        region="us-east-1",
        provider_call_count=2,
        successful_call_count=2,
        failed_attempt_count=0,
        observable_retry_count=None,
        usage_complete=True,
        input_tokens=10,
        output_tokens=5,
        total_tokens=15,
        reasoning_tokens=3,
    )
    estimate = estimate_model_usage_cost(
        usage,
        load_pricing_catalog(),
        pricing_mode="on-demand-standard",
        as_of=date(2026, 9, 25),
    )

    assert estimate.estimated_api_cost == Decimal("0.000044")
    assert usage.provider_call_count == 2


def test_historical_paid_smoke_without_usage_remains_cost_unavailable() -> None:
    report = ManagedRAGEvaluationReport.model_validate_json(
        Path("reports/evaluation/paid-smoke.json").read_text(encoding="utf-8")
    )

    assert report.format_version == "managed-rag-evaluation-report-v1"
    assert report.models.judge_model == "bedrock:/meta.llama3-70b-instruct-v1:0"
    assert report.answer_quality.ragas.aggregate_metrics
    assert report.answer_quality.ragas.usage is None
    estimate = estimate_model_usage_cost(
        report.answer_quality.ragas.usage,
        load_pricing_catalog(),
        pricing_mode="on-demand-standard",
        as_of=report.generated_at.date(),
    )
    assert format_estimated_usd(estimate) == "Cost unavailable"
