from __future__ import annotations

from decimal import Decimal

from consumer_finance_rag.costs import CostEstimate
from consumer_finance_rag.usage_presentation import (
    MeasurementKind,
    format_usd,
    render_usage_cost,
    summarize_usage_cost,
    usage_component,
)


class _RecordingStreamlit:
    def __init__(self) -> None:
        self.metric_values: list[tuple[str, str]] = []

    def dataframe(self, *args: object, **kwargs: object) -> None:
        del args, kwargs

    def metric(self, label: str, value: str) -> None:
        self.metric_values.append((label, value))

    def warning(self, value: str) -> None:
        del value

    def caption(self, value: str) -> None:
        del value


def test_small_cost_includes_precise_usd_and_human_cents() -> None:
    assert format_usd(Decimal("0.002202964")) == (
        "Estimated USD $0.00220296 (0.2203¢)"
    )
    assert format_usd(None, measurement=MeasurementKind.UNAVAILABLE) == "Cost unavailable"


def test_reasoning_tokens_are_a_subset_not_added_to_output_or_cost() -> None:
    component = usage_component(
        name="RAGAS judge",
        status="COMPLETED",
        model_id="judge",
        region="ap-southeast-2",
        provider_invocations=10,
        input_tokens=14_320,
        output_tokens=3_788,
        reasoning_tokens=2_996,
        estimate=CostEstimate(
            available=True,
            estimated_api_cost=Decimal("0.002202964"),
            currency="USD",
        ),
        pricing_version="2026-09-26",
    )

    view = summarize_usage_cost([component], same_request_scope=True)

    assert view.input_tokens == 14_320
    assert view.output_tokens == 3_788
    assert view.reasoning_tokens == 2_996
    assert view.provider_invocations == 10
    assert view.known_estimated_cost_usd == Decimal("0.002202964")


def test_unknown_component_makes_total_partial_not_zero() -> None:
    known = usage_component(
        name="generation",
        status="COMPLETED",
        model_id="generation",
        region="us-east-1",
        provider_invocations=1,
        input_tokens=10,
        output_tokens=2,
        reasoning_tokens=None,
        estimate=CostEstimate(
            available=True,
            estimated_api_cost=Decimal("0.00001"),
            currency="USD",
        ),
        pricing_version="v1",
    )
    unknown = usage_component(
        name="judge",
        status="UNAVAILABLE",
        model_id="judge",
        region="us-east-1",
        provider_invocations=None,
        input_tokens=None,
        output_tokens=None,
        reasoning_tokens=None,
        estimate=CostEstimate(available=False, reason="usage unavailable"),
        pricing_version="v1",
    )

    view = summarize_usage_cost([known, unknown], same_request_scope=True)

    assert view.known_estimated_cost_usd == Decimal("0.00001")
    assert view.total_complete is False
    assert view.unavailable_components == ("judge",)


def test_all_unknown_components_render_total_as_unavailable() -> None:
    unknown = usage_component(
        name="generation",
        status="USAGE_UNAVAILABLE",
        model_id="model",
        region="region",
        provider_invocations=None,
        input_tokens=None,
        output_tokens=None,
        reasoning_tokens=None,
        estimate=CostEstimate(available=False, reason="usage unavailable"),
        pricing_version="v1",
    )
    view = summarize_usage_cost([unknown], same_request_scope=True)
    streamlit = _RecordingStreamlit()

    render_usage_cost(view, streamlit=streamlit)

    assert streamlit.metric_values == [("Known estimated subtotal", "Cost unavailable")]
