"""Reusable, honest usage and estimated-cost presentation."""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from consumer_finance_rag.costs import CostEstimate
from consumer_finance_rag.ui_dataframes import (
    arrow_safe_dataframe,
    format_optional_count,
    render_arrow_dataframe,
)


class MeasurementKind(StrEnum):
    MEASURED = "MEASURED"
    ESTIMATED = "ESTIMATED"
    UNAVAILABLE = "UNAVAILABLE"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class UsageCostComponent(BaseModel):
    """One billable model component, counted at actual provider-call granularity."""

    model_config = ConfigDict(frozen=True)

    name: str
    status: str
    model_id: str
    region: str
    provider_invocations: int | None = Field(default=None, ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    reasoning_tokens: int | None = Field(default=None, ge=0)
    token_measurement: MeasurementKind
    estimated_cost_usd: Decimal | None = Field(default=None, ge=0)
    cost_measurement: MeasurementKind
    pricing_version: str | None = None
    pricing_source: str | None = None
    reason: str | None = None
    billable: bool = True


class UsageCostPresentation(BaseModel):
    model_config = ConfigDict(frozen=True)

    components: tuple[UsageCostComponent, ...]
    known_estimated_cost_usd: Decimal
    total_complete: bool
    unavailable_components: tuple[str, ...]
    provider_invocations: int | None = Field(default=None, ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    reasoning_tokens: int | None = Field(default=None, ge=0)
    same_request_scope: bool


def format_usd(
    amount: Decimal | None,
    *,
    measurement: MeasurementKind = MeasurementKind.ESTIMATED,
    include_cents: bool = True,
) -> str:
    """Format sub-cent amounts without substituting zero for unavailable cost."""

    if amount is None or measurement is MeasurementKind.UNAVAILABLE:
        return "Cost unavailable"
    if measurement is MeasurementKind.NOT_APPLICABLE:
        return "Not applicable"
    prefix = "Estimated " if measurement is MeasurementKind.ESTIMATED else "Measured "
    cents = f" ({amount * Decimal(100):.4f}¢)" if include_cents else ""
    return f"{prefix}USD ${amount:.8f}{cents}"


def usage_component(
    *,
    name: str,
    status: str,
    model_id: str,
    region: str,
    provider_invocations: int | None,
    input_tokens: int | None,
    output_tokens: int | None,
    reasoning_tokens: int | None,
    estimate: CostEstimate | None,
    pricing_version: str | None,
    billable: bool = True,
    reason: str | None = None,
) -> UsageCostComponent:
    usage_known = input_tokens is not None and output_tokens is not None
    cost_known = bool(
        estimate is not None
        and estimate.available
        and estimate.estimated_api_cost is not None
    )
    rule = estimate.rule if estimate is not None else None
    return UsageCostComponent(
        name=name,
        status=status,
        model_id=model_id,
        region=region,
        provider_invocations=provider_invocations,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        reasoning_tokens=reasoning_tokens,
        token_measurement=(
            MeasurementKind.MEASURED if usage_known else MeasurementKind.UNAVAILABLE
        ),
        estimated_cost_usd=(
            estimate.estimated_api_cost if cost_known and estimate is not None else None
        ),
        cost_measurement=(
            MeasurementKind.ESTIMATED
            if cost_known
            else MeasurementKind.UNAVAILABLE
        ),
        pricing_version=pricing_version,
        pricing_source=rule.source if rule is not None else None,
        reason=(
            reason
            or (estimate.reason if estimate is not None else None)
            or "Exact-match versioned pricing estimate."
        ),
        billable=billable,
    )


def summarize_usage_cost(
    components: Sequence[UsageCostComponent],
    *,
    same_request_scope: bool,
) -> UsageCostPresentation:
    """Aggregate only components that genuinely share a request/evaluation scope.

    Reasoning tokens are reported as an output-token subset and are never added to total
    tokens or cost. Parent metric aggregates are not represented as provider invocations.
    """

    billable = tuple(item for item in components if item.billable)
    known_cost = sum(
        (item.estimated_cost_usd or Decimal(0) for item in billable),
        Decimal(0),
    )
    unavailable = tuple(
        item.name for item in billable if item.estimated_cost_usd is None
    )
    if not same_request_scope:
        return UsageCostPresentation(
            components=tuple(components),
            known_estimated_cost_usd=known_cost,
            total_complete=False,
            unavailable_components=("components do not share one accounting scope",),
            same_request_scope=False,
        )
    return UsageCostPresentation(
        components=tuple(components),
        known_estimated_cost_usd=known_cost,
        total_complete=not unavailable,
        unavailable_components=unavailable,
        provider_invocations=_sum_if_complete(
            [item.provider_invocations for item in billable]
        ),
        input_tokens=_sum_if_complete([item.input_tokens for item in billable]),
        output_tokens=_sum_if_complete([item.output_tokens for item in billable]),
        reasoning_tokens=_sum_if_complete([item.reasoning_tokens for item in billable]),
        same_request_scope=True,
    )


def presentation_rows(view: UsageCostPresentation) -> list[dict[str, str]]:
    return [
        {
            "Component": item.name,
            "Status": item.status,
            "Model": item.model_id,
            "Region": item.region,
            "Provider invocations": format_optional_count(item.provider_invocations),
            "Input tokens": format_optional_count(item.input_tokens),
            "Output tokens": format_optional_count(item.output_tokens),
            "Reasoning tokens (output subset)": format_optional_count(
                item.reasoning_tokens
            ),
            "Cost": format_usd(
                item.estimated_cost_usd,
                measurement=item.cost_measurement,
            ),
            "Pricing version": item.pricing_version or "Unavailable",
            "Pricing source": item.pricing_source or "Unavailable",
            "Measurement": item.token_measurement.value,
            "Reason": item.reason or "",
        }
        for item in view.components
    ]


def usage_cost_dataframe(view: UsageCostPresentation) -> Any:
    """Return the Arrow-validated display table used by both Streamlit surfaces."""

    return arrow_safe_dataframe(presentation_rows(view))


def render_usage_cost(view: UsageCostPresentation, *, streamlit: Any) -> None:
    """Render the shared component without importing Streamlit into core services."""

    render_arrow_dataframe(streamlit, presentation_rows(view))
    label = "Complete estimated total" if view.total_complete else "Known estimated subtotal"
    has_known_cost = any(
        item.billable and item.estimated_cost_usd is not None for item in view.components
    )
    total_value = (
        format_usd(view.known_estimated_cost_usd)
        if view.total_complete or has_known_cost
        else "Cost unavailable"
    )
    streamlit.metric(label, total_value)
    if view.unavailable_components:
        streamlit.warning(
            "A complete total is unavailable for: "
            + ", ".join(view.unavailable_components)
            + "."
        )
    streamlit.caption(
        "Reasoning tokens are an output-token subset. Provider invocations—not metric "
        "aggregates or Instructor parsing retries—are the accounting unit."
    )


def _sum_if_complete(values: Sequence[int | None]) -> int | None:
    return sum(item for item in values if item is not None) if all(
        item is not None for item in values
    ) else None
