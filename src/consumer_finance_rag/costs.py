"""Privacy-safe usage accounting and exact-match Bedrock cost estimates."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class PricingRule(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: str
    model_id: str
    region: str
    pricing_mode: str
    input_usd_per_million_tokens: Decimal = Field(ge=0)
    output_usd_per_million_tokens: Decimal = Field(ge=0)
    currency: str
    effective_date: date
    source: str


class PricingCatalog(BaseModel):
    model_config = ConfigDict(frozen=True)

    version: str
    rules: tuple[PricingRule, ...]


class CostEstimate(BaseModel):
    model_config = ConfigDict(frozen=True)

    available: bool
    estimated_api_cost: Decimal | None = None
    currency: str | None = None
    rule: PricingRule | None = None
    reason: str | None = None


class ModelInvocationUsage(BaseModel):
    """Content-free accounting for one observable provider invocation."""

    model_config = ConfigDict(frozen=True)

    backend: Literal["ragas", "mlflow_genai"]
    metric_name: str
    case_id: str | None = None
    status: Literal["COMPLETED", "FAILED"]
    model_id: str
    region: str
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    reasoning_tokens: int | None = Field(default=None, ge=0)
    retry_count: int | None = Field(default=None, ge=0)
    latency_ms: float | None = Field(default=None, ge=0)
    error_category: str | None = None
    provider_request_id: str | None = None
    finish_reason: str | None = Field(default=None, max_length=64)
    effective_output_token_limit: int | None = Field(default=None, gt=0)
    response_type: str | None = Field(default=None, max_length=128)
    completion_returned_normally: bool | None = None
    estimated_cost_usd: Decimal | None = Field(default=None, ge=0)
    estimated_cost_reason: str | None = None
    pricing_version: str | None = None


class ModelUsageSummary(BaseModel):
    """Aggregate that remains honest when a provider omits usage."""

    model_config = ConfigDict(frozen=True)

    model_id: str
    region: str
    provider_call_count: int = Field(ge=0)
    successful_call_count: int = Field(ge=0)
    failed_attempt_count: int = Field(ge=0)
    observable_retry_count: int | None = Field(default=None, ge=0)
    usage_complete: bool
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    reasoning_tokens: int | None = Field(default=None, ge=0)
    reason: str | None = None
    invocations: tuple[ModelInvocationUsage, ...] = ()


class CombinedCostEstimate(BaseModel):
    """Known subtotal plus an explicit completeness flag."""

    model_config = ConfigDict(frozen=True)

    known_estimated_cost: Decimal
    currency: Literal["USD"] = "USD"
    complete: bool
    unavailable_components: tuple[str, ...] = ()


def load_pricing_catalog(path: str | Path = "config/bedrock_pricing.yaml") -> PricingCatalog:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return PricingCatalog.model_validate(payload)


def estimate_generation_cost(
    usage: dict[str, int] | None,
    catalog: PricingCatalog,
    *,
    provider: str,
    model_id: str,
    region: str,
    pricing_mode: str,
    as_of: date,
) -> CostEstimate:
    if usage is None:
        return CostEstimate(available=False, reason="Provider token usage unavailable")
    matches = [
        rule
        for rule in catalog.rules
        if rule.provider == provider
        and rule.model_id == model_id
        and rule.region == region
        and rule.pricing_mode == pricing_mode
        and rule.effective_date <= as_of
    ]
    if not matches:
        return CostEstimate(available=False, reason="No exact verified pricing rule")
    rule = max(matches, key=lambda item: item.effective_date)
    million = Decimal(1_000_000)
    amount = (
        Decimal(usage["input_tokens"]) * rule.input_usd_per_million_tokens
        + Decimal(usage["output_tokens"]) * rule.output_usd_per_million_tokens
    ) / million
    return CostEstimate(
        available=True,
        estimated_api_cost=amount,
        currency=rule.currency,
        rule=rule,
    )


def estimate_model_usage_cost(
    usage: ModelUsageSummary | None,
    catalog: PricingCatalog,
    *,
    pricing_mode: str,
    as_of: date,
) -> CostEstimate:
    """Price a complete provider usage summary without guessing missing tokens."""

    if usage is None:
        return CostEstimate(available=False, reason="Provider usage telemetry unavailable")
    if not usage.usage_complete:
        return CostEstimate(
            available=False,
            reason=usage.reason or "Provider token usage is incomplete",
        )
    if usage.input_tokens is None or usage.output_tokens is None:
        return CostEstimate(available=False, reason="Provider token usage is incomplete")
    return estimate_generation_cost(
        {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "total_tokens": usage.total_tokens
            if usage.total_tokens is not None
            else usage.input_tokens + usage.output_tokens,
        },
        catalog,
        provider="aws-bedrock",
        model_id=usage.model_id,
        region=usage.region,
        pricing_mode=pricing_mode,
        as_of=as_of,
    )


def combine_cost_estimates(
    components: Mapping[str, CostEstimate | None],
) -> CombinedCostEstimate:
    """Sum known estimates and never turn an unknown billable component into zero."""

    known = Decimal(0)
    unavailable: list[str] = []
    for name, estimate in components.items():
        if (
            estimate is None
            or not estimate.available
            or estimate.estimated_api_cost is None
            or estimate.currency != "USD"
        ):
            unavailable.append(name)
        else:
            known += estimate.estimated_api_cost
    return CombinedCostEstimate(
        known_estimated_cost=known,
        complete=not unavailable,
        unavailable_components=tuple(unavailable),
    )


def format_estimated_usd(estimate: CostEstimate) -> str:
    """Keep enough precision for sub-cent inference while labeling the estimate."""

    if not estimate.available or estimate.estimated_api_cost is None:
        return "Cost unavailable"
    return f"Estimated USD ${estimate.estimated_api_cost:.8f}"
