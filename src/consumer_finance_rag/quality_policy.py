"""Typed quality policy for runtime gating, monitoring, and offline evaluation.

The three workflows in this module are deliberately separate. A runtime decision can
block delivery, an online observation can only describe an already completed request,
and an offline evaluation operates on a reviewed dataset rather than arbitrary traffic.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from decimal import Decimal
from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from consumer_finance_rag.costs import ModelUsageSummary
from consumer_finance_rag.models import RetrievedChunk


class QualityProfile(StrEnum):
    STANDARD = "STANDARD"
    STRICT = "STRICT"
    EVALUATION = "EVALUATION"


class QualityStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    ABSTAIN = "ABSTAIN"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    NOT_RUN = "NOT_RUN"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    UNAVAILABLE = "UNAVAILABLE"
    ERROR = "ERROR"


class JudgeFailureBehavior(StrEnum):
    ABSTAIN = "ABSTAIN"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    FAIL_CLOSED = "FAIL_CLOSED"


class DeliveryOutcome(StrEnum):
    ANSWERED = "ANSWERED"
    ABSTAIN = "ABSTAIN"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    BLOCKED = "BLOCKED"
    ERROR = "ERROR"


class PolicyProfileSettings(BaseModel):
    """Versioned controls whose thresholds require reviewed-case validation."""

    model_config = ConfigDict(frozen=True)

    version: str = Field(min_length=1)
    require_evidence: bool = True
    require_relevant_evidence: bool = True
    relevance_policy_version: str = "cross-encoder-candidate-v1"
    minimum_relevance_score: float = 0.0
    require_valid_citations: bool = True
    reference_free_judge_enabled: bool = False
    judge_model_id: str | None = None
    judge_region: str | None = None
    groundedness_threshold: float = Field(default=0.8, ge=0, le=1)
    judge_timeout_seconds: float = Field(default=8.0, gt=0)
    judge_max_input_tokens: int = Field(default=8192, ge=256, le=131072)
    judge_max_output_tokens: int = Field(default=1024, ge=64, le=8192)
    judge_max_retries: int = Field(default=0, ge=0, le=2)
    judge_budget_usd: float = Field(default=0.01, ge=0)
    judge_failure_behavior: JudgeFailureBehavior = JudgeFailureBehavior.REVIEW_REQUIRED
    judge_timeout_behavior: JudgeFailureBehavior = JudgeFailureBehavior.ABSTAIN
    high_risk_financial_behavior: QualityStatus = QualityStatus.REVIEW_REQUIRED


class QualityPolicySettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    default_profile: QualityProfile = QualityProfile.STANDARD
    standard: PolicyProfileSettings = PolicyProfileSettings(version="standard-v1")
    strict: PolicyProfileSettings = PolicyProfileSettings(
        version="strict-v1",
        reference_free_judge_enabled=True,
    )
    evaluation: PolicyProfileSettings = PolicyProfileSettings(
        version="evaluation-v1",
        reference_free_judge_enabled=True,
    )

    def for_profile(self, profile: QualityProfile) -> PolicyProfileSettings:
        return {
            QualityProfile.STANDARD: self.standard,
            QualityProfile.STRICT: self.strict,
            QualityProfile.EVALUATION: self.evaluation,
        }[profile]


class RuntimeQualityContext(BaseModel):
    """Content-free facts used by deterministic pre-response controls."""

    model_config = ConfigDict(frozen=True)

    request_id: str
    user_authorized: bool
    corpus_authorized: bool
    evidence_available: bool
    evidence_relevant: bool | None = None
    evidence_relevance_score: float | None = None
    evidence_relevance_threshold: float | None = None
    citations_present: bool | None = None
    citations_valid: bool | None = None
    output_schema_valid: bool | None = None
    pii_policy_satisfied: bool | None = None
    safety_policy_satisfied: bool | None = None
    within_timeout: bool | None = None
    within_budget: bool | None = None
    high_risk_financial: bool = False


class QualityCheckResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    check: str
    status: QualityStatus
    reason: str
    reason_category: str | None = None
    score: float | None = None
    threshold: float | None = None


class GroundednessEvidence(BaseModel):
    """Exactly one prompt passage supplied to generation."""

    model_config = ConfigDict(frozen=True)

    citation_id: str
    chunk_id: str
    document_id: str
    text: str
    truncated: bool = False


class GroundednessJudgeInput(BaseModel):
    """Content passed only to the blocking judge, never to ordinary trace attributes."""

    model_config = ConfigDict(frozen=True)

    request_id: str
    question: str
    candidate_answer: str
    evidence: tuple[GroundednessEvidence, ...]


class RuntimeJudgeExecution(BaseModel):
    """Content-free per-request judge telemetry suitable for UI and MLflow."""

    model_config = ConfigDict(frozen=True)

    model_id: str
    region: str
    duration_ms: float | None = Field(default=None, ge=0)
    timed_out: bool = False
    usage: ModelUsageSummary | None = None
    estimated_cost_usd: Decimal | None = Field(default=None, ge=0)
    pricing_version: str | None = None
    provider_request_id: str | None = None


class ReferenceFreeJudgeResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    check: QualityCheckResult
    execution: RuntimeJudgeExecution


class QualityDecision(BaseModel):
    """A policy decision, intentionally distinct from any model quality score."""

    model_config = ConfigDict(frozen=True)

    request_id: str
    profile: QualityProfile
    policy_version: str
    status: QualityStatus
    deliver_answer: bool
    delivery_outcome: DeliveryOutcome
    checks: tuple[QualityCheckResult, ...]
    reason: str
    judge_execution: RuntimeJudgeExecution | None = None


class EvidenceAnswerabilityAssessment(BaseModel):
    """Versioned decision from the reranker scores of approved final evidence."""

    model_config = ConfigDict(frozen=True)

    check: QualityCheckResult
    policy_version: str


def assess_evidence_answerability(
    chunks: Sequence[RetrievedChunk],
    policy: PolicyProfileSettings,
) -> EvidenceAnswerabilityAssessment:
    """Require at least one final passage to clear the configured reranker boundary.

    This is deliberately a bounded candidate policy, not a universal relevance cutoff.
    A partially answerable request proceeds when at least one approved passage clears the
    boundary; post-generation citation and groundedness controls still apply.
    """

    scores = tuple(item.rerank_score for item in chunks if item.rerank_score is not None)
    if not chunks:
        check = QualityCheckResult(
            check="evidence_relevance",
            status=QualityStatus.ABSTAIN,
            reason="No approved evidence was available to answer the question.",
            reason_category="INSUFFICIENT_RELEVANT_EVIDENCE",
            threshold=policy.minimum_relevance_score,
        )
    elif not scores:
        check = QualityCheckResult(
            check="evidence_relevance",
            status=QualityStatus.UNAVAILABLE,
            reason=(
                "Evidence relevance could not be verified because reranker scores were unavailable."
            ),
            reason_category="EVIDENCE_RELEVANCE_UNAVAILABLE",
            threshold=policy.minimum_relevance_score,
        )
    else:
        best = max(scores)
        relevant = best >= policy.minimum_relevance_score
        check = QualityCheckResult(
            check="evidence_relevance",
            status=QualityStatus.PASS if relevant else QualityStatus.ABSTAIN,
            reason=(
                "At least one approved final passage met the configured reranker "
                "relevance boundary."
                if relevant
                else "No approved final passage met the configured reranker relevance boundary."
            ),
            reason_category=(
                "RELEVANT_EVIDENCE_AVAILABLE" if relevant else "INSUFFICIENT_RELEVANT_EVIDENCE"
            ),
            score=best,
            threshold=policy.minimum_relevance_score,
        )
    return EvidenceAnswerabilityAssessment(
        check=check,
        policy_version=policy.relevance_policy_version,
    )


class ReferenceFreeJudge(Protocol):
    def __call__(
        self,
        judge_input: GroundednessJudgeInput,
        *,
        threshold: float,
        timeout_seconds: float,
        budget_usd: float,
    ) -> ReferenceFreeJudgeResult: ...


_BLOCKING_STATUSES = {
    QualityStatus.FAIL,
    QualityStatus.ABSTAIN,
    QualityStatus.REVIEW_REQUIRED,
    QualityStatus.UNAVAILABLE,
    QualityStatus.ERROR,
}


def deterministic_checks(
    context: RuntimeQualityContext,
    policy: PolicyProfileSettings,
) -> tuple[QualityCheckResult, ...]:
    """Evaluate mandatory controls without invoking a model."""

    checks = [
        _measured_check(
            "user_authorization",
            context.user_authorized,
            "The request used the enforced public-user access context.",
            "User authorization failed.",
        ),
        _measured_check(
            "corpus_authorization",
            context.corpus_authorized,
            "Retrieved passages satisfied the configured corpus access policy.",
            "Corpus authorization failed.",
        ),
        _measured_check(
            "evidence_availability",
            context.evidence_available if policy.require_evidence else None,
            "At least one approved passage was supplied as generation evidence.",
            "Required retrieval evidence is unavailable.",
            not_applicable=not policy.require_evidence,
        ),
        _measured_check(
            "evidence_relevance",
            context.evidence_relevant if policy.require_relevant_evidence else None,
            "At least one approved passage met the configured reranker relevance boundary.",
            "No approved passage met the configured reranker relevance boundary.",
            not_applicable=not policy.require_relevant_evidence,
            false_status=QualityStatus.ABSTAIN,
            reason_category=(
                "INSUFFICIENT_RELEVANT_EVIDENCE" if context.evidence_relevant is False else None
            ),
            score=context.evidence_relevance_score,
            threshold=context.evidence_relevance_threshold,
        ),
        _measured_check(
            "citation_presence",
            context.citations_present if policy.require_valid_citations else None,
            "The generated answer contained at least one visible citation reference.",
            "The generated answer did not contain a required citation reference.",
            not_applicable=not policy.require_valid_citations,
        ),
        _measured_check(
            "citation_validity",
            context.citations_valid if policy.require_valid_citations else None,
            "Every visible citation reference resolved to supplied final evidence.",
            "A visible citation reference does not resolve to supplied final evidence.",
            not_applicable=not policy.require_valid_citations,
        ),
        _measured_check(
            "output_schema",
            context.output_schema_valid,
            "The generated response satisfied the required non-empty output contract.",
            "Output schema validation failed.",
        ),
        _measured_check(
            "pii_policy",
            context.pii_policy_satisfied,
            "The post-redaction answer scan found no configured PII types.",
            "PII policy was not satisfied.",
        ),
        _measured_check(
            "safety_policy",
            context.safety_policy_satisfied,
            "The configured safety policy executed and passed.",
            "Safety policy was not satisfied.",
        ),
        _measured_check(
            "timeout",
            context.within_timeout,
            "The measured full request completed within its configured timeout.",
            "The request exceeded its timeout policy.",
        ),
        _measured_check(
            "budget",
            context.within_budget,
            "The measured full request stayed within its configured total budget.",
            "The request exceeded its budget policy.",
        ),
    ]
    return tuple(checks)


def evaluate_pre_response(
    context: RuntimeQualityContext,
    settings: QualityPolicySettings,
    *,
    profile: QualityProfile | None = None,
    judge_input: GroundednessJudgeInput | None = None,
    judge: ReferenceFreeJudge | None = None,
) -> QualityDecision:
    """Return the blocking decision for one answer before it is delivered."""

    selected = profile or settings.default_profile
    policy = settings.for_profile(selected)
    checks = list(deterministic_checks(context, policy))
    failed = next(
        (item for item in checks if item.status in {QualityStatus.FAIL, QualityStatus.ABSTAIN}),
        None,
    )
    if failed is not None:
        return _decision(context, selected, policy, checks, failed.status, failed.reason)

    if context.high_risk_financial:
        status = policy.high_risk_financial_behavior
        if status is not QualityStatus.PASS:
            checks.append(
                QualityCheckResult(
                    check="high_risk_financial_route",
                    status=status,
                    reason="Configured controlled response or human review route is required.",
                )
            )
            return _decision(context, selected, policy, checks, status, checks[-1].reason)

    if selected is QualityProfile.EVALUATION:
        checks.append(
            QualityCheckResult(
                check="runtime_profile_route",
                status=QualityStatus.NOT_RUN,
                reason="EVALUATION is restricted to explicit reviewed-dataset workflows.",
                reason_category="OFFLINE_EVALUATION_ONLY",
            )
        )
        return _decision(
            context,
            selected,
            policy,
            checks,
            QualityStatus.REVIEW_REQUIRED,
            checks[-1].reason,
        )

    if selected is QualityProfile.STRICT:
        if not policy.reference_free_judge_enabled:
            status = _failure_status(policy.judge_failure_behavior)
            checks.append(
                QualityCheckResult(
                    check="reference_free_groundedness",
                    status=QualityStatus.UNAVAILABLE,
                    reason="STRICT requires a synchronous groundedness judge.",
                    reason_category="JUDGE_DISABLED",
                    threshold=policy.groundedness_threshold,
                )
            )
            return _decision(context, selected, policy, checks, status, checks[-1].reason)
        if judge is None or judge_input is None:
            status = _failure_status(policy.judge_failure_behavior)
            checks.append(
                QualityCheckResult(
                    check="reference_free_groundedness",
                    status=QualityStatus.UNAVAILABLE,
                    reason="The configured synchronous judge is unavailable.",
                    reason_category="JUDGE_UNAVAILABLE",
                )
            )
            return _decision(context, selected, policy, checks, status, checks[-1].reason)
        try:
            judged = judge(
                judge_input,
                threshold=policy.groundedness_threshold,
                timeout_seconds=policy.judge_timeout_seconds,
                budget_usd=policy.judge_budget_usd,
            )
        except Exception:
            status = _failure_status(policy.judge_failure_behavior)
            checks.append(
                QualityCheckResult(
                    check="reference_free_groundedness",
                    status=QualityStatus.ERROR,
                    reason="The synchronous judge failed safely.",
                    reason_category="JUDGE_ERROR",
                )
            )
            return _decision(context, selected, policy, checks, status, checks[-1].reason)
        judge_result = judged.check
        checks.append(judge_result)
        if judge_result.status is not QualityStatus.PASS:
            if judge_result.status in {QualityStatus.UNAVAILABLE, QualityStatus.ERROR}:
                behavior = (
                    policy.judge_timeout_behavior
                    if judged.execution.timed_out
                    else policy.judge_failure_behavior
                )
                status = _failure_status(behavior)
            else:
                status = (
                    judge_result.status
                    if judge_result.status in _BLOCKING_STATUSES
                    else _failure_status(policy.judge_failure_behavior)
                )
            return _decision(
                context,
                selected,
                policy,
                checks,
                status,
                judge_result.reason,
                judge_execution=judged.execution,
            )

        return _decision(
            context,
            selected,
            policy,
            checks,
            QualityStatus.PASS,
            "All applicable pre-response controls passed.",
            judge_execution=judged.execution,
        )

    checks.append(
        QualityCheckResult(
            check="reference_free_groundedness",
            status=QualityStatus.NOT_APPLICABLE,
            reason="The STANDARD runtime profile does not invoke a model-based judge.",
        )
    )
    return _decision(
        context,
        selected,
        policy,
        checks,
        QualityStatus.PASS,
        "All applicable pre-response controls passed.",
    )


class MonitoringSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = False
    approved_data_handling: bool = False
    sample_rate: float = Field(default=0.0, ge=0, le=1)
    max_concurrency: int = Field(default=1, ge=1)
    daily_budget_usd: float = Field(default=0.0, ge=0)


class MonitoringObservation(BaseModel):
    model_config = ConfigDict(frozen=True)

    request_id: str
    trace_id: str | None = None
    status: QualityStatus
    reason: str
    scores: dict[str, float] = Field(default_factory=dict)


Monitor = Callable[[str, str | None], MonitoringObservation]


def observe_completed_request(
    *,
    request_id: str,
    trace_id: str | None,
    settings: MonitoringSettings,
    sampled: bool,
    monitor: Monitor | None,
) -> MonitoringObservation:
    """Observe a delivered response; this result never changes answer delivery."""

    if not settings.enabled or not sampled:
        return MonitoringObservation(
            request_id=request_id,
            trace_id=trace_id,
            status=QualityStatus.NOT_RUN,
            reason="Online monitoring is disabled or this request was not sampled.",
        )
    if not settings.approved_data_handling or monitor is None:
        return MonitoringObservation(
            request_id=request_id,
            trace_id=trace_id,
            status=QualityStatus.UNAVAILABLE,
            reason="Approved monitoring data handling or an adapter is unavailable.",
        )
    try:
        return monitor(request_id, trace_id)
    except Exception:
        return MonitoringObservation(
            request_id=request_id,
            trace_id=trace_id,
            status=QualityStatus.ERROR,
            reason="Online monitoring failed after answer delivery.",
        )


class EvaluationMetricPlan(BaseModel):
    model_config = ConfigDict(frozen=True)

    reference_free: tuple[str, ...]
    reference_dependent: tuple[str, ...]
    skipped: dict[str, str]


def offline_metric_plan(
    *,
    has_reference: bool,
    requested_reference_metrics: Sequence[str],
) -> EvaluationMetricPlan:
    """Keep reference-dependent metrics out of arbitrary live request processing."""

    reference_free = ("faithfulness", "answer_relevancy")
    selected = tuple(requested_reference_metrics) if has_reference else ()
    skipped = (
        {}
        if has_reference
        else {
            name: "A reviewed reference answer is required." for name in requested_reference_metrics
        }
    )
    return EvaluationMetricPlan(
        reference_free=reference_free,
        reference_dependent=selected,
        skipped=skipped,
    )


def _failure_status(behavior: JudgeFailureBehavior) -> QualityStatus:
    return {
        JudgeFailureBehavior.ABSTAIN: QualityStatus.ABSTAIN,
        JudgeFailureBehavior.REVIEW_REQUIRED: QualityStatus.REVIEW_REQUIRED,
        JudgeFailureBehavior.FAIL_CLOSED: QualityStatus.FAIL,
    }[behavior]


def _measured_check(
    name: str,
    value: bool | None,
    pass_reason: str,
    failure_reason: str,
    *,
    not_applicable: bool = False,
    false_status: QualityStatus = QualityStatus.FAIL,
    reason_category: str | None = None,
    score: float | None = None,
    threshold: float | None = None,
) -> QualityCheckResult:
    if not_applicable:
        return QualityCheckResult(
            check=name,
            status=QualityStatus.NOT_APPLICABLE,
            reason="This control is not required by the selected policy.",
        )
    if value is None:
        return QualityCheckResult(
            check=name,
            status=QualityStatus.NOT_RUN,
            reason="No measured result was supplied for this control.",
        )
    return QualityCheckResult(
        check=name,
        status=QualityStatus.PASS if value else false_status,
        reason=pass_reason if value else failure_reason,
        reason_category=reason_category,
        score=score,
        threshold=threshold,
    )


def _decision(
    context: RuntimeQualityContext,
    profile: QualityProfile,
    policy: PolicyProfileSettings,
    checks: list[QualityCheckResult],
    status: QualityStatus,
    reason: str,
    judge_execution: RuntimeJudgeExecution | None = None,
) -> QualityDecision:
    return QualityDecision(
        request_id=context.request_id,
        profile=profile,
        policy_version=policy.version,
        status=status,
        deliver_answer=status is QualityStatus.PASS,
        delivery_outcome={
            QualityStatus.PASS: DeliveryOutcome.ANSWERED,
            QualityStatus.ABSTAIN: DeliveryOutcome.ABSTAIN,
            QualityStatus.REVIEW_REQUIRED: DeliveryOutcome.REVIEW_REQUIRED,
            QualityStatus.ERROR: DeliveryOutcome.ERROR,
        }.get(status, DeliveryOutcome.BLOCKED),
        checks=tuple(checks),
        reason=reason,
        judge_execution=judge_execution,
    )
