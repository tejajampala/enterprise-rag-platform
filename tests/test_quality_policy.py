from __future__ import annotations

from consumer_finance_rag.models import Chunk, RetrievedChunk
from consumer_finance_rag.quality_policy import (
    GroundednessJudgeInput,
    JudgeFailureBehavior,
    MonitoringObservation,
    MonitoringSettings,
    PolicyProfileSettings,
    QualityCheckResult,
    QualityPolicySettings,
    QualityProfile,
    QualityStatus,
    ReferenceFreeJudgeResult,
    RuntimeJudgeExecution,
    RuntimeQualityContext,
    assess_evidence_answerability,
    evaluate_pre_response,
    observe_completed_request,
    offline_metric_plan,
)


def _context(**updates: bool) -> RuntimeQualityContext:
    values = {
        "request_id": "request-1",
        "user_authorized": True,
        "corpus_authorized": True,
        "evidence_available": True,
        "citations_present": True,
        "citations_valid": True,
        "output_schema_valid": True,
        "pii_policy_satisfied": True,
        "safety_policy_satisfied": True,
    }
    values.update(updates)
    return RuntimeQualityContext.model_validate(values)


def test_standard_runs_deterministic_controls_without_a_judge() -> None:
    calls = 0

    def judge(
        judge_input: GroundednessJudgeInput,
        *,
        threshold: float,
        timeout_seconds: float,
        budget_usd: float,
    ) -> ReferenceFreeJudgeResult:
        nonlocal calls
        del judge_input, threshold, timeout_seconds, budget_usd
        calls += 1
        return ReferenceFreeJudgeResult(
            check=QualityCheckResult(check="judge", status=QualityStatus.PASS, reason="ok"),
            execution=RuntimeJudgeExecution(model_id="judge", region="region"),
        )

    decision = evaluate_pre_response(
        _context(),
        QualityPolicySettings(),
        profile=QualityProfile.STANDARD,
        judge=judge,
    )

    assert decision.status is QualityStatus.PASS
    assert decision.deliver_answer is True
    assert calls == 0


def test_deterministic_failure_blocks_before_any_judge() -> None:
    called = False

    def judge(
        judge_input: GroundednessJudgeInput,
        *,
        threshold: float,
        timeout_seconds: float,
        budget_usd: float,
    ) -> ReferenceFreeJudgeResult:
        nonlocal called
        del judge_input, threshold, timeout_seconds, budget_usd
        called = True
        raise AssertionError("must not run")

    decision = evaluate_pre_response(
        _context(citations_valid=False),
        QualityPolicySettings(),
        profile=QualityProfile.STRICT,
        judge=judge,
    )

    assert decision.status is QualityStatus.FAIL
    assert decision.deliver_answer is False
    assert called is False


def test_strict_unavailable_judge_never_becomes_pass() -> None:
    settings = QualityPolicySettings(
        strict=PolicyProfileSettings(
            version="strict-reviewed-v2",
            reference_free_judge_enabled=True,
            judge_failure_behavior=JudgeFailureBehavior.ABSTAIN,
        )
    )

    decision = evaluate_pre_response(_context(), settings, profile=QualityProfile.STRICT)

    assert decision.status is QualityStatus.ABSTAIN
    assert decision.checks[-1].status is QualityStatus.UNAVAILABLE
    assert decision.deliver_answer is False


def test_strict_disabled_judge_never_silently_passes() -> None:
    settings = QualityPolicySettings(
        strict=PolicyProfileSettings(
            version="strict-disabled-v1",
            reference_free_judge_enabled=False,
            judge_failure_behavior=JudgeFailureBehavior.REVIEW_REQUIRED,
        )
    )

    decision = evaluate_pre_response(_context(), settings, profile=QualityProfile.STRICT)

    assert decision.status is QualityStatus.REVIEW_REQUIRED
    assert decision.deliver_answer is False
    assert decision.checks[-1].status is QualityStatus.UNAVAILABLE
    assert decision.checks[-1].reason_category == "JUDGE_DISABLED"


def test_evaluation_profile_cannot_be_used_as_a_live_delivery_policy() -> None:
    decision = evaluate_pre_response(
        _context(),
        QualityPolicySettings(),
        profile=QualityProfile.EVALUATION,
    )

    assert decision.status is QualityStatus.REVIEW_REQUIRED
    assert decision.deliver_answer is False
    assert decision.checks[-1].status is QualityStatus.NOT_RUN
    assert decision.checks[-1].reason_category == "OFFLINE_EVALUATION_ONLY"


def test_high_risk_financial_route_requires_review() -> None:
    decision = evaluate_pre_response(
        _context(high_risk_financial=True),
        QualityPolicySettings(),
        profile=QualityProfile.STANDARD,
    )

    assert decision.status is QualityStatus.REVIEW_REQUIRED
    assert decision.deliver_answer is False


def test_reference_metrics_are_offline_and_require_reviewed_reference() -> None:
    plan = offline_metric_plan(
        has_reference=False,
        requested_reference_metrics=("context_recall", "context_precision"),
    )

    assert plan.reference_dependent == ()
    assert set(plan.skipped) == {"context_recall", "context_precision"}
    assert "faithfulness" in plan.reference_free


def test_async_monitoring_error_cannot_change_delivery_decision() -> None:
    delivered = evaluate_pre_response(
        _context(), QualityPolicySettings(), profile=QualityProfile.STANDARD
    )

    def failing_monitor(request_id: str, trace_id: str | None) -> MonitoringObservation:
        del request_id, trace_id
        raise RuntimeError("offline adapter failed")

    observation = observe_completed_request(
        request_id="request-1",
        trace_id="trace-1",
        settings=MonitoringSettings(
            enabled=True,
            approved_data_handling=True,
            sample_rate=1.0,
            daily_budget_usd=1.0,
        ),
        sampled=True,
        monitor=failing_monitor,
    )

    assert delivered.deliver_answer is True
    assert observation.status is QualityStatus.ERROR


def _evidence(score: float | None) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(
            chunk_id="chunk",
            document_id="document",
            document_version="v1",
            text="Reviewed evidence",
            ordinal=0,
            chunk_hash="a" * 64,
            metadata={},
        ),
        score=score or 0.0,
        rerank_score=score,
    )


def test_answerability_uses_versioned_reviewed_candidate_boundary() -> None:
    policy = PolicyProfileSettings(
        version="standard-v1",
        relevance_policy_version="reviewed-candidate-v1",
        minimum_relevance_score=0.0,
    )

    relevant = assess_evidence_answerability((_evidence(8.75),), policy)
    airplane = assess_evidence_answerability((_evidence(-9.4375),), policy)
    same_domain_insufficient = assess_evidence_answerability((_evidence(-0.5),), policy)
    partial = assess_evidence_answerability((_evidence(-2.0), _evidence(0.25)), policy)

    assert relevant.check.status is QualityStatus.PASS
    assert airplane.check.status is QualityStatus.ABSTAIN
    assert airplane.check.reason_category == "INSUFFICIENT_RELEVANT_EVIDENCE"
    assert same_domain_insufficient.check.status is QualityStatus.ABSTAIN
    assert partial.check.status is QualityStatus.PASS


def test_unavailable_relevance_and_unexecuted_controls_are_truthful() -> None:
    assessment = assess_evidence_answerability(
        (_evidence(None),), PolicyProfileSettings(version="standard-v1")
    )
    context = RuntimeQualityContext(
        request_id="request-1",
        user_authorized=True,
        corpus_authorized=True,
        evidence_available=True,
        evidence_relevant=True,
    )

    decision = evaluate_pre_response(context, QualityPolicySettings())
    statuses = {item.check: item.status for item in decision.checks}

    assert assessment.check.status is QualityStatus.UNAVAILABLE
    assert statuses["safety_policy"] is QualityStatus.NOT_RUN
    assert statuses["timeout"] is QualityStatus.NOT_RUN
    assert statuses["budget"] is QualityStatus.NOT_RUN
    assert statuses["citation_presence"] is QualityStatus.NOT_RUN
