from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from consumer_finance_rag.config import MLflowSettings
from consumer_finance_rag.context import ContextBuilder
from consumer_finance_rag.evaluation.managed_rag import (
    ApprovedAnswerEvaluation,
    JudgeStageSummary,
    ManagedAnswerCaseReport,
    ManagedEvaluationDatasetError,
    evaluate_approved_answer,
    load_managed_rag_dataset,
)
from consumer_finance_rag.evaluation.mlflow_genai import (
    MLflowJudgeResults,
    MLflowMetricDiagnostic,
    MLflowMetricStatus,
)
from consumer_finance_rag.evaluation.models import (
    EvaluationCase,
    JudgeMetricValue,
    RAGCaseResult,
)
from consumer_finance_rag.evaluation.ragas_adapter import RagasEvaluationInput
from consumer_finance_rag.evaluation.workbench import (
    RAGAS_METRIC_INPUT_FIELDS,
    EvaluationExecutionRegistry,
    EvaluationInputSnapshotStore,
    WorkbenchEvaluationOptions,
    WorkbenchEvaluationStatus,
    backend_state_from_summary,
    capture_evaluation_input_snapshot,
    complete_chat_evaluation,
    display_status,
    evaluation_key,
    evaluation_plan,
    find_eligible_case,
    proposed_chat_evaluation,
    saved_report_matches_batch,
)
from consumer_finance_rag.models import (
    Chunk,
    ModelMetadata,
    RAGResponse,
    RetrievalMetadata,
    RetrievedChunk,
)
from consumer_finance_rag.observability import NoOpTracer

DATASET_PATH = Path("data/evaluation/managed_public_cases.json")
JUDGE_MODEL = "bedrock:/meta.llama3-70b-instruct-v1:0"


class _RagasEvaluator:
    enabled = True

    def evaluate(
        self,
        results: Sequence[RAGCaseResult],
    ) -> dict[str, dict[str, JudgeMetricValue]]:
        return {
            result.case_id: {
                "faithfulness": JudgeMetricValue(value=0.9),
            }
            for result in results
        }


def _approved_result() -> tuple[Any, RAGCaseResult]:
    case = load_managed_rag_dataset(DATASET_PATH).cases[0]
    return case, RAGCaseResult(
        case_id=case.case_id,
        question=case.question,
        answer="Approved answer [C1].",
        contexts=["Reviewed PUBLIC evidence."],
        chunk_ids=[case.expected_chunk_ids[0]],
    )


def _snapshot_response(case: Any, *, request_id: str = "request-1") -> RAGResponse:
    candidate = RetrievedChunk(
        chunk=Chunk(
            chunk_id=case.expected_chunk_ids[0],
            document_id=case.expected_document_ids[0],
            document_version="v1",
            text="Reviewed PUBLIC evidence " * 20,
            ordinal=0,
            chunk_hash="a" * 64,
            metadata={
                "title": "Reviewed source",
                "source_id": case.expected_source_ids[0],
                "source_url": "https://example.test/reviewed",
                "access_visibility": "PUBLIC",
            },
        ),
        score=1.0,
    )
    context = ContextBuilder(max_characters=140).build([candidate])
    return RAGResponse(
        request_id=request_id,
        trace_id="tr-exact",
        question=case.question,
        answer="Generated answer [C1].",
        structured_context=context,
        citations=context.citations,
        retrieved_chunks=context.chunks,
        retrieval_metadata=RetrievalMetadata(
            dense_candidates=1,
            sparse_candidates=1,
            fused_candidates=1,
            reranked=1,
            final_contexts=1,
        ),
        model_metadata=ModelMetadata(
            embedding_model="embedding",
            reranker_model="reranker",
            generation_model="google.gemma-3-4b-it",
        ),
    )


def test_workbench_evaluation_defaults_are_independently_configurable() -> None:
    defaults = WorkbenchEvaluationOptions()

    assert defaults.mlflow_genai is True
    assert defaults.ragas is True
    assert defaults.save_to_mlflow is True
    assert defaults.capture_judge_io is False
    assert defaults.judge_io_retention_hours == 24
    assert WorkbenchEvaluationOptions(mlflow_genai=False).ragas is True
    assert WorkbenchEvaluationOptions(ragas=False).save_to_mlflow is True
    assert WorkbenchEvaluationOptions(save_to_mlflow=False).mlflow_genai is True


def test_execution_registry_prevents_implicit_repeat_but_allows_explicit_repeat() -> None:
    registry = EvaluationExecutionRegistry()
    key = evaluation_key(
        scope_id="request-1",
        generation=4,
        case_ids=["case-1"],
        generation_model="google.gemma-3-4b-it",
        judge_model=JUDGE_MODEL,
        options=WorkbenchEvaluationOptions(),
    )

    assert registry.should_execute(key) is True
    registry.record(key, "completed")
    assert registry.should_execute(key) is False
    assert registry.should_execute(key, explicit_re_evaluate=True) is True
    assert registry.get(key) == "completed"


def test_configuration_changes_create_a_distinct_execution_key() -> None:
    assert evaluation_key(
        scope_id="request-1",
        generation=4,
        case_ids=["case-1"],
        generation_model="google.gemma-3-4b-it",
        judge_model=JUDGE_MODEL,
        options=WorkbenchEvaluationOptions(),
    ) != evaluation_key(
        scope_id="request-1",
        generation=4,
        case_ids=["case-1"],
        generation_model="google.gemma-3-4b-it",
        judge_model=JUDGE_MODEL,
        options=WorkbenchEvaluationOptions(ragas=False),
    )


def test_metric_plan_marks_reference_dependent_metrics_without_fabricating_values() -> None:
    plan = evaluation_plan(has_reference=False, has_content_trace=False)

    assert plan.mlflow_genai.skipped["correctness"] is (
        WorkbenchEvaluationStatus.REFERENCE_REQUIRED
    )
    assert plan.ragas.skipped["context_precision"] is (
        WorkbenchEvaluationStatus.REFERENCE_REQUIRED
    )
    assert plan.ragas.skipped["context_recall"] is (
        WorkbenchEvaluationStatus.REFERENCE_REQUIRED
    )
    assert plan.mlflow_genai.skipped["retrieval_groundedness"] is (
        WorkbenchEvaluationStatus.SKIPPED
    )


def test_only_an_exact_reviewed_public_question_is_eligible() -> None:
    dataset = load_managed_rag_dataset(DATASET_PATH)
    case = dataset.cases[0]

    assert find_eligible_case(f"  {case.question.upper()}  ", dataset) == case
    assert find_eligible_case("arbitrary private request content", dataset) is None


def test_snapshot_uses_exact_reviewed_case_and_existing_five_field_ragas_contract() -> None:
    dataset = load_managed_rag_dataset(DATASET_PATH)
    case = dataset.cases[0]
    response = _snapshot_response(case)

    snapshot = capture_evaluation_input_snapshot(
        response=response,
        request_id=response.request_id,
        case=case,
        dataset=dataset,
        configuration_id="configuration-1",
        generation_region="ap-southeast-2",
        judge_model=JUDGE_MODEL,
        judge_region="us-east-1",
    )

    assert set(RagasEvaluationInput.model_fields) == {
        "case_id",
        "question",
        "answer",
        "contexts",
        "reference",
    }
    assert snapshot.request_id == "request-1"
    assert snapshot.case_id == case.case_id
    assert snapshot.original_question == case.question
    assert snapshot.ragas_input.question == case.question
    assert snapshot.ragas_input.answer == response.answer
    assert snapshot.ragas_input.contexts == [
        item.chunk.text for item in response.retrieved_chunks
    ]
    assert snapshot.ragas_input.reference == case.reference_answer
    assert snapshot.passages[0].truncated is True
    assert snapshot.prompt_context == response.structured_context.text  # type: ignore[union-attr]
    assert snapshot.generation_region == "ap-southeast-2"
    assert snapshot.judge_region == "us-east-1"


def test_metric_input_contract_uses_only_the_documented_field_subsets() -> None:
    assert RAGAS_METRIC_INPUT_FIELDS == {
        "faithfulness": ("question", "generated_answer", "ragas_contexts"),
        "context_precision": ("question", "reference_answer", "ragas_contexts"),
        "context_recall": ("question", "reference_answer", "ragas_contexts"),
        "response_groundedness": ("generated_answer", "ragas_contexts"),
    }


def test_snapshot_store_requires_exact_request_case_and_configuration_correlation() -> None:
    dataset = load_managed_rag_dataset(DATASET_PATH)
    case = dataset.cases[0]
    snapshot = capture_evaluation_input_snapshot(
        response=_snapshot_response(case, request_id="request-exact"),
        request_id="request-exact",
        case=case,
        dataset=dataset,
        configuration_id="configuration-exact",
        generation_region="ap-southeast-2",
        judge_model=JUDGE_MODEL,
        judge_region="us-east-1",
    )
    store = EvaluationInputSnapshotStore()
    store.record(snapshot)

    assert store.get_exact(
        "request-exact",
        case_id=case.case_id,
        configuration_id="configuration-exact",
    ) == snapshot
    assert store.get_exact("another-request") is None
    assert store.get_exact("request-exact", case_id="another-case") is None
    assert store.get_exact(
        "request-exact", configuration_id="another-configuration"
    ) is None
    updated = store.attach_evaluation_run(
        "request-exact",
        case_id=case.case_id,
        configuration_id="configuration-exact",
        evaluation_run_id="evaluation-run-exact",
    )
    assert updated is not None
    assert updated.trace_id == "tr-exact"
    assert updated.evaluation_run_id == "evaluation-run-exact"

    with pytest.raises(ValueError, match="reviewed evaluation question"):
        capture_evaluation_input_snapshot(
            response=_snapshot_response(case).model_copy(
                update={"question": "arbitrary private question"}
            ),
            request_id="request-1",
            case=case,
            dataset=dataset,
            configuration_id="configuration-exact",
            generation_region="ap-southeast-2",
            judge_model=JUDGE_MODEL,
            judge_region="us-east-1",
        )
    with pytest.raises(ValueError, match="exact request ID"):
        capture_evaluation_input_snapshot(
            response=_snapshot_response(case, request_id="response-request"),
            request_id="record-request",
            case=case,
            dataset=dataset,
            configuration_id="configuration-exact",
            generation_region="ap-southeast-2",
            judge_model=JUDGE_MODEL,
            judge_region="us-east-1",
        )


def test_expired_or_missing_snapshot_is_unavailable_without_historical_reconstruction() -> None:
    dataset = load_managed_rag_dataset(DATASET_PATH)
    case = dataset.cases[0]
    captured_at = datetime.now(UTC)
    snapshot = capture_evaluation_input_snapshot(
        response=_snapshot_response(case),
        request_id="request-1",
        case=case,
        dataset=dataset,
        configuration_id="configuration-1",
        generation_region="ap-southeast-2",
        judge_model=JUDGE_MODEL,
        judge_region="us-east-1",
        retention=timedelta(seconds=1),
        now=captured_at,
    )
    store = EvaluationInputSnapshotStore(retention=timedelta(seconds=1))
    store.record(snapshot)

    assert store.get_exact("missing-request") is None
    assert store.get_exact("request-1", now=captured_at + timedelta(seconds=2)) is None
    assert "generated_answer" not in ManagedAnswerCaseReport.model_fields
    assert "prompt_context" not in ManagedAnswerCaseReport.model_fields
    assert "ragas_input" not in ManagedAnswerCaseReport.model_fields


def test_approved_answer_reuses_same_judge_identity_for_both_backends() -> None:
    case, result = _approved_result()
    calls = {"mlflow": 0, "ragas": 0}

    def mlflow_runner(
        cases: Sequence[EvaluationCase],
        results: Sequence[RAGCaseResult],
        settings: MLflowSettings,
    ) -> MLflowJudgeResults:
        assert cases[0].case_id == case.case_id
        assert results[0].case_id == case.case_id
        assert settings.judge_model == JUDGE_MODEL
        calls["mlflow"] += 1
        return MLflowJudgeResults(
            aggregate={"correctness": 1.0},
            by_case={case.case_id: {}},
            run_id="judge-run",
        )

    def ragas_builder(**kwargs: Any) -> _RagasEvaluator:
        assert kwargs["judge_model"] == JUDGE_MODEL
        calls["ragas"] += 1
        return _RagasEvaluator()

    evaluated = evaluate_approved_answer(
        case,
        result,
        generation_model="google.gemma-3-4b-it",
        judge_model=JUDGE_MODEL,
        with_mlflow_judges=True,
        with_ragas=True,
        mlflow_settings=MLflowSettings(judge_model=JUDGE_MODEL),
        mlflow_judge_runner=mlflow_runner,
        ragas_builder=ragas_builder,
        ragas_execution_tracer=NoOpTracer(),
    )

    assert calls == {"mlflow": 1, "ragas": 1}
    assert evaluated.judge_model == JUDGE_MODEL
    assert evaluated.mlflow_genai.run_id == "judge-run"
    assert display_status(evaluated.mlflow_genai) is WorkbenchEvaluationStatus.COMPLETED
    assert display_status(evaluated.ragas) is WorkbenchEvaluationStatus.COMPLETED


def test_unavailable_judge_does_not_call_adapters_or_fail_answer() -> None:
    case, result = _approved_result()

    def unexpected(*_args: object, **_kwargs: object) -> Any:
        raise AssertionError("unavailable judge must not be invoked")

    evaluated = evaluate_approved_answer(
        case,
        result,
        generation_model="google.gemma-3-4b-it",
        judge_model=None,
        with_mlflow_judges=True,
        with_ragas=True,
        mlflow_settings=MLflowSettings(judge_model=None),
        mlflow_judge_runner=unexpected,
        ragas_builder=unexpected,
        ragas_execution_tracer=NoOpTracer(),
    )

    assert evaluated.mlflow_genai.status == "UNAVAILABLE"
    assert evaluated.ragas.status == "UNAVAILABLE"


def test_judge_failure_is_contained_and_identity_mismatch_is_rejected() -> None:
    case, result = _approved_result()

    def failing(*_args: object, **_kwargs: object) -> Any:
        raise RuntimeError("provider unavailable")

    evaluated = evaluate_approved_answer(
        case,
        result,
        generation_model="google.gemma-3-4b-it",
        judge_model=JUDGE_MODEL,
        with_mlflow_judges=True,
        with_ragas=True,
        mlflow_settings=MLflowSettings(judge_model=JUDGE_MODEL),
        mlflow_judge_runner=failing,
        ragas_builder=failing,
        ragas_execution_tracer=NoOpTracer(),
    )
    assert evaluated.mlflow_genai.status == "UNAVAILABLE"
    assert evaluated.ragas.status == "UNAVAILABLE"

    with pytest.raises(ValueError, match="configured judge model identity"):
        evaluate_approved_answer(
            case,
            result,
            generation_model="google.gemma-3-4b-it",
            judge_model="bedrock:/different-judge",
            with_mlflow_judges=True,
            with_ragas=False,
            mlflow_settings=MLflowSettings(judge_model=JUDGE_MODEL),
        )


def test_mismatched_answer_cannot_be_submitted_to_judges() -> None:
    case, result = _approved_result()

    with pytest.raises(ManagedEvaluationDatasetError, match="does not match"):
        evaluate_approved_answer(
            case,
            result.model_copy(update={"question": "private content"}),
            generation_model="google.gemma-3-4b-it",
            judge_model=JUDGE_MODEL,
            with_mlflow_judges=True,
            with_ragas=True,
            mlflow_settings=MLflowSettings(judge_model=JUDGE_MODEL),
        )


def test_public_showcase_does_not_import_or_invoke_judge_adapters() -> None:
    source = Path("app/streamlit_app.py").read_text(encoding="utf-8")

    assert "evaluate_managed_rag(" not in source
    assert "evaluate_approved_answer(" not in source
    assert "run_mlflow_judges(" not in source
    assert "build_ragas_evaluator(" not in source
    assert "JudgeCallDebugCapture" not in source
    assert "Individual Judge Calls" not in source


def test_request_scoped_states_cover_eligibility_pending_and_skips() -> None:
    options = WorkbenchEvaluationOptions(ragas=False)
    pending = proposed_chat_evaluation(
        request_id="request-1",
        case_id="case-1",
        evaluation_key_value="configuration-1",
        options=options,
        automatic_evaluation_enabled=True,
        evaluate_answer_selected=True,
        has_reference=True,
    )
    assert pending.eligible is True
    assert pending.mlflow_genai.status is WorkbenchEvaluationStatus.PENDING
    assert pending.ragas.status is WorkbenchEvaluationStatus.NOT_REQUESTED

    ineligible = proposed_chat_evaluation(
        request_id="request-2",
        case_id=None,
        evaluation_key_value=None,
        options=WorkbenchEvaluationOptions(),
        automatic_evaluation_enabled=True,
        evaluate_answer_selected=True,
        has_reference=False,
    )
    assert ineligible.eligible is False
    assert ineligible.mlflow_genai.status is WorkbenchEvaluationStatus.SKIPPED
    assert "ineligible" in ineligible.mlflow_genai.reason.casefold()


def test_request_state_preserves_completed_failed_and_unavailable_reasons() -> None:
    state = proposed_chat_evaluation(
        request_id="request-1",
        case_id="case-1",
        evaluation_key_value="configuration-1",
        options=WorkbenchEvaluationOptions(),
        automatic_evaluation_enabled=True,
        evaluate_answer_selected=True,
        has_reference=True,
    )
    result = ApprovedAnswerEvaluation(
        case_id="case-1",
        generation_model="google.gemma-3-4b-it",
        judge_model=JUDGE_MODEL,
        mlflow_genai=JudgeStageSummary(
            status="PARTIAL",
            reason="One requested scorer failed.",
            aggregate_metrics={"safety": 1.0},
            metric_statuses={"safety": "COMPLETED", "correctness": "FAILED"},
            failed_metrics={"correctness": "SCORER_ERROR"},
            run_id="run-1",
        ),
        ragas=JudgeStageSummary(
            status="UNAVAILABLE",
            reason="Judge provider unavailable.",
        ),
    )
    completed = complete_chat_evaluation(state, result)
    assert completed.mlflow_genai.status is WorkbenchEvaluationStatus.PARTIAL
    assert completed.mlflow_genai.run_id == "run-1"
    assert completed.mlflow_genai.metric_reasons["correctness"] == "SCORER_ERROR"
    assert completed.ragas.status is WorkbenchEvaluationStatus.UNAVAILABLE
    assert completed.completed_at is not None


def test_reference_required_and_stale_report_identity_are_explicit() -> None:
    state = proposed_chat_evaluation(
        request_id="request-1",
        case_id="case-1",
        evaluation_key_value="configuration-1",
        options=WorkbenchEvaluationOptions(),
        automatic_evaluation_enabled=True,
        evaluate_answer_selected=True,
        has_reference=False,
    )
    assert state.mlflow_genai.status is WorkbenchEvaluationStatus.REFERENCE_REQUIRED
    now = datetime.now(UTC)
    assert saved_report_matches_batch(now, now) is True
    assert saved_report_matches_batch(now - timedelta(seconds=1), now) is False
    assert saved_report_matches_batch(now, None) is False


def test_backend_failed_status_keeps_exact_safe_reason() -> None:
    backend = backend_state_from_summary(
        JudgeStageSummary(
            status="FAILED",
            reason="No requested metric produced a valid result.",
            metric_statuses={"safety": "FAILED"},
            failed_metrics={"safety": "AUTHENTICATION_CONFIGURATION"},
            run_id="failed-run",
        )
    )
    assert backend.status is WorkbenchEvaluationStatus.FAILED
    assert backend.reason == "No requested metric produced a valid result."
    assert backend.metric_reasons == {"safety": "AUTHENTICATION_CONFIGURATION"}


@pytest.mark.parametrize(
    ("diagnostics", "expected_status"),
    [
        (
            {
                "correctness": MLflowMetricDiagnostic(MLflowMetricStatus.COMPLETED),
                "safety": MLflowMetricDiagnostic(MLflowMetricStatus.COMPLETED),
            },
            "COMPLETED",
        ),
        (
            {
                "correctness": MLflowMetricDiagnostic(
                    MLflowMetricStatus.FAILED, "SCORER_ERROR"
                ),
                "safety": MLflowMetricDiagnostic(MLflowMetricStatus.COMPLETED),
            },
            "PARTIAL",
        ),
        (
            {
                "correctness": MLflowMetricDiagnostic(
                    MLflowMetricStatus.FAILED, "SCORER_ERROR"
                ),
                "safety": MLflowMetricDiagnostic(
                    MLflowMetricStatus.MISSING, "MISSING_RESULT"
                ),
            },
            "FAILED",
        ),
    ],
)
def test_mlflow_stage_requires_every_requested_metric_to_succeed(
    diagnostics: dict[str, MLflowMetricDiagnostic],
    expected_status: str,
) -> None:
    case, result = _approved_result()

    def runner(*_args: object, **_kwargs: object) -> MLflowJudgeResults:
        valid = {
            name: JudgeMetricValue(value=1.0)
            for name, diagnostic in diagnostics.items()
            if diagnostic.status is MLflowMetricStatus.COMPLETED
        }
        return MLflowJudgeResults(
            aggregate={name: 1.0 for name in valid},
            by_case={case.case_id: valid},
            run_id="preserved-run-id",
            metric_diagnostics=diagnostics,
        )

    evaluated = evaluate_approved_answer(
        case,
        result,
        generation_model="google.gemma-3-4b-it",
        judge_model=JUDGE_MODEL,
        with_mlflow_judges=True,
        with_ragas=False,
        mlflow_settings=MLflowSettings(judge_model=JUDGE_MODEL),
        mlflow_judge_runner=runner,
    )
    assert evaluated.mlflow_genai.status == expected_status
    assert evaluated.mlflow_genai.run_id == "preserved-run-id"
