from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import mlflow
import pandas as pd  # type: ignore[import-untyped]
import pytest

from consumer_finance_rag.config import MLflowSettings
from consumer_finance_rag.evaluation.managed_rag import (
    _run_mlflow_stage,
    load_managed_rag_dataset,
)
from consumer_finance_rag.evaluation.mlflow_genai import (
    MLflowJudgeResults,
    MLflowMetricStatus,
    _field_judge_rows,
    _judge_rows,
    _metric_diagnostics,
    _parse_aggregate_scores,
    _parse_case_scores,
    run_mlflow_judges,
)
from consumer_finance_rag.evaluation.models import EvaluationCase, RAGCaseResult


def _case_and_result() -> tuple[EvaluationCase, RAGCaseResult]:
    case = EvaluationCase(
        case_id="case-1",
        question="Reviewed question?",
        category="bank accounts",
        difficulty="easy",
        expected_facts=["Reviewed fact."],
        reference_answer="Reviewed reference.",
        relevant_document_ids=["document-1"],
        relevant_chunk_ids=["chunk-1"],
        review_status="reviewed",
    )
    result = RAGCaseResult(
        case_id="case-1",
        question="Reviewed question?",
        answer="Generated answer.",
        contexts=["Genuine authorized context."],
        chunk_ids=["chunk-1"],
    )
    return case, result


def test_judge_rows_use_actual_approved_inputs_and_one_correctness_expectation() -> None:
    case, result = _case_and_result()
    row = _judge_rows([case], [result])[0]

    assert row["inputs"] == {"question": "Reviewed question?"}
    assert row["outputs"] == {"answer": "Generated answer."}
    assert row["expectations"] == {"expected_response": "Reviewed reference."}
    assert "query_characters" not in str(row)
    assert "answer_characters" not in str(row)


def test_field_judge_rows_never_trigger_trace_based_autonomous_tools() -> None:
    case, result = _case_and_result()
    traced = result.model_copy(update={"trace_id": "tr-approved"})

    row = _field_judge_rows([case], [traced])[0]

    assert "trace" not in row
    assert row["inputs"] == {"question": case.question}
    assert row["outputs"] == {"answer": traced.answer}
    assert row["expectations"] == {"expected_response": case.reference_answer}


def test_mlflow_runner_executes_only_supported_field_scorers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case, result = _case_and_result()
    traced = result.model_copy(update={"trace_id": "tr-original"})
    captured: dict[str, Any] = {}

    monkeypatch.setattr(mlflow, "set_tracking_uri", lambda _uri: None)
    monkeypatch.setattr(mlflow, "set_experiment", lambda _name: None)

    def evaluate(*, data: Any, scorers: list[Any]) -> SimpleNamespace:
        captured["rows"] = data.to_dict(orient="records")
        captured["scorers"] = [scorer.name for scorer in scorers]
        return SimpleNamespace(
            metrics={
                "correctness/mean": 1.0,
                "relevance_to_query/mean": 1.0,
                "safety/mean": 1.0,
            },
            result_df=pd.DataFrame(
                [
                    {
                        "request_id": case.case_id,
                        "tags": {"case_id": case.case_id},
                        "correctness/value": True,
                        "relevance_to_query/value": True,
                        "safety/value": True,
                    }
                ]
            ),
            run_id="field-run",
        )

    monkeypatch.setattr(mlflow.genai, "evaluate", evaluate)

    evaluated = run_mlflow_judges(
        [case],
        [traced],
        MLflowSettings(judge_model="bedrock:/meta.llama3-70b-instruct-v1:0"),
    )

    assert captured["scorers"] == ["correctness", "relevance_to_query", "safety"]
    assert "trace" not in captured["rows"][0]
    assert captured["rows"][0]["expectations"] == {
        "expected_response": case.reference_answer
    }
    assert evaluated.run_id == "field-run"
    assert set(evaluated.aggregate) == {"correctness", "relevance_to_query", "safety"}


def test_scorer_diagnostics_reject_errors_missing_none_and_non_finite_values() -> None:
    frame = pd.DataFrame(
        [
            {
                "request_id": "case-1",
                "tags": {"case_id": "case-1"},
                "correctness/value": None,
                "correctness/error_code": "SCORER_ERROR",
                "correctness/error_message": (
                    "Both expected_response and expected_facts were provided"
                ),
                "safety/value": math.nan,
                "relevance_to_query/value": 1.0,
            }
        ]
    )
    diagnostics = _metric_diagnostics(
        frame,
        ["correctness", "safety", "relevance_to_query", "retrieval_groundedness"],
    )

    assert diagnostics["correctness"].status is MLflowMetricStatus.FAILED
    assert diagnostics["correctness"].error_category == "INVALID_EXPECTATIONS"
    assert diagnostics["safety"].status is MLflowMetricStatus.INVALID
    assert diagnostics["relevance_to_query"].status is MLflowMetricStatus.COMPLETED
    assert diagnostics["retrieval_groundedness"].status is MLflowMetricStatus.MISSING
    parsed = _parse_case_scores(frame)
    assert set(parsed["case-1"]) == {"relevance_to_query"}


def test_safe_error_category_does_not_expose_provider_message() -> None:
    frame = pd.DataFrame(
        [
            {
                "request_id": "case-1",
                "tags": {"case_id": "case-1"},
                "safety/value": None,
                "safety/error_message": "AWSIdAndKey requires an access key secret",
            }
        ]
    )
    diagnostic = _metric_diagnostics(frame, ["safety"])["safety"]
    assert diagnostic.error_category == "AUTHENTICATION_CONFIGURATION"
    assert "secret" not in diagnostic.error_category.casefold()


def test_installed_mlflow_bedrock_endpoint_failure_has_safe_category() -> None:
    frame = pd.DataFrame(
        [
            {
                "request_id": "case-1",
                "tags": {"case_id": "case-1"},
                "correctness/value": None,
                "correctness/error_message": (
                    "AmazonBedrockProvider does not implement get_endpoint_url"
                ),
            }
        ]
    )

    diagnostic = _metric_diagnostics(frame, ["correctness"])["correctness"]

    assert diagnostic.status is MLflowMetricStatus.FAILED
    assert diagnostic.error_category == "PROVIDER_UNSUPPORTED"


def test_exact_failed_report_shape_cannot_turn_expectations_into_metrics() -> None:
    managed_case = load_managed_rag_dataset(
        Path("data/evaluation/managed_public_cases.json")
    ).cases[0]
    result = RAGCaseResult(
        case_id=managed_case.case_id,
        question=managed_case.question,
        answer="Generated answer.",
        contexts=["Genuine reviewed PUBLIC context."],
        chunk_ids=[managed_case.expected_chunk_ids[0]],
        trace_id="trace-1",
    )
    frame = pd.DataFrame(
        [
            {
                "request_id": managed_case.case_id,
                "tags": {"case_id": managed_case.case_id},
                "inputs/question": managed_case.question,
                "outputs/answer": result.answer,
                "expected_facts/value": None,
                "expected_response/value": managed_case.reference_answer,
                "correctness/value": None,
                "correctness/error_message": (
                    "Both expected_response and expected_facts were provided"
                ),
                "relevance_to_query/value": None,
                "relevance_to_query/error_message": (
                    "AWSIdAndKey requires access key configuration"
                ),
                "safety/value": None,
                "safety/error_message": "AWSIdAndKey requires credentials",
                "retrieval_groundedness/value": None,
                "retrieval_groundedness/error_message": (
                    "AWSIdAndKey requires access key configuration"
                ),
            }
        ]
    )
    requested = (
        "correctness",
        "relevance_to_query",
        "safety",
        "retrieval_groundedness",
    )
    parsed = _parse_case_scores(frame, requested)
    diagnostics = _metric_diagnostics(frame, requested)
    aggregate = _parse_aggregate_scores(
        {
            "expected_facts/mean": 1.0,
            "expected_response/mean": 1.0,
            "inputs/mean": 1.0,
            "outputs/mean": 1.0,
        },
        requested,
    )

    def runner(*_args: object, **_kwargs: object) -> MLflowJudgeResults:
        return MLflowJudgeResults(
            aggregate=aggregate,
            by_case=parsed,
            run_id="2327fa18956d490fb59e472aa77aae8f",
            metric_diagnostics=diagnostics,
        )

    summary, by_case = _run_mlflow_stage(
        [managed_case],
        [result],
        requested=True,
        settings=MLflowSettings(
            judge_model="bedrock:/meta.llama3-70b-instruct-v1:0"
        ),
        runner=runner,
    )

    assert by_case == {managed_case.case_id: {}}
    assert aggregate == {}
    assert summary.status == "FAILED"
    assert summary.run_id == "2327fa18956d490fb59e472aa77aae8f"
    assert summary.failed_metrics == {
        "correctness": "INVALID_EXPECTATIONS",
        "relevance_to_query": "AUTHENTICATION_CONFIGURATION",
        "safety": "AUTHENTICATION_CONFIGURATION",
        "retrieval_groundedness": "AUTHENTICATION_CONFIGURATION",
    }
    safe_report_fields: dict[str, Any] = {
        "aggregate": summary.aggregate_metrics,
        "by_case": by_case,
    }
    assert managed_case.reference_answer not in repr(safe_report_fields)
