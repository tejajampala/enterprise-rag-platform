from __future__ import annotations

from consumer_finance_rag.evaluation.models import (
    CaseStageResult,
    RetrievalCaseResult,
)
from consumer_finance_rag.evaluation.retrieval_runner import aggregate_case_results


def test_stage_comparison_aggregation_keeps_stages_separate() -> None:
    results = [
        RetrievalCaseResult(
            case_id="one",
            category="credit cards",
            difficulty="easy",
            stages={
                "dense": CaseStageResult(retrieved_ids=["a"], metrics={"recall_at_1": 1.0}),
                "sparse": CaseStageResult(retrieved_ids=["b"], metrics={"recall_at_1": 0.0}),
            },
        ),
        RetrievalCaseResult(
            case_id="two",
            category="credit cards",
            difficulty="hard",
            stages={
                "dense": CaseStageResult(retrieved_ids=["x"], metrics={"recall_at_1": 0.0}),
                "sparse": CaseStageResult(retrieved_ids=["y"], metrics={"recall_at_1": 1.0}),
            },
        ),
    ]

    aggregate = aggregate_case_results(results)
    assert aggregate["dense"]["recall_at_1"] == 0.5
    assert aggregate["sparse"]["recall_at_1"] == 0.5
