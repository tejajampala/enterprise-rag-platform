"""Run deterministic metrics over every production retrieval stage."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from statistics import fmean

from consumer_finance_rag.evaluation.metrics import calculate_ir_metrics
from consumer_finance_rag.evaluation.models import (
    CaseStageResult,
    EvaluationCase,
    RetrievalCaseResult,
    RetrievalEvaluationReport,
    SeedEvaluationSet,
)
from consumer_finance_rag.retrieval import HybridRetrievalService, RetrievalTrace

STAGES = ("dense", "sparse", "fused", "reranked")
METRIC_NAMES = {
    "recall": "recall",
    "precision": "precision",
    "hit_rate": "hit_rate",
    "mrr": "mrr",
    "ndcg": "ndcg",
    "average_precision": "map",
}


def _stage_ids(trace: RetrievalTrace) -> dict[str, list[str]]:
    return {
        "dense": [item.chunk.chunk_id for item in trace.dense],
        "sparse": [item.chunk.chunk_id for item in trace.sparse],
        "fused": [item.chunk.chunk_id for item in trace.fused],
        "reranked": [item.chunk.chunk_id for item in trace.reranked],
    }


def evaluate_case(
    case: EvaluationCase, trace: RetrievalTrace, k_values: Sequence[int]
) -> RetrievalCaseResult:
    relevant = set(case.relevant_chunk_ids)
    grades = case.graded_relevance or {item: 1 for item in relevant}
    stage_results: dict[str, CaseStageResult] = {}
    for stage, retrieved_ids in _stage_ids(trace).items():
        metrics: dict[str, float] = {}
        for k in k_values:
            values = calculate_ir_metrics(retrieved_ids, relevant, k, grades)
            for metric_name, value in values.model_dump().items():
                metrics[f"{METRIC_NAMES[metric_name]}_at_{k}"] = value
        stage_results[stage] = CaseStageResult(
            retrieved_ids=retrieved_ids,
            metrics=metrics,
        )
    return RetrievalCaseResult(
        case_id=case.case_id,
        category=case.category,
        difficulty=case.difficulty,
        stages=stage_results,
    )


def aggregate_case_results(
    results: Iterable[RetrievalCaseResult],
) -> dict[str, dict[str, float]]:
    values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for result in results:
        for stage, stage_result in result.stages.items():
            for metric, value in stage_result.metrics.items():
                values[stage][metric].append(value)
    return {
        stage: {metric: fmean(metric_values) for metric, metric_values in metrics.items()}
        for stage, metrics in values.items()
    }


def _grouped_aggregates(
    results: Sequence[RetrievalCaseResult], attribute: str
) -> dict[str, dict[str, dict[str, float]]]:
    groups: dict[str, list[RetrievalCaseResult]] = defaultdict(list)
    for result in results:
        groups[str(getattr(result, attribute))].append(result)
    return {name: aggregate_case_results(group) for name, group in groups.items()}


def evaluate_retrieval(
    service: HybridRetrievalService,
    dataset: SeedEvaluationSet,
    *,
    k_values: Sequence[int],
    parameters: Mapping[str, str | int | float | bool | None],
) -> RetrievalEvaluationReport:
    if not k_values or any(k <= 0 for k in k_values):
        raise ValueError("k_values must contain positive integers")
    results = [
        evaluate_case(case, service.retrieve_debug(case.question), k_values)
        for case in dataset.cases
    ]
    return RetrievalEvaluationReport(
        created_at=datetime.now(UTC).isoformat(),
        dataset_name=dataset.name,
        dataset_version=dataset.version,
        dataset_status=dataset.status,
        case_count=len(results),
        k_values=list(k_values),
        parameters=dict(parameters),
        aggregate=aggregate_case_results(results),
        per_category=_grouped_aggregates(results, "category"),
        per_difficulty=_grouped_aggregates(results, "difficulty"),
        cases=results,
    )
