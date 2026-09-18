"""Transparent deterministic information-retrieval metric formulas."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from consumer_finance_rag.evaluation.models import IRMetrics


def _unique_at_k(retrieved: Sequence[str], k: int) -> list[str]:
    if k <= 0:
        raise ValueError("k must be positive")
    seen: set[str] = set()
    result: list[str] = []
    for item in retrieved:
        if item not in seen:
            seen.add(item)
            result.append(item)
        if len(result) == k:
            break
    return result


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(_unique_at_k(retrieved, k)) & relevant) / len(relevant)


def precision_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    return len(set(_unique_at_k(retrieved, k)) & relevant) / k


def hit_rate_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    return float(bool(set(_unique_at_k(retrieved, k)) & relevant))


def reciprocal_rank_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    for rank, item in enumerate(_unique_at_k(retrieved, k), start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def dcg_at_k(retrieved: Sequence[str], grades: Mapping[str, int], k: int) -> float:
    return float(
        sum(
            (2 ** grades.get(item, 0) - 1) / math.log2(rank + 1)
            for rank, item in enumerate(_unique_at_k(retrieved, k), start=1)
        )
    )


def ndcg_at_k(retrieved: Sequence[str], grades: Mapping[str, int], k: int) -> float:
    actual = dcg_at_k(retrieved, grades, k)
    ideal_grades = sorted(grades.values(), reverse=True)[:k]
    ideal = sum((2**grade - 1) / math.log2(rank + 1) for rank, grade in enumerate(ideal_grades, 1))
    return actual / ideal if ideal else 0.0


def average_precision_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for rank, item in enumerate(_unique_at_k(retrieved, k), start=1):
        if item in relevant:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / min(len(relevant), k)


def calculate_ir_metrics(
    retrieved: Sequence[str],
    relevant: set[str],
    k: int,
    graded_relevance: Mapping[str, int] | None = None,
) -> IRMetrics:
    grades = dict(graded_relevance or {item: 1 for item in relevant})
    return IRMetrics(
        recall=recall_at_k(retrieved, relevant, k),
        precision=precision_at_k(retrieved, relevant, k),
        hit_rate=hit_rate_at_k(retrieved, relevant, k),
        mrr=reciprocal_rank_at_k(retrieved, relevant, k),
        ndcg=ndcg_at_k(retrieved, grades, k),
        average_precision=average_precision_at_k(retrieved, relevant, k),
    )
