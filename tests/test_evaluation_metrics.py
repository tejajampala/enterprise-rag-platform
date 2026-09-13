from __future__ import annotations

import math

import pytest

from consumer_finance_rag.evaluation.metrics import (
    average_precision_at_k,
    dcg_at_k,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)


def test_binary_ir_metrics_match_hand_calculation() -> None:
    retrieved = ["x", "a", "b", "c"]
    relevant = {"a", "c"}

    assert recall_at_k(retrieved, relevant, 3) == 0.5
    assert precision_at_k(retrieved, relevant, 3) == pytest.approx(1 / 3)
    assert hit_rate_at_k(retrieved, relevant, 3) == 1.0
    assert reciprocal_rank_at_k(retrieved, relevant, 3) == 0.5
    assert average_precision_at_k(retrieved, relevant, 3) == 0.25


def test_dcg_and_graded_ndcg_match_hand_calculation() -> None:
    grades = {"high": 3, "low": 1}
    actual = 1.0 + 7.0 / math.log2(3)
    ideal = 7.0 + 1.0 / math.log2(3)

    assert dcg_at_k(["low", "high"], grades, 2) == pytest.approx(actual)
    assert ndcg_at_k(["low", "high"], grades, 2) == pytest.approx(actual / ideal)
    assert ndcg_at_k(["high", "low"], grades, 2) == 1.0


def test_metrics_deduplicate_retrieved_ids() -> None:
    assert precision_at_k(["a", "a", "b"], {"a", "b"}, 2) == 1.0
    assert average_precision_at_k(["a", "a", "b"], {"a", "b"}, 2) == 1.0


def test_empty_relevance_and_short_rankings_have_explicit_conventions() -> None:
    assert recall_at_k(["a"], set(), 5) == 0.0
    assert precision_at_k(["a"], {"a"}, 5) == 0.2
    assert hit_rate_at_k([], {"a"}, 5) == 0.0
    assert reciprocal_rank_at_k([], {"a"}, 5) == 0.0
    assert average_precision_at_k(["a"], set(), 5) == 0.0
    assert ndcg_at_k(["a"], {}, 5) == 0.0


def test_average_precision_at_k_uses_minimum_of_relevant_count_and_k() -> None:
    relevant = {"a", "b", "c"}
    assert average_precision_at_k(["a", "x", "b"], relevant, 2) == 0.5


@pytest.mark.parametrize(
    "metric",
    [recall_at_k, precision_at_k, hit_rate_at_k, reciprocal_rank_at_k],
)
def test_metrics_reject_non_positive_k(metric: object) -> None:
    with pytest.raises(ValueError, match="positive"):
        metric(["a"], {"a"}, 0)  # type: ignore[operator]
