from __future__ import annotations

from collections.abc import Callable

from consumer_finance_rag.models import RetrievedChunk
from consumer_finance_rag.retrieval import reciprocal_rank_fusion


def test_rrf_rewards_chunks_present_in_both_rankings(
    make_candidate: Callable[..., RetrievedChunk],
) -> None:
    dense = [make_candidate("a"), make_candidate("b")]
    sparse = [make_candidate("b"), make_candidate("c")]

    fused = reciprocal_rank_fusion([dense, sparse], rrf_k=60)

    assert fused[0].chunk.chunk_id == "b"
    assert fused[0].dense_rank == 2
    assert fused[0].sparse_rank == 1


def test_rrf_deduplicates_candidates_by_chunk_id(
    make_candidate: Callable[..., RetrievedChunk],
) -> None:
    duplicate = make_candidate("same")
    fused = reciprocal_rank_fusion([[duplicate], [duplicate]], rrf_k=10)

    assert len(fused) == 1
    assert fused[0].score == 2 / 11
