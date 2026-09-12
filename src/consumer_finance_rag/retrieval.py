"""Hybrid retrieval orchestration: HNSW + BM25 + RRF + reranking."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from consumer_finance_rag.config import RetrievalSettings
from consumer_finance_rag.models import RetrievalMetadata, RetrievedChunk
from consumer_finance_rag.providers import EmbeddingProvider, RerankerProvider
from consumer_finance_rag.sparse import BM25SparseEncoder


class HybridSearchStore(Protocol):
    def search_dense(self, vector: list[float], limit: int) -> list[RetrievedChunk]: ...

    def search_sparse(
        self, query: str, encoder: BM25SparseEncoder, limit: int
    ) -> list[RetrievedChunk]: ...


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[RetrievedChunk]], *, rrf_k: int = 60
) -> list[RetrievedChunk]:
    """Fuse rankings by chunk ID, deduplicating candidates across retrieval paths."""

    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive")
    by_id: dict[str, RetrievedChunk] = {}
    scores: dict[str, float] = {}
    dense_ranks: dict[str, int] = {}
    sparse_ranks: dict[str, int] = {}
    for list_index, ranked in enumerate(ranked_lists):
        for rank, candidate in enumerate(ranked, start=1):
            chunk_id = candidate.chunk.chunk_id
            by_id.setdefault(chunk_id, candidate)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)
            if candidate.dense_rank is not None or list_index == 0:
                dense_ranks.setdefault(chunk_id, candidate.dense_rank or rank)
            if candidate.sparse_rank is not None or list_index == 1:
                sparse_ranks.setdefault(chunk_id, candidate.sparse_rank or rank)

    fused = [
        RetrievedChunk(
            chunk=candidate.chunk,
            score=scores[chunk_id],
            dense_rank=dense_ranks.get(chunk_id),
            sparse_rank=sparse_ranks.get(chunk_id),
        )
        for chunk_id, candidate in by_id.items()
    ]
    return sorted(fused, key=lambda item: (-item.score, item.chunk.chunk_id))


@dataclass(frozen=True)
class RetrievalResult:
    chunks: list[RetrievedChunk]
    metadata: RetrievalMetadata


@dataclass(frozen=True)
class RetrievalTrace:
    """Intermediate rankings exposed for local debugging and inspection."""

    dense: list[RetrievedChunk]
    sparse: list[RetrievedChunk]
    fused: list[RetrievedChunk]
    reranked: list[RetrievedChunk]
    metadata: RetrievalMetadata


class HybridRetrievalService:
    """The single public retrieval entry point for the RAG pipeline."""

    def __init__(
        self,
        *,
        embedding_provider: EmbeddingProvider,
        reranker: RerankerProvider,
        store: HybridSearchStore,
        sparse_encoder: BM25SparseEncoder,
        settings: RetrievalSettings,
    ) -> None:
        self.embedding_provider = embedding_provider
        self.reranker = reranker
        self.store = store
        self.sparse_encoder = sparse_encoder
        self.settings = settings

    def retrieve(self, query: str, top_k: int | None = None) -> RetrievalResult:
        trace = self.retrieve_debug(query, top_k=top_k)
        return RetrievalResult(chunks=trace.reranked, metadata=trace.metadata)

    def retrieve_debug(self, query: str, top_k: int | None = None) -> RetrievalTrace:
        """Run the production retrieval path while retaining each ranked stage."""

        final_contexts = self.settings.final_contexts if top_k is None else top_k
        if final_contexts <= 0:
            raise ValueError("top_k must be positive")
        query_vector = self.embedding_provider.embed_query(query)
        dense = self.store.search_dense(query_vector, self.settings.dense_candidates)
        sparse = self.store.search_sparse(
            query, self.sparse_encoder, self.settings.sparse_candidates
        )
        fused = reciprocal_rank_fusion([dense, sparse], rrf_k=self.settings.rrf_k)
        candidates = fused[: self.settings.fused_candidates]
        reranked = self.reranker.rerank(query, candidates, final_contexts)
        final = reranked[:final_contexts]
        return RetrievalTrace(
            dense=dense,
            sparse=sparse,
            fused=candidates,
            reranked=final,
            metadata=RetrievalMetadata(
                dense_candidates=len(dense),
                sparse_candidates=len(sparse),
                fused_candidates=len(candidates),
                reranked=len(candidates),
                final_contexts=len(final),
            ),
        )
