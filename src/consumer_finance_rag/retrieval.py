"""Hybrid retrieval orchestration: HNSW + BM25 + RRF + reranking."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from consumer_finance_rag.config import RetrievalSettings
from consumer_finance_rag.models import RetrievalMetadata, RetrievedChunk
from consumer_finance_rag.observability import NoOpTracer, Tracer, chunk_summaries
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
        tracer: Tracer | None = None,
        collection_name: str | None = None,
    ) -> None:
        self.embedding_provider = embedding_provider
        self.reranker = reranker
        self.store = store
        self.sparse_encoder = sparse_encoder
        self.settings = settings
        self.tracer = tracer or NoOpTracer()
        self.collection_name = collection_name

    def retrieve(self, query: str, top_k: int | None = None) -> RetrievalResult:
        trace = self.retrieve_debug(query, top_k=top_k)
        return RetrievalResult(chunks=trace.reranked, metadata=trace.metadata)

    def retrieve_debug(self, query: str, top_k: int | None = None) -> RetrievalTrace:
        """Run the production retrieval path while retaining each ranked stage."""

        final_contexts = self.settings.final_contexts if top_k is None else top_k
        if final_contexts <= 0:
            raise ValueError("top_k must be positive")
        with self.tracer.span(
            "embed_query",
            span_type="EMBEDDING",
            attributes={"embedding.model": self.embedding_provider.model_name},
        ) as span:
            span.set_inputs({"query": query})
            query_vector = self.embedding_provider.embed_query(query)
            span.set_outputs({"dimensions": len(query_vector)})

        with self.tracer.span(
            "dense_hnsw_search",
            span_type="RETRIEVER",
            attributes={
                "retrieval.mode": "dense_hnsw_ann",
                "qdrant.collection": self.collection_name,
                "candidate_limit": self.settings.dense_candidates,
            },
        ) as span:
            span.set_inputs({"query": query})
            dense = self.store.search_dense(query_vector, self.settings.dense_candidates)
            span.set_outputs(chunk_summaries(dense))

        with self.tracer.span(
            "sparse_bm25_search",
            span_type="RETRIEVER",
            attributes={
                "retrieval.mode": "sparse_bm25",
                "qdrant.collection": self.collection_name,
                "candidate_limit": self.settings.sparse_candidates,
                "bm25.k1": self.sparse_encoder.state.k1,
                "bm25.b": self.sparse_encoder.state.b,
            },
        ) as span:
            span.set_inputs({"query": query})
            sparse = self.store.search_sparse(
                query, self.sparse_encoder, self.settings.sparse_candidates
            )
            span.set_outputs(chunk_summaries(sparse))

        with self.tracer.span(
            "rrf_fusion",
            span_type="CHAIN",
            attributes={
                "retrieval.mode": "reciprocal_rank_fusion",
                "rrf.k": self.settings.rrf_k,
            },
        ) as span:
            span.set_inputs(
                {
                    "dense_chunk_ids": [item.chunk.chunk_id for item in dense],
                    "sparse_chunk_ids": [item.chunk.chunk_id for item in sparse],
                }
            )
            fused = reciprocal_rank_fusion([dense, sparse], rrf_k=self.settings.rrf_k)
            span.set_outputs(chunk_summaries(fused[: self.settings.fused_candidates]))
        candidates = fused[: self.settings.fused_candidates]
        with self.tracer.span(
            "rerank",
            span_type="RERANKER",
            attributes={
                "reranker.model": self.reranker.model_name,
                "candidate_count": len(candidates),
                "final_context_limit": final_contexts,
            },
        ) as span:
            span.set_inputs(
                {"query": query, "chunk_ids": [item.chunk.chunk_id for item in candidates]}
            )
            reranked = self.reranker.rerank(query, candidates, final_contexts)
            final = reranked[:final_contexts]
            span.set_outputs(chunk_summaries(final))
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
