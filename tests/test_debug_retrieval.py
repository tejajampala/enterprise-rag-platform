from __future__ import annotations

from collections.abc import Callable, Sequence

from qdrant_client import models

from consumer_finance_rag.config import RetrievalSettings
from consumer_finance_rag.debug import format_retrieval_trace
from consumer_finance_rag.factory import warm_up_retrieval_service
from consumer_finance_rag.models import RetrievedChunk
from consumer_finance_rag.retrieval import HybridRetrievalService
from consumer_finance_rag.sparse import BM25SparseEncoder


class DebugEmbedding:
    model_name = "debug-embedding"

    def __init__(self) -> None:
        self.query_calls = 0

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        self.query_calls += 1
        return [1.0]


class DebugReranker:
    model_name = "debug-reranker"

    def __init__(self) -> None:
        self.calls = 0

    def rerank(
        self, query: str, candidates: Sequence[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        self.calls += 1
        return [
            candidate.model_copy(update={"rerank_score": 1.0 / rank})
            for rank, candidate in enumerate(candidates[:top_k], start=1)
        ]


class DebugStore:
    def __init__(self, dense: list[RetrievedChunk], sparse: list[RetrievedChunk]) -> None:
        self.dense = dense
        self.sparse = sparse

    def search_dense(
        self,
        vector: list[float],
        limit: int,
        *,
        query_filter: models.Filter | None = None,
    ) -> list[RetrievedChunk]:
        del query_filter
        return self.dense[:limit]

    def search_sparse(
        self,
        query: str,
        encoder: BM25SparseEncoder,
        limit: int,
        *,
        query_filter: models.Filter | None = None,
    ) -> list[RetrievedChunk]:
        del query_filter
        return self.sparse[:limit]


def test_debug_trace_retains_every_production_retrieval_stage(
    make_candidate: Callable[..., RetrievedChunk],
) -> None:
    dense = [make_candidate("dense-only"), make_candidate("shared")]
    sparse = [make_candidate("shared"), make_candidate("sparse-only")]
    service = HybridRetrievalService(
        embedding_provider=DebugEmbedding(),
        reranker=DebugReranker(),
        store=DebugStore(dense, sparse),
        sparse_encoder=BM25SparseEncoder.fit(["shared finance evidence"]),
        settings=RetrievalSettings(
            dense_candidates=2,
            sparse_candidates=2,
            fused_candidates=3,
            final_contexts=2,
        ),
    )

    trace = service.retrieve_debug("finance question")
    normal = service.retrieve("finance question")
    output = format_retrieval_trace("finance question", trace)

    assert [item.chunk.chunk_id for item in trace.dense] == ["dense-only", "shared"]
    assert [item.chunk.chunk_id for item in trace.sparse] == ["shared", "sparse-only"]
    assert trace.fused[0].chunk.chunk_id == "shared"
    assert len(trace.reranked) == 2
    assert normal.chunks == trace.reranked
    assert "Dense HNSW/ANN results" in output
    assert "Sparse BM25 results" in output
    assert "RRF-fused candidates" in output
    assert "Final reranked results" in output
    assert "chunk_id=shared" in output


def test_warmup_reuses_the_service_providers(make_candidate: Callable[..., RetrievedChunk]) -> None:
    embedding = DebugEmbedding()
    reranker = DebugReranker()
    service = HybridRetrievalService(
        embedding_provider=embedding,
        reranker=reranker,
        store=DebugStore([make_candidate("one")], []),
        sparse_encoder=BM25SparseEncoder.fit(["consumer finance evidence"]),
        settings=RetrievalSettings(),
    )
    messages: list[str] = []

    warm_up_retrieval_service(service, progress=messages.append)

    assert embedding.query_calls == 1
    assert reranker.calls == 1
    assert messages[-1] == "Embedding and reranker warm-up succeeded"
