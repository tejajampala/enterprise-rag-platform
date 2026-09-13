from __future__ import annotations

from collections.abc import Callable, Sequence

from consumer_finance_rag.config import RetrievalSettings
from consumer_finance_rag.context import ContextBuilder
from consumer_finance_rag.models import RetrievedChunk
from consumer_finance_rag.pipeline import RAGPipeline
from consumer_finance_rag.providers import LLMResult
from consumer_finance_rag.retrieval import HybridRetrievalService
from consumer_finance_rag.sparse import BM25SparseEncoder


class FakeEmbedding:
    model_name = "fake-embedding"

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0]


class FakeReranker:
    model_name = "fake-reranker"

    def rerank(
        self, query: str, candidates: Sequence[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        return list(candidates[:top_k])


class FakeStore:
    def __init__(self, candidate: RetrievedChunk) -> None:
        self.candidate = candidate

    def search_dense(self, vector: list[float], limit: int) -> list[RetrievedChunk]:
        return [self.candidate.model_copy(update={"dense_rank": 1})]

    def search_sparse(
        self, query: str, encoder: BM25SparseEncoder, limit: int
    ) -> list[RetrievedChunk]:
        return [self.candidate.model_copy(update={"sparse_rank": 1})]


class FakeLLM:
    model_name = "fake-generation"

    def generate(self, system_prompt: str, user_prompt: str) -> LLMResult:
        assert "[C1]" in user_prompt
        return LLMResult(text="Answer supported by evidence [C1].", model_id=self.model_name)


def test_pipeline_builds_citations_from_retrieved_chunks(
    make_candidate: Callable[..., RetrievedChunk],
) -> None:
    candidate = make_candidate("citation-chunk", title="Source title")
    embedding = FakeEmbedding()
    reranker = FakeReranker()
    retrieval = HybridRetrievalService(
        embedding_provider=embedding,
        reranker=reranker,
        store=FakeStore(candidate),
        sparse_encoder=BM25SparseEncoder.fit([candidate.chunk.text]),
        settings=RetrievalSettings(
            dense_candidates=2,
            sparse_candidates=2,
            fused_candidates=2,
            final_contexts=1,
        ),
    )
    response = RAGPipeline(
        retrieval=retrieval,
        context_builder=ContextBuilder(1000),
        llm=FakeLLM(),
        embedding_provider=embedding,
        reranker=reranker,
    ).answer("What is supported?")

    assert response.citations[0].citation_id == "C1"
    assert response.citations[0].chunk_id == "citation-chunk"
    assert response.retrieved_chunks[0].chunk.chunk_id == "citation-chunk"
    assert response.model_metadata.generation_model == "fake-generation"
