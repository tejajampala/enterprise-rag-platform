from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any

from qdrant_client import models

from consumer_finance_rag.bedrock import BedrockLLMProvider
from consumer_finance_rag.config import MLflowSettings, RetrievalSettings
from consumer_finance_rag.context import ContextBuilder
from consumer_finance_rag.models import RetrievedChunk
from consumer_finance_rag.observability import (
    MLflowTracer,
    TraceContentPolicy,
    TraceSpan,
    bedrock_token_usage,
)
from consumer_finance_rag.pipeline import RAGPipeline
from consumer_finance_rag.retrieval import HybridRetrievalService
from consumer_finance_rag.sparse import BM25SparseEncoder


class RecordingSpan:
    def __init__(self, name: str, records: list[tuple[str, str, Any]]) -> None:
        self.name = name
        self.records = records

    def set_inputs(self, inputs: Any) -> None:
        self.records.append((self.name, "inputs", inputs))

    def set_outputs(self, outputs: Any) -> None:
        self.records.append((self.name, "outputs", outputs))

    def set_attribute(self, key: str, value: Any) -> None:
        self.records.append((self.name, key, value))

    def set_attributes(self, attributes: dict[str, Any]) -> None:
        for key, value in attributes.items():
            self.set_attribute(key, value)


class RecordingTracer:
    def __init__(
        self,
        content_policy: TraceContentPolicy = TraceContentPolicy.METADATA_ONLY,
    ) -> None:
        self.content_policy = content_policy
        self.names: list[str] = []
        self.records: list[tuple[str, str, Any]] = []
        self.stack: list[str] = []
        self.parents: list[tuple[str, str | None]] = []

    @contextmanager
    def span(
        self,
        name: str,
        *,
        span_type: str,
        attributes: dict[str, Any] | None = None,
    ) -> Iterator[TraceSpan]:
        del span_type, attributes
        self.names.append(name)
        self.parents.append((name, self.stack[-1] if self.stack else None))
        self.stack.append(name)
        try:
            yield RecordingSpan(name, self.records)
        finally:
            self.stack.pop()


class FakeEmbedding:
    model_name = "embedding"

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0]


class FakeReranker:
    model_name = "reranker"

    def rerank(
        self, query: str, candidates: Sequence[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        return list(candidates[:top_k])


class FakeStore:
    def __init__(self, candidate: RetrievedChunk) -> None:
        self.candidate = candidate

    def search_dense(
        self,
        vector: list[float],
        limit: int,
        *,
        query_filter: models.Filter | None = None,
    ) -> list[RetrievedChunk]:
        del query_filter
        return [self.candidate.model_copy(update={"dense_rank": 1})]

    def search_sparse(
        self,
        query: str,
        encoder: BM25SparseEncoder,
        limit: int,
        *,
        query_filter: models.Filter | None = None,
    ) -> list[RetrievedChunk]:
        del query_filter
        return [self.candidate.model_copy(update={"sparse_rank": 1})]


class FakeBedrockClient:
    def converse(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "output": {"message": {"content": [{"text": "Grounded [C1]."}]}},
            "usage": {"inputTokens": 4, "outputTokens": 3, "totalTokens": 7},
        }


def test_tracing_preserves_output_and_has_expected_hierarchy(
    make_candidate: Callable[..., RetrievedChunk],
) -> None:
    candidate = make_candidate("chunk-1")
    tracer = RecordingTracer()
    embedding = FakeEmbedding()
    reranker = FakeReranker()
    retrieval = HybridRetrievalService(
        embedding_provider=embedding,
        reranker=reranker,
        store=FakeStore(candidate),
        sparse_encoder=BM25SparseEncoder.fit([candidate.chunk.text]),
        settings=RetrievalSettings(final_contexts=1),
        tracer=tracer,
    )
    llm = BedrockLLMProvider(
        region="us-east-1",
        model_id="generation",
        max_tokens=10,
        temperature=0.0,
        top_p=1.0,
        client=FakeBedrockClient(),
        tracer=tracer,
    )
    response = RAGPipeline(
        retrieval=retrieval,
        context_builder=ContextBuilder(1000, tracer=tracer),
        llm=llm,
        embedding_provider=embedding,
        reranker=reranker,
        tracer=tracer,
    ).answer("Question?")

    assert response.answer == "Grounded [C1]."
    assert tracer.names == [
        "rag_request",
        "embed_query",
        "dense_hnsw_search",
        "sparse_bm25_search",
        "rrf_fusion",
        "rerank",
        "context_build",
        "bedrock_generation",
    ]
    assert tracer.parents == [
        ("rag_request", None),
        ("embed_query", "rag_request"),
        ("dense_hnsw_search", "rag_request"),
        ("sparse_bm25_search", "rag_request"),
        ("rrf_fusion", "rag_request"),
        ("rerank", "rag_request"),
        ("context_build", "rag_request"),
        ("bedrock_generation", "rag_request"),
    ]
    recorded = repr(tracer.records)
    assert "Question?" not in recorded
    assert candidate.chunk.text not in recorded
    assert "Grounded [C1]." not in recorded
    assert ("bedrock_generation", "outputs", {"response_characters": 14}) in tracer.records


def test_bedrock_usage_maps_to_response_and_mlflow_token_attribute() -> None:
    tracer = RecordingTracer()
    result = BedrockLLMProvider(
        region="us-east-1",
        model_id="model",
        max_tokens=10,
        temperature=0.0,
        top_p=1.0,
        client=FakeBedrockClient(),
        tracer=tracer,
    ).generate("system", "user")

    expected = {"input_tokens": 4, "output_tokens": 3, "total_tokens": 7}
    assert bedrock_token_usage(result.response_metadata) == expected
    assert ("bedrock_generation", "mlflow.chat.tokenUsage", expected) in tracer.records
    assert bedrock_token_usage({"usage": {"inputTokens": 4}}) is None


def test_bedrock_text_trace_output_requires_approved_public_evaluation_policy() -> None:
    tracer = RecordingTracer(TraceContentPolicy.APPROVED_PUBLIC_EVALUATION)

    BedrockLLMProvider(
        region="us-east-1",
        model_id="model",
        max_tokens=10,
        temperature=0.0,
        top_p=1.0,
        client=FakeBedrockClient(),
        tracer=tracer,
    ).generate("system", "user")

    assert (
        "bedrock_generation",
        "outputs",
        {"response": "Grounded [C1]."},
    ) in tracer.records


def test_mlflow_root_span_uses_its_request_id_for_direct_trace_correlation(
    monkeypatch: Any,
) -> None:
    import mlflow

    updates: list[dict[str, str]] = []
    span = RecordingSpan("rag_request", [])
    span.trace_id = "trace-for-this-request"  # type: ignore[attr-defined]

    @contextmanager
    def fake_start_span(**_kwargs: Any) -> Iterator[RecordingSpan]:
        yield span

    monkeypatch.setattr(mlflow, "set_tracking_uri", lambda _value: None)
    monkeypatch.setattr(mlflow, "set_experiment", lambda _value: None)
    monkeypatch.setattr(mlflow, "start_span", fake_start_span)
    monkeypatch.setattr(mlflow, "update_current_trace", lambda **values: updates.append(values))
    tracer = MLflowTracer(MLflowSettings())

    with tracer.span(
        "rag_request",
        span_type="CHAIN",
        attributes={"request.id": "request-for-this-visitor"},
    ) as active:
        assert active.trace_id == "trace-for-this-request"  # type: ignore[attr-defined]

    assert updates == [
        {
            "client_request_id": "request-for-this-visitor",
            "request_preview": "[QUESTION OMITTED]",
            "response_preview": "[ANSWER OMITTED]",
        }
    ]
