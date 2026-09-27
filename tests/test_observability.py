from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import pytest
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
        self.types: list[tuple[str, str]] = []

    @contextmanager
    def span(
        self,
        name: str,
        *,
        span_type: str,
        attributes: dict[str, Any] | None = None,
        parent_span: TraceSpan | None = None,
    ) -> Iterator[TraceSpan]:
        del attributes, parent_span
        self.names.append(name)
        self.types.append((name, span_type))
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
    assert ("dense_hnsw_search", "CHAIN") in tracer.types
    assert ("sparse_bm25_search", "CHAIN") in tracer.types
    assert ("context_build", "CHAIN") in tracer.types
    assert "[CONTENT OMITTED FROM TELEMETRY]" not in recorded


def test_approved_public_context_uses_genuine_retriever_document_schema(
    make_candidate: Callable[..., RetrievedChunk],
) -> None:
    candidate = make_candidate("chunk-approved")
    tracer = RecordingTracer(TraceContentPolicy.APPROVED_PUBLIC_EVALUATION)

    ContextBuilder(1000, tracer=tracer).build([candidate])

    outputs = next(
        value
        for name, field, value in tracer.records
        if name == "context_build" and field == "outputs"
    )
    assert ("context_build", "RETRIEVER") in tracer.types
    assert outputs == [
        {
            "id": candidate.chunk.chunk_id,
            "page_content": candidate.chunk.text,
            "metadata": {
                "rank": 1,
                "document_id": candidate.chunk.document_id,
                "title": str(candidate.chunk.metadata.get("title") or "Untitled source"),
            },
        }
    ]
    assert "[CONTENT OMITTED FROM TELEMETRY]" not in repr(outputs)


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


def test_mlflow_ragas_root_uses_bounded_client_id_and_preserves_full_configuration(
    monkeypatch: Any,
) -> None:
    import mlflow

    updates: list[dict[str, str]] = []
    starts: list[dict[str, Any]] = []
    span = RecordingSpan("ragas_evaluation", [])

    @contextmanager
    def fake_start_span(**kwargs: Any) -> Iterator[RecordingSpan]:
        starts.append(kwargs)
        yield span

    monkeypatch.setattr(mlflow, "set_tracking_uri", lambda _value: None)
    monkeypatch.setattr(mlflow, "set_experiment", lambda _value: None)
    monkeypatch.setattr(mlflow, "start_span", fake_start_span)
    monkeypatch.setattr(mlflow, "update_current_trace", lambda **values: updates.append(values))
    tracer = MLflowTracer(MLflowSettings())
    client_id = "12345678-1234-5678-1234-567812345678"
    configuration_id = "a" * 64

    with tracer.span(
        "ragas_evaluation",
        span_type="CHAIN",
        attributes={
            "evaluation.client_request_id": client_id,
            "evaluation.configuration_id": configuration_id,
        },
    ):
        pass

    assert len(client_id) == 36
    assert updates[0]["client_request_id"] == client_id
    assert starts[0]["attributes"]["evaluation.configuration_id"] == configuration_id


def test_mlflow_client_request_id_is_validated_before_span_export(
    monkeypatch: Any,
) -> None:
    import mlflow

    starts: list[dict[str, Any]] = []

    @contextmanager
    def fake_start_span(**kwargs: Any) -> Iterator[RecordingSpan]:
        starts.append(kwargs)
        yield RecordingSpan("ragas_evaluation", [])

    monkeypatch.setattr(mlflow, "set_tracking_uri", lambda _value: None)
    monkeypatch.setattr(mlflow, "set_experiment", lambda _value: None)
    monkeypatch.setattr(mlflow, "start_span", fake_start_span)
    tracer = MLflowTracer(MLflowSettings())

    with pytest.raises(ValueError, match="1 to 50"), tracer.span(
        "ragas_evaluation",
        span_type="CHAIN",
        attributes={"evaluation.client_request_id": "f" * 64},
    ):
        pass

    assert starts == []


def test_mlflow_trace_verification_flushes_and_reads_exact_trace(
    monkeypatch: Any,
) -> None:
    import mlflow

    flushed: list[bool] = []
    seen: list[str] = []
    client_id = "12345678-1234-5678-1234-567812345678"
    trace = type(
        "Trace",
        (),
        {
            "info": type(
                "Info", (), {"client_request_id": client_id, "state": "OK"}
            )(),
            "data": type(
                "Data",
                (),
                {
                    "spans": [
                        type(
                            "Span",
                            (),
                            {
                                "name": "ragas_evaluation",
                                "span_id": "root",
                                "parent_id": None,
                            },
                        )(),
                        type(
                            "Span",
                            (),
                            {
                                "name": "ragas_metric:faithfulness",
                                "span_id": "metric",
                                "parent_id": "root",
                            },
                        )(),
                        type(
                            "Span",
                            (),
                            {
                                "name": "bedrock_provider_call",
                                "span_id": "provider",
                                "parent_id": "metric",
                            },
                        )(),
                    ]
                },
            )(),
        },
    )()

    class Client:
        def get_trace(self, trace_id: str, display: bool = False) -> Any:
            assert display is False
            seen.append(trace_id)
            return trace

    monkeypatch.setattr(mlflow, "set_tracking_uri", lambda _value: None)
    monkeypatch.setattr(mlflow, "set_experiment", lambda _value: None)
    monkeypatch.setattr(
        mlflow, "flush_trace_async_logging", lambda: flushed.append(True)
    )
    monkeypatch.setattr(mlflow, "MlflowClient", lambda **_kwargs: Client())
    tracer = MLflowTracer(MLflowSettings())

    result = tracer.verify_trace(
        "tr-exact-evaluation",
        expected_client_request_id=client_id,
        expected_span_count=3,
        expected_metric_names=("faithfulness",),
        expected_provider_call_count=1,
    )

    assert flushed == [True]
    assert seen == ["tr-exact-evaluation"]
    assert result.status == "CONFIRMED"
    assert result.persisted_span_count == 3


def test_mlflow_trace_verification_separates_sync_delay_from_export_failure(
    monkeypatch: Any,
) -> None:
    import mlflow

    class MissingClient:
        def get_trace(self, _trace_id: str, display: bool = False) -> Any:
            assert display is False
            raise RuntimeError("RESOURCE_DOES_NOT_EXIST")

    class FailedClient:
        def get_trace(self, _trace_id: str, display: bool = False) -> Any:
            assert display is False
            raise RuntimeError("StringDataRightTruncation")

    monkeypatch.setattr(mlflow, "set_tracking_uri", lambda _value: None)
    monkeypatch.setattr(mlflow, "set_experiment", lambda _value: None)
    monkeypatch.setattr(mlflow, "flush_trace_async_logging", lambda: None)
    tracer = MLflowTracer(MLflowSettings())
    client_id = "12345678-1234-5678-1234-567812345678"

    monkeypatch.setattr(mlflow, "MlflowClient", lambda **_kwargs: MissingClient())
    pending = tracer.verify_trace(
        "tr-pending",
        expected_client_request_id=client_id,
        expected_span_count=3,
    )
    monkeypatch.setattr(mlflow, "MlflowClient", lambda **_kwargs: FailedClient())
    failed = tracer.verify_trace(
        "tr-failed",
        expected_client_request_id=client_id,
        expected_span_count=3,
    )

    assert pending.status == "PENDING"
    assert failed.status == "FAILED"
    assert "StringDataRightTruncation" not in failed.reason


def test_mlflow_trace_verification_rejects_persisted_root_without_children(
    monkeypatch: Any,
) -> None:
    import mlflow

    client_id = "12345678-1234-5678-1234-567812345678"
    trace = type(
        "Trace",
        (),
        {
            "info": type(
                "Info", (), {"client_request_id": client_id, "state": "OK"}
            )(),
            "data": type(
                "Data",
                (),
                {
                    "spans": [
                        type(
                            "Span",
                            (),
                            {
                                "name": "ragas_evaluation",
                                "span_id": "root-only",
                                "parent_id": None,
                            },
                        )()
                    ]
                },
            )(),
        },
    )()

    class Client:
        def get_trace(self, _trace_id: str, display: bool = False) -> Any:
            assert display is False
            return trace

    monkeypatch.setattr(mlflow, "set_tracking_uri", lambda _value: None)
    monkeypatch.setattr(mlflow, "set_experiment", lambda _value: None)
    monkeypatch.setattr(mlflow, "flush_trace_async_logging", lambda: None)
    monkeypatch.setattr(mlflow, "MlflowClient", lambda **_kwargs: Client())
    tracer = MLflowTracer(MLflowSettings())

    result = tracer.verify_trace(
        "tr-root-only",
        expected_client_request_id=client_id,
        expected_span_count=9,
        expected_metric_names=(
            "faithfulness",
            "context_precision",
            "context_recall",
            "response_groundedness",
        ),
        expected_provider_call_count=4,
    )

    assert result.status == "HIERARCHY_INCOMPLETE"
    assert result.persisted_span_count == 1
    assert "missing expected evaluation child spans" in result.reason


def test_mlflow_trace_verification_rejects_five_span_metric_only_trace(
    monkeypatch: Any,
) -> None:
    import mlflow

    client_id = "12345678-1234-5678-1234-567812345678"
    root = type(
        "Span",
        (),
        {"name": "ragas_evaluation", "span_id": "root", "parent_id": None},
    )()
    metric_names = (
        "faithfulness",
        "context_precision",
        "context_recall",
        "response_groundedness",
    )
    metrics = [
        type(
            "Span",
            (),
            {
                "name": f"ragas_metric:{name}",
                "span_id": f"metric-{name}",
                "parent_id": "root",
            },
        )()
        for name in metric_names
    ]
    trace = type(
        "Trace",
        (),
        {
            "info": type(
                "Info", (), {"client_request_id": client_id, "state": "OK"}
            )(),
            "data": type("Data", (), {"spans": [root, *metrics]})(),
        },
    )()

    class Client:
        def get_trace(self, _trace_id: str, display: bool = False) -> Any:
            assert display is False
            return trace

    monkeypatch.setattr(mlflow, "set_tracking_uri", lambda _value: None)
    monkeypatch.setattr(mlflow, "set_experiment", lambda _value: None)
    monkeypatch.setattr(mlflow, "flush_trace_async_logging", lambda: None)
    monkeypatch.setattr(mlflow, "MlflowClient", lambda **_kwargs: Client())
    tracer = MLflowTracer(MLflowSettings())

    result = tracer.verify_trace(
        "tr-five-spans",
        expected_client_request_id=client_id,
        expected_span_count=5,
        expected_metric_names=metric_names,
        expected_provider_call_count=0,
    )

    assert result.status == "HIERARCHY_INCOMPLETE"
    assert result.persisted_span_count == 5
    assert result.persisted_provider_call_count == 0
    assert "No LiteLLM completion invocation" in result.reason
