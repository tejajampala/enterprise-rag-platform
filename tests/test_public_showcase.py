from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
from qdrant_client import models

from consumer_finance_rag.config import (
    AppSettings,
    PublicShowcaseSettings,
)
from consumer_finance_rag.context import ContextBuilder
from consumer_finance_rag.factory import ApplicationRuntime, RetrievalRuntime
from consumer_finance_rag.models import Chunk, RetrievedChunk
from consumer_finance_rag.observability import (
    RetrievalTelemetryContext,
    TraceContentPolicy,
    TraceSpan,
)
from consumer_finance_rag.pipeline import RAGPipeline
from consumer_finance_rag.providers import LLMResult
from consumer_finance_rag.public_showcase import (
    PublicInferenceGate,
    PublicRAGService,
    PublicRequestError,
    PublicShowcaseConfigurationError,
    RequestTelemetryCollector,
    validate_public_showcase_runtime,
)
from consumer_finance_rag.retrieval import HybridRetrievalService
from consumer_finance_rag.sparse import BM25SparseEncoder


class FakeEmbedding:
    model_name = "test/embedding"

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0] for _text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(bool(text))]


class FakeReranker:
    model_name = "test/reranker"

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        del query
        return list(candidates[:top_k])


class PublicStore:
    def __init__(self, candidate: RetrievedChunk) -> None:
        self.candidate = candidate
        self.filters: list[models.Filter | None] = []
        self.write_count = 0

    def search_dense(
        self,
        vector: list[float],
        limit: int,
        *,
        query_filter: models.Filter | None = None,
    ) -> list[RetrievedChunk]:
        del vector
        self.filters.append(query_filter)
        return [self.candidate.model_copy(update={"dense_rank": 1})][:limit]

    def search_sparse(
        self,
        query: str,
        encoder: BM25SparseEncoder,
        limit: int,
        *,
        query_filter: models.Filter | None = None,
    ) -> list[RetrievedChunk]:
        del query, encoder
        self.filters.append(query_filter)
        return [self.candidate.model_copy(update={"sparse_rank": 1})][:limit]


class RecordingSpan:
    trace_id = "trace-request-bound"

    def set_inputs(self, inputs: Any) -> None:
        del inputs

    def set_outputs(self, outputs: Any) -> None:
        del outputs

    def set_attribute(self, key: str, value: Any) -> None:
        del key, value

    def set_attributes(self, attributes: dict[str, Any]) -> None:
        del attributes


class RequestBoundTracer:
    content_policy = TraceContentPolicy.METADATA_ONLY

    def __init__(self) -> None:
        self.root_attributes: dict[str, Any] | None = None

    @contextmanager
    def span(
        self,
        name: str,
        *,
        span_type: str,
        attributes: dict[str, Any] | None = None,
    ) -> Iterator[TraceSpan]:
        del span_type
        if name == "rag_request":
            self.root_attributes = attributes
        yield RecordingSpan()


class SafeLLM:
    model_name = "test/bedrock-generation"

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(self, system_prompt: str, user_prompt: str) -> LLMResult:
        del system_prompt
        self.prompts.append(user_prompt)
        assert "alice@example.com" not in user_prompt
        return LLMResult(
            text="Call +61 2 9374 4000 for an answer [C1].",
            model_id=self.model_name,
            response_metadata={
                "usage": {"inputTokens": 12, "outputTokens": 8, "totalTokens": 20}
            },
        )


class FailingLLM(SafeLLM):
    def generate(self, system_prompt: str, user_prompt: str) -> LLMResult:
        del system_prompt, user_prompt
        raise RuntimeError("credential=secret-value alice@example.com")


def _settings(
    *,
    profile: str = "managed",
    public: PublicShowcaseSettings | None = None,
) -> AppSettings:
    return AppSettings.model_validate(
        {
            "models": {
                "embedding": {"model": "test/embedding"},
                "reranker": {"model": "test/reranker"},
                "generation": {"model": "test/bedrock-generation"},
            },
            "runtime": {"profile": profile},
            "public_showcase": (
                public
                or PublicShowcaseSettings(
                    enabled=True,
                    inference_enabled=True,
                    max_requests_per_window=5,
                    max_concurrent_requests=1,
                )
            ).model_dump(),
        }
    )


def _candidate() -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(
            chunk_id="public-doc:v1:0:hash",
            document_id="public-doc:v1",
            document_version="v1",
            text="PUBLIC evidence. Contact alice@example.com for the old workflow.",
            ordinal=0,
            chunk_hash="a" * 64,
            metadata={
                "title": "Reviewed public source",
                "source_url": "https://example.test/public",
                "access_visibility": "PUBLIC",
                "allowed_groups": [],
                "contains_pii": True,
                "pii_types": ["EMAIL"],
                "pii_detector_version": "deterministic-pii-v1",
            },
        ),
        score=1.0,
    )


def _application(
    settings: AppSettings,
    telemetry: RequestTelemetryCollector,
    *,
    llm: SafeLLM | None = None,
) -> tuple[ApplicationRuntime, PublicStore, SafeLLM, RequestBoundTracer]:
    candidate = _candidate()
    store = PublicStore(candidate)
    tracer = RequestBoundTracer()
    embedding = FakeEmbedding()
    reranker = FakeReranker()
    retrieval = HybridRetrievalService(
        embedding_provider=embedding,
        reranker=reranker,
        store=store,
        sparse_encoder=BM25SparseEncoder.fit([candidate.chunk.text]),
        settings=settings.retrieval.model_copy(
            update={
                "dense_candidates": 2,
                "sparse_candidates": 2,
                "fused_candidates": 2,
                "final_contexts": 1,
            }
        ),
        tracer=tracer,
        collection_name="consumer_finance_managed_g000009",
        access_control_enabled=True,
        pii_handling_enabled=True,
        telemetry_context=RetrievalTelemetryContext(
            profile="managed",
            index_state_id=UUID(int=9),
            generation=9,
            qdrant_collection="consumer_finance_managed_g000009",
        ),
        telemetry_sink=telemetry,
    )
    generation = llm or SafeLLM()
    pipeline = RAGPipeline(
        retrieval=retrieval,
        context_builder=ContextBuilder(2000, tracer=tracer),
        llm=generation,
        embedding_provider=embedding,
        reranker=reranker,
        tracer=tracer,
    )
    return (
        ApplicationRuntime(
            retrieval=RetrievalRuntime(
                profile="managed",
                retrieval_service=retrieval,
                qdrant_collection="consumer_finance_managed_g000009",
                generation=9,
                index_state_id=UUID(int=9),
            ),
            pipeline=pipeline,
        ),
        store,
        generation,
        tracer,
    )


def _service(
    settings: AppSettings,
    *,
    llm: SafeLLM | None = None,
) -> tuple[PublicRAGService, PublicStore, SafeLLM, RequestBoundTracer]:
    telemetry = RequestTelemetryCollector()
    application, store, generation, tracer = _application(
        settings,
        telemetry,
        llm=llm,
    )
    return (
        PublicRAGService(
            settings,
            application,
            telemetry,
            PublicInferenceGate(settings.public_showcase),
        ),
        store,
        generation,
        tracer,
    )


def test_public_request_is_managed_anonymous_redacted_and_request_correlated() -> None:
    raw_question = "How do I contact alice@example.com?"
    service, store, llm, tracer = _service(_settings())

    result = service.answer(raw_question)

    assert result.trace.mlflow_trace_id == "trace-request-bound"
    assert tracer.root_attributes is not None
    assert tracer.root_attributes["request.id"] == result.trace.request_id
    assert result.trace.request_group_count == 0
    assert result.trace.acl_filter_applied is True
    assert result.trace.pii_policy == "REDACT"
    assert result.trace.redacted_context_count == 1
    assert result.trace.redacted_pii_type_counts == {"EMAIL": 1}
    assert all(query_filter is not None for query_filter in store.filters)
    assert "alice@example.com" not in llm.prompts[0]
    assert "+61 2 9374 4000" not in result.answer
    assert "[REDACTED:PHONE]" in result.answer
    assert result.token_usage == {
        "input_tokens": 12,
        "output_tokens": 8,
        "total_tokens": 20,
    }
    serialized = result.model_dump_json()
    assert raw_question not in serialized
    assert "alice@example.com" not in serialized
    assert "+61 2 9374 4000" not in serialized
    assert "access-group-secret" not in serialized
    assert store.write_count == 0


def test_public_runtime_rejects_legacy_profile_and_unapproved_corpus() -> None:
    settings = _settings(profile="fixture")
    telemetry = RequestTelemetryCollector()
    application, _store, _llm, _tracer = _application(settings, telemetry)

    with pytest.raises(PublicShowcaseConfigurationError, match="managed"):
        validate_public_showcase_runtime(settings, application)

    managed = _settings(
        public=PublicShowcaseSettings(
            enabled=True,
            inference_enabled=True,
            approved_corpus_key="different-corpus",
        )
    )
    application, _store, _llm, _tracer = _application(managed, telemetry)
    with pytest.raises(PublicShowcaseConfigurationError, match="not approved"):
        validate_public_showcase_runtime(managed, application)


def test_public_error_hides_internal_exception_details() -> None:
    service, _store, _llm, _tracer = _service(_settings(), llm=FailingLLM())

    with pytest.raises(PublicRequestError) as captured:
        service.answer("What should I do?")

    assert captured.value.code == "request_failed"
    assert "secret-value" not in captured.value.public_message
    assert "alice@example.com" not in captured.value.public_message


def test_question_length_and_kill_switch_fail_before_inference() -> None:
    disabled = _settings(
        public=PublicShowcaseSettings(
            enabled=True,
            inference_enabled=False,
            max_question_characters=20,
        )
    )
    service, store, _llm, _tracer = _service(disabled)

    with pytest.raises(PublicRequestError, match="20 characters"):
        service.answer("x" * 21)
    with pytest.raises(PublicRequestError) as captured:
        service.answer("short question")

    assert captured.value.code == "inference_disabled"
    assert store.filters == []


def test_process_wide_usage_and_concurrency_limits_are_bounded() -> None:
    settings = PublicShowcaseSettings(
        enabled=True,
        inference_enabled=True,
        max_requests_per_window=1,
        rate_window_seconds=60,
        max_concurrent_requests=1,
    )
    gate = PublicInferenceGate(settings, clock=lambda: 10.0)

    with gate.admit(), pytest.raises(PublicRequestError) as concurrent, gate.admit():
        pass
    assert concurrent.value.code == "capacity_reached"

    with pytest.raises(PublicRequestError) as limited, gate.admit():
        pass
    assert limited.value.code == "usage_limit_reached"


def test_public_ui_reads_reports_without_invoking_paid_evaluation() -> None:
    source = Path("app/streamlit_app.py").read_text(encoding="utf-8")

    assert "load_managed_showcase_report" in source
    assert "run_mlflow_judges" not in source
    assert "build_ragas_evaluator" not in source
    assert "evaluate_managed_rag(" not in source
    assert 'st.set_option("client.showErrorDetails", "none")' in source
