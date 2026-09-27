"""End-to-end plain-Python retrieval-augmented generation service."""

from __future__ import annotations

from collections.abc import Callable
from time import perf_counter
from uuid import uuid4

from consumer_finance_rag.access import AccessContext
from consumer_finance_rag.context import ContextBuilder
from consumer_finance_rag.models import ModelMetadata, RAGResponse, StructuredContext
from consumer_finance_rag.observability import (
    NoOpTracer,
    TraceContentPolicy,
    Tracer,
    TraceSpan,
    trace_id_from_span,
)
from consumer_finance_rag.pii import PIIHandlingPolicy
from consumer_finance_rag.prompts import SYSTEM_PROMPT, build_user_prompt
from consumer_finance_rag.providers import EmbeddingProvider, LLMProvider, RerankerProvider
from consumer_finance_rag.retrieval import HybridRetrievalService

PreResponseGate = Callable[[RAGResponse, TraceSpan], RAGResponse]
PreGenerationGate = Callable[[StructuredContext, TraceSpan], None]


class RAGPipeline:
    def __init__(
        self,
        *,
        retrieval: HybridRetrievalService,
        context_builder: ContextBuilder,
        llm: LLMProvider,
        embedding_provider: EmbeddingProvider,
        reranker: RerankerProvider,
        tracer: Tracer | None = None,
        trace_attributes: dict[str, object] | None = None,
    ) -> None:
        self.retrieval = retrieval
        self.context_builder = context_builder
        self.llm = llm
        self.embedding_provider = embedding_provider
        self.reranker = reranker
        self.tracer = tracer or NoOpTracer()
        self.trace_attributes = trace_attributes or {}

    def answer(
        self,
        question: str,
        *,
        access_context: AccessContext | None = None,
        pii_policy: PIIHandlingPolicy | None = None,
        request_id: str | None = None,
        pre_generation_gate: PreGenerationGate | None = None,
        pre_response_gate: PreResponseGate | None = None,
    ) -> RAGResponse:
        request_started = perf_counter()
        resolved_request_id = request_id or str(uuid4())
        attributes: dict[str, object] = {
            "request.id": resolved_request_id,
            "embedding.model": self.embedding_provider.model_name,
            "reranker.model": self.reranker.model_name,
            "generation.model": self.llm.model_name,
            "retrieval.mode": "dense_hnsw_ann+sparse_bm25+rrf+rerank",
            **self.trace_attributes,
        }
        with self.tracer.span("rag_request", span_type="CHAIN", attributes=attributes) as span:
            trace_id = trace_id_from_span(span)
            approved_content = (
                getattr(
                    self.tracer,
                    "content_policy",
                    TraceContentPolicy.METADATA_ONLY,
                )
                is TraceContentPolicy.APPROVED_PUBLIC_EVALUATION
            )
            span.set_inputs(
                {"question": question} if approved_content else {"query_characters": len(question)}
            )
            stage_started = perf_counter()
            retrieval_result = self.retrieval.retrieve(
                question,
                access_context=access_context,
                pii_policy=pii_policy,
                request_id=resolved_request_id,
            )
            retrieval_ms = _elapsed_ms(stage_started)
            stage_started = perf_counter()
            context = self.context_builder.build(retrieval_result.chunks)
            context_ms = _elapsed_ms(stage_started)
            if pre_generation_gate is not None:
                pre_generation_gate(context, span)
            stage_started = perf_counter()
            result = self.llm.generate(SYSTEM_PROMPT, build_user_prompt(question, context.text))
            generation_ms = _elapsed_ms(stage_started)
            response = RAGResponse(
                request_id=resolved_request_id,
                trace_id=trace_id,
                question=question,
                answer=result.text,
                structured_context=context,
                citations=context.citations,
                retrieved_chunks=context.chunks,
                retrieval_metadata=retrieval_result.metadata.model_copy(
                    update={"final_contexts": len(context.chunks)}
                ),
                model_metadata=ModelMetadata(
                    embedding_model=self.embedding_provider.model_name,
                    reranker_model=self.reranker.model_name,
                    generation_model=result.model_id,
                ),
                generation_metadata=result.response_metadata,
                stage_timings_ms={
                    "retrieval": retrieval_ms,
                    "context_build": context_ms,
                    "bedrock_generation": generation_ms,
                    "total": _elapsed_ms(request_started),
                },
            )
            if pre_response_gate is not None:
                response = pre_response_gate(response, span)
            span.set_attributes(
                {
                    "retrieval.dense_candidates": response.retrieval_metadata.dense_candidates,
                    "retrieval.sparse_candidates": response.retrieval_metadata.sparse_candidates,
                    "retrieval.fused_candidates": response.retrieval_metadata.fused_candidates,
                    "retrieval.final_contexts": response.retrieval_metadata.final_contexts,
                }
            )
            span.set_outputs(
                {"answer": response.answer}
                if approved_content
                else {
                    "answer_characters": len(response.answer),
                    "citation_count": len(response.citations),
                }
            )
            return response


def _elapsed_ms(started: float) -> float:
    return max(0.0, (perf_counter() - started) * 1000.0)
