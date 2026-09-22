"""End-to-end plain-Python retrieval-augmented generation service."""

from __future__ import annotations

from consumer_finance_rag.access import AccessContext
from consumer_finance_rag.context import ContextBuilder
from consumer_finance_rag.models import ModelMetadata, RAGResponse
from consumer_finance_rag.observability import NoOpTracer, Tracer
from consumer_finance_rag.pii import PIIHandlingPolicy
from consumer_finance_rag.prompts import SYSTEM_PROMPT, build_user_prompt
from consumer_finance_rag.providers import EmbeddingProvider, LLMProvider, RerankerProvider
from consumer_finance_rag.retrieval import HybridRetrievalService


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
    ) -> RAGResponse:
        attributes: dict[str, object] = {
            "embedding.model": self.embedding_provider.model_name,
            "reranker.model": self.reranker.model_name,
            "generation.model": self.llm.model_name,
            "retrieval.mode": "dense_hnsw_ann+sparse_bm25+rrf+rerank",
            **self.trace_attributes,
        }
        with self.tracer.span("rag_request", span_type="CHAIN", attributes=attributes) as span:
            span.set_inputs({"query_characters": len(question)})
            retrieval_result = self.retrieval.retrieve(
                question,
                access_context=access_context,
                pii_policy=pii_policy,
            )
            context = self.context_builder.build(retrieval_result.chunks)
            result = self.llm.generate(SYSTEM_PROMPT, build_user_prompt(question, context.text))
            response = RAGResponse(
                question=question,
                answer=result.text,
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
            )
            span.set_attributes(
                {
                    "retrieval.dense_candidates": response.retrieval_metadata.dense_candidates,
                    "retrieval.sparse_candidates": response.retrieval_metadata.sparse_candidates,
                    "retrieval.fused_candidates": response.retrieval_metadata.fused_candidates,
                    "retrieval.final_contexts": response.retrieval_metadata.final_contexts,
                }
            )
            span.set_outputs(
                {
                    "answer_characters": len(response.answer),
                    "citation_count": len(response.citations),
                }
            )
            return response
