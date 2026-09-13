"""End-to-end plain-Python retrieval-augmented generation service."""

from __future__ import annotations

from consumer_finance_rag.context import ContextBuilder
from consumer_finance_rag.models import ModelMetadata, RAGResponse
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
    ) -> None:
        self.retrieval = retrieval
        self.context_builder = context_builder
        self.llm = llm
        self.embedding_provider = embedding_provider
        self.reranker = reranker

    def answer(self, question: str) -> RAGResponse:
        retrieval_result = self.retrieval.retrieve(question)
        context = self.context_builder.build(retrieval_result.chunks)
        result = self.llm.generate(SYSTEM_PROMPT, build_user_prompt(question, context.text))
        return RAGResponse(
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
        )
