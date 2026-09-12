"""Composition root: configuration selects providers without changing RAG logic."""

from __future__ import annotations

from pathlib import Path

from consumer_finance_rag.bedrock import BedrockLLMProvider
from consumer_finance_rag.config import AppSettings
from consumer_finance_rag.context import ContextBuilder
from consumer_finance_rag.pipeline import RAGPipeline
from consumer_finance_rag.providers import (
    HuggingFaceEmbeddingProvider,
    HuggingFaceRerankerProvider,
)
from consumer_finance_rag.retrieval import HybridRetrievalService
from consumer_finance_rag.sparse import BM25SparseEncoder
from consumer_finance_rag.vector_store import QdrantHybridStore


def build_retrieval_service(
    settings: AppSettings, *, bm25_state_path: str | Path = "data/index/bm25_state.json"
) -> HybridRetrievalService:
    embedding = HuggingFaceEmbeddingProvider(
        settings.models.embedding.model,
        settings.models.embedding.device,
        settings.models.embedding.trust_remote_code,
        settings.models.embedding.query_prompt_name,
    )
    reranker = HuggingFaceRerankerProvider(
        settings.models.reranker.model,
        settings.models.reranker.device,
        settings.models.reranker.trust_remote_code,
    )
    sparse_encoder = BM25SparseEncoder.load(bm25_state_path)
    store = QdrantHybridStore(settings.qdrant)
    return HybridRetrievalService(
        embedding_provider=embedding,
        reranker=reranker,
        store=store,
        sparse_encoder=sparse_encoder,
        settings=settings.retrieval,
    )


def build_pipeline(
    settings: AppSettings, *, bm25_state_path: str | Path = "data/index/bm25_state.json"
) -> RAGPipeline:
    retrieval = build_retrieval_service(settings, bm25_state_path=bm25_state_path)
    llm = BedrockLLMProvider(
        region=settings.aws.region,
        model_id=settings.models.generation.model,
        max_tokens=settings.models.generation.max_tokens,
        temperature=settings.models.generation.temperature,
        top_p=settings.models.generation.top_p,
        connect_timeout_seconds=settings.aws.connect_timeout_seconds,
        read_timeout_seconds=settings.aws.read_timeout_seconds,
        retry_mode=settings.aws.retry_mode,
        total_max_attempts=settings.aws.total_max_attempts,
        tcp_keepalive=settings.aws.tcp_keepalive,
    )
    return RAGPipeline(
        retrieval=retrieval,
        context_builder=ContextBuilder(settings.context.max_characters),
        llm=llm,
        embedding_provider=retrieval.embedding_provider,
        reranker=retrieval.reranker,
    )
