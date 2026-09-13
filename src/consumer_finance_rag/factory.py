"""Composition root: configuration selects providers without changing RAG logic."""

from __future__ import annotations

from pathlib import Path

from consumer_finance_rag.bedrock import BedrockLLMProvider
from consumer_finance_rag.config import AppSettings
from consumer_finance_rag.context import ContextBuilder
from consumer_finance_rag.observability import Tracer, build_tracer
from consumer_finance_rag.pipeline import RAGPipeline
from consumer_finance_rag.providers import (
    HuggingFaceEmbeddingProvider,
    HuggingFaceRerankerProvider,
)
from consumer_finance_rag.retrieval import HybridRetrievalService
from consumer_finance_rag.sparse import BM25SparseEncoder
from consumer_finance_rag.vector_store import QdrantHybridStore


def build_retrieval_service(
    settings: AppSettings,
    *,
    bm25_state_path: str | Path = "data/index/bm25_state.json",
    tracer: Tracer | None = None,
) -> HybridRetrievalService:
    active_tracer = tracer or build_tracer(settings.mlflow)
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
        tracer=active_tracer,
        collection_name=settings.qdrant.collection_name,
    )


def build_pipeline(
    settings: AppSettings, *, bm25_state_path: str | Path = "data/index/bm25_state.json"
) -> RAGPipeline:
    tracer = build_tracer(settings.mlflow)
    retrieval = build_retrieval_service(settings, bm25_state_path=bm25_state_path, tracer=tracer)
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
        tracer=tracer,
    )
    return RAGPipeline(
        retrieval=retrieval,
        context_builder=ContextBuilder(settings.context.max_characters, tracer=tracer),
        llm=llm,
        embedding_provider=retrieval.embedding_provider,
        reranker=retrieval.reranker,
        tracer=tracer,
        trace_attributes={
            "qdrant.collection": settings.qdrant.collection_name,
            "qdrant.hnsw_m": settings.qdrant.hnsw_m,
            "qdrant.hnsw_ef_construct": settings.qdrant.hnsw_ef_construct,
            "retrieval.dense_candidate_limit": settings.retrieval.dense_candidates,
            "retrieval.sparse_candidate_limit": settings.retrieval.sparse_candidates,
            "retrieval.fused_candidate_limit": settings.retrieval.fused_candidates,
            "retrieval.final_context_limit": settings.retrieval.final_contexts,
            "retrieval.rrf_k": settings.retrieval.rrf_k,
        },
    )
