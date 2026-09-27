"""Composition root: configuration selects providers without changing RAG logic."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal
from uuid import UUID

from sqlalchemy.orm import Session

from consumer_finance_rag.bedrock import BedrockLLMProvider
from consumer_finance_rag.config import AppSettings
from consumer_finance_rag.context import ContextBuilder
from consumer_finance_rag.corpus import evaluation_runtime_settings
from consumer_finance_rag.lineage.database import (
    create_database_engine,
    require_database_url,
)
from consumer_finance_rag.lineage.managed_retrieval import (
    ManagedRetrievalRuntime,
    build_managed_retrieval_runtime,
)
from consumer_finance_rag.lineage.repository import LineageRepository
from consumer_finance_rag.observability import (
    RetrievalTelemetryContext,
    RetrievalTelemetrySink,
    TraceContentPolicy,
    Tracer,
    build_tracer,
)
from consumer_finance_rag.pipeline import RAGPipeline
from consumer_finance_rag.providers import (
    HuggingFaceEmbeddingProvider,
    HuggingFaceRerankerProvider,
)
from consumer_finance_rag.retrieval import HybridRetrievalService
from consumer_finance_rag.sparse import BM25SparseEncoder
from consumer_finance_rag.vector_store import QdrantHybridStore

WarmupProgress = Callable[[str], None]
RuntimeProfile = Literal["fixture", "evaluation", "managed"]


@dataclass(frozen=True, slots=True)
class RetrievalRuntime:
    """Small profile-neutral runtime identity plus its pinned retrieval service."""

    profile: RuntimeProfile
    retrieval_service: HybridRetrievalService
    qdrant_collection: str
    generation: int | None = None
    index_state_id: UUID | None = None
    managed_runtime: ManagedRetrievalRuntime | None = None

    def display_metadata(self) -> tuple[tuple[str, str], ...]:
        """Return only metadata appropriate for unobtrusive UI display."""

        values: list[tuple[str, str]] = [("Retrieval profile", self.profile)]
        if self.profile == "managed":
            if self.generation is None or self.index_state_id is None:
                raise ValueError("Managed retrieval metadata is incomplete")
            values.extend(
                [
                    ("Generation", f"g{self.generation:06d}"),
                    ("Collection", self.qdrant_collection),
                ]
            )
        return tuple(values)


@dataclass(frozen=True, slots=True)
class ApplicationRuntime:
    """Generation pipeline paired with its explicit retrieval runtime identity."""

    retrieval: RetrievalRuntime
    pipeline: RAGPipeline


def build_retrieval_service(
    settings: AppSettings,
    *,
    bm25_state_path: str | Path = "data/index/bm25_state.json",
    tracer: Tracer | None = None,
    telemetry_sink: RetrievalTelemetrySink | None = None,
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
        telemetry_context=RetrievalTelemetryContext(
            profile=(
                "evaluation"
                if settings.runtime.profile == "evaluation"
                else "fixture"
            ),
            qdrant_collection=settings.qdrant.collection_name,
        ),
        telemetry_sink=telemetry_sink,
    )


def build_runtime_retrieval(
    settings: AppSettings,
    *,
    tracer: Tracer | None = None,
    telemetry_sink: RetrievalTelemetrySink | None = None,
) -> RetrievalRuntime:
    """Dispatch explicitly without changing legacy fixture/evaluation builders."""

    active_tracer = tracer or build_tracer(settings.mlflow)
    profile = settings.runtime.profile
    if profile == "fixture":
        if telemetry_sink is None:
            service = build_retrieval_service(settings, tracer=active_tracer)
        else:
            service = build_retrieval_service(
                settings,
                tracer=active_tracer,
                telemetry_sink=telemetry_sink,
            )
        return RetrievalRuntime(
            profile="fixture",
            retrieval_service=service,
            qdrant_collection=settings.qdrant.collection_name,
        )
    if profile == "evaluation":
        evaluation_settings = evaluation_runtime_settings(settings)
        if telemetry_sink is None:
            service = build_retrieval_service(
                evaluation_settings,
                bm25_state_path=settings.evaluation.bm25_state_path,
                tracer=active_tracer,
            )
        else:
            service = build_retrieval_service(
                evaluation_settings,
                bm25_state_path=settings.evaluation.bm25_state_path,
                tracer=active_tracer,
                telemetry_sink=telemetry_sink,
            )
        return RetrievalRuntime(
            profile="evaluation",
            retrieval_service=service,
            qdrant_collection=settings.evaluation.qdrant_collection_name,
        )

    require_database_url(settings.database)
    engine = create_database_engine(settings.database)
    try:
        with Session(engine) as session:
            managed = build_managed_retrieval_runtime(
                settings,
                LineageRepository(session),
                tracer=active_tracer,
                telemetry_sink=telemetry_sink,
            )
    finally:
        engine.dispose()
    return RetrievalRuntime(
        profile="managed",
        retrieval_service=managed.retrieval_service,
        qdrant_collection=managed.qdrant_collection,
        generation=managed.generation,
        index_state_id=managed.index_state_id,
        managed_runtime=managed,
    )


def warm_up_retrieval_service(
    service: HybridRetrievalService,
    *,
    progress: WarmupProgress | None = None,
) -> None:
    """Initialize the service's configured models once through their production interfaces."""

    notify = progress or (lambda message: None)
    query = "consumer finance model warm-up"
    notify(f"Loading embedding model once: {service.embedding_provider.model_name}")
    vector = service.embedding_provider.embed_query(query)
    notify("Embedding model initialized; loading one candidate from Qdrant")
    candidates = service.store.search_dense(vector, 1)
    if not candidates:
        raise RuntimeError("Model warm-up requires at least one indexed Qdrant chunk")
    notify(f"Loading reranker model once: {service.reranker.model_name}")
    service.reranker.rerank(query, candidates, 1)
    notify("Embedding and reranker warm-up succeeded")


def build_pipeline(
    settings: AppSettings, *, bm25_state_path: str | Path = "data/index/bm25_state.json"
) -> RAGPipeline:
    tracer = build_tracer(settings.mlflow)
    retrieval = build_retrieval_service(settings, bm25_state_path=bm25_state_path, tracer=tracer)
    return _compose_pipeline(
        settings,
        retrieval,
        tracer=tracer,
        qdrant_collection=settings.qdrant.collection_name,
        hnsw_m=settings.qdrant.hnsw_m,
        hnsw_ef_construct=settings.qdrant.hnsw_ef_construct,
    )


def build_application_runtime(
    settings: AppSettings,
    *,
    retrieval_runtime: RetrievalRuntime | None = None,
    telemetry_sink: RetrievalTelemetrySink | None = None,
    trace_content_policy: TraceContentPolicy = TraceContentPolicy.METADATA_ONLY,
) -> ApplicationRuntime:
    """Build the explicitly selected retrieval profile and backend-agnostic RAG pipeline."""

    tracer = (
        build_tracer(settings.mlflow)
        if trace_content_policy is TraceContentPolicy.METADATA_ONLY
        else build_tracer(settings.mlflow, content_policy=trace_content_policy)
    )
    if retrieval_runtime is None:
        runtime = build_runtime_retrieval(
            settings,
            tracer=tracer,
            telemetry_sink=telemetry_sink,
        )
    else:
        service = _retrieval_with_tracer(
            retrieval_runtime.retrieval_service,
            tracer,
            telemetry_sink=telemetry_sink,
        )
        managed_runtime = retrieval_runtime.managed_runtime
        if managed_runtime is not None:
            managed_runtime = replace(managed_runtime, retrieval_service=service)
        runtime = replace(
            retrieval_runtime,
            retrieval_service=service,
            managed_runtime=managed_runtime,
        )
    managed = runtime.profile == "managed"
    pipeline = _compose_pipeline(
        settings,
        runtime.retrieval_service,
        tracer=tracer,
        qdrant_collection=runtime.qdrant_collection,
        hnsw_m=(settings.managed_index.hnsw_m if managed else settings.qdrant.hnsw_m),
        hnsw_ef_construct=(
            settings.managed_index.hnsw_ef_construct
            if managed
            else settings.qdrant.hnsw_ef_construct
        ),
    )
    return ApplicationRuntime(retrieval=runtime, pipeline=pipeline)


def _retrieval_with_tracer(
    service: HybridRetrievalService,
    tracer: Tracer,
    *,
    telemetry_sink: RetrievalTelemetrySink | None = None,
) -> HybridRetrievalService:
    """Reuse one pinned backend/model set while enabling request-time tracing."""

    return HybridRetrievalService(
        embedding_provider=service.embedding_provider,
        reranker=service.reranker,
        store=service.store,
        sparse_encoder=service.sparse_encoder,
        settings=service.settings,
        tracer=tracer,
        collection_name=service.collection_name,
        access_control_enabled=service.access_control_enabled,
        pii_handling_enabled=service.pii_handling_enabled,
        pii_detector=service.pii_detector,
        telemetry_context=service.telemetry_context,
        telemetry_sink=telemetry_sink or service.telemetry_sink,
        inspection_sink=service.inspection_sink,
    )


def _compose_pipeline(
    settings: AppSettings,
    retrieval: HybridRetrievalService,
    *,
    tracer: Tracer,
    qdrant_collection: str,
    hnsw_m: int,
    hnsw_ef_construct: int,
) -> RAGPipeline:
    """Inject a resolved retriever without exposing its persistence backend to RAG."""

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
            "qdrant.collection": qdrant_collection,
            "qdrant.hnsw_m": hnsw_m,
            "qdrant.hnsw_ef_construct": hnsw_ef_construct,
            "retrieval.dense_candidate_limit": settings.retrieval.dense_candidates,
            "retrieval.sparse_candidate_limit": settings.retrieval.sparse_candidates,
            "retrieval.fused_candidate_limit": settings.retrieval.fused_candidates,
            "retrieval.final_context_limit": settings.retrieval.final_contexts,
            "retrieval.rrf_k": settings.retrieval.rrf_k,
        },
    )
