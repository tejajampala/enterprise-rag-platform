"""Hybrid retrieval orchestration: HNSW + BM25 + RRF + reranking."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from time import perf_counter
from typing import Protocol
from uuid import uuid4

from qdrant_client import models

from consumer_finance_rag.access import (
    AccessContext,
    access_policy_from_payload,
    build_qdrant_access_filter,
    is_access_allowed,
)
from consumer_finance_rag.config import RetrievalSettings
from consumer_finance_rag.models import RetrievalMetadata, RetrievedChunk
from consumer_finance_rag.observability import (
    NoOpRetrievalTelemetrySink,
    NoOpTracer,
    PIITypeCount,
    RetrievalACLMode,
    RetrievalFailureStage,
    RetrievalMetrics,
    RetrievalStatus,
    RetrievalTelemetryContext,
    RetrievalTelemetrySink,
    Tracer,
    chunk_summaries,
)
from consumer_finance_rag.pii import (
    DeterministicPIIDetector,
    PIIHandlingPolicy,
    PIIType,
)
from consumer_finance_rag.providers import EmbeddingProvider, RerankerProvider
from consumer_finance_rag.sparse import BM25SparseEncoder


class HybridSearchStore(Protocol):
    def search_dense(
        self,
        vector: list[float],
        limit: int,
        *,
        query_filter: models.Filter | None = None,
    ) -> list[RetrievedChunk]: ...

    def search_sparse(
        self,
        query: str,
        encoder: BM25SparseEncoder,
        limit: int,
        *,
        query_filter: models.Filter | None = None,
    ) -> list[RetrievedChunk]: ...


class AuthorizationInvariantError(RuntimeError):
    """Raised when filtered managed retrieval returns malformed/unauthorized data."""

    def __init__(self, message: str, *, rejected_candidate_count: int = 1) -> None:
        super().__init__(message)
        self.rejected_candidate_count = rejected_candidate_count


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[RetrievedChunk]], *, rrf_k: int = 60
) -> list[RetrievedChunk]:
    """Fuse rankings by chunk ID, deduplicating candidates across retrieval paths."""

    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive")
    by_id: dict[str, RetrievedChunk] = {}
    scores: dict[str, float] = {}
    dense_ranks: dict[str, int] = {}
    sparse_ranks: dict[str, int] = {}
    for list_index, ranked in enumerate(ranked_lists):
        for rank, candidate in enumerate(ranked, start=1):
            chunk_id = candidate.chunk.chunk_id
            by_id.setdefault(chunk_id, candidate)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)
            if candidate.dense_rank is not None or list_index == 0:
                dense_ranks.setdefault(chunk_id, candidate.dense_rank or rank)
            if candidate.sparse_rank is not None or list_index == 1:
                sparse_ranks.setdefault(chunk_id, candidate.sparse_rank or rank)

    fused = [
        RetrievedChunk(
            chunk=candidate.chunk,
            score=scores[chunk_id],
            dense_rank=dense_ranks.get(chunk_id),
            sparse_rank=sparse_ranks.get(chunk_id),
        )
        for chunk_id, candidate in by_id.items()
    ]
    return sorted(fused, key=lambda item: (-item.score, item.chunk.chunk_id))


@dataclass(frozen=True)
class RetrievalResult:
    chunks: list[RetrievedChunk]
    metadata: RetrievalMetadata


@dataclass(frozen=True)
class RetrievalTrace:
    """Intermediate rankings exposed for local debugging and inspection."""

    dense: list[RetrievedChunk]
    sparse: list[RetrievedChunk]
    fused: list[RetrievedChunk]
    reranked: list[RetrievedChunk]
    metadata: RetrievalMetadata


class HybridRetrievalService:
    """The single public retrieval entry point for the RAG pipeline."""

    def __init__(
        self,
        *,
        embedding_provider: EmbeddingProvider,
        reranker: RerankerProvider,
        store: HybridSearchStore,
        sparse_encoder: BM25SparseEncoder,
        settings: RetrievalSettings,
        tracer: Tracer | None = None,
        collection_name: str | None = None,
        access_control_enabled: bool = False,
        pii_handling_enabled: bool = False,
        pii_detector: DeterministicPIIDetector | None = None,
        telemetry_context: RetrievalTelemetryContext | None = None,
        telemetry_sink: RetrievalTelemetrySink | None = None,
    ) -> None:
        self.embedding_provider = embedding_provider
        self.reranker = reranker
        self.store = store
        self.sparse_encoder = sparse_encoder
        self.settings = settings
        self.tracer = tracer or NoOpTracer()
        self.collection_name = collection_name
        self.access_control_enabled = access_control_enabled
        self.pii_handling_enabled = pii_handling_enabled
        self.pii_detector = pii_detector or DeterministicPIIDetector()
        self.telemetry_context = telemetry_context or RetrievalTelemetryContext(
            profile="fixture",
            qdrant_collection=collection_name,
        )
        self.telemetry_sink = telemetry_sink or NoOpRetrievalTelemetrySink()

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        *,
        access_context: AccessContext | None = None,
        pii_policy: PIIHandlingPolicy | None = None,
    ) -> RetrievalResult:
        trace = self.retrieve_debug(
            query,
            top_k=top_k,
            access_context=access_context,
            pii_policy=pii_policy,
        )
        return RetrievalResult(chunks=trace.reranked, metadata=trace.metadata)

    def retrieve_debug(
        self,
        query: str,
        top_k: int | None = None,
        *,
        access_context: AccessContext | None = None,
        pii_policy: PIIHandlingPolicy | None = None,
    ) -> RetrievalTrace:
        """Run the production retrieval path while retaining each ranked stage."""

        started = perf_counter()
        request_id = str(uuid4())
        query_timestamp = datetime.now(UTC)
        resolved_access = access_context or AccessContext()
        resolved_pii_policy = pii_policy or (
            PIIHandlingPolicy.REDACT
            if self.pii_handling_enabled
            else PIIHandlingPolicy.ALLOW
        )
        acl_mode = (
            RetrievalACLMode.FILTER_AND_VALIDATE
            if self.access_control_enabled
            else RetrievalACLMode.DISABLED
        )
        dense_latency: float | None = None
        sparse_latency: float | None = None
        fusion_latency: float | None = None
        reranker_latency: float | None = None
        pii_latency: float | None = None
        dense_count = 0
        sparse_count = 0
        fused_count = 0
        final_count = 0
        rejected_count = 0
        redacted_count = 0
        detected_types: tuple[PIITypeCount, ...] = ()
        redacted_types: tuple[PIITypeCount, ...] = ()
        failure_stage = RetrievalFailureStage.OTHER

        def completed_metrics(
            status: RetrievalStatus,
            *,
            error_stage: RetrievalFailureStage | None = None,
            error_type: str | None = None,
        ) -> RetrievalMetrics:
            return RetrievalMetrics(
                request_id=request_id,
                profile=self.telemetry_context.profile,
                index_state_id=self.telemetry_context.index_state_id,
                generation=self.telemetry_context.generation,
                qdrant_collection=self.telemetry_context.qdrant_collection,
                query_timestamp=query_timestamp,
                total_retrieval_latency_ms=_elapsed_ms(started),
                dense_search_latency_ms=dense_latency,
                sparse_search_latency_ms=sparse_latency,
                fusion_latency_ms=fusion_latency,
                reranker_latency_ms=reranker_latency,
                pii_processing_latency_ms=pii_latency,
                dense_candidate_count=dense_count,
                sparse_candidate_count=sparse_count,
                fused_candidate_count=fused_count,
                final_context_count=final_count,
                acl_mode=acl_mode,
                acl_filter_applied=self.access_control_enabled,
                request_group_count=len(resolved_access.groups),
                rejected_candidate_count=rejected_count,
                pii_policy=resolved_pii_policy.value,
                redacted_context_count=redacted_count,
                detected_pii_type_counts=detected_types,
                redacted_pii_type_counts=redacted_types,
                reranker_enabled=True,
                status=status,
                error_stage=error_stage,
                error_type=error_type,
            )

        try:
            final_contexts = self.settings.final_contexts if top_k is None else top_k
            if final_contexts <= 0:
                raise ValueError("top_k must be positive")
            failure_stage = RetrievalFailureStage.ACL
            query_filter = (
                build_qdrant_access_filter(resolved_access)
                if self.access_control_enabled
                else None
            )
            failure_stage = RetrievalFailureStage.DENSE
            with self.tracer.span(
                "embed_query",
                span_type="EMBEDDING",
                attributes={"embedding.model": self.embedding_provider.model_name},
            ) as span:
                span.set_inputs({"query_characters": len(query)})
                query_vector = self.embedding_provider.embed_query(query)
                span.set_outputs({"dimensions": len(query_vector)})

            with self.tracer.span(
                "dense_hnsw_search",
                span_type="RETRIEVER",
                attributes={
                    "retrieval.mode": "dense_hnsw_ann",
                    "qdrant.collection": self.collection_name,
                    "candidate_limit": self.settings.dense_candidates,
                },
            ) as span:
                span.set_inputs({"query_characters": len(query)})
                stage_started = perf_counter()
                try:
                    dense = self.store.search_dense(
                        query_vector,
                        self.settings.dense_candidates,
                        query_filter=query_filter,
                    )
                finally:
                    dense_latency = _elapsed_ms(stage_started)
                dense_count = len(dense)
                failure_stage = RetrievalFailureStage.ACL
                self._validate_authorized(dense, resolved_access, stage="dense")
                span.set_outputs(chunk_summaries(dense))

            failure_stage = RetrievalFailureStage.SPARSE
            with self.tracer.span(
                "sparse_bm25_search",
                span_type="RETRIEVER",
                attributes={
                    "retrieval.mode": "sparse_bm25",
                    "qdrant.collection": self.collection_name,
                    "candidate_limit": self.settings.sparse_candidates,
                    "bm25.k1": self.sparse_encoder.state.k1,
                    "bm25.b": self.sparse_encoder.state.b,
                },
            ) as span:
                span.set_inputs({"query_characters": len(query)})
                stage_started = perf_counter()
                try:
                    sparse = self.store.search_sparse(
                        query,
                        self.sparse_encoder,
                        self.settings.sparse_candidates,
                        query_filter=query_filter,
                    )
                finally:
                    sparse_latency = _elapsed_ms(stage_started)
                sparse_count = len(sparse)
                failure_stage = RetrievalFailureStage.ACL
                self._validate_authorized(sparse, resolved_access, stage="sparse")
                span.set_outputs(chunk_summaries(sparse))

            failure_stage = RetrievalFailureStage.FUSION
            with self.tracer.span(
                "rrf_fusion",
                span_type="CHAIN",
                attributes={
                    "retrieval.mode": "reciprocal_rank_fusion",
                    "rrf.k": self.settings.rrf_k,
                },
            ) as span:
                span.set_inputs(
                    {
                        "dense_candidate_count": len(dense),
                        "sparse_candidate_count": len(sparse),
                    }
                )
                stage_started = perf_counter()
                try:
                    fused = reciprocal_rank_fusion(
                        [dense, sparse],
                        rrf_k=self.settings.rrf_k,
                    )
                    candidates = fused[: self.settings.fused_candidates]
                finally:
                    fusion_latency = _elapsed_ms(stage_started)
                fused_count = len(candidates)
                span.set_outputs(chunk_summaries(candidates))

            failure_stage = RetrievalFailureStage.RERANKER
            with self.tracer.span(
                "rerank",
                span_type="RERANKER",
                attributes={
                    "reranker.model": self.reranker.model_name,
                    "candidate_count": len(candidates),
                    "final_context_limit": final_contexts,
                },
            ) as span:
                span.set_inputs({"candidate_count": len(candidates)})
                stage_started = perf_counter()
                try:
                    reranked = self.reranker.rerank(
                        query,
                        candidates,
                        final_contexts,
                    )
                finally:
                    reranker_latency = _elapsed_ms(stage_started)
                unredacted_final = reranked[:final_contexts]
                failure_stage = RetrievalFailureStage.ACL
                self._validate_authorized(
                    unredacted_final,
                    resolved_access,
                    stage="reranker",
                )
                failure_stage = RetrievalFailureStage.PII
                stage_started = perf_counter()
                try:
                    (
                        final,
                        redacted_count,
                        detected_types,
                        redacted_types,
                    ) = self._apply_pii_policy(
                        unredacted_final,
                        resolved_pii_policy,
                    )
                finally:
                    pii_latency = _elapsed_ms(stage_started)
                final_count = len(final)
                span.set_outputs(chunk_summaries(final))
            result = RetrievalTrace(
                dense=dense,
                sparse=sparse,
                fused=candidates,
                reranked=final,
                metadata=RetrievalMetadata(
                    dense_candidates=len(dense),
                    sparse_candidates=len(sparse),
                    fused_candidates=len(candidates),
                    reranked=len(candidates),
                    final_contexts=len(final),
                ),
            )
        except Exception as exc:
            if isinstance(exc, AuthorizationInvariantError):
                failure_stage = RetrievalFailureStage.ACL
                rejected_count += exc.rejected_candidate_count
            self._emit_telemetry(
                completed_metrics(
                    RetrievalStatus.ERROR,
                    error_stage=failure_stage,
                    error_type=type(exc).__name__,
                )
            )
            raise
        self._emit_telemetry(completed_metrics(RetrievalStatus.SUCCESS))
        return result

    def _validate_authorized(
        self,
        candidates: Sequence[RetrievedChunk],
        access_context: AccessContext,
        *,
        stage: str,
    ) -> None:
        if not self.access_control_enabled:
            return
        rejected = 0
        first_issue: str | None = None
        for candidate in candidates:
            try:
                policy = access_policy_from_payload(candidate.chunk.metadata)
            except (TypeError, ValueError):
                rejected += 1
                first_issue = first_issue or (
                    f"Managed {stage} result {candidate.chunk.chunk_id!r} "
                    "has invalid ACL metadata"
                )
                continue
            if not is_access_allowed(policy, access_context):
                rejected += 1
                first_issue = first_issue or (
                    f"Managed {stage} result {candidate.chunk.chunk_id!r} "
                    "violated ACL filter"
                )
        if rejected:
            raise AuthorizationInvariantError(
                first_issue or f"Managed {stage} results violated ACL validation",
                rejected_candidate_count=rejected,
            )

    def _apply_pii_policy(
        self,
        candidates: Sequence[RetrievedChunk],
        policy: PIIHandlingPolicy,
    ) -> tuple[
        list[RetrievedChunk],
        int,
        tuple[PIITypeCount, ...],
        tuple[PIITypeCount, ...],
    ]:
        if policy is PIIHandlingPolicy.ALLOW:
            return list(candidates), 0, (), ()
        counts: Counter[PIIType] = Counter()
        redacted_contexts = 0
        redacted: list[RetrievedChunk] = []
        for candidate in candidates:
            matches = self.pii_detector.detect(candidate.chunk.text)
            counts.update(match.pii_type for match in matches)
            if matches:
                redacted_contexts += 1
            redacted.append(
                candidate.model_copy(
                    update={
                        "chunk": candidate.chunk.model_copy(
                            update={
                                "text": self.pii_detector.redact(candidate.chunk.text)
                            }
                        )
                    }
                )
            )
        type_counts = tuple(
            PIITypeCount(pii_type=pii_type.value, count=count)
            for pii_type, count in sorted(
                counts.items(),
                key=lambda item: item[0].value,
            )
        )
        return redacted, redacted_contexts, type_counts, type_counts

    def _emit_telemetry(self, metrics: RetrievalMetrics) -> None:
        """Keep telemetry failures from changing retrieval behavior or exceptions."""

        try:
            self.telemetry_sink.emit(metrics)
        except Exception:
            return


def _elapsed_ms(started: float) -> float:
    return max(0.0, (perf_counter() - started) * 1000.0)
