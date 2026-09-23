from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import FrozenInstanceError
from typing import Any
from uuid import uuid4

import pytest
from qdrant_client import models

from consumer_finance_rag.access import (
    AccessContext,
    AccessVisibility,
    DocumentAccessPolicy,
)
from consumer_finance_rag.config import RetrievalSettings
from consumer_finance_rag.models import Chunk, RetrievedChunk
from consumer_finance_rag.observability import (
    RetrievalFailureStage,
    RetrievalMetrics,
    RetrievalStatus,
    RetrievalTelemetryContext,
    StructuredLoggingRetrievalTelemetrySink,
)
from consumer_finance_rag.pii import DeterministicPIIDetector, PIIHandlingPolicy
from consumer_finance_rag.retrieval import (
    AuthorizationInvariantError,
    HybridRetrievalService,
)
from consumer_finance_rag.sparse import BM25SparseEncoder


class RecordingSink:
    def __init__(self) -> None:
        self.events: list[RetrievalMetrics] = []

    def emit(self, metrics: RetrievalMetrics) -> None:
        self.events.append(metrics)


class FakeEmbedding:
    model_name = "telemetry-embedding"

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, float(bool(text))]


class FakeReranker:
    model_name = "telemetry-reranker"

    def __init__(self, failure: Exception | None = None) -> None:
        self.failure = failure

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        del query
        if self.failure is not None:
            raise self.failure
        return list(candidates[:top_k])


class FakeStore:
    def __init__(
        self,
        dense: Sequence[RetrievedChunk],
        sparse: Sequence[RetrievedChunk] | None = None,
        *,
        dense_failure: Exception | None = None,
        sparse_failure: Exception | None = None,
    ) -> None:
        self.dense = list(dense)
        self.sparse = list(sparse if sparse is not None else dense)
        self.dense_failure = dense_failure
        self.sparse_failure = sparse_failure
        self.dense_calls = 0
        self.sparse_calls = 0

    def search_dense(
        self,
        vector: list[float],
        limit: int,
        *,
        query_filter: models.Filter | None = None,
    ) -> list[RetrievedChunk]:
        del vector, query_filter
        self.dense_calls += 1
        if self.dense_failure is not None:
            raise self.dense_failure
        return self.dense[:limit]

    def search_sparse(
        self,
        query: str,
        encoder: BM25SparseEncoder,
        limit: int,
        *,
        query_filter: models.Filter | None = None,
    ) -> list[RetrievedChunk]:
        del query, encoder, query_filter
        self.sparse_calls += 1
        if self.sparse_failure is not None:
            raise self.sparse_failure
        return self.sparse[:limit]


def candidate(
    chunk_id: str,
    text: str = "Consumer finance evidence",
    *,
    policy: DocumentAccessPolicy | None = None,
) -> RetrievedChunk:
    access = policy or DocumentAccessPolicy()
    return RetrievedChunk(
        chunk=Chunk(
            chunk_id=chunk_id,
            document_id=f"document-{chunk_id}",
            document_version="v1",
            text=text,
            ordinal=0,
            chunk_hash="a" * 64,
            metadata={
                "access_visibility": access.visibility.value,
                "allowed_groups": list(access.canonical_groups),
            },
        ),
        score=1.0,
    )


def service(
    store: FakeStore,
    sink: Any,
    context: RetrievalTelemetryContext,
    *,
    reranker: FakeReranker | None = None,
    acl: bool = False,
    pii: bool = False,
    pii_detector: DeterministicPIIDetector | None = None,
) -> HybridRetrievalService:
    return HybridRetrievalService(
        embedding_provider=FakeEmbedding(),
        reranker=reranker or FakeReranker(),
        store=store,
        sparse_encoder=BM25SparseEncoder.fit(["Consumer finance evidence"]),
        settings=RetrievalSettings(
            dense_candidates=5,
            sparse_candidates=5,
            fused_candidates=5,
            final_contexts=2,
        ),
        collection_name=context.qdrant_collection,
        access_control_enabled=acl,
        pii_handling_enabled=pii,
        pii_detector=pii_detector or DeterministicPIIDetector(),
        telemetry_context=context,
        telemetry_sink=sink,
    )


def test_managed_trace_contains_pinned_identity_counts_and_latencies() -> None:
    sink = RecordingSink()
    state_id = uuid4()
    context = RetrievalTelemetryContext(
        profile="managed",
        index_state_id=state_id,
        generation=7,
        qdrant_collection="consumer_finance_managed_g000007",
    )
    first = candidate("chunk-1")
    second = candidate("chunk-2")
    store = FakeStore([first, second], [second])
    retrieval = service(store, sink, context)

    result = retrieval.retrieve("dispute evidence")

    assert len(result.chunks) == 2
    assert len(sink.events) == 1
    event = sink.events[0]
    assert event.profile == "managed"
    assert event.index_state_id == state_id
    assert event.generation == 7
    assert event.qdrant_collection == "consumer_finance_managed_g000007"
    assert event.dense_candidate_count == 2
    assert event.sparse_candidate_count == 1
    assert event.fused_candidate_count == 2
    assert event.final_context_count == 2
    assert event.reranker_enabled is True
    assert event.status is RetrievalStatus.SUCCESS
    for latency in (
        event.total_retrieval_latency_ms,
        event.dense_search_latency_ms,
        event.sparse_search_latency_ms,
        event.fusion_latency_ms,
        event.reranker_latency_ms,
        event.pii_processing_latency_ms,
    ):
        assert latency is not None and latency >= 0
    with pytest.raises(FrozenInstanceError):
        event.generation = 8  # type: ignore[misc]


@pytest.mark.parametrize("profile", ["fixture", "evaluation"])
def test_nonmanaged_trace_has_no_fabricated_generation(profile: str) -> None:
    sink = RecordingSink()
    context = RetrievalTelemetryContext(
        profile=profile,
        qdrant_collection=f"consumer_finance_{profile}",
    )

    service(FakeStore([candidate("chunk-1")]), sink, context).retrieve("evidence")

    event = sink.events[0]
    assert event.profile == profile
    assert event.generation is None
    assert event.index_state_id is None


def test_acl_trace_records_mode_group_count_without_group_names() -> None:
    sink = RecordingSink()
    context = RetrievalTelemetryContext(
        profile="managed",
        index_state_id=uuid4(),
        generation=3,
        qdrant_collection="consumer_finance_managed_g000003",
    )
    retrieval = service(FakeStore([candidate("public")]), sink, context, acl=True)

    retrieval.retrieve(
        "evidence",
        access_context=AccessContext(
            principal_id="principal-secret",
            groups=frozenset({"fraud-analysts", "audit-reviewers"}),
        ),
    )

    event = sink.events[0]
    assert event.acl_filter_applied is True
    assert event.request_group_count == 2
    assert event.rejected_candidate_count == 0
    serialized = str(event.as_log_record())
    assert "principal-secret" not in serialized
    assert "fraud-analysts" not in serialized
    assert "audit-reviewers" not in serialized


def test_acl_rejections_are_counted_and_original_exception_propagates() -> None:
    sink = RecordingSink()
    restricted = DocumentAccessPolicy(
        visibility=AccessVisibility.RESTRICTED,
        allowed_groups=frozenset({"restricted-team"}),
    )
    retrieval = service(
        FakeStore(
            [
                candidate("restricted-1", policy=restricted),
                candidate("restricted-2", policy=restricted),
            ]
        ),
        sink,
        RetrievalTelemetryContext(
            profile="managed",
            index_state_id=uuid4(),
            generation=4,
            qdrant_collection="consumer_finance_managed_g000004",
        ),
        acl=True,
    )

    with pytest.raises(AuthorizationInvariantError) as captured:
        retrieval.retrieve(
            "restricted evidence",
            access_context=AccessContext(groups=frozenset({"different-team"})),
        )

    assert captured.value.rejected_candidate_count == 2
    event = sink.events[0]
    assert event.status is RetrievalStatus.ERROR
    assert event.error_stage is RetrievalFailureStage.ACL
    assert event.rejected_candidate_count == 2


def test_pii_redaction_trace_contains_only_aggregate_type_counts() -> None:
    sink = RecordingSink()
    raw = "Email alice@example.com or call +61 2 9374 4000."
    retrieval = service(
        FakeStore([candidate("pii", raw)]),
        sink,
        RetrievalTelemetryContext(
            profile="managed",
            index_state_id=uuid4(),
            generation=5,
            qdrant_collection="consumer_finance_managed_g000005",
        ),
        acl=True,
        pii=True,
    )

    result = retrieval.retrieve("contact", pii_policy=PIIHandlingPolicy.REDACT)

    assert "alice@example.com" not in result.chunks[0].chunk.text
    event = sink.events[0]
    assert event.pii_policy == "REDACT"
    assert event.redacted_context_count == 1
    detected = {item.pii_type: item.count for item in event.detected_pii_type_counts}
    redacted = {item.pii_type: item.count for item in event.redacted_pii_type_counts}
    assert detected == {"EMAIL": 1, "PHONE": 1}
    assert redacted == detected
    serialized = str(event.as_log_record())
    assert "alice@example.com" not in serialized
    assert "+61 2 9374 4000" not in serialized


def test_structured_logging_sink_does_not_log_query_chunk_or_identity_secrets(
    caplog: pytest.LogCaptureFixture,
) -> None:
    logger = logging.getLogger("tests.retrieval.telemetry")
    caplog.set_level(logging.INFO, logger=logger.name)
    sink = StructuredLoggingRetrievalTelemetrySink(logger)
    raw_chunk = "Card 4111111111111111 belongs to bob@example.com"
    retrieval = service(
        FakeStore([candidate("safe-id", raw_chunk)]),
        sink,
        RetrievalTelemetryContext(
            profile="managed",
            index_state_id=uuid4(),
            generation=6,
            qdrant_collection="consumer_finance_managed_g000006",
        ),
        acl=True,
        pii=True,
    )
    raw_query = "Find bob@example.com for ultra-secret-group"

    retrieval.retrieve(
        raw_query,
        access_context=AccessContext(
            principal_id="principal@example.com",
            groups=frozenset({"ultra-secret-group"}),
        ),
    )

    logged = caplog.text
    assert "retrieval.completed" in logged
    assert raw_query not in logged
    assert raw_chunk not in logged
    assert "bob@example.com" not in logged
    assert "4111111111111111" not in logged
    assert "principal@example.com" not in logged
    assert "ultra-secret-group" not in logged


@pytest.mark.parametrize(
    ("failure_location", "expected_stage"),
    [
        ("dense", RetrievalFailureStage.DENSE),
        ("sparse", RetrievalFailureStage.SPARSE),
        ("reranker", RetrievalFailureStage.RERANKER),
    ],
)
def test_failure_trace_records_stage_and_preserves_original_exception(
    failure_location: str,
    expected_stage: RetrievalFailureStage,
) -> None:
    sink = RecordingSink()
    original = RuntimeError(f"injected {failure_location} failure")
    item = candidate("chunk-1")
    store = FakeStore(
        [item],
        dense_failure=original if failure_location == "dense" else None,
        sparse_failure=original if failure_location == "sparse" else None,
    )
    reranker = FakeReranker(
        failure=original if failure_location == "reranker" else None
    )
    retrieval = service(
        store,
        sink,
        RetrievalTelemetryContext(profile="fixture", qdrant_collection="fixture"),
        reranker=reranker,
    )

    with pytest.raises(RuntimeError) as captured:
        retrieval.retrieve("query text that must not be emitted")

    assert captured.value is original
    event = sink.events[0]
    assert event.status is RetrievalStatus.ERROR
    assert event.error_stage is expected_stage
    assert event.error_type == "RuntimeError"
    assert event.total_retrieval_latency_ms >= 0


def test_fusion_failure_is_classified_and_propagated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sink = RecordingSink()
    original = RuntimeError("injected fusion failure")

    def fail_fusion(*_args: object, **_kwargs: object) -> list[RetrievedChunk]:
        raise original

    monkeypatch.setattr(
        "consumer_finance_rag.retrieval.reciprocal_rank_fusion",
        fail_fusion,
    )
    retrieval = service(
        FakeStore([candidate("chunk-1")]),
        sink,
        RetrievalTelemetryContext(profile="fixture", qdrant_collection="fixture"),
    )

    with pytest.raises(RuntimeError) as captured:
        retrieval.retrieve("query")

    assert captured.value is original
    assert sink.events[0].error_stage is RetrievalFailureStage.FUSION


def test_pii_failure_is_classified_and_propagated() -> None:
    class FailingDetector(DeterministicPIIDetector):
        def redact(self, text: str) -> str:
            del text
            raise RuntimeError("injected PII failure")

    sink = RecordingSink()
    retrieval = service(
        FakeStore([candidate("chunk-1", "alice@example.com")]),
        sink,
        RetrievalTelemetryContext(
            profile="managed",
            index_state_id=uuid4(),
            generation=9,
            qdrant_collection="consumer_finance_managed_g000009",
        ),
        pii=True,
        pii_detector=FailingDetector(),
    )

    with pytest.raises(RuntimeError, match="injected PII failure"):
        retrieval.retrieve("query")

    assert sink.events[0].error_stage is RetrievalFailureStage.PII


def test_preflight_failure_is_classified_as_other() -> None:
    sink = RecordingSink()
    retrieval = service(
        FakeStore([candidate("chunk-1")]),
        sink,
        RetrievalTelemetryContext(profile="fixture", qdrant_collection="fixture"),
    )

    with pytest.raises(ValueError, match="top_k"):
        retrieval.retrieve("query", top_k=0)

    assert sink.events[0].error_stage is RetrievalFailureStage.OTHER


def test_telemetry_path_performs_searches_only_and_no_store_writes() -> None:
    sink = RecordingSink()
    store = FakeStore([candidate("chunk-1")])

    service(
        store,
        sink,
        RetrievalTelemetryContext(profile="evaluation", qdrant_collection="evaluation"),
    ).retrieve("evidence")

    assert store.dense_calls == 1
    assert store.sparse_calls == 1
    assert len(sink.events) == 1
