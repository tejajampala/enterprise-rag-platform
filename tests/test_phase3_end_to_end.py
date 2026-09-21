from __future__ import annotations

import math
import re
from collections.abc import Generator, Sequence
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import pytest
from qdrant_client import QdrantClient
from sqlalchemy import Engine, create_engine, event, func, select
from sqlalchemy.orm import Session, sessionmaker

from consumer_finance_rag.access import (
    AccessContext,
    AccessVisibility,
    DocumentAccessPolicy,
)
from consumer_finance_rag.acquisition import SourceRecord
from consumer_finance_rag.chunking import ParagraphChunker
from consumer_finance_rag.config import AppSettings, load_settings
from consumer_finance_rag.corpus import CorpusSnapshot
from consumer_finance_rag.lineage.chunk_materialization import (
    ChunkMaterializationService,
)
from consumer_finance_rag.lineage.index_types import (
    ActiveChunkManifest,
    BM25PlanAction,
)
from consumer_finance_rag.lineage.indexing import (
    ManagedIndexPublishOutcome,
    ManagedIndexPublishResult,
    ManagedIndexService,
)
from consumer_finance_rag.lineage.managed_retrieval import (
    ManagedRetrievalRuntime,
    build_managed_retrieval_runtime,
)
from consumer_finance_rag.lineage.managed_vector_store import ManagedQdrantStore
from consumer_finance_rag.lineage.models import (
    Base,
    Corpus,
    DocumentVersion,
    IndexState,
    IndexStateChunk,
    IngestionRun,
    IngestionRunDocument,
    LineageChunk,
    LineageDocument,
)
from consumer_finance_rag.lineage.repository import LineageRepository
from consumer_finance_rag.lineage.service import (
    LineageApplyResult,
    LineageApplyService,
    LineagePlanningService,
)
from consumer_finance_rag.models import RetrievedChunk
from consumer_finance_rag.normalization import normalize_record
from consumer_finance_rag.sparse import (
    BM25SparseEncoder,
    bm25_corpus_hash,
    bm25_state_from_json_dict,
    bm25_state_hash,
)

CORPUS_KEY = "consumer-finance-managed-test"
MANAGED_PREFIX = "consumer_finance_managed_test"
CHUNK_SIZE = 105
CHUNK_OVERLAP = 0
EMBEDDING_DIMENSION = 5


@pytest.fixture
def engine() -> Generator[Engine]:
    sqlite_engine = create_engine("sqlite+pysqlite:///:memory:")

    @event.listens_for(sqlite_engine, "connect")
    def enable_foreign_keys(dbapi_connection: Any, _record: Any) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(sqlite_engine)
    yield sqlite_engine
    sqlite_engine.dispose()


class DeterministicEmbeddingProvider:
    """Small lexical embedding used by both publication and query-time retrieval."""

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self.dimension = EMBEDDING_DIMENSION
        self.document_batches: list[tuple[str, ...]] = []
        self.query_calls: list[str] = []

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.document_batches.append(tuple(texts))
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self.query_calls.append(text)
        return self._embed(text)

    @property
    def embedded_document_count(self) -> int:
        return sum(len(batch) for batch in self.document_batches)

    @staticmethod
    def _embed(text: str) -> list[float]:
        lowered = text.lower()
        vector = [
            float(sum(lowered.count(term) for term in ("omega", "chargeback", "replacement"))),
            float(sum(lowered.count(term) for term in ("alpha", "dispute", "legacy"))),
            float(sum(lowered.count(term) for term in ("mortgage", "escrow", "servicer"))),
            float(sum(lowered.count(term) for term in ("savings", "overdraft", "account"))),
            0.25,
        ]
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector]


class DeterministicReranker:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    @property
    def model_name(self) -> str:
        return "test-token-overlap-reranker"

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        self.calls.append((query, tuple(item.chunk.chunk_id for item in candidates)))
        query_terms = set(re.findall(r"[a-z0-9]+", query.lower()))
        rescored = [
            item.model_copy(
                update={
                    "rerank_score": float(
                        len(
                            query_terms
                            & set(re.findall(r"[a-z0-9]+", item.chunk.text.lower()))
                        )
                    )
                }
            )
            for item in candidates
        ]
        return sorted(
            rescored,
            key=lambda item: (
                -(item.rerank_score or 0.0),
                item.chunk.chunk_id,
            ),
        )[:top_k]


@dataclass(frozen=True, slots=True)
class AppliedSnapshot:
    snapshot: CorpusSnapshot
    result: LineageApplyResult
    manifest: ActiveChunkManifest


def test_two_generation_managed_lifecycle_is_incremental_and_pinned(
    engine: Engine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bm25_fits: list[tuple[str, ...]] = []
    original_bm25_fit = BM25SparseEncoder.fit

    def recording_bm25_fit(
        cls: type[BM25SparseEncoder],
        texts: Sequence[str],
        *,
        k1: float = 1.2,
        b: float = 0.75,
    ) -> BM25SparseEncoder:
        del cls
        bm25_fits.append(tuple(texts))
        return original_bm25_fit(texts, k1=k1, b=b)

    monkeypatch.setattr(
        BM25SparseEncoder,
        "fit",
        classmethod(recording_bm25_fit),
    )
    sessions = sessionmaker(bind=engine, class_=Session, expire_on_commit=False)
    settings = build_test_settings()
    qdrant = QdrantClient(":memory:")
    publication_embedding = DeterministicEmbeddingProvider(
        settings.models.embedding.model
    )
    publisher = build_publisher(
        sessions,
        settings,
        qdrant,
        publication_embedding,
    )

    # Scenario 1: the real planner/apply path creates three version-1 documents.
    first = apply_snapshot(sessions, initial_records())
    assert first.result.status == "SUCCEEDED"
    assert first.result.new_count == 3
    assert first.result.changed_count == 0
    assert first.result.unchanged_count == 0
    assert first.manifest.document_count == 3
    assert first.manifest.chunk_count == len(first.snapshot.chunks)
    with sessions() as session:
        corpus = session.scalar(select(Corpus))
        assert corpus is not None
        assert corpus.corpus_key == CORPUS_KEY
        assert corpus.qdrant_collection_prefix == MANAGED_PREFIX
        assert session.scalar(select(func.count()).select_from(IngestionRun)) == 1
        assert session.scalar(select(func.count()).select_from(LineageDocument)) == 3
        assert session.scalar(select(func.count()).select_from(DocumentVersion)) == 3
        assert session.scalar(select(func.count()).select_from(LineageChunk)) == len(
            first.snapshot.chunks
        )
        assert set(session.scalars(select(DocumentVersion.version_number))) == {1}
        assert set(session.scalars(select(IngestionRunDocument.change_type))) == {"NEW"}
        assert session.scalar(select(func.count()).select_from(IndexState)) == 0

    # Scenario 2: generation 1 is fully computed and activated.
    generation_1 = publisher.publish(first.manifest, first.result.run_id)
    assert_published_generation(
        generation_1,
        expected_generation=1,
        expected_chunks=first.manifest.chunk_count,
    )
    assert generation_1.bm25_action is BM25PlanAction.REFIT
    assert generation_1.dense_computed_count == first.manifest.chunk_count
    assert generation_1.dense_reused_count == 0
    assert generation_1.collection == f"{MANAGED_PREFIX}_g000001"
    assert bm25_fits == [tuple(chunk.text for chunk in first.manifest.chunks)]
    assert_generation_invariants(sessions, qdrant, generation_1, "ACTIVE")
    with sessions() as session:
        memberships_g1 = membership_rows(session, generation_1.index_state_id)
        assert all(row.dense_action == "COMPUTED" for row in memberships_g1)
        assert all(row.dense_source_chunk_pk is None for row in memberships_g1)
        assert_exactly_one_active(session)

    # Scenario 3: resolve and query a generation-pinned production retriever.
    fits_before_runtime_g1 = len(bm25_fits)
    runtime_g1, runtime_g1_embedding, reranker_g1 = build_runtime(
        sessions,
        settings,
        qdrant,
    )
    assert len(bm25_fits) == fits_before_runtime_g1
    assert runtime_g1.generation == 1
    assert runtime_g1.qdrant_collection == f"{MANAGED_PREFIX}_g000001"
    state_g1 = runtime_g1.retrieval_service.sparse_encoder.state
    old_result = runtime_g1.retrieval_service.retrieve_debug("legacy alpha dispute", top_k=3)
    assert old_result.dense
    assert old_result.sparse
    assert old_result.fused
    assert old_result.reranked
    assert any("Legacy alpha dispute" in item.chunk.text for item in old_result.reranked)
    assert runtime_g1_embedding.query_calls == ["legacy alpha dispute"]
    assert reranker_g1.calls

    # Scenario 4: exactly one source changes through the same authoritative apply path.
    old_payments = next(
        document for document in first.snapshot.documents if document.source_id == "payments"
    )
    second = apply_snapshot(sessions, changed_records())
    assert second.result.status == "SUCCEEDED"
    assert second.result.new_count == 0
    assert second.result.changed_count == 1
    assert second.result.unchanged_count == 2
    assert second.result.deleted_count == 0
    with sessions() as session:
        classifications = classification_by_source(session, second.result.run_id)
        assert classifications == {
            "accounts": "UNCHANGED",
            "mortgages": "UNCHANGED",
            "payments": "CHANGED",
        }
        payments = session.scalar(
            select(LineageDocument).where(LineageDocument.source_id == "payments")
        )
        assert payments is not None
        versions = list(
            session.scalars(
                select(DocumentVersion)
                .where(DocumentVersion.document_pk == payments.document_pk)
                .order_by(DocumentVersion.version_number)
            )
        )
        assert [version.version_number for version in versions] == [1, 2]
        assert versions[0].normalized_text == old_payments.text
        assert versions[0].content_hash == old_payments.content_hash
        assert "omega chargeback replacement" in versions[1].normalized_text.lower()
        assert session.scalar(select(func.count()).select_from(DocumentVersion)) == 4
        state = session.get(IndexState, generation_1.index_state_id)
        assert state is not None and state.status == "ACTIVE"

    # Scenario 5: generation 2 refits BM25 and mixes reuse with fresh computation.
    embedded_before_g2 = publication_embedding.embedded_document_count
    generation_2 = publisher.publish(second.manifest, second.result.run_id)
    assert_published_generation(
        generation_2,
        expected_generation=2,
        expected_chunks=second.manifest.chunk_count,
    )
    assert generation_2.collection == f"{MANAGED_PREFIX}_g000002"
    assert generation_2.bm25_action is BM25PlanAction.REFIT
    exact_prior_texts = {(chunk.chunk_hash, chunk.text) for chunk in first.manifest.chunks}
    expected_reused = sum(
        (chunk.chunk_hash, chunk.text) in exact_prior_texts
        for chunk in second.manifest.chunks
    )
    assert 0 < expected_reused < second.manifest.chunk_count
    assert generation_2.dense_reused_count == expected_reused
    assert generation_2.dense_computed_count == second.manifest.chunk_count - expected_reused
    assert (
        generation_2.dense_reused_count + generation_2.dense_computed_count
        == second.manifest.chunk_count
    )
    assert (
        publication_embedding.embedded_document_count - embedded_before_g2
        == generation_2.dense_computed_count
    )
    assert bm25_fits == [
        tuple(chunk.text for chunk in first.manifest.chunks),
        tuple(chunk.text for chunk in second.manifest.chunks),
    ]
    assert qdrant.collection_exists(generation_1.collection)
    assert qdrant.collection_exists(generation_2.collection)
    assert_generation_invariants(sessions, qdrant, generation_1, "RETIRED")
    assert_generation_invariants(sessions, qdrant, generation_2, "ACTIVE")

    with sessions() as session:
        assert_exactly_one_active(session)
        memberships_g2 = membership_rows(session, generation_2.index_state_id)
        reused = [row for row in memberships_g2 if row.dense_action == "REUSED"]
        computed = [row for row in memberships_g2 if row.dense_action == "COMPUTED"]
        assert reused and computed
        assert all(row.dense_source_chunk_pk is not None for row in reused)
        assert all(row.dense_source_chunk_pk is None for row in computed)
        assert any(
            exact_reuse_crosses_versions(session, row)
            for row in reused
            if row.dense_source_chunk_pk is not None
        )
        assert session.scalar(select(func.count()).select_from(DocumentVersion)) == 4
        historical_chunk_count = (
            session.scalar(select(func.count()).select_from(LineageChunk)) or 0
        )
        assert historical_chunk_count > len(second.manifest.chunks)

    # Persisted BM25 state belongs to each immutable generation and validates exactly.
    with sessions() as session:
        persisted_g1 = session.get(IndexState, generation_1.index_state_id)
        persisted_g2 = session.get(IndexState, generation_2.index_state_id)
        assert persisted_g1 is not None and persisted_g2 is not None
        validate_bm25_state(persisted_g1, first.manifest)
        validate_bm25_state(persisted_g2, second.manifest)
        assert persisted_g1.bm25_state is not None
        assert persisted_g2.bm25_state is not None
        assert state_g1 == bm25_state_from_json_dict(persisted_g1.bm25_state)
        assert persisted_g1.bm25_corpus_hash != persisted_g2.bm25_corpus_hash

    # Scenario 6: a newly constructed runtime resolves generation 2 and changed text.
    fits_before_runtime_g2 = len(bm25_fits)
    runtime_g2, runtime_g2_embedding, reranker_g2 = build_runtime(
        sessions,
        settings,
        qdrant,
    )
    assert len(bm25_fits) == fits_before_runtime_g2
    assert runtime_g2.generation == 2
    assert runtime_g2.qdrant_collection == f"{MANAGED_PREFIX}_g000002"
    state_g2 = runtime_g2.retrieval_service.sparse_encoder.state
    assert state_g2.corpus_hash != state_g1.corpus_hash
    with sessions() as session:
        persisted_g2 = session.get(IndexState, generation_2.index_state_id)
        assert persisted_g2 is not None and persisted_g2.bm25_state is not None
        assert state_g2 == bm25_state_from_json_dict(persisted_g2.bm25_state)
    changed_result = runtime_g2.retrieval_service.retrieve_debug(
        "omega chargeback replacement",
        top_k=3,
    )
    assert changed_result.dense
    assert changed_result.sparse
    assert changed_result.fused
    assert changed_result.reranked
    assert any(
        "omega chargeback replacement" in item.chunk.text.lower()
        for item in changed_result.reranked
    )
    unchanged_result = runtime_g2.retrieval_service.retrieve_debug(
        "mortgage escrow servicer",
        top_k=3,
    )
    assert any(
        "mortgage servicer" in item.chunk.text.lower()
        for item in unchanged_result.reranked
    )
    assert runtime_g2_embedding.query_calls == [
        "omega chargeback replacement",
        "mortgage escrow servicer",
    ]
    assert reranker_g2.calls

    # Scenario 7: the first runtime and its generation-specific BM25 state stay pinned.
    assert runtime_g1.generation == 1
    assert runtime_g1.qdrant_collection == generation_1.collection
    assert runtime_g1.retrieval_service.sparse_encoder.state == state_g1
    assert runtime_g1.retrieval_service.sparse_encoder.state.corpus_hash != state_g2.corpus_hash
    pinned_result = runtime_g1.retrieval_service.retrieve_debug(
        "legacy alpha dispute",
        top_k=3,
    )
    assert any("Legacy alpha dispute" in item.chunk.text for item in pinned_result.reranked)
    assert all(
        "omega chargeback replacement" not in item.chunk.text.lower()
        for item in pinned_result.reranked
    )

    # Scenario 8: a true no-op neither allocates g3 nor performs embedding inference.
    embedding_before_noop = publication_embedding.embedded_document_count
    states_before_noop = index_state_count(sessions)
    noop = publisher.publish(second.manifest, second.result.run_id)
    assert noop.outcome is ManagedIndexPublishOutcome.NO_INDEX_CHANGE
    assert noop.index_state_id == generation_2.index_state_id
    assert noop.generation == 2
    assert noop.collection == generation_2.collection
    assert noop.bm25_action is BM25PlanAction.REUSE
    assert publication_embedding.embedded_document_count == embedding_before_noop
    assert len(bm25_fits) == fits_before_runtime_g2
    assert index_state_count(sessions) == states_before_noop == 2
    assert not qdrant.collection_exists(f"{MANAGED_PREFIX}_g000003")
    assert_generation_invariants(sessions, qdrant, generation_1, "RETIRED")
    assert_generation_invariants(sessions, qdrant, generation_2, "ACTIVE")
    with sessions() as session:
        assert_exactly_one_active(session)

    collection_names = {item.name for item in qdrant.get_collections().collections}
    assert collection_names == {
        f"{MANAGED_PREFIX}_g000001",
        f"{MANAGED_PREFIX}_g000002",
    }
    assert "consumer_finance_phase1" not in collection_names
    assert "consumer_finance_evaluation" not in collection_names


def test_acl_filtered_retrieval_and_acl_only_generation_are_pinned(
    engine: Engine,
) -> None:
    sessions = sessionmaker(bind=engine, class_=Session, expire_on_commit=False)
    settings = build_test_settings()
    qdrant = QdrantClient(":memory:")
    embedding = DeterministicEmbeddingProvider(settings.models.embedding.model)
    publisher = build_publisher(sessions, settings, qdrant, embedding)
    restricted_fraud = DocumentAccessPolicy(
        visibility=AccessVisibility.RESTRICTED,
        allowed_groups=frozenset({"fraud-analysts", "audit-reviewers"}),
    )
    restricted_service = DocumentAccessPolicy(
        visibility=AccessVisibility.RESTRICTED,
        allowed_groups=frozenset({"customer-service"}),
    )
    first_records = (
        SourceRecord(
            source_id="public-cards",
            title="Public card guidance",
            source_url="https://example.test/public-cards",
            category="credit-cards",
            source_version="v1",
            text="Omega public card replacement guidance.",
        ),
        SourceRecord(
            source_id="fraud-only",
            title="Restricted fraud procedure",
            source_url="https://example.test/fraud-only",
            category="payments-fraud",
            source_version="v1",
            text="Alpha fraud dispute investigation procedure.",
            access_policy=restricted_fraud,
        ),
        SourceRecord(
            source_id="service-only",
            title="Restricted mortgage procedure",
            source_url="https://example.test/service-only",
            category="mortgages",
            source_version="v1",
            text="Mortgage escrow customer service procedure.",
            access_policy=restricted_service,
        ),
    )

    first = apply_snapshot(sessions, first_records)
    generation_1 = publisher.publish(first.manifest, first.result.run_id)
    assert generation_1.bm25_action is BM25PlanAction.REFIT
    runtime_g1, _, reranker_g1 = build_runtime(sessions, settings, qdrant)

    query = "omega alpha fraud dispute mortgage escrow customer service"

    def sources(
        runtime: ManagedRetrievalRuntime,
        context: AccessContext | None,
    ) -> set[str]:
        trace = runtime.retrieval_service.retrieve_debug(
            query,
            top_k=10,
            access_context=context,
        )
        return {str(item.chunk.metadata["source_id"]) for item in trace.reranked}

    assert sources(runtime_g1, None) == {"public-cards"}
    assert sources(
        runtime_g1,
        AccessContext(groups=frozenset({"fraud-analysts"})),
    ) == {
        "public-cards",
        "fraud-only",
    }
    assert sources(
        runtime_g1,
        AccessContext(groups=frozenset({"customer-service"})),
    ) == {
        "public-cards",
        "service-only",
    }
    assert sources(
        runtime_g1,
        AccessContext(
            groups=frozenset({"customer-service", "FRAUD-ANALYSTS"})
        ),
    ) == {"public-cards", "fraud-only", "service-only"}
    assert all(
        set(call[1])
        <= {
            chunk.compatibility_chunk_id
            for chunk in first.manifest.chunks
        }
        for call in reranker_g1.calls
    )

    second_records = (
        first_records[0].model_copy(update={"access_policy": restricted_fraud}),
        first_records[1],
        first_records[2],
    )
    second = apply_snapshot(sessions, second_records)
    assert second.result.changed_count == 1
    assert second.result.unchanged_count == 2
    with sessions() as session:
        public_document = session.scalar(
            select(LineageDocument).where(
                LineageDocument.source_id == "public-cards"
            )
        )
        assert public_document is not None
        versions = list(
            session.scalars(
                select(DocumentVersion)
                .where(DocumentVersion.document_pk == public_document.document_pk)
                .order_by(DocumentVersion.version_number)
            )
        )
        assert [version.version_number for version in versions] == [1, 2]
        assert versions[0].content_hash == versions[1].content_hash
        assert versions[0].metadata_hash != versions[1].metadata_hash
        assert versions[0].access_visibility == "PUBLIC"
        assert versions[1].access_visibility == "RESTRICTED"
        classification = session.scalar(
            select(IngestionRunDocument).where(
                IngestionRunDocument.run_id == second.result.run_id,
                IngestionRunDocument.document_pk == public_document.document_pk,
            )
        )
        assert classification is not None
        assert classification.content_changed is False
        assert classification.metadata_changed is True

    embedded_before = embedding.embedded_document_count
    generation_2 = publisher.publish(second.manifest, second.result.run_id)
    assert generation_2.outcome is ManagedIndexPublishOutcome.PUBLISHED
    assert generation_2.generation == 2
    assert generation_2.bm25_action is BM25PlanAction.REUSE
    assert generation_2.dense_reused_count == second.manifest.chunk_count
    assert generation_2.dense_computed_count == 0
    assert embedding.embedded_document_count == embedded_before
    assert_generation_invariants(sessions, qdrant, generation_1, "RETIRED")
    assert_generation_invariants(sessions, qdrant, generation_2, "ACTIVE")

    def payload_for(collection: str, source_id: str) -> dict[str, Any]:
        records, _ = qdrant.scroll(
            collection_name=collection,
            limit=100,
            with_payload=True,
            with_vectors=False,
        )
        return next(
            dict(record.payload or {})
            for record in records
            if (record.payload or {}).get("source_id") == source_id
        )

    old_payload = payload_for(generation_1.collection, "public-cards")
    new_payload = payload_for(generation_2.collection, "public-cards")
    assert old_payload["access_visibility"] == "PUBLIC"
    assert old_payload["allowed_groups"] == []
    assert new_payload["access_visibility"] == "RESTRICTED"
    assert new_payload["allowed_groups"] == ["audit-reviewers", "fraud-analysts"]

    runtime_g2, _, _ = build_runtime(sessions, settings, qdrant)
    database_counts_before = lineage_counts(sessions)
    qdrant_counts_before = {
        generation_1.collection: qdrant.count(generation_1.collection, exact=True).count,
        generation_2.collection: qdrant.count(generation_2.collection, exact=True).count,
    }
    assert sources(runtime_g2, None) == set()
    assert sources(
        runtime_g2,
        AccessContext(groups=frozenset({"fraud-analysts"})),
    ) == {
        "public-cards",
        "fraud-only",
    }
    assert sources(
        runtime_g2,
        AccessContext(groups=frozenset({"customer-service"})),
    ) == {
        "service-only"
    }
    assert sources(runtime_g1, None) == {"public-cards"}
    assert runtime_g1.generation == 1
    assert runtime_g2.generation == 2
    assert lineage_counts(sessions) == database_counts_before
    assert {
        generation_1.collection: qdrant.count(generation_1.collection, exact=True).count,
        generation_2.collection: qdrant.count(generation_2.collection, exact=True).count,
    } == qdrant_counts_before


def build_test_settings() -> AppSettings:
    base = load_settings("config/config.yaml")
    return base.model_copy(
        update={
            "managed_ingestion": base.managed_ingestion.model_copy(
                update={
                    "corpus_key": CORPUS_KEY,
                    "qdrant_collection_prefix": MANAGED_PREFIX,
                }
            ),
            "chunking": base.chunking.model_copy(
                update={"chunk_size": CHUNK_SIZE, "overlap": CHUNK_OVERLAP}
            ),
            "models": base.models.model_copy(
                update={
                    "embedding": base.models.embedding.model_copy(
                        update={
                            "model": "test/deterministic-embedding",
                            "device": "cpu",
                            "trust_remote_code": False,
                            "query_prompt_name": None,
                        }
                    )
                }
            ),
            "retrieval": base.retrieval.model_copy(
                update={
                    "dense_candidates": 10,
                    "sparse_candidates": 10,
                    "fused_candidates": 10,
                    "final_contexts": 3,
                    "rrf_k": 60,
                    "hnsw_ef": 32,
                }
            ),
        }
    )


def initial_records() -> tuple[SourceRecord, ...]:
    return (
        SourceRecord(
            source_id="payments",
            title="Card payment disputes",
            source_url="https://example.test/payments",
            category="payments-fraud",
            source_version="2026-09-01",
            text=(
                "Keep account notices and promptly contact the card issuer about "
                "an unfamiliar charge.\n\n"
                "Legacy alpha dispute guidance explains how to preserve card "
                "transaction records."
            ),
        ),
        SourceRecord(
            source_id="mortgages",
            title="Mortgage escrow servicing",
            source_url="https://example.test/mortgages",
            category="mortgages",
            source_version="2026-09-01",
            text=(
                "A mortgage servicer provides escrow statements showing taxes and "
                "insurance payments.\n\n"
                "Borrowers should review the annual escrow analysis and contact the "
                "servicer about errors."
            ),
        ),
        SourceRecord(
            source_id="accounts",
            title="Savings account overdrafts",
            source_url="https://example.test/accounts",
            category="bank-accounts",
            source_version="2026-09-01",
            text=(
                "Savings account customers should review overdraft terms and periodic "
                "account statements."
            ),
        ),
    )


def changed_records() -> tuple[SourceRecord, ...]:
    records = list(initial_records())
    records[0] = records[0].model_copy(
        update={
            "source_version": "2026-09-02",
            "text": (
                "Keep account notices and promptly contact the card issuer about "
                "an unfamiliar charge.\n\n"
                "Omega chargeback replacement guidance explains how to preserve card "
                "transaction records."
            ),
        }
    )
    return tuple(records)


def corpus_snapshot(records: Sequence[SourceRecord]) -> CorpusSnapshot:
    documents = [normalize_record(record) for record in records]
    chunker = ParagraphChunker(CHUNK_SIZE, CHUNK_OVERLAP)
    chunks = [chunk for document in documents for chunk in chunker.chunk(document)]
    categories: dict[str, int] = {}
    for document in documents:
        categories[document.category] = categories.get(document.category, 0) + 1
    return CorpusSnapshot(
        documents=documents,
        chunks=chunks,
        fixture_document_count=len(documents),
        expanded_document_count=0,
        category_distribution=categories,
    )


def apply_snapshot(
    sessions: sessionmaker[Session],
    records: Sequence[SourceRecord],
) -> AppliedSnapshot:
    snapshot = corpus_snapshot(records)
    with sessions() as session, session.begin():
        planned = LineagePlanningService(LineageRepository(session)).plan_snapshot(
            snapshot,
            corpus_key=CORPUS_KEY,
            authoritative_snapshot=True,
        )
    with sessions() as session:
        result = LineageApplyService(
            session,
            chunk_size=CHUNK_SIZE,
            overlap=CHUNK_OVERLAP,
        ).apply_plan(
            planned.plan,
            corpus_key=CORPUS_KEY,
            qdrant_collection_prefix=MANAGED_PREFIX,
            snapshot_hash=planned.snapshot_hash,
        )
    with sessions() as session, session.begin():
        manifest = ChunkMaterializationService(
            session,
            chunk_size=CHUNK_SIZE,
            overlap=CHUNK_OVERLAP,
        ).materialize_active_manifest(CORPUS_KEY)
    return AppliedSnapshot(snapshot=snapshot, result=result, manifest=manifest)


def build_publisher(
    sessions: sessionmaker[Session],
    settings: AppSettings,
    client: QdrantClient,
    embedding: DeterministicEmbeddingProvider,
) -> ManagedIndexService:
    def store_factory(
        managed_prefix: str,
        configuration: Any,
        embedding_dimension: int,
    ) -> ManagedQdrantStore:
        return ManagedQdrantStore(
            client,
            managed_prefix=managed_prefix,
            configuration=configuration,
            embedding_dimension=embedding_dimension,
        )

    return ManagedIndexService(
        session_factory=sessions,
        embedding_provider=embedding,
        embedding_settings=settings.models.embedding,
        chunking_settings=settings.chunking,
        managed_index_settings=settings.managed_index,
        store_factory=store_factory,
    )


def build_runtime(
    sessions: sessionmaker[Session],
    settings: AppSettings,
    client: QdrantClient,
) -> tuple[
    ManagedRetrievalRuntime,
    DeterministicEmbeddingProvider,
    DeterministicReranker,
]:
    embedding = DeterministicEmbeddingProvider(settings.models.embedding.model)
    reranker = DeterministicReranker()
    with sessions() as session:
        runtime = build_managed_retrieval_runtime(
            settings,
            LineageRepository(session),
            qdrant_client=client,
            embedding_provider=embedding,
            reranker_provider=reranker,
        )
    return runtime, embedding, reranker


def assert_published_generation(
    result: ManagedIndexPublishResult,
    *,
    expected_generation: int,
    expected_chunks: int,
) -> None:
    assert result.outcome is ManagedIndexPublishOutcome.PUBLISHED
    assert result.generation == expected_generation
    assert result.chunk_count == expected_chunks


def membership_rows(
    session: Session,
    index_state_id: UUID,
) -> list[IndexStateChunk]:
    return list(
        session.scalars(
            select(IndexStateChunk)
            .where(IndexStateChunk.index_state_id == index_state_id)
            .order_by(IndexStateChunk.chunk_pk)
        )
    )


def assert_generation_invariants(
    sessions: sessionmaker[Session],
    client: QdrantClient,
    result: ManagedIndexPublishResult,
    expected_status: str,
) -> None:
    with sessions() as session:
        state = session.get(IndexState, result.index_state_id)
        assert state is not None, f"missing generation {result.generation} index state"
        membership_count = session.scalar(
            select(func.count())
            .select_from(IndexStateChunk)
            .where(IndexStateChunk.index_state_id == result.index_state_id)
        )
        point_count = client.count(collection_name=result.collection, exact=True).count
        assert state.status == expected_status
        assert state.chunk_count == membership_count == point_count, (
            f"generation {result.generation}: state={state.chunk_count}, "
            f"memberships={membership_count}, Qdrant={point_count}"
        )


def assert_exactly_one_active(session: Session) -> None:
    active = list(session.scalars(select(IndexState).where(IndexState.status == "ACTIVE")))
    assert len(active) == 1, [
        (state.generation, state.status, state.qdrant_collection) for state in active
    ]


def classification_by_source(session: Session, run_id: UUID) -> dict[str, str]:
    rows = session.execute(
        select(LineageDocument.source_id, IngestionRunDocument.change_type)
        .join(
            IngestionRunDocument,
            IngestionRunDocument.document_pk == LineageDocument.document_pk,
        )
        .where(IngestionRunDocument.run_id == run_id)
    )
    return {source_id: change_type for source_id, change_type in rows}


def exact_reuse_crosses_versions(session: Session, membership: IndexStateChunk) -> bool:
    if membership.dense_source_chunk_pk is None:
        return False
    current = session.get(LineageChunk, membership.chunk_pk)
    source = session.get(LineageChunk, membership.dense_source_chunk_pk)
    assert current is not None and source is not None
    if current.document_version_id == source.document_version_id:
        return False
    return current.chunk_hash == source.chunk_hash and current.text == source.text


def validate_bm25_state(state: IndexState, manifest: ActiveChunkManifest) -> None:
    assert state.bm25_state is not None
    parsed = bm25_state_from_json_dict(state.bm25_state)
    expected_corpus_hash = bm25_corpus_hash([chunk.text for chunk in manifest.chunks])
    assert state.bm25_state_hash == bm25_state_hash(parsed)
    assert parsed.corpus_hash == state.bm25_corpus_hash == expected_corpus_hash
    assert parsed.document_count == state.chunk_count == manifest.chunk_count


def index_state_count(sessions: sessionmaker[Session]) -> int:
    with sessions() as session:
        return session.scalar(select(func.count()).select_from(IndexState)) or 0


def lineage_counts(sessions: sessionmaker[Session]) -> tuple[int, ...]:
    with sessions() as session:
        return tuple(
            session.scalar(select(func.count()).select_from(model)) or 0
            for model in (
                Corpus,
                IngestionRun,
                LineageDocument,
                DocumentVersion,
                LineageChunk,
                IngestionRunDocument,
                IndexState,
                IndexStateChunk,
            )
        )
