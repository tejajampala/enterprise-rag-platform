from __future__ import annotations

from collections.abc import Sequence

import pytest
from pydantic import ValidationError
from qdrant_client import models

from consumer_finance_rag.access import (
    AccessContext,
    AccessVisibility,
    DocumentAccessPolicy,
    build_qdrant_access_filter,
    is_access_allowed,
)
from consumer_finance_rag.acquisition import SourceRecord
from consumer_finance_rag.config import RetrievalSettings
from consumer_finance_rag.lineage.fingerprints import metadata_hash
from consumer_finance_rag.models import Chunk, RetrievedChunk
from consumer_finance_rag.retrieval import (
    AuthorizationInvariantError,
    HybridRetrievalService,
)
from consumer_finance_rag.sparse import BM25SparseEncoder


class FakeEmbedding:
    model_name = "test-embedding"

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0]


class RecordingReranker:
    model_name = "test-reranker"

    def __init__(self) -> None:
        self.seen: list[RetrievedChunk] = []

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        self.seen = list(candidates)
        return list(candidates[:top_k])


class RecordingStore:
    def __init__(
        self,
        *,
        dense: Sequence[RetrievedChunk],
        sparse: Sequence[RetrievedChunk],
    ) -> None:
        self.dense = list(dense)
        self.sparse = list(sparse)
        self.dense_filter: models.Filter | None = None
        self.sparse_filter: models.Filter | None = None

    def search_dense(
        self,
        vector: list[float],
        limit: int,
        *,
        query_filter: models.Filter | None = None,
    ) -> list[RetrievedChunk]:
        self.dense_filter = query_filter
        return self.dense[:limit]

    def search_sparse(
        self,
        query: str,
        encoder: BM25SparseEncoder,
        limit: int,
        *,
        query_filter: models.Filter | None = None,
    ) -> list[RetrievedChunk]:
        self.sparse_filter = query_filter
        return self.sparse[:limit]


def candidate(
    chunk_id: str,
    policy: DocumentAccessPolicy,
    *,
    dense_rank: int | None = None,
    sparse_rank: int | None = None,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(
            chunk_id=chunk_id,
            document_id=f"document-{chunk_id}",
            document_version="v1",
            text=f"Evidence for {chunk_id}",
            ordinal=0,
            chunk_hash="a" * 64,
            metadata={
                "access_visibility": policy.visibility.value,
                "allowed_groups": list(policy.canonical_groups),
            },
        ),
        score=1.0,
        dense_rank=dense_rank,
        sparse_rank=sparse_rank,
    )


def retrieval(
    store: RecordingStore,
    reranker: RecordingReranker,
    *,
    access_control_enabled: bool = True,
) -> HybridRetrievalService:
    return HybridRetrievalService(
        embedding_provider=FakeEmbedding(),
        reranker=reranker,
        store=store,
        sparse_encoder=BM25SparseEncoder.fit(["Evidence"]),
        settings=RetrievalSettings(
            dense_candidates=5,
            sparse_candidates=5,
            fused_candidates=5,
            final_contexts=5,
        ),
        access_control_enabled=access_control_enabled,
    )


def test_existing_sources_default_to_public() -> None:
    source = SourceRecord(
        source_id="reports",
        title="Credit reports",
        source_url=None,
        category="credit-reporting",
        source_version="v1",
        text="Credit report evidence.",
    )

    assert source.access_policy == DocumentAccessPolicy()
    assert source.access_policy.visibility is AccessVisibility.PUBLIC


def test_policy_and_context_normalize_groups_deterministically() -> None:
    first = DocumentAccessPolicy.model_validate(
        {
            "visibility": "RESTRICTED",
            "allowed_groups": [
                " Fraud-Analysts ",
                "AU-Employees",
                "fraud-analysts",
            ],
        }
    )
    second = DocumentAccessPolicy.model_validate(
        {
            "visibility": "RESTRICTED",
            "allowed_groups": ["au-employees", "FRAUD-ANALYSTS"],
        }
    )
    context = AccessContext.model_validate(
        {"groups": [" FRAUD-ANALYSTS ", "fraud-analysts"]}
    )

    assert first == second
    assert first.canonical_groups == ("au-employees", "fraud-analysts")
    assert context.groups == frozenset({"fraud-analysts"})


@pytest.mark.parametrize("group", ["", " ", "\t"])
def test_empty_group_identifier_is_rejected(group: str) -> None:
    with pytest.raises(ValidationError, match="cannot be empty"):
        AccessContext.model_validate({"groups": [group]})


def test_invalid_visibility_group_combinations_are_rejected() -> None:
    with pytest.raises(ValidationError, match="requires at least one"):
        DocumentAccessPolicy(visibility=AccessVisibility.RESTRICTED)
    with pytest.raises(ValidationError, match="cannot contain"):
        DocumentAccessPolicy(
            visibility=AccessVisibility.PUBLIC,
            allowed_groups=frozenset({"finance"}),
        )


@pytest.mark.parametrize(
    ("policy", "groups", "allowed"),
    [
        (DocumentAccessPolicy(), frozenset(), True),
        (DocumentAccessPolicy(), frozenset({"any-group"}), True),
        (
            DocumentAccessPolicy(
                visibility=AccessVisibility.RESTRICTED,
                allowed_groups=frozenset({"fraud-analysts"}),
            ),
            frozenset(),
            False,
        ),
        (
            DocumentAccessPolicy(
                visibility=AccessVisibility.RESTRICTED,
                allowed_groups=frozenset({"fraud-analysts"}),
            ),
            frozenset({"customer-service"}),
            False,
        ),
        (
            DocumentAccessPolicy(
                visibility=AccessVisibility.RESTRICTED,
                allowed_groups=frozenset({"fraud-analysts"}),
            ),
            frozenset({"fraud-analysts"}),
            True,
        ),
        (
            DocumentAccessPolicy(
                visibility=AccessVisibility.RESTRICTED,
                allowed_groups=frozenset({"fraud-analysts"}),
            ),
            frozenset({"customer-service", "fraud-analysts"}),
            True,
        ),
    ],
)
def test_access_semantics(
    policy: DocumentAccessPolicy,
    groups: frozenset[str],
    allowed: bool,
) -> None:
    assert is_access_allowed(policy, AccessContext(groups=groups)) is allowed


def test_policy_changes_metadata_hash_but_group_order_does_not() -> None:
    public = metadata_hash(
        title="Fraud guidance",
        source_url="https://example.test/fraud",
        category="payments-fraud",
        source_version="v1",
    )
    first = metadata_hash(
        title="Fraud guidance",
        source_url="https://example.test/fraud",
        category="payments-fraud",
        source_version="v1",
        access_policy=DocumentAccessPolicy(
            visibility=AccessVisibility.RESTRICTED,
            allowed_groups=frozenset({"b", "a"}),
        ),
    )
    second = metadata_hash(
        title="Fraud guidance",
        source_url="https://example.test/fraud",
        category="payments-fraud",
        source_version="v1",
        access_policy=DocumentAccessPolicy(
            visibility=AccessVisibility.RESTRICTED,
            allowed_groups=frozenset({"A", "B", "a"}),
        ),
    )
    changed_groups = metadata_hash(
        title="Fraud guidance",
        source_url="https://example.test/fraud",
        category="payments-fraud",
        source_version="v1",
        access_policy=DocumentAccessPolicy(
            visibility=AccessVisibility.RESTRICTED,
            allowed_groups=frozenset({"a", "c"}),
        ),
    )

    assert first == second
    assert public != first != changed_groups


def test_managed_retrieval_forwards_one_filter_to_both_candidate_searches() -> None:
    public = candidate("public", DocumentAccessPolicy(), dense_rank=1, sparse_rank=1)
    store = RecordingStore(dense=[public], sparse=[public])
    reranker = RecordingReranker()

    result = retrieval(store, reranker).retrieve_debug(
        "evidence",
        access_context=AccessContext(groups=frozenset({"fraud-analysts"})),
    )

    assert store.dense_filter is not None
    assert store.dense_filter is store.sparse_filter
    assert result.reranked == reranker.seen
    assert [item.chunk.chunk_id for item in reranker.seen] == ["public"]


def test_missing_managed_access_context_builds_public_only_filter() -> None:
    public = candidate("public", DocumentAccessPolicy(), dense_rank=1, sparse_rank=1)
    store = RecordingStore(dense=[public], sparse=[public])

    retrieval(store, RecordingReranker()).retrieve("evidence")

    assert store.dense_filter == build_qdrant_access_filter(AccessContext())
    assert store.dense_filter is store.sparse_filter


@pytest.mark.parametrize("stage", ["dense", "sparse"])
def test_unauthorized_candidate_is_rejected_before_fusion_and_reranking(
    stage: str,
) -> None:
    restricted = candidate(
        "restricted",
        DocumentAccessPolicy(
            visibility=AccessVisibility.RESTRICTED,
            allowed_groups=frozenset({"fraud-analysts"}),
        ),
        dense_rank=1 if stage == "dense" else None,
        sparse_rank=1 if stage == "sparse" else None,
    )
    store = RecordingStore(
        dense=[restricted] if stage == "dense" else [],
        sparse=[restricted] if stage == "sparse" else [],
    )
    reranker = RecordingReranker()

    with pytest.raises(AuthorizationInvariantError, match=stage):
        retrieval(store, reranker).retrieve(
            "evidence",
            access_context=AccessContext(groups=frozenset({"other-group"})),
        )

    assert reranker.seen == []


def test_legacy_retrieval_remains_unfiltered_and_accepts_payload_without_acl() -> None:
    legacy = RetrievedChunk(
        chunk=Chunk(
            chunk_id="legacy",
            document_id="document-legacy",
            document_version="v1",
            text="Legacy public evidence",
            ordinal=0,
            chunk_hash="b" * 64,
            metadata={},
        ),
        score=1.0,
    )
    store = RecordingStore(dense=[legacy], sparse=[legacy])

    result = retrieval(
        store,
        RecordingReranker(),
        access_control_enabled=False,
    ).retrieve("evidence")

    assert store.dense_filter is None
    assert store.sparse_filter is None
    assert [item.chunk.chunk_id for item in result.chunks] == ["legacy"]
