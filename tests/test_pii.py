from __future__ import annotations

from collections.abc import Sequence

import pytest
from qdrant_client import models

from consumer_finance_rag.access import (
    AccessContext,
    AccessVisibility,
    DocumentAccessPolicy,
)
from consumer_finance_rag.config import RetrievalSettings
from consumer_finance_rag.context import ContextBuilder
from consumer_finance_rag.models import Chunk, RetrievedChunk
from consumer_finance_rag.pii import (
    DETERMINISTIC_PII_VERSION,
    DeterministicPIIDetector,
    PIIHandlingPolicy,
    PIIMatch,
    PIIRedactionError,
    PIIType,
    resolve_pii_matches,
)
from consumer_finance_rag.pipeline import RAGPipeline
from consumer_finance_rag.providers import LLMResult
from consumer_finance_rag.retrieval import (
    AuthorizationInvariantError,
    HybridRetrievalService,
)
from consumer_finance_rag.sparse import BM25SparseEncoder


@pytest.mark.parametrize(
    ("text", "pii_type"),
    [
        ("Contact alice@example.com today.", PIIType.EMAIL),
        ("TFN 123 456 782 requires review.", PIIType.AU_TFN),
        ("Card 4111-1111-1111-1111 was reported.", PIIType.CREDIT_CARD),
        ("Call +61 412 345 678 for assistance.", PIIType.PHONE),
    ],
)
def test_detector_recognizes_supported_pii(text: str, pii_type: PIIType) -> None:
    matches = DeterministicPIIDetector().detect(text)

    assert [match.pii_type for match in matches] == [pii_type]
    assert matches[0].detector_version == DETERMINISTIC_PII_VERSION


@pytest.mark.parametrize(
    "text",
    [
        "This sentence contains no personal data.",
        "not-an-email@example",
        "TFN 123 456 789 is checksum-invalid.",
        "Card 4111 1111 1111 1112 fails Luhn.",
        "Reference number 12345 is too short for a phone.",
    ],
)
def test_detector_conservatively_ignores_invalid_candidates(text: str) -> None:
    assert DeterministicPIIDetector().detect(text) == ()


def test_detector_returns_multiple_types_in_offset_order_without_raw_values() -> None:
    text = (
        "Email alice@example.com, call 02 9876 5432, use TFN 123456782, "
        "and card 4111 1111 1111 1111."
    )

    first = DeterministicPIIDetector().detect(text)
    second = DeterministicPIIDetector().detect(text)

    assert first == second
    assert [match.pii_type for match in first] == [
        PIIType.EMAIL,
        PIIType.PHONE,
        PIIType.AU_TFN,
        PIIType.CREDIT_CARD,
    ]
    assert all("value" not in match.model_dump() for match in first)
    assert all(
        first[index].end <= first[index + 1].start
        for index in range(len(first) - 1)
    )


def test_overlapping_candidates_resolve_deterministically_without_duplicates() -> None:
    detector_version = DETERMINISTIC_PII_VERSION
    overlapping = [
        PIIMatch(
            pii_type=PIIType.PHONE,
            start=0,
            end=12,
            detector_version=detector_version,
        ),
        PIIMatch(
            pii_type=PIIType.AU_TFN,
            start=2,
            end=11,
            detector_version=detector_version,
        ),
        PIIMatch(
            pii_type=PIIType.AU_TFN,
            start=2,
            end=11,
            detector_version=detector_version,
        ),
    ]

    first = resolve_pii_matches(overlapping)
    second = resolve_pii_matches(list(reversed(overlapping)))

    assert first == second
    assert len(first) == 1
    assert first[0].pii_type is PIIType.AU_TFN


def test_classification_hash_is_deterministic_versioned_and_contains_no_raw_pii() -> None:
    text = "Contact alice@example.com about TFN 123456782."
    first = DeterministicPIIDetector().classify(text)
    second = DeterministicPIIDetector().classify(text)
    newer = DeterministicPIIDetector("deterministic-pii-v2-test").classify(text)

    assert first == second
    assert first.classification_hash != newer.classification_hash
    assert first.pii_types == (PIIType.AU_TFN, PIIType.EMAIL)
    assert first.match_count == 2
    serialized = str(first.model_dump())
    assert "alice@example.com" not in serialized
    assert "123456782" not in serialized


def test_redaction_replaces_all_supported_values_and_preserves_surrounding_text() -> None:
    original = (
        "Start alice@example.com; TFN 123 456 782; card 4111 1111 1111 1111; "
        "phone +61 412 345 678. End"
    )
    detector = DeterministicPIIDetector()

    redacted = detector.redact(original)

    assert redacted == (
        "Start [REDACTED:EMAIL]; TFN [REDACTED:AU_TFN]; "
        "card [REDACTED:CREDIT_CARD]; phone [REDACTED:PHONE]. End"
    )
    assert detector.detect(redacted) == ()
    assert detector.redact(redacted) == redacted
    assert detector.redact("No PII here.") == "No PII here."


class FakeEmbedding:
    model_name = "test-embedding"

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0]


class RecordingReranker:
    model_name = "test-reranker"

    def __init__(self) -> None:
        self.seen_texts: list[str] = []

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        self.seen_texts = [candidate.chunk.text for candidate in candidates]
        return list(candidates[:top_k])


class FakeStore:
    def __init__(self, candidate: RetrievedChunk) -> None:
        self.candidate = candidate
        self.filters: list[models.Filter | None] = []

    def search_dense(
        self,
        vector: list[float],
        limit: int,
        *,
        query_filter: models.Filter | None = None,
    ) -> list[RetrievedChunk]:
        self.filters.append(query_filter)
        return [self.candidate.model_copy(update={"dense_rank": 1})]

    def search_sparse(
        self,
        query: str,
        encoder: BM25SparseEncoder,
        limit: int,
        *,
        query_filter: models.Filter | None = None,
    ) -> list[RetrievedChunk]:
        self.filters.append(query_filter)
        return [self.candidate.model_copy(update={"sparse_rank": 1})]


class CountingDetector(DeterministicPIIDetector):
    def __init__(self) -> None:
        super().__init__()
        self.redaction_calls = 0

    def redact(self, text: str) -> str:
        self.redaction_calls += 1
        return super().redact(text)


class ResidualDetector(DeterministicPIIDetector):
    def detect(self, text: str) -> tuple[PIIMatch, ...]:
        return (
            PIIMatch(
                pii_type=PIIType.EMAIL,
                start=0,
                end=1,
                detector_version=self.version,
            ),
        )


class SafePromptLLM:
    model_name = "test-generation"

    def generate(self, system_prompt: str, user_prompt: str) -> LLMResult:
        assert "alice@example.com" not in user_prompt
        assert "[REDACTED:EMAIL]" in user_prompt
        return LLMResult(text="Safe answer [C1].", model_id=self.model_name)


def retrieved_candidate(
    text: str,
    policy: DocumentAccessPolicy | None = None,
) -> RetrievedChunk:
    resolved = policy or DocumentAccessPolicy()
    return RetrievedChunk(
        chunk=Chunk(
            chunk_id="chunk-1",
            document_id="document-1",
            document_version="v1",
            text=text,
            ordinal=0,
            chunk_hash="a" * 64,
            metadata={
                "access_visibility": resolved.visibility.value,
                "allowed_groups": list(resolved.canonical_groups),
            },
        ),
        score=1.0,
    )


def managed_retrieval(
    candidate: RetrievedChunk,
    reranker: RecordingReranker,
    detector: DeterministicPIIDetector,
) -> tuple[HybridRetrievalService, FakeStore]:
    store = FakeStore(candidate)
    service = HybridRetrievalService(
        embedding_provider=FakeEmbedding(),
        reranker=reranker,
        store=store,
        sparse_encoder=BM25SparseEncoder.fit([candidate.chunk.text]),
        settings=RetrievalSettings(
            dense_candidates=2,
            sparse_candidates=2,
            fused_candidates=2,
            final_contexts=1,
        ),
        access_control_enabled=True,
        pii_handling_enabled=True,
        pii_detector=detector,
    )
    return service, store


def test_managed_default_redacts_after_reranking_and_preserves_lineage() -> None:
    original = "Contact support at alice@example.com"
    reranker = RecordingReranker()
    service, store = managed_retrieval(
        retrieved_candidate(original),
        reranker,
        DeterministicPIIDetector(),
    )

    result = service.retrieve("support")

    assert reranker.seen_texts == [original]
    assert result.chunks[0].chunk.text == "Contact support at [REDACTED:EMAIL]"
    assert result.chunks[0].chunk.chunk_id == "chunk-1"
    assert result.chunks[0].chunk.document_id == "document-1"
    assert store.filters[0] is store.filters[1]
    assert store.filters[0] is not None


def test_pipeline_sends_only_redacted_final_context_to_generation() -> None:
    candidate = retrieved_candidate("Contact support at alice@example.com")
    reranker = RecordingReranker()
    service, _ = managed_retrieval(
        candidate,
        reranker,
        DeterministicPIIDetector(),
    )
    embedding = FakeEmbedding()
    pipeline = RAGPipeline(
        retrieval=service,
        context_builder=ContextBuilder(1000),
        llm=SafePromptLLM(),
        embedding_provider=embedding,
        reranker=reranker,
    )

    response = pipeline.answer("How can I contact support?")

    assert response.retrieved_chunks[0].chunk.text == (
        "Contact support at [REDACTED:EMAIL]"
    )


def test_explicit_allow_returns_original_authorized_text() -> None:
    original = "Contact support at alice@example.com"
    service, _ = managed_retrieval(
        retrieved_candidate(original),
        RecordingReranker(),
        DeterministicPIIDetector(),
    )

    result = service.retrieve("support", pii_policy=PIIHandlingPolicy.ALLOW)

    assert result.chunks[0].chunk.text == original


def test_unauthorized_chunk_fails_acl_before_redactor() -> None:
    detector = CountingDetector()
    restricted = DocumentAccessPolicy(
        visibility=AccessVisibility.RESTRICTED,
        allowed_groups=frozenset({"fraud-analysts"}),
    )
    service, _ = managed_retrieval(
        retrieved_candidate("TFN 123456782", restricted),
        RecordingReranker(),
        detector,
    )

    with pytest.raises(AuthorizationInvariantError):
        service.retrieve("TFN", access_context=AccessContext())

    assert detector.redaction_calls == 0


def test_redaction_failure_fails_closed() -> None:
    service, _ = managed_retrieval(
        retrieved_candidate("x"),
        RecordingReranker(),
        ResidualDetector(),
    )

    with pytest.raises(PIIRedactionError, match="failed closed"):
        service.retrieve("x")


def test_legacy_retrieval_defaults_to_allow() -> None:
    original = "Contact support at alice@example.com"
    candidate = retrieved_candidate(original)
    store = FakeStore(candidate)
    service = HybridRetrievalService(
        embedding_provider=FakeEmbedding(),
        reranker=RecordingReranker(),
        store=store,
        sparse_encoder=BM25SparseEncoder.fit([candidate.chunk.text]),
        settings=RetrievalSettings(final_contexts=1),
    )

    result = service.retrieve("support")

    assert result.chunks[0].chunk.text == original
    assert store.filters == [None, None]
