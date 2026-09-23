from __future__ import annotations

import pytest

from consumer_finance_rag.lineage.index_state import (
    IndexStateReadyMetadata,
    IndexStateValidationError,
    validate_index_state_ready,
    validate_index_state_transition,
)


@pytest.mark.parametrize(
    ("current", "target"),
    [
        ("BUILDING", "READY"),
        ("BUILDING", "FAILED"),
        ("READY", "ACTIVE"),
        ("READY", "FAILED"),
        ("ACTIVE", "RETIRED"),
    ],
)
def test_allowed_index_state_transitions(current: str, target: str) -> None:
    validate_index_state_transition(current, target)


@pytest.mark.parametrize(
    ("current", "target"),
    [
        ("FAILED", "ACTIVE"),
        ("ACTIVE", "READY"),
        ("RETIRED", "BUILDING"),
        ("RETIRED", "ACTIVE"),
        ("BUILDING", "ACTIVE"),
    ],
)
def test_invalid_index_state_transitions(current: str, target: str) -> None:
    with pytest.raises(IndexStateValidationError, match="not allowed"):
        validate_index_state_transition(current, target)


def ready_metadata(**updates: object) -> IndexStateReadyMetadata:
    values = {
        "qdrant_collection": "consumer_finance_managed_g000001",
        "embedding_dimension": 1024,
        "bm25_state": {"vocabulary": {"credit": 0}},
        "bm25_state_hash": "a" * 64,
        "bm25_corpus_hash": "b" * 64,
        "index_configuration": {"format_version": "managed-hybrid-index-v1"},
        "index_configuration_fingerprint": "c" * 64,
        "document_count": 6,
        "chunk_count": 12,
    }
    values.update(updates)
    return IndexStateReadyMetadata(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"embedding_dimension": None}, "embedding_dimension"),
        ({"embedding_dimension": 0}, "embedding_dimension"),
        ({"bm25_state": None}, "bm25_state"),
        ({"bm25_state_hash": None}, "bm25_state_hash"),
        ({"bm25_corpus_hash": None}, "bm25_corpus_hash"),
        ({"index_configuration": None}, "index_configuration"),
        ({"index_configuration_fingerprint": None}, "index_configuration_fingerprint"),
        ({"document_count": -1}, "document_count"),
        ({"chunk_count": -1}, "chunk_count"),
        ({"qdrant_collection": " "}, "qdrant_collection"),
    ],
)
def test_ready_validation_rejects_incomplete_metadata(
    updates: dict[str, object], message: str
) -> None:
    with pytest.raises(IndexStateValidationError, match=message):
        validate_index_state_ready(ready_metadata(**updates))


def test_ready_validation_accepts_complete_metadata() -> None:
    validate_index_state_ready(ready_metadata())
