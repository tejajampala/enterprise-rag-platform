"""Pure state-machine and readiness validation for managed index generations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

IndexStateStatus = Literal["BUILDING", "READY", "ACTIVE", "RETIRED", "FAILED"]

ALLOWED_INDEX_STATE_TRANSITIONS: Mapping[str, frozenset[str]] = {
    "BUILDING": frozenset({"READY", "FAILED"}),
    "READY": frozenset({"ACTIVE", "FAILED"}),
    "ACTIVE": frozenset({"RETIRED"}),
    "RETIRED": frozenset(),
    "FAILED": frozenset(),
}


class IndexStateValidationError(ValueError):
    """Raised when generation metadata or a requested transition is invalid."""


@dataclass(frozen=True, slots=True)
class IndexStateReadyMetadata:
    """Persistence-independent values required before a generation becomes READY."""

    qdrant_collection: str
    embedding_dimension: int | None
    bm25_state: Mapping[str, Any] | None
    bm25_state_hash: str | None
    bm25_corpus_hash: str | None
    index_configuration: Mapping[str, Any] | None
    index_configuration_fingerprint: str | None
    document_count: int
    chunk_count: int


def validate_index_state_transition(current_status: str, target_status: str) -> None:
    """Reject transitions outside the first managed-generation state machine."""

    if current_status not in ALLOWED_INDEX_STATE_TRANSITIONS:
        raise IndexStateValidationError(f"Unknown current index status: {current_status}")
    if target_status not in ALLOWED_INDEX_STATE_TRANSITIONS:
        raise IndexStateValidationError(f"Unknown target index status: {target_status}")
    if target_status not in ALLOWED_INDEX_STATE_TRANSITIONS[current_status]:
        raise IndexStateValidationError(
            f"Index state transition is not allowed: {current_status} -> {target_status}"
        )


def validate_index_state_ready(metadata: IndexStateReadyMetadata) -> None:
    """Validate persisted build metadata without contacting Qdrant."""

    missing: list[str] = []
    if metadata.embedding_dimension is None or metadata.embedding_dimension <= 0:
        missing.append("positive embedding_dimension")
    if not metadata.bm25_state:
        missing.append("bm25_state")
    if not metadata.bm25_state_hash:
        missing.append("bm25_state_hash")
    if not metadata.bm25_corpus_hash:
        missing.append("bm25_corpus_hash")
    if not metadata.index_configuration:
        missing.append("index_configuration")
    if not metadata.index_configuration_fingerprint:
        missing.append("index_configuration_fingerprint")
    if metadata.document_count < 0:
        missing.append("non-negative document_count")
    if metadata.chunk_count < 0:
        missing.append("non-negative chunk_count")
    if not metadata.qdrant_collection.strip():
        missing.append("qdrant_collection")
    if missing:
        raise IndexStateValidationError(
            "Index state is not ready: " + ", ".join(missing)
        )
