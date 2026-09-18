"""Deterministic, versioned fingerprints for Phase 3 lineage and index compatibility."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass

DOCUMENT_METADATA_FINGERPRINT_VERSION = "document-metadata-v1"
DOCUMENT_VERSION_FINGERPRINT_VERSION = "document-version-v1"
CHUNKING_FINGERPRINT_VERSION = "chunking-v1"
DEFAULT_CHUNKER_IMPLEMENTATION = "paragraph-aware-recursive-v1"
EMBEDDING_FINGERPRINT_VERSION = "embedding-index-v1"
MANAGED_SNAPSHOT_FINGERPRINT_VERSION = "managed-snapshot-v1"

def _canonical_fingerprint(payload: object) -> str:
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ManagedSnapshotFingerprintRecord:
    """Source/version identity included in a managed snapshot fingerprint."""

    source_key: str
    content_hash: str
    metadata_hash: str
    version_hash: str


def managed_snapshot_hash(records: Sequence[ManagedSnapshotFingerprintRecord]) -> str:
    """Fingerprint an order-independent current managed-corpus snapshot."""

    ordered = sorted(
        records,
        key=lambda record: (
            record.source_key,
            record.content_hash,
            record.metadata_hash,
            record.version_hash,
        ),
    )
    return _canonical_fingerprint(
        {
            "documents": [
                {
                    "content_hash": record.content_hash,
                    "metadata_hash": record.metadata_hash,
                    "source_key": record.source_key,
                    "version_hash": record.version_hash,
                }
                for record in ordered
            ],
            "fingerprint_version": MANAGED_SNAPSHOT_FINGERPRINT_VERSION,
        }
    )


def metadata_hash(
    *,
    title: str,
    source_url: str | None,
    category: str,
    source_version: str,
) -> str:
    """Fingerprint material source metadata using canonical JSON."""

    return _canonical_fingerprint(
        {
            "category": category,
            "fingerprint_version": DOCUMENT_METADATA_FINGERPRINT_VERSION,
            "source_url": source_url,
            "source_version": source_version,
            "title": title,
        }
    )


def version_hash(*, content_hash: str, metadata_hash: str) -> str:
    """Combine normalized content and material metadata identities."""

    return _canonical_fingerprint(
        {
            "content_hash": content_hash,
            "fingerprint_version": DOCUMENT_VERSION_FINGERPRINT_VERSION,
            "metadata_hash": metadata_hash,
        }
    )


def chunking_fingerprint(
    *,
    chunk_size: int,
    overlap: int,
    implementation: str = DEFAULT_CHUNKER_IMPLEMENTATION,
) -> str:
    """Fingerprint the explicit chunker implementation and behavior settings."""

    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("require chunk_size > overlap >= 0")
    return _canonical_fingerprint(
        {
            "chunk_size": chunk_size,
            "fingerprint_version": CHUNKING_FINGERPRINT_VERSION,
            "implementation": implementation,
            "overlap": overlap,
        }
    )


def embedding_fingerprint(
    *,
    provider: str,
    model: str,
    document_encoding_mode: str = "encode_document",
    normalize_embeddings: bool = True,
    trust_remote_code: bool = True,
    device: str | None = None,
) -> str:
    """Fingerprint settings that determine persisted document-vector compatibility.

    Execution device is accepted for composition-root convenience but deliberately omitted:
    CPU versus GPU placement does not change the stored vector contract. Query prompt settings
    are likewise omitted because they do not affect persisted document vectors.
    """

    del device
    return _canonical_fingerprint(
        {
            "document_encoding_mode": document_encoding_mode,
            "fingerprint_version": EMBEDDING_FINGERPRINT_VERSION,
            "model": model,
            "normalize_embeddings": normalize_embeddings,
            "provider": provider,
            "trust_remote_code": trust_remote_code,
        }
    )
