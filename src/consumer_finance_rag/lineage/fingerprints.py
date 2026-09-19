"""Deterministic, versioned fingerprints for Phase 3 lineage and index compatibility."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from consumer_finance_rag.config import ManagedIndexSettings

DOCUMENT_METADATA_FINGERPRINT_VERSION = "document-metadata-v1"
DOCUMENT_VERSION_FINGERPRINT_VERSION = "document-version-v1"
CHUNKING_FINGERPRINT_VERSION = "chunking-v1"
DEFAULT_CHUNKER_IMPLEMENTATION = "paragraph-aware-recursive-v1"
EMBEDDING_FINGERPRINT_VERSION = "embedding-index-v1"
MANAGED_SNAPSHOT_FINGERPRINT_VERSION = "managed-snapshot-v1"
MANAGED_INDEX_CONFIGURATION_FINGERPRINT_VERSION = "managed-index-configuration-v1"
BM25_IMPLEMENTATION_VERSION = "client-bm25-v1"
BM25_TOKENIZER_VERSION = "lowercase-alphanumeric-apostrophe-v1"
POINT_ID_FORMAT_VERSION = "lineage-chunk-uuid5-v1"
PAYLOAD_SCHEMA_VERSION = "managed-qdrant-payload-v1"


@dataclass(frozen=True, slots=True)
class ManagedIndexConfiguration:
    """Build-time values that define one immutable managed index format."""

    embedding_fingerprint: str
    chunking_fingerprint: str
    bm25_k1: float
    bm25_b: float
    dense_vector_name: str
    sparse_vector_name: str
    distance_metric: str
    hnsw_m: int
    hnsw_ef_construct: int
    sparse_modifier: str
    sparse_on_disk: bool
    managed_index_format_version: str
    bm25_implementation_version: str = BM25_IMPLEMENTATION_VERSION
    bm25_tokenizer_version: str = BM25_TOKENIZER_VERSION
    point_id_format_version: str = POINT_ID_FORMAT_VERSION
    payload_schema_version: str = PAYLOAD_SCHEMA_VERSION

    def canonical_payload(self) -> dict[str, object]:
        """Return the JSON-compatible configuration covered by its fingerprint."""

        return {
            "bm25": {
                "b": self.bm25_b,
                "implementation_version": self.bm25_implementation_version,
                "k1": self.bm25_k1,
                "tokenizer_version": self.bm25_tokenizer_version,
            },
            "chunking_fingerprint": self.chunking_fingerprint,
            "dense": {
                "distance_metric": self.distance_metric,
                "hnsw_ef_construct": self.hnsw_ef_construct,
                "hnsw_m": self.hnsw_m,
                "vector_name": self.dense_vector_name,
            },
            "embedding_fingerprint": self.embedding_fingerprint,
            "fingerprint_version": MANAGED_INDEX_CONFIGURATION_FINGERPRINT_VERSION,
            "managed_index_format_version": self.managed_index_format_version,
            "payload_schema_version": self.payload_schema_version,
            "point_id_format_version": self.point_id_format_version,
            "sparse": {
                "modifier": self.sparse_modifier,
                "on_disk": self.sparse_on_disk,
                "vector_name": self.sparse_vector_name,
            },
        }


@dataclass(frozen=True, slots=True)
class ManagedIndexConfigurationIdentity:
    """Typed configuration plus its deterministic canonical SHA-256."""

    configuration: ManagedIndexConfiguration
    fingerprint: str

    @property
    def payload(self) -> dict[str, object]:
        """Return a fresh JSON-compatible payload safe to persist as JSONB."""

        return self.configuration.canonical_payload()


def managed_index_configuration_identity(
    configuration: ManagedIndexConfiguration,
) -> ManagedIndexConfigurationIdentity:
    """Fingerprint persisted build settings; embedding dimension is validated separately."""

    return ManagedIndexConfigurationIdentity(
        configuration=configuration,
        fingerprint=_canonical_fingerprint(configuration.canonical_payload()),
    )


def managed_index_configuration_payload_fingerprint(
    payload: Mapping[str, object],
) -> str:
    """Fingerprint a persisted canonical managed-index configuration payload."""

    return _canonical_fingerprint(dict(payload))


def managed_index_configuration_identity_from_settings(
    settings: ManagedIndexSettings,
    *,
    embedding_fingerprint: str,
    chunking_fingerprint: str,
) -> ManagedIndexConfigurationIdentity:
    """Compose the persisted identity from build settings, excluding retrieval tuning."""

    return managed_index_configuration_identity(
        ManagedIndexConfiguration(
            embedding_fingerprint=embedding_fingerprint,
            chunking_fingerprint=chunking_fingerprint,
            bm25_k1=settings.bm25_k1,
            bm25_b=settings.bm25_b,
            dense_vector_name=settings.dense_vector_name,
            sparse_vector_name=settings.sparse_vector_name,
            distance_metric=settings.distance_metric,
            hnsw_m=settings.hnsw_m,
            hnsw_ef_construct=settings.hnsw_ef_construct,
            sparse_modifier=settings.sparse_modifier,
            sparse_on_disk=settings.sparse_on_disk,
            managed_index_format_version=settings.format_version,
        )
    )

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
