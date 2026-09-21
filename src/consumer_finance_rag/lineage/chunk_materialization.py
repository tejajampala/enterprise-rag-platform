"""Materialize and fingerprint the active textual corpus without building an index."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from uuid import UUID

from sqlalchemy.orm import Session

from consumer_finance_rag.chunking import ParagraphChunker
from consumer_finance_rag.hashing import sha256_text
from consumer_finance_rag.lineage.fingerprints import (
    DEFAULT_CHUNKER_IMPLEMENTATION,
    ManagedSnapshotFingerprintRecord,
    chunking_fingerprint,
    managed_snapshot_hash,
)
from consumer_finance_rag.lineage.index_types import (
    ActiveChunk,
    ActiveChunkManifest,
    ActiveDocumentVersion,
)
from consumer_finance_rag.lineage.models import IngestionRun, LineageChunk
from consumer_finance_rag.lineage.repository import (
    InvalidLineageStateError,
    LineageNotFoundError,
    LineageRepository,
)
from consumer_finance_rag.lineage.types import ChunkPersistenceInput
from consumer_finance_rag.models import Chunk, Document

ACTIVE_CHUNK_MANIFEST_VERSION = "active-chunk-manifest-v1"


class ChunkMaterializationError(InvalidLineageStateError):
    """Base exception for invalid active-corpus materialization state."""


class DocumentContentMismatchError(ChunkMaterializationError):
    """Raised when immutable normalized text no longer matches its content hash."""


class ChunkDerivationMismatchError(ChunkMaterializationError):
    """Raised when persisted chunks disagree with deterministic current derivation."""


class ActiveSnapshotResolutionError(ChunkMaterializationError):
    """Raised when current ACTIVE lineage cannot be tied to successful ingestion state."""


class ChunkMaterializationService:
    """Build an exact active chunk manifest inside one caller-owned transaction."""

    def __init__(
        self,
        session: Session,
        *,
        chunk_size: int,
        overlap: int,
        chunker_implementation: str = DEFAULT_CHUNKER_IMPLEMENTATION,
    ) -> None:
        self._session = session
        self._repository = LineageRepository(session)
        self._chunker = ParagraphChunker(chunk_size=chunk_size, overlap=overlap)
        self._chunking_fingerprint = chunking_fingerprint(
            chunk_size=chunk_size,
            overlap=overlap,
            implementation=chunker_implementation,
        )

    @property
    def chunking_fingerprint(self) -> str:
        """Return the one fingerprint used for lookup, writes, and the manifest."""

        return self._chunking_fingerprint

    def materialize_active_manifest(self, corpus_key: str) -> ActiveChunkManifest:
        """Validate or create every requested derivation and return its manifest.

        The caller must wrap this complete operation in ``with session.begin():`` so a
        failure for any document rolls back every new derivation from the pass.
        """

        if not self._session.in_transaction():
            raise ChunkMaterializationError(
                "Chunk materialization requires one caller-owned transaction"
            )
        corpus = self._repository.get_corpus_by_key(corpus_key)
        if corpus is None:
            raise LineageNotFoundError(f"Corpus not found: {corpus_key}")

        source_run = self._resolve_successful_source_run(corpus.corpus_id)
        documents = tuple(
            sorted(
                self._repository.list_active_latest_document_versions(corpus.corpus_id),
                key=lambda document: document.source_key,
            )
        )
        source_snapshot_hash = _active_source_snapshot_hash(documents)
        if (
            source_run.authoritative_snapshot
            and source_run.snapshot_hash != source_snapshot_hash
        ):
            raise ActiveSnapshotResolutionError(
                "Latest authoritative successful ingestion snapshot does not match "
                "the current ACTIVE document versions"
            )

        chunks: list[ActiveChunk] = []
        for document in documents:
            domain_document = _reconstruct_document(document)
            expected = self._chunker.chunk(domain_document)
            persisted = self._repository.list_chunks_for_version_and_fingerprint(
                document.document_version_id,
                self._chunking_fingerprint,
            )
            if persisted:
                _validate_derivation(
                    document,
                    expected,
                    persisted,
                    self._chunking_fingerprint,
                )
                selected = persisted
            else:
                selected = self._repository.create_chunks(
                    document.document_version_id,
                    _persistence_inputs(expected, self._chunking_fingerprint),
                )
            chunks.extend(
                _active_chunk(document, chunk, self._chunking_fingerprint)
                for chunk in selected
            )

        ordered_chunks = tuple(
            sorted(
                chunks,
                key=lambda chunk: (
                    chunk.source_key,
                    chunk.ordinal,
                    chunk.compatibility_chunk_id,
                ),
            )
        )
        manifest_hash = active_chunk_manifest_hash(
            corpus_key=corpus.corpus_key,
            source_snapshot_hash=source_snapshot_hash,
            chunking_fingerprint=self._chunking_fingerprint,
            documents=documents,
            chunks=ordered_chunks,
        )
        return ActiveChunkManifest(
            corpus_id=corpus.corpus_id,
            corpus_key=corpus.corpus_key,
            source_snapshot_hash=source_snapshot_hash,
            chunking_fingerprint=self._chunking_fingerprint,
            documents=documents,
            chunks=ordered_chunks,
            document_count=len(documents),
            chunk_count=len(ordered_chunks),
            manifest_hash=manifest_hash,
        )

    def _resolve_successful_source_run(self, corpus_id: UUID) -> IngestionRun:
        run = self._repository.get_latest_successful_ingestion_run(corpus_id)
        if run is None:
            raise ActiveSnapshotResolutionError(
                f"Corpus {corpus_id} has no successful or no-op ingestion run"
            )
        return run


def active_chunk_manifest_hash(
    *,
    corpus_key: str,
    source_snapshot_hash: str,
    chunking_fingerprint: str,
    documents: Sequence[ActiveDocumentVersion],
    chunks: Sequence[ActiveChunk],
) -> str:
    """Hash logical corpus identity while excluding database-generated UUIDs."""

    ordered_documents = sorted(documents, key=lambda document: document.source_key)
    ordered_chunks = sorted(
        chunks,
        key=lambda chunk: (
            chunk.source_key,
            chunk.ordinal,
            chunk.compatibility_chunk_id,
        ),
    )
    payload = {
        "chunking_fingerprint": chunking_fingerprint,
        "chunks": [
            {
                "chunk_hash": chunk.chunk_hash,
                "compatibility_chunk_id": chunk.compatibility_chunk_id,
                "compatibility_document_id": chunk.compatibility_document_id,
                "content_hash": chunk.content_hash,
                "ordinal": chunk.ordinal,
                "source_key": chunk.source_key,
                "text": chunk.text,
                "access_visibility": chunk.access_policy.visibility.value,
                "allowed_groups": list(chunk.access_policy.canonical_groups),
            }
            for chunk in ordered_chunks
        ],
        "corpus_key": corpus_key,
        "documents": [
            {
                "compatibility_document_id": document.compatibility_document_id,
                "content_hash": document.content_hash,
                "source_key": document.source_key,
                "version_hash": document.version_hash,
                "version_number": document.version_number,
                "access_visibility": document.access_policy.visibility.value,
                "allowed_groups": list(document.access_policy.canonical_groups),
            }
            for document in ordered_documents
        ],
        "manifest_format_version": ACTIVE_CHUNK_MANIFEST_VERSION,
        "source_snapshot_hash": source_snapshot_hash,
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _active_source_snapshot_hash(
    documents: Sequence[ActiveDocumentVersion],
) -> str:
    return managed_snapshot_hash(
        [
            ManagedSnapshotFingerprintRecord(
                source_key=document.source_key,
                content_hash=document.content_hash,
                metadata_hash=document.metadata_hash,
                version_hash=document.version_hash,
            )
            for document in documents
        ]
    )


def _reconstruct_document(document: ActiveDocumentVersion) -> Document:
    actual_hash = sha256_text(document.normalized_text)
    if actual_hash != document.content_hash:
        raise DocumentContentMismatchError(
            f"Document version {document.document_version_id} content hash mismatch: "
            f"persisted {document.content_hash}, calculated {actual_hash}"
        )
    return Document(
        document_id=document.compatibility_document_id,
        source_id=document.source_id,
        title=document.title,
        source_url=document.source_url,
        category=document.category,
        text=document.normalized_text,
        source_version=document.source_version,
        content_hash=document.content_hash,
        access_policy=document.access_policy,
    )


def _persistence_inputs(
    chunks: Sequence[Chunk], chunking_fingerprint_value: str
) -> list[ChunkPersistenceInput]:
    return [
        ChunkPersistenceInput(
            chunking_fingerprint=chunking_fingerprint_value,
            compatibility_chunk_id=chunk.chunk_id,
            ordinal=chunk.ordinal,
            chunk_hash=chunk.chunk_hash,
            text=chunk.text,
            metadata=chunk.metadata,
        )
        for chunk in chunks
    ]


def _validate_derivation(
    document: ActiveDocumentVersion,
    expected: Sequence[Chunk],
    persisted: Sequence[LineageChunk],
    chunking_fingerprint_value: str,
) -> None:
    if len(expected) != len(persisted):
        raise ChunkDerivationMismatchError(
            f"Chunk derivation mismatch for {document.source_key}: expected "
            f"{len(expected)} chunks, found {len(persisted)}"
        )
    for expected_chunk, persisted_chunk in zip(expected, persisted, strict=True):
        expected_values = (
            expected_chunk.ordinal,
            expected_chunk.chunk_id,
            expected_chunk.chunk_hash,
            expected_chunk.text,
            chunking_fingerprint_value,
        )
        persisted_values = (
            persisted_chunk.ordinal,
            persisted_chunk.compatibility_chunk_id,
            persisted_chunk.chunk_hash,
            persisted_chunk.text,
            persisted_chunk.chunking_fingerprint,
        )
        if expected_values != persisted_values:
            raise ChunkDerivationMismatchError(
                f"Chunk derivation mismatch for {document.source_key} at ordinal "
                f"{expected_chunk.ordinal}"
            )


def _active_chunk(
    document: ActiveDocumentVersion,
    chunk: LineageChunk,
    chunking_fingerprint_value: str,
) -> ActiveChunk:
    return ActiveChunk(
        document_pk=document.document_pk,
        document_version_id=document.document_version_id,
        version_number=document.version_number,
        source_key=document.source_key,
        source_id=document.source_id,
        source_url=document.source_url,
        title=document.title,
        category=document.category,
        source_version=document.source_version,
        content_hash=document.content_hash,
        compatibility_document_id=document.compatibility_document_id,
        chunk_pk=chunk.chunk_pk,
        compatibility_chunk_id=chunk.compatibility_chunk_id,
        ordinal=chunk.ordinal,
        chunk_hash=chunk.chunk_hash,
        text=chunk.text,
        chunking_fingerprint=chunking_fingerprint_value,
        access_policy=document.access_policy,
    )
