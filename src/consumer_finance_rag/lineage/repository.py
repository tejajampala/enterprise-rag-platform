"""Session-backed persistence primitives for Phase 3 application lineage."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import and_, func, select
from sqlalchemy.orm import Session, selectinload

from consumer_finance_rag.lineage.index_state import (
    IndexStateReadyMetadata,
    IndexStateValidationError,
    validate_index_state_ready,
    validate_index_state_transition,
)
from consumer_finance_rag.lineage.index_types import (
    ActiveDocumentVersion,
    PreviousBM25GenerationState,
)
from consumer_finance_rag.lineage.models import (
    DENSE_ACTIONS,
    DOCUMENT_CHANGE_TYPES,
    Corpus,
    DocumentVersion,
    IndexState,
    IndexStateChunk,
    IngestionRun,
    IngestionRunDocument,
    LineageChunk,
    LineageDocument,
    utc_now,
)
from consumer_finance_rag.lineage.types import ChunkPersistenceInput

FINISHED_INGESTION_STATUSES = frozenset({"SUCCEEDED", "NOOP"})


class LineageRepositoryError(RuntimeError):
    """Base exception for expected lineage repository failures."""


class LineageConflictError(LineageRepositoryError):
    """Raised when a requested write conflicts with existing lineage state."""


class LineageNotFoundError(LineageRepositoryError):
    """Raised when a required lineage record does not exist."""


class InvalidLineageStateError(LineageRepositoryError):
    """Raised when a requested state or transition is invalid."""


class MissingActiveDocumentVersionError(InvalidLineageStateError):
    """Raised when an ACTIVE logical document has no unambiguous latest version."""


class LineageRepository:
    """Persist lineage entities through a caller-owned SQLAlchemy session."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def get_corpus_by_key(self, corpus_key: str) -> Corpus | None:
        """Return a corpus by stable key, if present."""

        return self._session.scalar(select(Corpus).where(Corpus.corpus_key == corpus_key))

    def create_corpus(self, corpus_key: str, qdrant_collection_prefix: str) -> Corpus:
        """Create a managed corpus without committing the caller's transaction."""

        if self.get_corpus_by_key(corpus_key) is not None:
            raise LineageConflictError(f"Corpus key already exists: {corpus_key}")
        prefix_owner = self._session.scalar(
            select(Corpus).where(
                Corpus.qdrant_collection_prefix == qdrant_collection_prefix
            )
        )
        if prefix_owner is not None:
            raise LineageConflictError(
                "Qdrant collection prefix is already assigned to corpus "
                f"{prefix_owner.corpus_key}: {qdrant_collection_prefix}"
            )
        corpus = Corpus(
            corpus_key=corpus_key,
            qdrant_collection_prefix=qdrant_collection_prefix,
        )
        self._session.add(corpus)
        self._session.flush()
        return corpus

    def get_or_create_corpus(
        self, corpus_key: str, qdrant_collection_prefix: str
    ) -> Corpus:
        """Return an exact corpus configuration or create it deterministically."""

        corpus = self.get_corpus_by_key(corpus_key)
        if corpus is None:
            return self.create_corpus(corpus_key, qdrant_collection_prefix)
        if corpus.qdrant_collection_prefix != qdrant_collection_prefix:
            raise LineageConflictError(
                f"Corpus {corpus_key} already uses Qdrant prefix "
                f"{corpus.qdrant_collection_prefix!r}, not {qdrant_collection_prefix!r}"
            )
        return corpus

    def start_ingestion_run(
        self,
        corpus_id: UUID,
        authoritative_snapshot: bool,
        snapshot_hash: str | None = None,
    ) -> IngestionRun:
        """Start a RUNNING ingestion attempt for an existing corpus."""

        self._require_corpus(corpus_id)
        run = IngestionRun(
            corpus_id=corpus_id,
            status="RUNNING",
            authoritative_snapshot=authoritative_snapshot,
            snapshot_hash=snapshot_hash,
        )
        self._session.add(run)
        self._session.flush()
        return run

    def get_ingestion_run(self, run_id: UUID) -> IngestionRun | None:
        """Return an ingestion run by ID, if present."""

        return self._session.get(IngestionRun, run_id)

    def finish_ingestion_run(
        self,
        run_id: UUID,
        status: str,
        completed_at: datetime,
        documents_new: int,
        documents_changed: int,
        documents_unchanged: int,
        documents_deleted: int,
    ) -> IngestionRun:
        """Finish a RUNNING attempt as SUCCEEDED or NOOP and persist its counts."""

        run = self._require_ingestion_run(run_id)
        self._require_running(run)
        if status not in FINISHED_INGESTION_STATUSES:
            raise InvalidLineageStateError(
                "finish_ingestion_run status must be SUCCEEDED or NOOP"
            )
        self._require_aware_timestamp(completed_at)
        counts = (
            documents_new,
            documents_changed,
            documents_unchanged,
            documents_deleted,
        )
        if any(count < 0 for count in counts):
            raise InvalidLineageStateError("Ingestion document counts cannot be negative")
        run.status = status
        run.completed_at = completed_at
        run.documents_new = documents_new
        run.documents_changed = documents_changed
        run.documents_unchanged = documents_unchanged
        run.documents_deleted = documents_deleted
        self._session.flush()
        return run

    def fail_ingestion_run(
        self,
        run_id: UUID,
        error_message: str,
        completed_at: datetime,
        *,
        documents_new: int = 0,
        documents_changed: int = 0,
        documents_unchanged: int = 0,
        documents_deleted: int = 0,
    ) -> IngestionRun:
        """Finish a RUNNING attempt as FAILED with its planned counts."""

        run = self._require_ingestion_run(run_id)
        self._require_running(run)
        self._require_aware_timestamp(completed_at)
        counts = (
            documents_new,
            documents_changed,
            documents_unchanged,
            documents_deleted,
        )
        if any(count < 0 for count in counts):
            raise InvalidLineageStateError("Ingestion document counts cannot be negative")
        run.status = "FAILED"
        run.error_message = error_message
        run.completed_at = completed_at
        run.documents_new = documents_new
        run.documents_changed = documents_changed
        run.documents_unchanged = documents_unchanged
        run.documents_deleted = documents_deleted
        self._session.flush()
        return run

    def get_document_by_source_key(
        self, corpus_id: UUID, source_key: str
    ) -> LineageDocument | None:
        """Return a logical document by its corpus-scoped source key."""

        return self._session.scalar(
            select(LineageDocument).where(
                LineageDocument.corpus_id == corpus_id,
                LineageDocument.source_key == source_key,
            )
        )

    def list_active_documents(self, corpus_id: UUID) -> list[LineageDocument]:
        """List ACTIVE logical documents in deterministic source-key order."""

        return list(
            self._session.scalars(
                select(LineageDocument)
                .where(
                    LineageDocument.corpus_id == corpus_id,
                    LineageDocument.state == "ACTIVE",
                )
                .order_by(LineageDocument.source_key)
            )
        )

    def list_active_documents_with_latest_versions(
        self, corpus_id: UUID
    ) -> list[tuple[LineageDocument, DocumentVersion | None]]:
        """Load ACTIVE documents and their latest versions without N+1 queries."""

        latest_numbers = (
            select(
                DocumentVersion.document_pk.label("document_pk"),
                func.max(DocumentVersion.version_number).label("version_number"),
            )
            .group_by(DocumentVersion.document_pk)
            .subquery()
        )
        statement = (
            select(LineageDocument, DocumentVersion)
            .outerjoin(
                latest_numbers,
                latest_numbers.c.document_pk == LineageDocument.document_pk,
            )
            .outerjoin(
                DocumentVersion,
                and_(
                    DocumentVersion.document_pk == LineageDocument.document_pk,
                    DocumentVersion.version_number == latest_numbers.c.version_number,
                ),
            )
            .where(
                LineageDocument.corpus_id == corpus_id,
                LineageDocument.state == "ACTIVE",
            )
            .order_by(LineageDocument.source_key)
        )
        return list(self._session.execute(statement).tuples())

    def list_active_latest_document_versions(
        self, corpus_id: UUID
    ) -> list[ActiveDocumentVersion]:
        """Project ACTIVE documents and latest versions into immutable ordered values."""

        rows = self.list_active_documents_with_latest_versions(corpus_id)
        projected: list[ActiveDocumentVersion] = []
        seen_documents: set[UUID] = set()
        for document, version in rows:
            if document.document_pk in seen_documents:
                raise MissingActiveDocumentVersionError(
                    "ACTIVE document has multiple rows for its maximum version number: "
                    f"{document.document_pk}"
                )
            seen_documents.add(document.document_pk)
            if version is None:
                raise MissingActiveDocumentVersionError(
                    f"ACTIVE document {document.document_pk} has no document version"
                )
            projected.append(
                ActiveDocumentVersion(
                    document_pk=document.document_pk,
                    document_version_id=version.document_version_id,
                    version_number=version.version_number,
                    source_key=document.source_key,
                    source_id=document.source_id,
                    source_url=version.source_url,
                    title=version.title,
                    category=version.category,
                    source_version=version.source_version,
                    normalized_text=version.normalized_text,
                    content_hash=version.content_hash,
                    metadata_hash=version.metadata_hash,
                    version_hash=version.version_hash,
                    compatibility_document_id=version.compatibility_document_id,
                )
            )
        return projected

    def get_latest_successful_ingestion_run(
        self, corpus_id: UUID
    ) -> IngestionRun | None:
        """Return the latest SUCCEEDED/NOOP run, ignoring later failed attempts."""

        self._require_corpus(corpus_id)
        return self._session.scalar(
            select(IngestionRun)
            .where(
                IngestionRun.corpus_id == corpus_id,
                IngestionRun.status.in_(FINISHED_INGESTION_STATUSES),
                IngestionRun.completed_at.is_not(None),
            )
            .order_by(
                IngestionRun.completed_at.desc(),
                IngestionRun.started_at.desc(),
                IngestionRun.run_id.desc(),
            )
            .limit(1)
        )

    def create_document(
        self,
        corpus_id: UUID,
        source_key: str,
        source_id: str,
        source_url: str | None,
        first_seen_run_id: UUID,
    ) -> LineageDocument:
        """Create a stable logical source document in ACTIVE state."""

        self._require_corpus(corpus_id)
        run = self._require_ingestion_run(first_seen_run_id)
        self._require_same_corpus(corpus_id, run.corpus_id, "ingestion run")
        if self.get_document_by_source_key(corpus_id, source_key) is not None:
            raise LineageConflictError(
                f"Source key already exists in corpus {corpus_id}: {source_key}"
            )
        document = LineageDocument(
            corpus_id=corpus_id,
            source_key=source_key,
            source_id=source_id,
            source_url=source_url,
            state="ACTIVE",
            first_seen_run_id=first_seen_run_id,
            last_seen_run_id=first_seen_run_id,
        )
        self._session.add(document)
        self._session.flush()
        return document

    def mark_document_seen(
        self,
        document_pk: UUID,
        run_id: UUID,
        source_id: str,
        source_url: str | None,
    ) -> LineageDocument:
        """Update observation metadata and ensure the logical document is ACTIVE."""

        document = self._require_document(document_pk)
        run = self._require_ingestion_run(run_id)
        self._require_same_corpus(document.corpus_id, run.corpus_id, "ingestion run")
        document.last_seen_run_id = run_id
        document.source_id = source_id
        document.source_url = source_url
        document.state = "ACTIVE"
        document.deleted_run_id = None
        document.updated_at = utc_now()
        self._session.flush()
        return document

    def mark_document_deleted(
        self, document_pk: UUID, run_id: UUID
    ) -> LineageDocument:
        """Soft-delete a logical document in the specified corpus run."""

        document = self._require_document(document_pk)
        run = self._require_ingestion_run(run_id)
        self._require_same_corpus(document.corpus_id, run.corpus_id, "ingestion run")
        document.state = "DELETED"
        document.deleted_run_id = run_id
        document.updated_at = utc_now()
        self._session.flush()
        return document

    def get_latest_document_version(self, document_pk: UUID) -> DocumentVersion | None:
        """Return the greatest version number for a logical document."""

        return self._session.scalar(
            select(DocumentVersion)
            .where(DocumentVersion.document_pk == document_pk)
            .order_by(DocumentVersion.version_number.desc())
            .limit(1)
        )

    def get_document_version_by_hash(
        self, document_pk: UUID, version_hash: str
    ) -> DocumentVersion | None:
        """Return a document version by its document-scoped version hash."""

        return self._session.scalar(
            select(DocumentVersion).where(
                DocumentVersion.document_pk == document_pk,
                DocumentVersion.version_hash == version_hash,
            )
        )

    def create_document_version(
        self,
        document_pk: UUID,
        version_number: int,
        content_hash: str,
        metadata_hash: str,
        version_hash: str,
        compatibility_document_id: str,
        title: str,
        category: str,
        source_url: str | None,
        source_version: str,
        normalized_text: str,
        created_run_id: UUID,
    ) -> DocumentVersion:
        """Persist precomputed immutable document-version values."""

        document = self._require_document(document_pk)
        run = self._require_ingestion_run(created_run_id)
        self._require_same_corpus(document.corpus_id, run.corpus_id, "ingestion run")
        if version_number <= 0:
            raise InvalidLineageStateError("Document version number must be positive")
        existing_number = self._session.scalar(
            select(DocumentVersion.document_version_id).where(
                DocumentVersion.document_pk == document_pk,
                DocumentVersion.version_number == version_number,
            )
        )
        if existing_number is not None:
            raise LineageConflictError(
                f"Document {document_pk} already has version number {version_number}"
            )
        if self.get_document_version_by_hash(document_pk, version_hash) is not None:
            raise LineageConflictError(
                f"Document {document_pk} already has version hash {version_hash}"
            )
        version = DocumentVersion(
            document_pk=document_pk,
            version_number=version_number,
            content_hash=content_hash,
            metadata_hash=metadata_hash,
            version_hash=version_hash,
            compatibility_document_id=compatibility_document_id,
            title=title,
            category=category,
            source_url=source_url,
            source_version=source_version,
            normalized_text=normalized_text,
            created_run_id=created_run_id,
        )
        self._session.add(version)
        self._session.flush()
        return version

    def list_chunks_for_version(self, document_version_id: UUID) -> list[LineageChunk]:
        """List chunks deterministically by ordinal and persisted identity."""

        return list(
            self._session.scalars(
                select(LineageChunk)
                .where(LineageChunk.document_version_id == document_version_id)
                .order_by(
                    LineageChunk.ordinal,
                    LineageChunk.chunking_fingerprint,
                    LineageChunk.compatibility_chunk_id,
                )
            )
        )

    def list_chunks_for_version_and_fingerprint(
        self,
        document_version_id: UUID,
        chunking_fingerprint: str,
    ) -> list[LineageChunk]:
        """List one immutable chunk derivation in canonical ordinal order."""

        return list(
            self._session.scalars(
                select(LineageChunk)
                .where(
                    LineageChunk.document_version_id == document_version_id,
                    LineageChunk.chunking_fingerprint == chunking_fingerprint,
                )
                .order_by(
                    LineageChunk.ordinal,
                    LineageChunk.compatibility_chunk_id,
                )
            )
        )

    def create_chunks(
        self,
        document_version_id: UUID,
        chunks: Sequence[ChunkPersistenceInput],
    ) -> list[LineageChunk]:
        """Persist precomputed chunks for one version without deriving identities."""

        self._require_document_version(document_version_id)
        existing_chunks = self.list_chunks_for_version(document_version_id)
        ordinal_keys = {
            (chunk.chunking_fingerprint, chunk.ordinal) for chunk in existing_chunks
        }
        compatibility_keys = {
            (chunk.chunking_fingerprint, chunk.compatibility_chunk_id)
            for chunk in existing_chunks
        }
        created: list[LineageChunk] = []
        for values in chunks:
            if values.ordinal < 0:
                raise InvalidLineageStateError("Chunk ordinal cannot be negative")
            ordinal_key = (values.chunking_fingerprint, values.ordinal)
            compatibility_key = (
                values.chunking_fingerprint,
                values.compatibility_chunk_id,
            )
            if ordinal_key in ordinal_keys:
                raise LineageConflictError(
                    "Chunk ordinal already exists for document version and fingerprint: "
                    f"{values.ordinal}"
                )
            if compatibility_key in compatibility_keys:
                raise LineageConflictError(
                    "Compatibility chunk ID already exists for document version and fingerprint: "
                    f"{values.compatibility_chunk_id}"
                )
            ordinal_keys.add(ordinal_key)
            compatibility_keys.add(compatibility_key)
            created.append(
                LineageChunk(
                    document_version_id=document_version_id,
                    chunking_fingerprint=values.chunking_fingerprint,
                    compatibility_chunk_id=values.compatibility_chunk_id,
                    ordinal=values.ordinal,
                    chunk_hash=values.chunk_hash,
                    text=values.text,
                    chunk_metadata=dict(values.metadata),
                )
            )
        self._session.add_all(created)
        self._session.flush()
        return created

    def record_run_document(
        self,
        run_id: UUID,
        document_pk: UUID,
        change_type: str,
        previous_version_id: UUID | None = None,
        observed_version_id: UUID | None = None,
        content_changed: bool = False,
        metadata_changed: bool = False,
    ) -> IngestionRunDocument:
        """Persist a caller-supplied run classification without calculating it."""

        run = self._require_ingestion_run(run_id)
        document = self._require_document(document_pk)
        self._require_same_corpus(document.corpus_id, run.corpus_id, "document")
        self._require_domain_value("change_type", change_type, DOCUMENT_CHANGE_TYPES)
        existing = self._session.get(IngestionRunDocument, (run_id, document_pk))
        if existing is not None:
            raise LineageConflictError(
                f"Run {run_id} already records document {document_pk}"
            )
        if previous_version_id is not None:
            previous = self._require_document_version(previous_version_id)
            self._require_same_document(document_pk, previous.document_pk, "previous version")
        if observed_version_id is not None:
            observed = self._require_document_version(observed_version_id)
            self._require_same_document(document_pk, observed.document_pk, "observed version")
        record = IngestionRunDocument(
            run_id=run_id,
            document_pk=document_pk,
            change_type=change_type,
            previous_version_id=previous_version_id,
            observed_version_id=observed_version_id,
            content_changed=content_changed,
            metadata_changed=metadata_changed,
        )
        self._session.add(record)
        self._session.flush()
        return record

    def get_run_documents(self, run_id: UUID) -> list[IngestionRunDocument]:
        """Return persisted document classifications for one run."""

        return list(
            self._session.scalars(
                select(IngestionRunDocument)
                .where(IngestionRunDocument.run_id == run_id)
                .order_by(IngestionRunDocument.document_pk)
            )
        )

    def get_active_index_state(self, corpus_id: UUID) -> IndexState | None:
        """Return the sole ACTIVE index or fail on invalid multiple-active state."""

        active = list(
            self._session.scalars(
                select(IndexState)
                .where(
                    IndexState.corpus_id == corpus_id,
                    IndexState.status == "ACTIVE",
                )
                .order_by(IndexState.generation.desc())
                .limit(2)
            )
        )
        if len(active) > 1:
            raise InvalidLineageStateError(
                f"Corpus {corpus_id} has more than one ACTIVE index state"
            )
        return active[0] if active else None

    def get_active_bm25_generation_state(
        self, corpus_id: UUID
    ) -> PreviousBM25GenerationState | None:
        """Detach BM25/configuration fields from the optional ACTIVE generation."""

        state = self.get_active_index_state(corpus_id)
        if state is None:
            return None
        return PreviousBM25GenerationState(
            index_state_id=state.index_state_id,
            bm25_state=deepcopy(state.bm25_state),
            bm25_state_hash=state.bm25_state_hash,
            bm25_corpus_hash=state.bm25_corpus_hash,
            index_configuration=deepcopy(state.index_configuration),
            index_configuration_fingerprint=state.index_configuration_fingerprint,
            chunking_fingerprint=state.chunking_fingerprint,
            chunk_count=state.chunk_count,
        )

    def get_index_state(self, index_state_id: UUID) -> IndexState | None:
        """Return one managed index generation, if present."""

        return self._session.get(IndexState, index_state_id)

    def list_index_states(self, corpus_id: UUID) -> list[IndexState]:
        """List a corpus's immutable generation attempts in generation order."""

        self._require_corpus(corpus_id)
        return list(
            self._session.scalars(
                select(IndexState)
                .where(IndexState.corpus_id == corpus_id)
                .order_by(IndexState.generation, IndexState.index_state_id)
            )
        )

    def get_latest_index_state(self, corpus_id: UUID) -> IndexState | None:
        """Return the greatest index generation for a corpus."""

        return self._session.scalar(
            select(IndexState)
            .where(IndexState.corpus_id == corpus_id)
            .order_by(IndexState.generation.desc())
            .limit(1)
        )

    def get_latest_generation_number(self, corpus_id: UUID) -> int:
        """Return the latest generation number, or zero; callers must lock allocation."""

        self._require_corpus(corpus_id)
        return (
            self._session.scalar(
                select(func.max(IndexState.generation)).where(
                    IndexState.corpus_id == corpus_id
                )
            )
            or 0
        )

    def create_building_index_state(
        self,
        corpus_id: UUID,
        generation: int,
        ingestion_run_id: UUID,
        qdrant_collection: str,
        snapshot_hash: str,
        embedding_fingerprint: str,
        chunking_fingerprint: str,
        document_count: int = 0,
        chunk_count: int = 0,
    ) -> IndexState:
        """Create an incomplete BUILDING generation without committing."""

        self._require_corpus(corpus_id)
        run = self._require_ingestion_run(ingestion_run_id)
        self._require_same_corpus(corpus_id, run.corpus_id, "ingestion run")
        if generation <= 0:
            raise InvalidLineageStateError("Index generation must be positive")
        if document_count < 0 or chunk_count < 0:
            raise InvalidLineageStateError("Index document and chunk counts cannot be negative")
        if not qdrant_collection.strip():
            raise InvalidLineageStateError("Qdrant collection cannot be empty")
        existing_generation = self._session.scalar(
            select(IndexState.index_state_id).where(
                IndexState.corpus_id == corpus_id,
                IndexState.generation == generation,
            )
        )
        if existing_generation is not None:
            raise LineageConflictError(
                f"Corpus {corpus_id} already has index generation {generation}"
            )
        existing_collection = self._session.scalar(
            select(IndexState.index_state_id).where(
                IndexState.qdrant_collection == qdrant_collection
            )
        )
        if existing_collection is not None:
            raise LineageConflictError(
                f"Qdrant collection is already registered: {qdrant_collection}"
            )
        state = IndexState(
            corpus_id=corpus_id,
            generation=generation,
            ingestion_run_id=ingestion_run_id,
            status="BUILDING",
            qdrant_collection=qdrant_collection,
            snapshot_hash=snapshot_hash,
            embedding_fingerprint=embedding_fingerprint,
            embedding_dimension=None,
            chunking_fingerprint=chunking_fingerprint,
            document_count=document_count,
            chunk_count=chunk_count,
        )
        self._session.add(state)
        self._session.flush()
        return state

    def update_index_build_metadata(
        self,
        index_state_id: UUID,
        *,
        bm25_state: Mapping[str, Any],
        bm25_state_hash: str,
        bm25_corpus_hash: str,
        index_configuration: Mapping[str, Any],
        index_configuration_fingerprint: str,
        embedding_dimension: int,
        document_count: int,
        chunk_count: int,
    ) -> IndexState:
        """Populate validated build outputs while a generation remains BUILDING."""

        state = self._require_index_state(index_state_id)
        if state.status != "BUILDING":
            raise InvalidLineageStateError(
                f"Index state {index_state_id} is {state.status}, not BUILDING"
            )
        if embedding_dimension <= 0:
            raise InvalidLineageStateError("Embedding dimension must be positive")
        if document_count < 0 or chunk_count < 0:
            raise InvalidLineageStateError("Index document and chunk counts cannot be negative")
        state.bm25_state = dict(bm25_state)
        state.bm25_state_hash = bm25_state_hash
        state.bm25_corpus_hash = bm25_corpus_hash
        state.index_configuration = dict(index_configuration)
        state.index_configuration_fingerprint = index_configuration_fingerprint
        state.embedding_dimension = embedding_dimension
        state.document_count = document_count
        state.chunk_count = chunk_count
        self._session.flush()
        return state

    def transition_index_state(
        self,
        index_state_id: UUID,
        expected_current_status: str,
        target_status: str,
        *,
        occurred_at: datetime | None = None,
        error_message: str | None = None,
    ) -> IndexState:
        """Persist one validated state transition without committing."""

        state = self._require_index_state(index_state_id)
        if state.status != expected_current_status:
            raise InvalidLineageStateError(
                f"Index state {index_state_id} is {state.status}, "
                f"not {expected_current_status}"
            )
        try:
            validate_index_state_transition(expected_current_status, target_status)
        except IndexStateValidationError as exc:
            raise InvalidLineageStateError(str(exc)) from exc

        if target_status == "READY":
            try:
                validate_index_state_ready(self._ready_metadata(state))
            except IndexStateValidationError as exc:
                raise InvalidLineageStateError(str(exc)) from exc
        elif target_status == "ACTIVE":
            self._require_transition_timestamp(occurred_at)
            active = self.get_active_index_state(state.corpus_id)
            if active is not None and active.index_state_id != state.index_state_id:
                raise InvalidLineageStateError(
                    f"Corpus {state.corpus_id} already has ACTIVE generation "
                    f"{active.generation}"
                )
            state.activated_at = occurred_at
        elif target_status == "RETIRED":
            self._require_transition_timestamp(occurred_at)
            state.retired_at = occurred_at
        elif target_status == "FAILED":
            self._require_transition_timestamp(occurred_at)
            if not error_message or not error_message.strip():
                raise InvalidLineageStateError(
                    "FAILED index state requires a non-empty error message"
                )
            state.error_message = error_message
            state.failed_at = occurred_at

        state.status = target_status
        self._session.flush()
        return state

    def mark_index_state_failed(
        self,
        index_state_id: UUID,
        expected_current_status: str,
        error_message: str,
        failed_at: datetime,
    ) -> IndexState:
        """Mark a BUILDING or READY generation FAILED without committing."""

        return self.transition_index_state(
            index_state_id,
            expected_current_status,
            "FAILED",
            occurred_at=failed_at,
            error_message=error_message,
        )

    def list_index_state_chunks(self, index_state_id: UUID) -> list[IndexStateChunk]:
        """List generation membership in deterministic chunk order."""

        return list(
            self._session.scalars(
                select(IndexStateChunk)
                .join(LineageChunk, IndexStateChunk.chunk_pk == LineageChunk.chunk_pk)
                .where(IndexStateChunk.index_state_id == index_state_id)
                .order_by(
                    LineageChunk.ordinal,
                    LineageChunk.compatibility_chunk_id,
                    IndexStateChunk.qdrant_point_id,
                )
            )
        )

    def record_index_state_chunk(
        self,
        index_state_id: UUID,
        chunk_pk: UUID,
        qdrant_point_id: UUID,
        dense_action: str,
        dense_source_chunk_pk: UUID | None = None,
    ) -> IndexStateChunk:
        """Persist caller-supplied membership and dense-vector provenance."""

        state = self._require_index_state(index_state_id)
        chunk = self._require_chunk(chunk_pk)
        self._require_same_corpus(
            state.corpus_id,
            chunk.document_version.document.corpus_id,
            "chunk",
        )
        self._require_domain_value("dense_action", dense_action, DENSE_ACTIONS)
        if dense_action == "COMPUTED" and dense_source_chunk_pk is not None:
            raise InvalidLineageStateError(
                "COMPUTED dense provenance cannot reference a source chunk"
            )
        if dense_action == "REUSED" and dense_source_chunk_pk is None:
            raise InvalidLineageStateError(
                "REUSED dense provenance requires a source chunk"
            )
        if self._session.get(IndexStateChunk, (index_state_id, chunk_pk)) is not None:
            raise LineageConflictError(
                f"Index state {index_state_id} already contains chunk {chunk_pk}"
            )
        existing_point = self._session.scalar(
            select(IndexStateChunk.chunk_pk).where(
                IndexStateChunk.index_state_id == index_state_id,
                IndexStateChunk.qdrant_point_id == qdrant_point_id,
            )
        )
        if existing_point is not None:
            raise LineageConflictError(
                f"Index state {index_state_id} already uses Qdrant point {qdrant_point_id}"
            )
        if dense_source_chunk_pk is not None:
            dense_source = self._require_chunk(dense_source_chunk_pk)
            self._require_same_corpus(
                state.corpus_id,
                dense_source.document_version.document.corpus_id,
                "dense source chunk",
            )
        membership = IndexStateChunk(
            index_state_id=index_state_id,
            chunk_pk=chunk_pk,
            qdrant_point_id=qdrant_point_id,
            dense_action=dense_action,
            dense_source_chunk_pk=dense_source_chunk_pk,
        )
        self._session.add(membership)
        self._session.flush()
        return membership

    def find_reusable_chunk(
        self, corpus_id: UUID, chunk_hash: str
    ) -> list[LineageChunk]:
        """Return exact-hash candidates with memberships loaded for compatibility checks."""

        self._require_corpus(corpus_id)
        return list(
            self._session.scalars(
                select(LineageChunk)
                .join(DocumentVersion)
                .join(LineageDocument)
                .where(
                    LineageDocument.corpus_id == corpus_id,
                    LineageChunk.chunk_hash == chunk_hash,
                )
                .options(
                    selectinload(LineageChunk.index_memberships).selectinload(
                        IndexStateChunk.index_state
                    )
                )
                .order_by(
                    DocumentVersion.version_number.desc(),
                    LineageDocument.document_pk,
                    LineageChunk.ordinal,
                    LineageChunk.chunk_pk,
                )
            )
        )

    def _require_corpus(self, corpus_id: UUID) -> Corpus:
        corpus = self._session.get(Corpus, corpus_id)
        if corpus is None:
            raise LineageNotFoundError(f"Corpus not found: {corpus_id}")
        return corpus

    def _require_ingestion_run(self, run_id: UUID) -> IngestionRun:
        run = self.get_ingestion_run(run_id)
        if run is None:
            raise LineageNotFoundError(f"Ingestion run not found: {run_id}")
        return run

    def _require_document(self, document_pk: UUID) -> LineageDocument:
        document = self._session.get(LineageDocument, document_pk)
        if document is None:
            raise LineageNotFoundError(f"Document not found: {document_pk}")
        return document

    def _require_document_version(self, document_version_id: UUID) -> DocumentVersion:
        version = self._session.get(DocumentVersion, document_version_id)
        if version is None:
            raise LineageNotFoundError(
                f"Document version not found: {document_version_id}"
            )
        return version

    def _require_chunk(self, chunk_pk: UUID) -> LineageChunk:
        chunk = self._session.get(LineageChunk, chunk_pk)
        if chunk is None:
            raise LineageNotFoundError(f"Chunk not found: {chunk_pk}")
        return chunk

    def _require_index_state(self, index_state_id: UUID) -> IndexState:
        state = self._session.get(IndexState, index_state_id)
        if state is None:
            raise LineageNotFoundError(f"Index state not found: {index_state_id}")
        return state

    @staticmethod
    def _ready_metadata(state: IndexState) -> IndexStateReadyMetadata:
        return IndexStateReadyMetadata(
            qdrant_collection=state.qdrant_collection,
            embedding_dimension=state.embedding_dimension,
            bm25_state=state.bm25_state,
            bm25_state_hash=state.bm25_state_hash,
            bm25_corpus_hash=state.bm25_corpus_hash,
            index_configuration=state.index_configuration,
            index_configuration_fingerprint=state.index_configuration_fingerprint,
            document_count=state.document_count,
            chunk_count=state.chunk_count,
        )

    @classmethod
    def _require_transition_timestamp(cls, value: datetime | None) -> None:
        if value is None:
            raise InvalidLineageStateError("Index state transition requires a timestamp")
        cls._require_aware_timestamp(value)

    @staticmethod
    def _require_running(run: IngestionRun) -> None:
        if run.status != "RUNNING":
            raise InvalidLineageStateError(
                f"Ingestion run {run.run_id} is {run.status}, not RUNNING"
            )

    @staticmethod
    def _require_aware_timestamp(value: datetime) -> None:
        if value.tzinfo is None or value.utcoffset() is None:
            raise InvalidLineageStateError("Timestamps must be timezone-aware")

    @staticmethod
    def _require_same_corpus(expected: UUID, actual: UUID, label: str) -> None:
        if expected != actual:
            raise InvalidLineageStateError(
                f"{label.capitalize()} belongs to a different corpus"
            )

    @staticmethod
    def _require_same_document(expected: UUID, actual: UUID, label: str) -> None:
        if expected != actual:
            raise InvalidLineageStateError(
                f"{label.capitalize()} belongs to a different document"
            )

    @staticmethod
    def _require_domain_value(
        attribute: str, value: str, allowed: frozenset[str]
    ) -> None:
        if value not in allowed:
            choices = ", ".join(sorted(allowed))
            raise InvalidLineageStateError(
                f"{attribute} must be one of: {choices}"
            )
