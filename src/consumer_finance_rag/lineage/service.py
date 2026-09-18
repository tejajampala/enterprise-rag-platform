"""Read-only bridge from current/persisted lineage state into the pure planner."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy.orm import Session

from consumer_finance_rag.corpus import CorpusSnapshot
from consumer_finance_rag.lineage.diff import (
    ChangePlan,
    ChangeType,
    CurrentDocumentState,
    PersistedDocumentState,
    PlannedDocumentChange,
    current_document_state,
    plan_document_changes,
)
from consumer_finance_rag.lineage.fingerprints import (
    ManagedSnapshotFingerprintRecord,
    chunking_fingerprint,
    managed_snapshot_hash,
)
from consumer_finance_rag.lineage.models import (
    DocumentVersion,
    LineageDocument,
    utc_now,
)
from consumer_finance_rag.lineage.repository import (
    InvalidLineageStateError,
    LineageConflictError,
    LineageRepository,
)
from consumer_finance_rag.lineage.types import ChunkPersistenceInput
from consumer_finance_rag.models import Chunk, Document


@dataclass(frozen=True, slots=True)
class LineageDryRunResult:
    """Read-only planning result plus source-state counts for presentation."""

    plan: ChangePlan
    corpus_initialized: bool
    current_document_count: int
    previous_active_document_count: int
    snapshot_hash: str


@dataclass(frozen=True, slots=True)
class LineageApplyResult:
    """Committed ingestion-run identity and plan-derived persistence counts."""

    run_id: UUID
    status: str
    new_count: int
    changed_count: int
    unchanged_count: int
    deleted_count: int


class UnsupportedResurrectionError(InvalidLineageStateError):
    """Raised when a DELETED logical source reappears before restoration is supported."""


class StaleChangePlanError(InvalidLineageStateError):
    """Raised when persisted latest-version state no longer matches a plan."""


class LineagePlanningService:
    """Adapt current and persisted snapshots without writing either system."""

    def __init__(self, repository: LineageRepository) -> None:
        self._repository = repository

    def plan_snapshot(
        self,
        snapshot: CorpusSnapshot,
        *,
        corpus_key: str,
        authoritative_snapshot: bool,
    ) -> LineageDryRunResult:
        """Read previous state and produce a pure deterministic change plan."""

        current = current_states_from_snapshot(snapshot)
        previous, corpus_initialized = self.load_previous_states(corpus_key)
        plan = plan_document_changes(
            current,
            previous,
            authoritative_snapshot=authoritative_snapshot,
        )
        return LineageDryRunResult(
            plan=plan,
            corpus_initialized=corpus_initialized,
            current_document_count=len(current),
            previous_active_document_count=len(previous),
            snapshot_hash=snapshot_hash_for_current_states(current),
        )

    def load_previous_states(
        self, corpus_key: str
    ) -> tuple[tuple[PersistedDocumentState, ...], bool]:
        """Load ACTIVE lineage state, treating an absent managed corpus as uninitialized."""

        corpus = self._repository.get_corpus_by_key(corpus_key)
        if corpus is None:
            return (), False
        rows = self._repository.list_active_documents_with_latest_versions(corpus.corpus_id)
        states: list[PersistedDocumentState] = []
        for document, version in rows:
            if version is None:
                raise InvalidLineageStateError(
                    f"ACTIVE document {document.document_pk} has no document version"
                )
            states.append(persisted_document_state(document, version))
        return tuple(states), True


class LineageApplyService:
    """Apply an existing ChangePlan using explicit caller-visible transactions."""

    def __init__(self, session: Session, *, chunk_size: int, overlap: int) -> None:
        self._session = session
        self._repository = LineageRepository(session)
        self._chunking_fingerprint = chunking_fingerprint(
            chunk_size=chunk_size,
            overlap=overlap,
        )

    def apply_plan(
        self,
        plan: ChangePlan,
        *,
        corpus_key: str,
        qdrant_collection_prefix: str,
        snapshot_hash: str,
    ) -> LineageApplyResult:
        """Persist one planned attempt and preserve a FAILED audit row after rollback."""

        if self._session.in_transaction():
            raise InvalidLineageStateError(
                "LineageApplyService requires a session with no active transaction"
            )

        with self._session.begin():
            corpus = self._repository.get_or_create_corpus(
                corpus_key,
                qdrant_collection_prefix,
            )
            run = self._repository.start_ingestion_run(
                corpus.corpus_id,
                authoritative_snapshot=plan.authoritative_snapshot,
                snapshot_hash=snapshot_hash,
            )
            corpus_id = corpus.corpus_id
            run_id = run.run_id

        try:
            with self._session.begin():
                for item in plan.items:
                    self._apply_item(item, corpus_id=corpus_id, run_id=run_id)
                status = (
                    "NOOP"
                    if plan.new_count == 0
                    and plan.changed_count == 0
                    and plan.deleted_count == 0
                    else "SUCCEEDED"
                )
                self._repository.finish_ingestion_run(
                    run_id,
                    status,
                    utc_now(),
                    documents_new=plan.new_count,
                    documents_changed=plan.changed_count,
                    documents_unchanged=plan.unchanged_count,
                    documents_deleted=plan.deleted_count,
                )
        except Exception as exc:
            with self._session.begin():
                self._repository.fail_ingestion_run(
                    run_id,
                    f"{type(exc).__name__}: {exc}",
                    utc_now(),
                    documents_new=plan.new_count,
                    documents_changed=plan.changed_count,
                    documents_unchanged=plan.unchanged_count,
                    documents_deleted=plan.deleted_count,
                )
            raise

        return LineageApplyResult(
            run_id=run_id,
            status=status,
            new_count=plan.new_count,
            changed_count=plan.changed_count,
            unchanged_count=plan.unchanged_count,
            deleted_count=plan.deleted_count,
        )

    def _apply_item(
        self, item: PlannedDocumentChange, *, corpus_id: UUID, run_id: UUID
    ) -> None:
        if item.change_type is ChangeType.NEW:
            self._apply_new(item, corpus_id=corpus_id, run_id=run_id)
        elif item.change_type is ChangeType.CHANGED:
            self._apply_changed(item, run_id=run_id)
        elif item.change_type is ChangeType.UNCHANGED:
            self._apply_unchanged(item, run_id=run_id)
        elif item.change_type is ChangeType.DELETED:
            self._apply_deleted(item, run_id=run_id)

    def _apply_new(
        self, item: PlannedDocumentChange, *, corpus_id: UUID, run_id: UUID
    ) -> None:
        current = self._require_current(item)
        existing = self._repository.get_document_by_source_key(corpus_id, item.source_key)
        if existing is not None:
            if existing.state == "DELETED":
                raise UnsupportedResurrectionError(
                    f"Restoring deleted source key is not supported: {item.source_key}"
                )
            raise LineageConflictError(
                f"Source key already exists in corpus {corpus_id}: {item.source_key}"
            )
        document = self._repository.create_document(
            corpus_id,
            item.source_key,
            current.source_id,
            current.source_url,
            run_id,
        )
        version = self._create_version(
            item,
            document_pk=document.document_pk,
            version_number=1,
            run_id=run_id,
        )
        self._repository.record_run_document(
            run_id,
            document.document_pk,
            ChangeType.NEW.value,
            observed_version_id=version.document_version_id,
            content_changed=item.content_changed,
            metadata_changed=item.metadata_changed,
        )

    def _apply_changed(self, item: PlannedDocumentChange, *, run_id: UUID) -> None:
        current = self._require_current(item)
        document_pk, previous_version_id, previous_number = self._existing_identifiers(item)
        self._require_plan_is_current(document_pk, previous_version_id)
        self._repository.mark_document_seen(
            document_pk,
            run_id,
            current.source_id,
            current.source_url,
        )
        version = self._create_version(
            item,
            document_pk=document_pk,
            version_number=previous_number + 1,
            run_id=run_id,
        )
        self._repository.record_run_document(
            run_id,
            document_pk,
            ChangeType.CHANGED.value,
            previous_version_id=previous_version_id,
            observed_version_id=version.document_version_id,
            content_changed=item.content_changed,
            metadata_changed=item.metadata_changed,
        )

    def _apply_unchanged(self, item: PlannedDocumentChange, *, run_id: UUID) -> None:
        current = self._require_current(item)
        document_pk, previous_version_id, _previous_number = self._existing_identifiers(item)
        self._require_plan_is_current(document_pk, previous_version_id)
        self._repository.mark_document_seen(
            document_pk,
            run_id,
            current.source_id,
            current.source_url,
        )
        self._repository.record_run_document(
            run_id,
            document_pk,
            ChangeType.UNCHANGED.value,
            previous_version_id=previous_version_id,
            observed_version_id=previous_version_id,
            content_changed=False,
            metadata_changed=False,
        )

    def _apply_deleted(self, item: PlannedDocumentChange, *, run_id: UUID) -> None:
        document_pk, previous_version_id, _previous_number = self._existing_identifiers(item)
        self._require_plan_is_current(document_pk, previous_version_id)
        self._repository.mark_document_deleted(document_pk, run_id)
        self._repository.record_run_document(
            run_id,
            document_pk,
            ChangeType.DELETED.value,
            previous_version_id=previous_version_id,
            observed_version_id=None,
            content_changed=False,
            metadata_changed=False,
        )

    def _create_version(
        self,
        item: PlannedDocumentChange,
        *,
        document_pk: UUID,
        version_number: int,
        run_id: UUID,
    ) -> DocumentVersion:
        current = self._require_current(item)
        if item.current_metadata_hash is None or item.current_version_hash is None:
            raise InvalidLineageStateError(
                f"{item.change_type.value} item lacks current fingerprints: {item.source_key}"
            )
        version = self._repository.create_document_version(
            document_pk,
            version_number,
            current.content_hash,
            item.current_metadata_hash,
            item.current_version_hash,
            current.document_id,
            current.title,
            current.category,
            current.source_url,
            current.source_version,
            current.text,
            run_id,
        )
        self._repository.create_chunks(
            version.document_version_id,
            [
                ChunkPersistenceInput(
                    chunking_fingerprint=self._chunking_fingerprint,
                    compatibility_chunk_id=chunk.chunk_id,
                    ordinal=chunk.ordinal,
                    chunk_hash=chunk.chunk_hash,
                    text=chunk.text,
                    metadata=chunk.metadata,
                )
                for chunk in item.current_chunks
            ],
        )
        return version

    def _require_plan_is_current(
        self, document_pk: UUID, previous_version_id: UUID
    ) -> None:
        latest = self._repository.get_latest_document_version(document_pk)
        if (
            latest is None
            or latest.document_version_id != previous_version_id
            or latest.document.state != "ACTIVE"
        ):
            raise StaleChangePlanError(
                f"Document state or latest version changed after planning for {document_pk}"
            )

    @staticmethod
    def _require_current(item: PlannedDocumentChange) -> Document:
        if item.current_document is None:
            raise InvalidLineageStateError(
                f"{item.change_type.value} item lacks a current document: {item.source_key}"
            )
        return item.current_document

    @staticmethod
    def _existing_identifiers(item: PlannedDocumentChange) -> tuple[UUID, UUID, int]:
        if (
            item.document_pk is None
            or item.previous_document_version_id is None
            or item.previous_version_number is None
        ):
            raise InvalidLineageStateError(
                f"{item.change_type.value} item lacks previous identifiers: {item.source_key}"
            )
        return (
            item.document_pk,
            item.previous_document_version_id,
            item.previous_version_number,
        )


def current_states_from_snapshot(
    snapshot: CorpusSnapshot,
) -> tuple[CurrentDocumentState, ...]:
    """Convert existing Document/Chunk values without recalculating their identities."""

    chunks_by_document: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in snapshot.chunks:
        chunks_by_document[chunk.document_id].append(chunk)
    states = [
        current_document_state(
            document,
            chunks_by_document.get(document.document_id, ()),
        )
        for document in snapshot.documents
    ]
    return tuple(sorted(states, key=lambda state: state.source_key))


def snapshot_hash_for_current_states(states: Sequence[CurrentDocumentState]) -> str:
    """Fingerprint current source/version identities without runtime-generated values."""

    return managed_snapshot_hash(
        [
            ManagedSnapshotFingerprintRecord(
                source_key=state.source_key,
                content_hash=state.document.content_hash,
                metadata_hash=state.metadata_hash,
                version_hash=state.version_hash,
            )
            for state in states
        ]
    )


def persisted_document_state(
    document: LineageDocument, version: DocumentVersion
) -> PersistedDocumentState:
    """Project ORM values into the persistence-independent planner type."""

    return PersistedDocumentState(
        document_pk=document.document_pk,
        source_key=document.source_key,
        source_id=document.source_id,
        source_url=document.source_url,
        state=document.state,
        latest_document_version_id=version.document_version_id,
        version_number=version.version_number,
        content_hash=version.content_hash,
        metadata_hash=version.metadata_hash,
        version_hash=version.version_hash,
        compatibility_document_id=version.compatibility_document_id,
    )
