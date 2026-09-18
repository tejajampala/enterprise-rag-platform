"""Pure deterministic source-document change planning."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol
from uuid import UUID

from consumer_finance_rag.lineage.fingerprints import metadata_hash, version_hash
from consumer_finance_rag.models import Chunk, Document


class SourceIdentity(Protocol):
    """Minimal source identity shared by acquired and normalized records."""

    @property
    def source_id(self) -> str: ...

    @property
    def source_url(self) -> str | None: ...


class ChangeType(StrEnum):
    """Supported source-document classifications for this milestone."""

    NEW = "NEW"
    CHANGED = "CHANGED"
    UNCHANGED = "UNCHANGED"
    DELETED = "DELETED"


class DuplicateSourceKeyError(ValueError):
    """Raised when one planner input contains a duplicate source identity."""


class InvalidPreviousDocumentStateError(ValueError):
    """Raised when previous-state input is not an ACTIVE document snapshot."""


@dataclass(frozen=True, slots=True)
class PersistedDocumentState:
    """Persistence-independent projection of an ACTIVE document and latest version."""

    document_pk: UUID
    source_key: str
    source_id: str
    source_url: str | None
    state: str
    latest_document_version_id: UUID
    version_number: int
    content_hash: str
    metadata_hash: str
    version_hash: str
    compatibility_document_id: str


@dataclass(frozen=True, slots=True)
class CurrentDocumentState:
    """One normalized current document and its already-generated chunks."""

    source_key: str
    document: Document
    chunks: tuple[Chunk, ...]
    metadata_hash: str
    version_hash: str


@dataclass(frozen=True, slots=True)
class PlannedDocumentChange:
    """One immutable source-document classification for a future service layer."""

    change_type: ChangeType
    source_key: str
    document_pk: UUID | None
    previous_document_version_id: UUID | None
    previous_version_number: int | None
    previous_content_hash: str | None
    previous_metadata_hash: str | None
    current_document: Document | None
    current_chunks: tuple[Chunk, ...]
    current_content_hash: str | None
    current_metadata_hash: str | None
    current_version_hash: str | None
    content_changed: bool
    metadata_changed: bool


@dataclass(frozen=True, slots=True)
class ChangePlan:
    """Deterministically ordered classifications with derived counters."""

    items: tuple[PlannedDocumentChange, ...]
    authoritative_snapshot: bool

    @property
    def deletion_detection_performed(self) -> bool:
        return self.authoritative_snapshot

    @property
    def new_items(self) -> tuple[PlannedDocumentChange, ...]:
        return self._items_of_type(ChangeType.NEW)

    @property
    def changed_items(self) -> tuple[PlannedDocumentChange, ...]:
        return self._items_of_type(ChangeType.CHANGED)

    @property
    def unchanged_items(self) -> tuple[PlannedDocumentChange, ...]:
        return self._items_of_type(ChangeType.UNCHANGED)

    @property
    def deleted_items(self) -> tuple[PlannedDocumentChange, ...]:
        return self._items_of_type(ChangeType.DELETED)

    @property
    def new_count(self) -> int:
        return len(self.new_items)

    @property
    def changed_count(self) -> int:
        return len(self.changed_items)

    @property
    def unchanged_count(self) -> int:
        return len(self.unchanged_items)

    @property
    def deleted_count(self) -> int:
        return len(self.deleted_items)

    def _items_of_type(
        self, change_type: ChangeType
    ) -> tuple[PlannedDocumentChange, ...]:
        return tuple(item for item in self.items if item.change_type is change_type)


def source_key_for(source: SourceIdentity) -> str:
    """Use the unchanged corpus-deduplication identity rule."""

    return source.source_url or source.source_id


def current_document_state(
    document: Document, chunks: Sequence[Chunk] = ()
) -> CurrentDocumentState:
    """Build a current planner snapshot without recalculating content or chunk identities."""

    document_metadata_hash = metadata_hash(
        title=document.title,
        source_url=document.source_url,
        category=document.category,
        source_version=document.source_version,
    )
    return CurrentDocumentState(
        source_key=source_key_for(document),
        document=document,
        chunks=tuple(sorted(chunks, key=lambda chunk: (chunk.ordinal, chunk.chunk_id))),
        metadata_hash=document_metadata_hash,
        version_hash=version_hash(
            content_hash=document.content_hash,
            metadata_hash=document_metadata_hash,
        ),
    )


def plan_document_changes(
    current_documents: Sequence[CurrentDocumentState],
    previous_documents: Sequence[PersistedDocumentState],
    *,
    authoritative_snapshot: bool,
) -> ChangePlan:
    """Classify a current snapshot without persistence or index side effects."""

    current_by_key = _unique_current_documents(current_documents)
    previous_by_key = _unique_previous_documents(previous_documents)
    planned: list[PlannedDocumentChange] = []

    all_source_keys = set(current_by_key)
    if authoritative_snapshot:
        all_source_keys.update(previous_by_key)

    for source_key in sorted(all_source_keys):
        current = current_by_key.get(source_key)
        previous = previous_by_key.get(source_key)
        if current is None:
            if previous is None:  # pragma: no cover - protected by the key union
                raise AssertionError("planned source key has no current or previous state")
            planned.append(_deleted_change(previous))
        elif previous is None:
            planned.append(_new_change(current))
        else:
            planned.append(_existing_change(current, previous))

    return ChangePlan(items=tuple(planned), authoritative_snapshot=authoritative_snapshot)


def _unique_current_documents(
    documents: Sequence[CurrentDocumentState],
) -> dict[str, CurrentDocumentState]:
    unique: dict[str, CurrentDocumentState] = {}
    for document in documents:
        if document.source_key in unique:
            raise DuplicateSourceKeyError(
                f"Duplicate current source key: {document.source_key}"
            )
        unique[document.source_key] = document
    return unique


def _unique_previous_documents(
    documents: Sequence[PersistedDocumentState],
) -> dict[str, PersistedDocumentState]:
    unique: dict[str, PersistedDocumentState] = {}
    for document in documents:
        if document.state != "ACTIVE":
            raise InvalidPreviousDocumentStateError(
                f"Previous document {document.document_pk} is {document.state}, not ACTIVE"
            )
        if document.source_key in unique:
            raise DuplicateSourceKeyError(
                f"Duplicate previous source key: {document.source_key}"
            )
        unique[document.source_key] = document
    return unique


def _new_change(current: CurrentDocumentState) -> PlannedDocumentChange:
    return PlannedDocumentChange(
        change_type=ChangeType.NEW,
        source_key=current.source_key,
        document_pk=None,
        previous_document_version_id=None,
        previous_version_number=None,
        previous_content_hash=None,
        previous_metadata_hash=None,
        current_document=current.document,
        current_chunks=_ordered_chunks(current.chunks),
        current_content_hash=current.document.content_hash,
        current_metadata_hash=current.metadata_hash,
        current_version_hash=current.version_hash,
        content_changed=True,
        metadata_changed=True,
    )


def _existing_change(
    current: CurrentDocumentState, previous: PersistedDocumentState
) -> PlannedDocumentChange:
    content_changed = current.document.content_hash != previous.content_hash
    metadata_changed = current.metadata_hash != previous.metadata_hash
    change_type = (
        ChangeType.CHANGED
        if content_changed or metadata_changed
        else ChangeType.UNCHANGED
    )
    return PlannedDocumentChange(
        change_type=change_type,
        source_key=current.source_key,
        document_pk=previous.document_pk,
        previous_document_version_id=previous.latest_document_version_id,
        previous_version_number=previous.version_number,
        previous_content_hash=previous.content_hash,
        previous_metadata_hash=previous.metadata_hash,
        current_document=current.document,
        current_chunks=_ordered_chunks(current.chunks),
        current_content_hash=current.document.content_hash,
        current_metadata_hash=current.metadata_hash,
        current_version_hash=current.version_hash,
        content_changed=content_changed,
        metadata_changed=metadata_changed,
    )


def _deleted_change(previous: PersistedDocumentState) -> PlannedDocumentChange:
    return PlannedDocumentChange(
        change_type=ChangeType.DELETED,
        source_key=previous.source_key,
        document_pk=previous.document_pk,
        previous_document_version_id=previous.latest_document_version_id,
        previous_version_number=previous.version_number,
        previous_content_hash=previous.content_hash,
        previous_metadata_hash=previous.metadata_hash,
        current_document=None,
        current_chunks=(),
        current_content_hash=None,
        current_metadata_hash=None,
        current_version_hash=None,
        content_changed=False,
        metadata_changed=False,
    )


def _ordered_chunks(chunks: Sequence[Chunk]) -> tuple[Chunk, ...]:
    return tuple(sorted(chunks, key=lambda chunk: (chunk.ordinal, chunk.chunk_id)))
