"""Read-only bridge from current/persisted lineage state into the pure planner."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from consumer_finance_rag.corpus import CorpusSnapshot
from consumer_finance_rag.lineage.diff import (
    ChangePlan,
    CurrentDocumentState,
    PersistedDocumentState,
    current_document_state,
    plan_document_changes,
)
from consumer_finance_rag.lineage.models import DocumentVersion, LineageDocument
from consumer_finance_rag.lineage.repository import (
    InvalidLineageStateError,
    LineageRepository,
)
from consumer_finance_rag.models import Chunk


@dataclass(frozen=True, slots=True)
class LineageDryRunResult:
    """Read-only planning result plus source-state counts for presentation."""

    plan: ChangePlan
    corpus_initialized: bool
    current_document_count: int
    previous_active_document_count: int


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
