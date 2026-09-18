from __future__ import annotations

from dataclasses import replace
from uuid import UUID

import pytest

from consumer_finance_rag.acquisition import SourceRecord
from consumer_finance_rag.chunking import ParagraphChunker
from consumer_finance_rag.lineage.diff import (
    ChangePlan,
    ChangeType,
    CurrentDocumentState,
    DuplicateSourceKeyError,
    PersistedDocumentState,
    current_document_state,
    plan_document_changes,
    source_key_for,
)
from consumer_finance_rag.lineage.fingerprints import (
    chunking_fingerprint,
    embedding_fingerprint,
)
from consumer_finance_rag.normalization import normalize_record


def record(
    source_id: str,
    *,
    title: str | None = None,
    source_url: str | None = None,
    category: str = "credit-cards",
    source_version: str = "2026-09-18",
    text: str | None = None,
) -> SourceRecord:
    return SourceRecord(
        source_id=source_id,
        title=title or f"Guidance for {source_id}",
        source_url=source_url,
        category=category,
        source_version=source_version,
        text=text or f"Public consumer-finance guidance for {source_id}.",
    )


def current(source: SourceRecord, *, with_chunks: bool = True) -> CurrentDocumentState:
    document = normalize_record(source)
    chunks = ParagraphChunker(chunk_size=200, overlap=20).chunk(document) if with_chunks else []
    return current_document_state(document, chunks)


def persisted(
    state: CurrentDocumentState,
    *,
    document_pk: int = 1,
    version_number: int = 1,
) -> PersistedDocumentState:
    return PersistedDocumentState(
        document_pk=UUID(int=document_pk),
        source_key=state.source_key,
        source_id=state.document.source_id,
        source_url=state.document.source_url,
        state="ACTIVE",
        latest_document_version_id=UUID(int=document_pk + 100),
        version_number=version_number,
        content_hash=state.document.content_hash,
        metadata_hash=state.metadata_hash,
        version_hash=state.version_hash,
        compatibility_document_id=state.document.document_id,
    )


def plan(
    current_states: list[CurrentDocumentState],
    previous_states: list[PersistedDocumentState] | None = None,
    *,
    authoritative: bool = True,
) -> ChangePlan:
    return plan_document_changes(
        current_states,
        previous_states or [],
        authoritative_snapshot=authoritative,
    )


def test_source_key_prefers_source_url() -> None:
    source = record(
        "source-id", source_url="https://www.consumerfinance.gov/ask-cfpb/example"
    )

    assert source_key_for(source) == source.source_url


def test_source_key_falls_back_to_source_id() -> None:
    source = record("source-id", source_url=None)

    assert source_key_for(source) == "source-id"


def test_first_authoritative_snapshot_is_all_new() -> None:
    states = [current(record("b")), current(record("a"))]

    result = plan(states)

    assert [item.source_key for item in result.items] == ["a", "b"]
    assert [item.change_type for item in result.items] == [ChangeType.NEW, ChangeType.NEW]
    assert result.new_count == 2
    assert all(item.content_changed and item.metadata_changed for item in result.items)


def test_identical_second_snapshot_is_unchanged() -> None:
    state = current(record("cards"))

    result = plan([state], [persisted(state)])

    item = result.items[0]
    assert item.change_type is ChangeType.UNCHANGED
    assert not item.content_changed
    assert not item.metadata_changed
    assert result.unchanged_items == (item,)


def test_content_only_modification_is_changed() -> None:
    previous_current = current(record("cards", text="Original content."))
    changed = current(record("cards", text="Changed content."))

    item = plan([changed], [persisted(previous_current)]).items[0]

    assert item.change_type is ChangeType.CHANGED
    assert item.content_changed
    assert not item.metadata_changed


def test_metadata_only_modification_is_changed() -> None:
    previous_current = current(record("cards", title="Original title"))
    changed = current(record("cards", title="Changed title"))

    item = plan([changed], [persisted(previous_current)]).items[0]

    assert item.change_type is ChangeType.CHANGED
    assert not item.content_changed
    assert item.metadata_changed


def test_content_and_metadata_modification_is_changed() -> None:
    previous_current = current(record("cards", title="Original", text="Original."))
    changed = current(record("cards", title="Changed", text="Changed."))

    item = plan([changed], [persisted(previous_current)]).items[0]

    assert item.change_type is ChangeType.CHANGED
    assert item.content_changed
    assert item.metadata_changed


def test_new_document_is_added_alongside_unchanged_document() -> None:
    existing = current(record("existing"))
    added = current(record("added"))

    result = plan([existing, added], [persisted(existing)])

    assert [(item.source_key, item.change_type) for item in result.items] == [
        ("added", ChangeType.NEW),
        ("existing", ChangeType.UNCHANGED),
    ]


def test_missing_document_is_deleted_for_authoritative_snapshot() -> None:
    prior = current(record("removed"))

    result = plan([], [persisted(prior)], authoritative=True)

    item = result.items[0]
    assert item.change_type is ChangeType.DELETED
    assert item.current_document is None
    assert item.document_pk == UUID(int=1)
    assert result.deletion_detection_performed
    assert result.deleted_count == 1


def test_missing_document_is_omitted_when_snapshot_is_not_authoritative() -> None:
    prior = current(record("not-observed"))

    result = plan([], [persisted(prior)], authoritative=False)

    assert result.items == ()
    assert not result.deletion_detection_performed
    assert result.deleted_count == 0
    assert result.unchanged_count == 0


def test_mixed_plan_has_derived_counts() -> None:
    unchanged = current(record("unchanged"))
    changed_before = current(record("changed", text="Before."))
    changed_now = current(record("changed", text="After."))
    added = current(record("new"))
    removed = current(record("deleted"))

    result = plan(
        [unchanged, changed_now, added],
        [
            persisted(unchanged, document_pk=1),
            persisted(changed_before, document_pk=2),
            persisted(removed, document_pk=3),
        ],
    )

    assert [(item.source_key, item.change_type) for item in result.items] == [
        ("changed", ChangeType.CHANGED),
        ("deleted", ChangeType.DELETED),
        ("new", ChangeType.NEW),
        ("unchanged", ChangeType.UNCHANGED),
    ]
    assert (
        result.new_count,
        result.changed_count,
        result.unchanged_count,
        result.deleted_count,
    ) == (1, 1, 1, 1)


def test_plan_is_independent_of_input_and_chunk_order() -> None:
    first = current(record("a"))
    second = current(record("b", text="First paragraph.\n\nSecond paragraph."))
    reversed_chunks = replace(second, chunks=tuple(reversed(second.chunks)))
    previous = [persisted(first, document_pk=1), persisted(second, document_pk=2)]

    forward = plan([first, second], previous)
    reversed_input = plan([reversed_chunks, first], list(reversed(previous)))

    assert forward == reversed_input


def test_duplicate_current_source_keys_are_rejected() -> None:
    state = current(record("duplicate"))

    with pytest.raises(DuplicateSourceKeyError, match="Duplicate current"):
        plan([state, state])


def test_duplicate_previous_source_keys_are_rejected() -> None:
    state = current(record("duplicate"))
    previous = persisted(state)

    with pytest.raises(DuplicateSourceKeyError, match="Duplicate previous"):
        plan([state], [previous, previous])


def test_current_content_derived_document_id_is_preserved() -> None:
    state = current(record("cards", text="Stable normalized content."))

    item = plan([state]).items[0]

    assert item.current_document is not None
    assert item.current_document.document_id == state.document.document_id
    assert item.current_version_hash == state.version_hash


def test_index_compatibility_fingerprints_do_not_change_document_classification() -> None:
    state = current(record("cards"))
    previous = persisted(state)

    baseline = plan([state], [previous])
    assert chunking_fingerprint(chunk_size=900, overlap=120) != chunking_fingerprint(
        chunk_size=1000, overlap=120
    )
    assert embedding_fingerprint(provider="huggingface", model="model-a") != (
        embedding_fingerprint(provider="huggingface", model="model-b")
    )
    repeated = plan([state], [previous])

    assert baseline == repeated
    assert repeated.items[0].change_type is ChangeType.UNCHANGED
