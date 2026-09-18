from __future__ import annotations

import json
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from consumer_finance_rag.config import QdrantSettings, load_settings
from consumer_finance_rag.corpus import (
    CorpusIndexValidationError,
    load_corpus_snapshot,
    validate_indexed_corpus,
)
from consumer_finance_rag.sparse import BM25SparseEncoder
from consumer_finance_rag.vector_store import QdrantHybridStore


def _write_records(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(json.dumps(records), encoding="utf-8")


def test_fixture_precedes_and_deduplicates_expanded_records(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture.json"
    expanded = tmp_path / "expanded.json"
    common: dict[str, object] = {
        "source_id": "fixture-one",
        "title": "Fixture",
        "source_url": "https://example.test/same",
        "category": "credit cards",
        "source_version": "fixture-v1",
        "text": "Fixture evidence remains stable.",
    }
    _write_records(fixture, [common])
    _write_records(
        expanded,
        [
            common | {"source_id": "downloaded-copy", "title": "Duplicate"},
            common
            | {
                "source_id": "downloaded-new",
                "title": "New",
                "source_url": "https://example.test/new",
                "category": "mortgages",
                "text": "New official evidence.",
            },
        ],
    )

    snapshot = load_corpus_snapshot(
        load_settings(), fixture_path=fixture, expanded_path=expanded
    )

    assert snapshot.fixture_document_count == 1
    assert snapshot.expanded_document_count == 1
    assert len(snapshot.documents) == 2
    assert snapshot.category_distribution == {"credit cards": 1, "mortgages": 1}


def test_index_validation_compares_bm25_and_live_qdrant_payloads(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture.json"
    _write_records(
        fixture,
        [
            {
                "source_id": "fixture-one",
                "title": "Fixture",
                "source_url": "https://example.test/fixture",
                "category": "credit cards",
                "source_version": "fixture-v1",
                "text": "First evidence paragraph.",
            }
        ],
    )
    snapshot = load_corpus_snapshot(load_settings(), fixture_path=fixture, include_expanded=False)
    encoder = BM25SparseEncoder.fit([chunk.text for chunk in snapshot.chunks])
    store = QdrantHybridStore(
        QdrantSettings(collection_name="corpus-validation"),
        client=QdrantClient(":memory:"),
    )
    store.recreate_collection(dense_size=2)
    store.upsert(snapshot.chunks, [[1.0, 0.0]], encoder)

    validate_indexed_corpus(snapshot, sparse_encoder=encoder, store=store)

    wrong_encoder = BM25SparseEncoder.fit(["different fixture-only corpus"])
    with pytest.raises(CorpusIndexValidationError, match="BM25 corpus hash"):
        validate_indexed_corpus(snapshot, sparse_encoder=wrong_encoder, store=store)
