from __future__ import annotations

import json
from pathlib import Path

from consumer_finance_rag.config import load_settings
from consumer_finance_rag.corpus import load_corpus_snapshot


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
