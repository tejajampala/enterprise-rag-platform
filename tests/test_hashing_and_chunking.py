from __future__ import annotations

from consumer_finance_rag.acquisition import SourceRecord
from consumer_finance_rag.chunking import ParagraphChunker
from consumer_finance_rag.normalization import normalize_record


def _record() -> SourceRecord:
    return SourceRecord(
        source_id="source-1",
        title="Credit reporting",
        source_url="https://example.test/report",
        category="credit reporting",
        source_version="2026-01-01",
        text="First paragraph has useful information.\r\n\r\nSecond   paragraph has more detail.",
    )


def test_document_hash_is_deterministic_across_newline_styles() -> None:
    first = normalize_record(_record())
    second = normalize_record(
        _record().model_copy(update={"text": _record().text.replace("\r\n", "\n")})
    )
    assert first.content_hash == second.content_hash
    assert first.document_id == second.document_id
    assert len(first.content_hash) == 64


def test_chunk_hash_and_metadata_are_deterministic_and_preserved() -> None:
    document = normalize_record(_record())
    chunker = ParagraphChunker(chunk_size=55, overlap=10)
    first = chunker.chunk(document)
    second = chunker.chunk(document)

    assert [chunk.chunk_hash for chunk in first] == [chunk.chunk_hash for chunk in second]
    assert all(len(chunk.chunk_hash) == 64 for chunk in first)
    assert [chunk.ordinal for chunk in first] == list(range(len(first)))
    assert all(len(chunk.text) <= 55 for chunk in first)
    assert all(chunk.metadata["title"] == document.title for chunk in first)
    assert all(chunk.metadata["source_url"] == document.source_url for chunk in first)
    assert all(chunk.document_version == document.source_version for chunk in first)
