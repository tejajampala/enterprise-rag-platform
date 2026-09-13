"""Assemble the stable fixture and optional downloaded evaluation corpus."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from consumer_finance_rag.acquisition import JsonFixtureSource, SourceRecord
from consumer_finance_rag.chunking import ParagraphChunker
from consumer_finance_rag.config import AppSettings
from consumer_finance_rag.models import Chunk, Document
from consumer_finance_rag.normalization import normalize_records

DEFAULT_FIXTURE_PATH = Path("data/fixtures/cfpb_consumer_finance.json")


@dataclass(frozen=True)
class CorpusSnapshot:
    """Normalized corpus plus counts suitable for indexing and experiment logging."""

    documents: list[Document]
    chunks: list[Chunk]
    fixture_document_count: int
    expanded_document_count: int
    category_distribution: dict[str, int]


def _record_key(record: SourceRecord) -> str:
    return record.source_url or record.source_id


def load_corpus_snapshot(
    settings: AppSettings,
    *,
    fixture_path: str | Path = DEFAULT_FIXTURE_PATH,
    expanded_path: str | Path | None = None,
    include_expanded: bool = True,
    require_expanded: bool = False,
) -> CorpusSnapshot:
    """Load fixture first, then distinct downloaded records, preserving fixture qrel IDs."""

    fixture_records = JsonFixtureSource(fixture_path).acquire()
    records = list(fixture_records)
    expanded_count = 0
    if include_expanded:
        path = Path(expanded_path or settings.corpus_expansion.output_path)
        if path.is_file():
            expanded_records = JsonFixtureSource(path).acquire()
            known = {_record_key(record) for record in fixture_records}
            for record in expanded_records:
                key = _record_key(record)
                if key in known:
                    continue
                known.add(key)
                records.append(record)
                expanded_count += 1
        elif require_expanded:
            raise FileNotFoundError(
                f"Expanded corpus not found at {path}. Run scripts/fetch_evaluation_corpus.py."
            )

    documents = normalize_records(records)
    chunker = ParagraphChunker(settings.chunking.chunk_size, settings.chunking.overlap)
    chunks = [chunk for document in documents for chunk in chunker.chunk(document)]
    distribution = dict(sorted(Counter(document.category for document in documents).items()))
    return CorpusSnapshot(
        documents=documents,
        chunks=chunks,
        fixture_document_count=len(fixture_records),
        expanded_document_count=expanded_count,
        category_distribution=distribution,
    )
