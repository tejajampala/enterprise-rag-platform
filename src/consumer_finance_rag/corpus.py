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
from consumer_finance_rag.sparse import BM25SparseEncoder, bm25_corpus_hash
from consumer_finance_rag.vector_store import QdrantHybridStore

DEFAULT_FIXTURE_PATH = Path("data/fixtures/cfpb_consumer_finance.json")


@dataclass(frozen=True)
class CorpusSnapshot:
    """Normalized corpus plus counts suitable for indexing and experiment logging."""

    documents: list[Document]
    chunks: list[Chunk]
    fixture_document_count: int
    expanded_document_count: int
    category_distribution: dict[str, int]


class CorpusIndexValidationError(ValueError):
    """Raised when persisted BM25/Qdrant state does not match an assembled corpus."""


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


def evaluation_runtime_settings(settings: AppSettings) -> AppSettings:
    """Select the dedicated expanded-corpus collection without changing production defaults."""

    qdrant = settings.qdrant.model_copy(
        update={"collection_name": settings.evaluation.qdrant_collection_name}
    )
    return settings.model_copy(update={"qdrant": qdrant})


def validate_indexed_corpus(
    snapshot: CorpusSnapshot,
    *,
    sparse_encoder: BM25SparseEncoder,
    store: QdrantHybridStore,
) -> None:
    """Prove that loaded BM25 state and live Qdrant payloads match the evaluation corpus."""

    texts = [chunk.text for chunk in snapshot.chunks]
    expected = {chunk.chunk_id: chunk.chunk_hash for chunk in snapshot.chunks}
    issues: list[str] = []
    if sparse_encoder.state.document_count != len(snapshot.chunks):
        issues.append(
            "BM25 document count "
            f"{sparse_encoder.state.document_count} != corpus chunk count {len(snapshot.chunks)}"
        )
    expected_bm25_hash = bm25_corpus_hash(texts)
    if sparse_encoder.state.corpus_hash != expected_bm25_hash:
        issues.append("BM25 corpus hash does not match the assembled evaluation corpus")

    actual = store.indexed_chunk_hashes()
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    mismatched = sorted(
        chunk_id
        for chunk_id in set(expected) & set(actual)
        if expected[chunk_id] != actual[chunk_id]
    )
    if missing:
        issues.append(f"Qdrant is missing {len(missing)} expected chunks")
    if unexpected:
        issues.append(f"Qdrant contains {len(unexpected)} unexpected chunks")
    if mismatched:
        issues.append(f"Qdrant contains {len(mismatched)} chunk-hash mismatches")
    if issues:
        detail = "\n- ".join(issues)
        raise CorpusIndexValidationError(
            "Evaluation index is incompatible; run "
            "`uv run python scripts/ingest.py --mode evaluation`:\n- "
            f"{detail}"
        )
