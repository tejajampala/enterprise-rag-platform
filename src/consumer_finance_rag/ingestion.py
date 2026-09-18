"""Offline-first ingestion workflow for the Phase 1 development corpus."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

from consumer_finance_rag.acquisition import DocumentSource
from consumer_finance_rag.chunking import ParagraphChunker
from consumer_finance_rag.models import Chunk
from consumer_finance_rag.normalization import normalize_records
from consumer_finance_rag.providers import EmbeddingProvider
from consumer_finance_rag.sparse import BM25SparseEncoder
from consumer_finance_rag.vector_store import QdrantHybridStore


def prepare_chunks(source: DocumentSource, chunker: ParagraphChunker) -> list[Chunk]:
    documents = normalize_records(source.acquire())
    return [chunk for document in documents for chunk in chunker.chunk(document)]


def index_chunks(
    chunks: Sequence[Chunk],
    *,
    embedding_provider: EmbeddingProvider,
    store: QdrantHybridStore,
    bm25_state_path: str | Path,
    embedding_batch_size: int = 32,
    progress: Callable[[int, int], None] | None = None,
) -> None:
    if not chunks:
        raise ValueError("no chunks to index")
    if embedding_batch_size <= 0:
        raise ValueError("embedding_batch_size must be positive")
    dense_vectors: list[list[float]] = []
    total = len(chunks)
    for start in range(0, total, embedding_batch_size):
        batch = chunks[start : start + embedding_batch_size]
        dense_vectors.extend(embedding_provider.embed_documents([chunk.text for chunk in batch]))
        if progress is not None:
            progress(min(start + len(batch), total), total)
    sparse_encoder = BM25SparseEncoder.fit([chunk.text for chunk in chunks])
    state_path = Path(bm25_state_path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    sparse_encoder.save(state_path)
    store.recreate_collection(dense_size=len(dense_vectors[0]))
    store.upsert(chunks, dense_vectors, sparse_encoder)
