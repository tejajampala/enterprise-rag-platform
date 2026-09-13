"""Validate seed evaluation qrels against the assembled local corpus."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from consumer_finance_rag.config import load_settings
from consumer_finance_rag.corpus import (
    evaluation_runtime_settings,
    load_corpus_snapshot,
    validate_indexed_corpus,
)
from consumer_finance_rag.evaluation.dataset import load_evaluation_set, validate_evaluation_set
from consumer_finance_rag.sparse import BM25SparseEncoder
from consumer_finance_rag.vector_store import QdrantHybridStore


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    parser.add_argument("--dataset", type=Path, default=None)
    args = parser.parse_args(argv)
    settings = evaluation_runtime_settings(load_settings(args.config))
    dataset = load_evaluation_set(args.dataset or settings.evaluation.dataset_path)
    corpus = load_corpus_snapshot(settings, require_expanded=True)
    validate_evaluation_set(dataset, documents=corpus.documents, chunks=corpus.chunks)
    validate_indexed_corpus(
        corpus,
        sparse_encoder=BM25SparseEncoder.load(settings.evaluation.bm25_state_path),
        store=QdrantHybridStore(settings.qdrant),
    )
    print(
        f"Validated {len(dataset.cases)} cases against "
        f"{len(corpus.documents)} documents / {len(corpus.chunks)} chunks."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
