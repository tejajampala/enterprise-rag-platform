"""Validate seed evaluation qrels against the local Phase 1 fixture."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from consumer_finance_rag.acquisition import JsonFixtureSource
from consumer_finance_rag.chunking import ParagraphChunker
from consumer_finance_rag.config import load_settings
from consumer_finance_rag.evaluation.dataset import load_evaluation_set, validate_evaluation_set
from consumer_finance_rag.normalization import normalize_records


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    parser.add_argument("--dataset", type=Path, default=None)
    args = parser.parse_args(argv)
    settings = load_settings(args.config)
    dataset = load_evaluation_set(args.dataset or settings.evaluation.dataset_path)
    source = JsonFixtureSource("data/fixtures/cfpb_consumer_finance.json")
    documents = normalize_records(source.acquire())
    chunker = ParagraphChunker(settings.chunking.chunk_size, settings.chunking.overlap)
    chunks = [chunk for document in documents for chunk in chunker.chunk(document)]
    validate_evaluation_set(dataset, documents=documents, chunks=chunks)
    print(f"Validated {len(dataset.cases)} cases against {len(chunks)} chunks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
