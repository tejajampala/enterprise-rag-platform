"""Validate seed evaluation qrels against the assembled local corpus."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from consumer_finance_rag.config import load_settings
from consumer_finance_rag.corpus import load_corpus_snapshot
from consumer_finance_rag.evaluation.dataset import load_evaluation_set, validate_evaluation_set


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    parser.add_argument("--dataset", type=Path, default=None)
    args = parser.parse_args(argv)
    settings = load_settings(args.config)
    dataset = load_evaluation_set(args.dataset or settings.evaluation.dataset_path)
    corpus = load_corpus_snapshot(settings)
    validate_evaluation_set(dataset, documents=corpus.documents, chunks=corpus.chunks)
    print(
        f"Validated {len(dataset.cases)} cases against "
        f"{len(corpus.documents)} documents / {len(corpus.chunks)} chunks."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
