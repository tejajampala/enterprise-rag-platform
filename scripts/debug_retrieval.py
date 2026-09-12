"""Print every stage of the production hybrid retrieval path for debugging."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from consumer_finance_rag.config import load_settings
from consumer_finance_rag.debug import format_retrieval_trace
from consumer_finance_rag.factory import build_retrieval_service


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect dense HNSW/ANN, sparse BM25, RRF-fused, and reranked results "
            "from the production retrieval service."
        )
    )
    parser.add_argument("question", help="Consumer-finance question to retrieve for.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/config.yaml"),
        help="Application YAML configuration path (default: config/config.yaml).",
    )
    parser.add_argument(
        "--bm25-state",
        type=Path,
        default=Path("data/index/bm25_state.json"),
        help="Persisted BM25 state created by ingestion.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Override the configured final reranked result count.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    question = str(args.question).strip()
    if not question:
        parser.error("question must not be empty")
    top_k = args.top_k
    if top_k is not None and top_k <= 0:
        parser.error("--top-k must be positive")
    bm25_state = Path(args.bm25_state)
    if not bm25_state.is_file():
        parser.error(
            f"BM25 state not found at {bm25_state}. Run `uv run python scripts/ingest.py` first."
        )

    settings = load_settings(Path(args.config))
    service = build_retrieval_service(settings, bm25_state_path=bm25_state)
    trace = service.retrieve_debug(question, top_k=top_k)
    print(format_retrieval_trace(question, trace))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
