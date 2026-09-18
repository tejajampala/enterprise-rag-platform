"""Build the local hybrid Qdrant index from fixture plus downloaded CFPB records."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from consumer_finance_rag.config import load_settings
from consumer_finance_rag.corpus import load_corpus_snapshot
from consumer_finance_rag.ingestion import index_chunks
from consumer_finance_rag.providers import HuggingFaceEmbeddingProvider
from consumer_finance_rag.vector_store import QdrantHybridStore


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    parser.add_argument(
        "--mode",
        choices=("fixture", "evaluation"),
        default="fixture",
        help="Select the isolated Phase 1 fixture index or combined evaluation index.",
    )
    args = parser.parse_args(argv)
    settings = load_settings(args.config)
    evaluation_mode = args.mode == "evaluation"
    snapshot = load_corpus_snapshot(
        settings, include_expanded=evaluation_mode, require_expanded=evaluation_mode
    )
    if evaluation_mode:
        settings = settings.model_copy(
            update={
                "qdrant": settings.qdrant.model_copy(
                    update={"collection_name": settings.evaluation.qdrant_collection_name}
                )
            }
        )
        bm25_state_path = settings.evaluation.bm25_state_path
    else:
        bm25_state_path = Path("data/index/bm25_state.json")
    chunks = snapshot.chunks
    model = settings.models.embedding
    embedding = HuggingFaceEmbeddingProvider(
        model.model, model.device, model.trust_remote_code, model.query_prompt_name
    )
    print(
        f"Preparing {len(snapshot.documents)} documents / {len(chunks)} chunks "
        f"({snapshot.fixture_document_count} fixture, "
        f"{snapshot.expanded_document_count} downloaded).",
        flush=True,
    )
    print(f"Category distribution: {snapshot.category_distribution}", flush=True)
    print(f"Loading embedding model once: {embedding.model_name}", flush=True)
    index_chunks(
        chunks,
        embedding_provider=embedding,
        store=QdrantHybridStore(settings.qdrant),
        bm25_state_path=bm25_state_path,
        progress=lambda completed, total: print(
            f"Embedded {completed}/{total} chunks", flush=True
        ),
    )
    print(
        f"Indexed {len(chunks)} chunks in {args.mode} mode into "
        f"'{settings.qdrant.collection_name}'.",
        flush=True,
    )


if __name__ == "__main__":
    main()
