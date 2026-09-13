"""Human-readable rendering for the local retrieval debugging CLI."""

from __future__ import annotations

from collections.abc import Sequence

from consumer_finance_rag.models import RetrievedChunk
from consumer_finance_rag.retrieval import RetrievalTrace


def _score_text(candidate: RetrievedChunk, score_kind: str) -> str:
    if score_kind == "reranker":
        value = candidate.rerank_score
        return "n/a" if value is None else f"{value:.6f}"
    return f"{candidate.score:.6f}"


def format_stage(
    title: str,
    candidates: Sequence[RetrievedChunk],
    *,
    score_kind: str,
    snippet_characters: int = 240,
) -> str:
    """Format one ranking without changing its order or scores."""

    lines = [f"=== {title} ({len(candidates)}) ==="]
    if not candidates:
        lines.append("No results.")
        return "\n".join(lines)

    for rank, candidate in enumerate(candidates, start=1):
        chunk = candidate.chunk
        text = " ".join(chunk.text.split())
        if len(text) > snippet_characters:
            text = f"{text[: snippet_characters - 1].rstrip()}…"
        lines.extend(
            [
                (f"{rank}. score={_score_text(candidate, score_kind)} chunk_id={chunk.chunk_id}"),
                f"   title={chunk.metadata.get('title') or 'Untitled source'}",
                f"   category={chunk.metadata.get('category') or 'unknown'}",
                (
                    f"   dense_rank={candidate.dense_rank or '-'} "
                    f"sparse_rank={candidate.sparse_rank or '-'}"
                ),
                f"   text={text}",
            ]
        )
    return "\n".join(lines)


def format_retrieval_trace(question: str, trace: RetrievalTrace) -> str:
    """Render all production retrieval stages for terminal inspection."""

    sections = [
        f"Question: {question}",
        format_stage("Dense HNSW/ANN results", trace.dense, score_kind="retrieval"),
        format_stage("Sparse BM25 results", trace.sparse, score_kind="retrieval"),
        format_stage("RRF-fused candidates", trace.fused, score_kind="rrf"),
        format_stage("Final reranked results", trace.reranked, score_kind="reranker"),
    ]
    return "\n\n".join(sections)
