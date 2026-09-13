"""Bounded, ordered context construction with stable citation IDs."""

from __future__ import annotations

from collections.abc import Sequence

from consumer_finance_rag.models import Citation, RetrievedChunk, StructuredContext


class ContextBuilder:
    def __init__(self, max_characters: int) -> None:
        if max_characters <= 0:
            raise ValueError("max_characters must be positive")
        self.max_characters = max_characters

    def build(self, chunks: Sequence[RetrievedChunk]) -> StructuredContext:
        sections: list[str] = []
        citations: list[Citation] = []
        included: list[RetrievedChunk] = []
        seen: set[str] = set()
        used = 0
        for candidate in chunks:
            chunk = candidate.chunk
            if chunk.chunk_id in seen:
                continue
            citation_id = f"C{len(citations) + 1}"
            title = str(chunk.metadata.get("title") or "Untitled source")
            section = f"[{citation_id}] {title}\n{chunk.text}"
            separator_length = 2 if sections else 0
            remaining = self.max_characters - used - separator_length
            if remaining <= 0:
                break
            if len(section) > remaining:
                # Include a final partial passage only when enough evidence remains to be useful.
                if remaining < 100:
                    break
                section = section[: remaining - 1].rstrip() + "…"
            sections.append(section)
            citations.append(
                Citation(
                    citation_id=citation_id,
                    document_title=title,
                    source_url=chunk.metadata.get("source_url"),
                    chunk_id=chunk.chunk_id,
                )
            )
            included.append(candidate)
            seen.add(chunk.chunk_id)
            used += separator_length + len(section)
            if used >= self.max_characters:
                break
        return StructuredContext(text="\n\n".join(sections), citations=citations, chunks=included)
