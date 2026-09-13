"""Bounded, ordered context construction with stable citation IDs."""

from __future__ import annotations

from collections.abc import Sequence

from consumer_finance_rag.models import Citation, RetrievedChunk, StructuredContext
from consumer_finance_rag.observability import NoOpTracer, Tracer, retriever_documents


class ContextBuilder:
    def __init__(self, max_characters: int, tracer: Tracer | None = None) -> None:
        if max_characters <= 0:
            raise ValueError("max_characters must be positive")
        self.max_characters = max_characters
        self.tracer = tracer or NoOpTracer()

    def build(self, chunks: Sequence[RetrievedChunk]) -> StructuredContext:
        with self.tracer.span(
            "context_build",
            span_type="RETRIEVER",
            attributes={"context.max_characters": self.max_characters},
        ) as span:
            span.set_inputs({"chunk_ids": [item.chunk.chunk_id for item in chunks]})
            result = self._build(chunks)
            span.set_attributes(
                {
                    "context.final_count": len(result.chunks),
                    "context.characters": len(result.text),
                }
            )
            span.set_outputs(retriever_documents(result.chunks))
            return result

    def _build(self, chunks: Sequence[RetrievedChunk]) -> StructuredContext:
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
