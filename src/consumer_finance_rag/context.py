"""Bounded, ordered context construction with stable citation IDs."""

from __future__ import annotations

from collections.abc import Sequence

from consumer_finance_rag.models import (
    Citation,
    ContextPassage,
    RetrievedChunk,
    StructuredContext,
)
from consumer_finance_rag.observability import (
    NoOpTracer,
    TraceContentPolicy,
    Tracer,
    approved_retriever_documents,
)


class ContextBuilder:
    def __init__(self, max_characters: int, tracer: Tracer | None = None) -> None:
        if max_characters <= 0:
            raise ValueError("max_characters must be positive")
        self.max_characters = max_characters
        self.tracer = tracer or NoOpTracer()

    def build(self, chunks: Sequence[RetrievedChunk]) -> StructuredContext:
        content_policy = getattr(
            self.tracer,
            "content_policy",
            TraceContentPolicy.METADATA_ONLY,
        )
        approved_content = (
            content_policy is TraceContentPolicy.APPROVED_PUBLIC_EVALUATION
        )
        with self.tracer.span(
            "context_build",
            span_type="RETRIEVER" if approved_content else "CHAIN",
            attributes={"context.max_characters": self.max_characters},
        ) as span:
            span.set_inputs({"candidate_count": len(chunks)})
            result = self._build(chunks)
            span.set_attributes(
                {
                    "context.final_count": len(result.chunks),
                    "context.characters": len(result.text),
                }
            )
            if approved_content:
                span.set_outputs(
                    approved_retriever_documents(
                        result.chunks,
                        content_policy=content_policy,
                    )
                )
            else:
                span.set_outputs(
                    {
                        "context_count": len(result.chunks),
                        "context_characters": len(result.text),
                    }
                )
            return result

    def _build(self, chunks: Sequence[RetrievedChunk]) -> StructuredContext:
        sections: list[str] = []
        citations: list[Citation] = []
        included: list[RetrievedChunk] = []
        passages: list[ContextPassage] = []
        seen: set[str] = set()
        used = 0
        for candidate in chunks:
            chunk = candidate.chunk
            if chunk.chunk_id in seen:
                continue
            citation_id = f"C{len(citations) + 1}"
            title = str(chunk.metadata.get("title") or "Untitled source")
            full_section = f"[{citation_id}] {title}\n{chunk.text}"
            section = full_section
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
            passages.append(
                ContextPassage(
                    order=len(passages) + 1,
                    citation_id=citation_id,
                    document_title=title,
                    source_url=chunk.metadata.get("source_url"),
                    source_id=(
                        str(chunk.metadata["source_id"])
                        if chunk.metadata.get("source_id") is not None
                        else None
                    ),
                    document_id=chunk.document_id,
                    chunk_id=chunk.chunk_id,
                    original_text=chunk.text,
                    prompt_section_text=section,
                    truncated=section != full_section,
                )
            )
            seen.add(chunk.chunk_id)
            used += separator_length + len(section)
            if used >= self.max_characters:
                break
        return StructuredContext(
            text="\n\n".join(sections),
            citations=citations,
            chunks=included,
            passages=passages,
        )
