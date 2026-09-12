from __future__ import annotations

from collections.abc import Callable

from consumer_finance_rag.context import ContextBuilder
from consumer_finance_rag.models import RetrievedChunk


def test_context_preserves_order_deduplicates_and_builds_citations(
    make_candidate: Callable[..., RetrievedChunk],
) -> None:
    first = make_candidate("chunk-1", title="First")
    second = make_candidate("chunk-2", title="Second", source_url=None)
    context = ContextBuilder(max_characters=1000).build([first, first, second])

    assert [item.chunk.chunk_id for item in context.chunks] == ["chunk-1", "chunk-2"]
    assert [citation.citation_id for citation in context.citations] == ["C1", "C2"]
    assert context.citations[0].chunk_id == "chunk-1"
    assert context.citations[0].document_title == "First"
    assert context.citations[1].source_url is None
    assert context.text.index("[C1]") < context.text.index("[C2]")


def test_context_respects_maximum_size(
    make_candidate: Callable[..., RetrievedChunk],
) -> None:
    context = ContextBuilder(max_characters=100).build([make_candidate("chunk-1")])
    assert len(context.text) <= 100
