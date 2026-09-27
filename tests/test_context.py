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
    assert [passage.truncated for passage in context.passages] == [False, False]


def test_context_respects_maximum_size(
    make_candidate: Callable[..., RetrievedChunk],
) -> None:
    context = ContextBuilder(max_characters=100).build([make_candidate("chunk-1")])
    assert len(context.text) <= 100


def test_context_records_exact_prompt_text_when_final_passage_is_truncated(
    make_candidate: Callable[..., RetrievedChunk],
) -> None:
    candidate = make_candidate("chunk-truncated", title="Long evidence")
    candidate = candidate.model_copy(
        update={"chunk": candidate.chunk.model_copy(update={"text": "evidence " * 80})}
    )

    context = ContextBuilder(max_characters=140).build([candidate])

    assert len(context.text) <= 140
    assert len(context.passages) == 1
    passage = context.passages[0]
    assert passage.truncated is True
    assert passage.original_text == candidate.chunk.text
    assert passage.prompt_section_text == context.text
    assert passage.original_text not in passage.prompt_section_text
