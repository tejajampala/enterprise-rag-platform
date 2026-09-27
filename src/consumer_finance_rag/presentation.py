"""Shared presentation-safe selection helpers for Streamlit entrypoints."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Protocol


class CitationLike(Protocol):
    citation_id: str


def cited_only[CitationT: CitationLike](
    answer: str, citations: Sequence[CitationT]
) -> tuple[CitationT, ...]:
    """Return citations explicitly referenced by the generated answer."""

    referenced = citation_references(answer)
    return tuple(item for item in citations if item.citation_id in referenced)


def citation_references(answer: str) -> set[str]:
    """Return every visible citation ID, including IDs absent from the catalogue."""

    return set(re.findall(r"\[(C[1-9][0-9]*)\]", answer))
