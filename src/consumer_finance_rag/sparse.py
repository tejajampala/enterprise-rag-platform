"""Transparent BM25 sparse-vector encoding for Qdrant lexical retrieval."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


class SparseVectorData(BaseModel):
    model_config = ConfigDict(frozen=True)

    indices: list[int]
    values: list[float]


class BM25State(BaseModel):
    model_config = ConfigDict(frozen=True)

    vocabulary: dict[str, int]
    document_frequencies: dict[str, int]
    document_count: int = Field(gt=0)
    average_document_length: float = Field(gt=0)
    k1: float = 1.2
    b: float = 0.75


class BM25SparseEncoder:
    """Fit BM25 statistics once, then create document/query sparse vectors.

    Qdrant computes sparse dot products. Document-side vectors contain the BM25
    term saturation and inverse document frequency; query vectors contain term
    frequency, producing standard BM25 scores for ordinary keyword queries.
    """

    def __init__(self, state: BM25State) -> None:
        self.state = state

    @classmethod
    def fit(cls, texts: Sequence[str], *, k1: float = 1.2, b: float = 0.75) -> BM25SparseEncoder:
        if not texts:
            raise ValueError("cannot fit BM25 on an empty corpus")
        tokenized = [tokenize(text) for text in texts]
        terms = sorted({term for tokens in tokenized for term in tokens})
        frequencies = {term: sum(term in tokens for tokens in tokenized) for term in terms}
        average_length = sum(len(tokens) for tokens in tokenized) / len(tokenized)
        state = BM25State(
            vocabulary={term: index for index, term in enumerate(terms)},
            document_frequencies=frequencies,
            document_count=len(tokenized),
            average_document_length=max(average_length, 1.0),
            k1=k1,
            b=b,
        )
        return cls(state)

    def encode_document(self, text: str) -> SparseVectorData:
        tokens = tokenize(text)
        counts = Counter(tokens)
        values: list[tuple[int, float]] = []
        for term, tf in counts.items():
            index = self.state.vocabulary.get(term)
            if index is None:
                continue
            df = self.state.document_frequencies[term]
            idf = math.log(1 + (self.state.document_count - df + 0.5) / (df + 0.5))
            length_norm = (
                1 - self.state.b + self.state.b * len(tokens) / self.state.average_document_length
            )
            saturated_tf = tf * (self.state.k1 + 1) / (tf + self.state.k1 * length_norm)
            values.append((index, idf * saturated_tf))
        values.sort()
        return SparseVectorData(
            indices=[item[0] for item in values], values=[item[1] for item in values]
        )

    def encode_query(self, text: str) -> SparseVectorData:
        counts = Counter(tokenize(text))
        values = sorted(
            (self.state.vocabulary[term], float(count))
            for term, count in counts.items()
            if term in self.state.vocabulary
        )
        return SparseVectorData(
            indices=[item[0] for item in values], values=[item[1] for item in values]
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.state.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> BM25SparseEncoder:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(BM25State.model_validate(data))
