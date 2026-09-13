from __future__ import annotations

import math

import pytest

from consumer_finance_rag.sparse import BM25SparseEncoder, tokenize


def test_bm25_state_uses_deterministic_vocabulary_document_frequency_and_token_lengths() -> None:
    texts = ["Alpha alpha, beta!", "Beta gamma gamma gamma."]

    encoder = BM25SparseEncoder.fit(texts)

    assert encoder.state.vocabulary == {"alpha": 0, "beta": 1, "gamma": 2}
    assert encoder.state.document_frequencies == {"alpha": 1, "beta": 2, "gamma": 1}
    assert encoder.state.document_count == 2
    assert encoder.state.average_document_length == 3.5
    assert tokenize(texts[0]) == ["alpha", "alpha", "beta"]


def test_bm25_document_and_query_use_separate_weights_with_shared_tokenization() -> None:
    encoder = BM25SparseEncoder.fit(["Alpha alpha, beta!", "Beta gamma gamma gamma."])

    document = encoder.encode_document("ALPHA alpha; beta")
    query = encoder.encode_query("ALPHA alpha; unknown")

    alpha_idf = math.log(1 + (2 - 1 + 0.5) / (1 + 0.5))
    alpha_length_norm = 1 - 0.75 + 0.75 * 3 / 3.5
    alpha_saturated_tf = 2 * (1.2 + 1) / (2 + 1.2 * alpha_length_norm)
    beta_idf = math.log(1 + (2 - 2 + 0.5) / (2 + 0.5))
    beta_saturated_tf = 1 * (1.2 + 1) / (1 + 1.2 * alpha_length_norm)

    assert document.indices == [0, 1]
    assert document.values == pytest.approx(
        [alpha_idf * alpha_saturated_tf, beta_idf * beta_saturated_tf]
    )
    assert query.indices == [0]
    assert query.values == [2.0]
