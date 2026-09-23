from __future__ import annotations

import math
from pathlib import Path

import pytest

from consumer_finance_rag.sparse import (
    BM25SparseEncoder,
    bm25_corpus_hash,
    bm25_state_from_json_dict,
    bm25_state_hash,
    bm25_state_to_json_dict,
    canonical_bm25_state,
    tokenize,
)


def test_bm25_state_json_round_trip_and_hash_are_deterministic() -> None:
    state = BM25SparseEncoder.fit(["alpha beta", "beta gamma"]).state

    payload = bm25_state_to_json_dict(state)
    restored = bm25_state_from_json_dict(payload)

    assert restored == state
    assert canonical_bm25_state(restored) == canonical_bm25_state(state)
    assert bm25_state_hash(restored) == bm25_state_hash(state)
    assert len(bm25_state_hash(state)) == 64


def test_bm25_state_hash_changes_with_vocabulary() -> None:
    first = BM25SparseEncoder.fit(["alpha beta"]).state
    second = BM25SparseEncoder.fit(["alpha gamma"]).state

    assert bm25_state_hash(first) != bm25_state_hash(second)


def test_existing_bm25_file_save_and_load_remain_compatible(tmp_path: Path) -> None:
    path = tmp_path / "bm25.json"
    encoder = BM25SparseEncoder.fit(["credit report", "mortgage disclosure"])

    encoder.save(path)
    loaded = BM25SparseEncoder.load(path)

    assert loaded.state == encoder.state
    assert loaded.encode_query("credit disclosure") == encoder.encode_query(
        "credit disclosure"
    )


def test_bm25_state_uses_deterministic_vocabulary_document_frequency_and_token_lengths() -> None:
    texts = ["Alpha alpha, beta!", "Beta gamma gamma gamma."]

    encoder = BM25SparseEncoder.fit(texts)

    assert encoder.state.vocabulary == {"alpha": 0, "beta": 1, "gamma": 2}
    assert encoder.state.document_frequencies == {"alpha": 1, "beta": 2, "gamma": 1}
    assert encoder.state.document_count == 2
    assert encoder.state.average_document_length == 3.5
    assert encoder.state.corpus_hash == bm25_corpus_hash(texts)
    assert tokenize(texts[0]) == ["alpha", "alpha", "beta"]


def test_bm25_corpus_hash_is_ordered_and_length_framed() -> None:
    assert bm25_corpus_hash(["ab", "c"]) != bm25_corpus_hash(["a", "bc"])
    assert bm25_corpus_hash(["first", "second"]) != bm25_corpus_hash(["second", "first"])


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
