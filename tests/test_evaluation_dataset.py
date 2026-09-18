from __future__ import annotations

import pytest

from consumer_finance_rag.acquisition import JsonFixtureSource
from consumer_finance_rag.chunking import ParagraphChunker
from consumer_finance_rag.evaluation.dataset import (
    EvaluationDatasetValidationError,
    load_evaluation_set,
    validate_evaluation_set,
)
from consumer_finance_rag.normalization import normalize_records


def test_seed_evaluation_set_is_reviewable_and_matches_fixture() -> None:
    dataset = load_evaluation_set("data/evaluation/seed_cases.json")
    documents = normalize_records(
        JsonFixtureSource("data/fixtures/cfpb_consumer_finance.json").acquire()
    )
    chunks = [
        chunk
        for document in documents
        for chunk in ParagraphChunker(chunk_size=900, overlap=120).chunk(document)
    ]

    validate_evaluation_set(dataset, documents=documents, chunks=chunks)

    assert len(dataset.cases) == 36
    assert dataset.status == "seed_review_required"
    assert {case.category for case in dataset.cases} == {
        "credit cards",
        "credit reporting",
        "mortgages",
        "bank accounts",
        "debt collection",
        "payments/fraud",
    }
    assert {case.difficulty for case in dataset.cases} == {"easy", "medium", "hard"}


def test_dataset_validator_rejects_duplicate_cases_and_unknown_qrels() -> None:
    dataset = load_evaluation_set("data/evaluation/seed_cases.json")
    bad_case = dataset.cases[0].model_copy(
        update={"relevant_chunk_ids": ["missing-chunk"], "graded_relevance": {}}
    )
    invalid = dataset.model_copy(update={"cases": [bad_case, bad_case]})

    with pytest.raises(EvaluationDatasetValidationError) as error:
        validate_evaluation_set(invalid, chunks=[])
    assert "duplicate case IDs" in str(error.value)
    assert "unknown chunk IDs" in str(error.value)
