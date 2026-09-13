"""Seed evaluation-set loading and corpus compatibility validation."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from consumer_finance_rag.evaluation.models import SeedEvaluationSet
from consumer_finance_rag.models import Chunk, Document


class EvaluationDatasetValidationError(ValueError):
    """Raised when seed qrels do not match the selected corpus."""


def load_evaluation_set(path: str | Path) -> SeedEvaluationSet:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return SeedEvaluationSet.model_validate(data)


def validate_evaluation_set(
    dataset: SeedEvaluationSet,
    *,
    documents: Sequence[Document] = (),
    chunks: Sequence[Chunk],
) -> None:
    """Validate unique cases and every document/chunk qrel against normalized corpus IDs."""

    issues: list[str] = []
    case_ids = [case.case_id for case in dataset.cases]
    duplicates = sorted({case_id for case_id in case_ids if case_ids.count(case_id) > 1})
    if duplicates:
        issues.append(f"duplicate case IDs: {', '.join(duplicates)}")

    known_chunk_ids = {chunk.chunk_id for chunk in chunks}
    known_document_ids = {document.document_id for document in documents} | {
        chunk.document_id for chunk in chunks
    }
    for case in dataset.cases:
        missing_documents = sorted(set(case.relevant_document_ids) - known_document_ids)
        missing_chunks = sorted(set(case.relevant_chunk_ids) - known_chunk_ids)
        missing_grades = sorted(set(case.graded_relevance) - known_chunk_ids)
        if missing_documents:
            issues.append(f"{case.case_id}: unknown document IDs: {', '.join(missing_documents)}")
        if missing_chunks:
            issues.append(f"{case.case_id}: unknown chunk IDs: {', '.join(missing_chunks)}")
        if missing_grades:
            issues.append(
                f"{case.case_id}: graded relevance refers to unknown chunks: "
                f"{', '.join(missing_grades)}"
            )
    if issues:
        detail = "\n- ".join(issues)
        raise EvaluationDatasetValidationError(f"Evaluation dataset is incompatible:\n- {detail}")
