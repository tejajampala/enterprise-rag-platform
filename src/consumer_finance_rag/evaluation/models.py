"""Typed evaluation datasets and regression-report schemas."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

EvaluationCategory = Literal[
    "credit cards",
    "credit reporting",
    "mortgages",
    "bank accounts",
    "debt collection",
    "payments/fraud",
]
Difficulty = Literal["easy", "medium", "hard"]
ReviewStatus = Literal["review_required", "reviewed", "rejected"]
RelevanceGrade = Annotated[int, Field(ge=1, le=3)]


class EvaluationCase(BaseModel):
    """A reviewable question and its source-grounded relevance judgements."""

    model_config = ConfigDict(frozen=True)

    case_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    category: EvaluationCategory
    difficulty: Difficulty
    expected_facts: list[str] = Field(min_length=1)
    reference_answer: str = Field(min_length=1)
    relevant_document_ids: list[str] = Field(min_length=1)
    relevant_chunk_ids: list[str] = Field(min_length=1)
    graded_relevance: dict[str, RelevanceGrade] = Field(default_factory=dict)
    notes: str | None = None
    review_status: ReviewStatus = "review_required"

    @model_validator(mode="after")
    def references_are_unique_and_grades_are_relevant(self) -> EvaluationCase:
        if len(set(self.relevant_document_ids)) != len(self.relevant_document_ids):
            raise ValueError("relevant_document_ids must be unique")
        if len(set(self.relevant_chunk_ids)) != len(self.relevant_chunk_ids):
            raise ValueError("relevant_chunk_ids must be unique")
        unknown_grades = set(self.graded_relevance) - set(self.relevant_chunk_ids)
        if unknown_grades:
            raise ValueError("graded_relevance keys must also be relevant_chunk_ids")
        return self


class SeedEvaluationSet(BaseModel):
    """Source-controlled seed cases; not a production golden dataset."""

    model_config = ConfigDict(frozen=True)

    name: str
    version: str
    status: Literal["seed_review_required", "reviewed"] = "seed_review_required"
    source_corpus: str
    cases: list[EvaluationCase] = Field(min_length=1)


class IRMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    recall: float
    precision: float
    hit_rate: float
    mrr: float
    ndcg: float
    average_precision: float


class CaseStageResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    retrieved_ids: list[str]
    metrics: dict[str, float]


class RetrievalCaseResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    case_id: str
    category: EvaluationCategory
    difficulty: Difficulty
    stages: dict[str, CaseStageResult]


class RetrievalEvaluationReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    report_type: Literal["retrieval_evaluation"] = "retrieval_evaluation"
    created_at: str
    dataset_name: str
    dataset_version: str
    dataset_status: str
    case_count: int
    k_values: list[int]
    parameters: dict[str, str | int | float | bool | None]
    aggregate: dict[str, dict[str, float]]
    per_category: dict[str, dict[str, dict[str, float]]]
    per_difficulty: dict[str, dict[str, dict[str, float]]]
    cases: list[RetrievalCaseResult]


class JudgeMetricValue(BaseModel):
    model_config = ConfigDict(frozen=True)

    value: float | str | bool | None
    reason: str | None = None


class RAGCaseResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    case_id: str
    question: str
    answer: str
    contexts: list[str]
    chunk_ids: list[str]
    token_usage: dict[str, int] | None = None
    request_id: str | None = None
    trace_id: str | None = None
    mlflow_judges: dict[str, JudgeMetricValue] = Field(default_factory=dict)
    ragas: dict[str, JudgeMetricValue] = Field(default_factory=dict)


class RAGEvaluationReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    report_type: Literal["rag_evaluation"] = "rag_evaluation"
    created_at: str
    dataset_name: str
    dataset_version: str
    case_count: int
    retrieval: dict[str, float]
    rag: dict[str, float | None]
    operations: dict[str, int | float | None]
    cases: list[RAGCaseResult]
