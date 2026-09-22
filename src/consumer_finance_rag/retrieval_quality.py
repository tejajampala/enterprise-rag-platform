"""Deterministic, generation-pinned managed retrieval quality evaluation."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean
from typing import Final, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from consumer_finance_rag.access import AccessContext
from consumer_finance_rag.evaluation.metrics import (
    hit_rate_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)
from consumer_finance_rag.evaluation.models import EvaluationCase, SeedEvaluationSet
from consumer_finance_rag.lineage.managed_retrieval import ManagedRetrievalRuntime
from consumer_finance_rag.models import RetrievedChunk

REPORT_FORMAT_VERSION: Final[Literal["retrieval-quality-report-v1"]] = (
    "retrieval-quality-report-v1"
)
ExpectedIdentityKind = Literal["source_id", "document_id", "chunk_id"]
QualityGateStatus = Literal["PASS", "FAIL"]
ProgressCallback = Callable[[int, int, "RetrievalEvaluationCase"], None]


class RetrievalEvaluationCase(BaseModel):
    """One private evaluation input scored by stable lineage identity.

    Query text and access-group details are deliberately excluded from serialization and
    representation. Exactly one identity family is used so metric denominators are unambiguous.
    """

    model_config = ConfigDict(frozen=True)

    case_id: str = Field(min_length=1)
    query: str = Field(min_length=1, exclude=True, repr=False)
    expected_source_ids: tuple[str, ...] = ()
    expected_document_ids: tuple[str, ...] = ()
    expected_chunk_ids: tuple[str, ...] = ()
    expected_access: bool = True
    access_context: AccessContext = Field(
        default_factory=AccessContext,
        exclude=True,
        repr=False,
    )
    scenario: str | None = None
    category: str | None = None

    @model_validator(mode="after")
    def one_expected_identity_family(self) -> RetrievalEvaluationCase:
        families = (
            self.expected_source_ids,
            self.expected_document_ids,
            self.expected_chunk_ids,
        )
        if sum(bool(values) for values in families) != 1:
            raise ValueError("exactly one expected identity family must be provided")
        for values in families:
            if len(set(values)) != len(values):
                raise ValueError("expected identities must be unique")
            if any(not value.strip() for value in values):
                raise ValueError("expected identities cannot be blank")
        return self

    @property
    def identity_kind(self) -> ExpectedIdentityKind:
        if self.expected_source_ids:
            return "source_id"
        if self.expected_document_ids:
            return "document_id"
        return "chunk_id"

    @property
    def expected_ids(self) -> tuple[str, ...]:
        if self.expected_source_ids:
            return self.expected_source_ids
        if self.expected_document_ids:
            return self.expected_document_ids
        return self.expected_chunk_ids


class RetrievedIdentity(BaseModel):
    """Safe result identity without raw chunk text or arbitrary payload metadata."""

    model_config = ConfigDict(frozen=True)

    source_id: str | None
    document_id: str
    chunk_id: str


class RetrievalCaseResult(BaseModel):
    """Deterministic result for one case against one pinned generation."""

    model_config = ConfigDict(frozen=True)

    case_id: str
    generation: int = Field(ge=1)
    qdrant_collection: str
    identity_kind: ExpectedIdentityKind
    expected_ids: tuple[str, ...]
    expected_access: bool
    access_expectation_met: bool
    retrieved_identities: tuple[RetrievedIdentity, ...]
    hit_at_k: bool
    recall_at_k: float = Field(ge=0.0, le=1.0)
    reciprocal_rank: float = Field(ge=0.0, le=1.0)
    expected_rank: int | None = Field(default=None, ge=1)
    final_context_count: int = Field(ge=0)
    scenario: str | None = None
    category: str | None = None

    @property
    def collection(self) -> str:
        """Short compatibility alias used in human-readable output."""

        return self.qdrant_collection


class RetrievalQualityMetrics(BaseModel):
    """Aggregate deterministic metrics at one cutoff."""

    model_config = ConfigDict(frozen=True)

    k: int = Field(gt=0)
    recall_at_k: float = Field(ge=0.0, le=1.0)
    mrr: float = Field(ge=0.0, le=1.0)
    hit_rate: float = Field(ge=0.0, le=1.0)
    missed_case_count: int = Field(ge=0)


class RetrievalRuntimeConfiguration(BaseModel):
    """Mutable-at-runtime retrieval settings recorded separately from index identity."""

    model_config = ConfigDict(frozen=True)

    dense_candidates: int = Field(gt=0)
    sparse_candidates: int = Field(gt=0)
    fused_candidates: int = Field(gt=0)
    final_contexts: int = Field(gt=0)
    rrf_k: int = Field(gt=0)
    hnsw_ef: int | None = Field(default=None, gt=0)
    reranker_model: str
    acl_filtering_enabled: bool
    pii_policy: Literal["REDACT"] = "REDACT"


class RetrievalQualityReport(BaseModel):
    """Auditable quality report for one immutable managed generation."""

    model_config = ConfigDict(frozen=True)

    format_version: Literal["retrieval-quality-report-v1"] = REPORT_FORMAT_VERSION
    generated_at: datetime
    generation: int = Field(ge=1)
    index_state_id: UUID
    qdrant_collection: str
    snapshot_hash: str
    embedding_fingerprint: str
    chunking_fingerprint: str
    index_configuration_fingerprint: str
    retrieval_settings: RetrievalRuntimeConfiguration
    case_count: int = Field(ge=0)
    metrics: RetrievalQualityMetrics
    cases: tuple[RetrievalCaseResult, ...]

    @model_validator(mode="after")
    def counts_and_identity_are_consistent(self) -> RetrievalQualityReport:
        if self.case_count != len(self.cases):
            raise ValueError("case_count must equal the number of case results")
        if any(case.generation != self.generation for case in self.cases):
            raise ValueError("all cases must use the report generation")
        if any(case.qdrant_collection != self.qdrant_collection for case in self.cases):
            raise ValueError("all cases must use the report collection")
        return self

    @property
    def collection(self) -> str:
        return self.qdrant_collection

    @property
    def recall_at_k(self) -> float:
        return self.metrics.recall_at_k

    @property
    def mrr(self) -> float:
        return self.metrics.mrr

    @property
    def hit_rate(self) -> float:
        return self.metrics.hit_rate

    @property
    def missed_case_count(self) -> int:
        return self.metrics.missed_case_count


class RetrievalQualityComparison(BaseModel):
    """Objective deltas between reports containing the same evaluation cases."""

    model_config = ConfigDict(frozen=True)

    baseline_generation: int
    candidate_generation: int
    baseline_collection: str
    candidate_collection: str
    recall_delta: float
    mrr_delta: float
    hit_rate_delta: float
    cases_improved: tuple[str, ...]
    cases_regressed: tuple[str, ...]
    cases_unchanged: tuple[str, ...]
    newly_missed_cases: tuple[str, ...]
    newly_recovered_cases: tuple[str, ...]


class RetrievalQualityThresholds(BaseModel):
    """Explicit CI policy kept separate from immutable index configuration."""

    model_config = ConfigDict(frozen=True)

    max_recall_drop: float = Field(default=0.0, ge=0.0, le=1.0)
    max_mrr_drop: float = Field(default=0.0, ge=0.0, le=1.0)


class RetrievalQualityGateResult(BaseModel):
    """PASS/FAIL assessment without publication or deployment side effects."""

    model_config = ConfigDict(frozen=True)

    status: QualityGateStatus
    thresholds: RetrievalQualityThresholds
    violations: tuple[str, ...]


def evaluation_cases_from_seed(
    dataset: SeedEvaluationSet,
    *,
    identity_kind: Literal["chunk_id", "document_id"] = "chunk_id",
) -> tuple[RetrievalEvaluationCase, ...]:
    """Adapt the Phase 2 golden-case format instead of creating another corpus."""

    return tuple(_case_from_phase2(case, identity_kind=identity_kind) for case in dataset.cases)


def evaluate_managed_retrieval(
    runtime: ManagedRetrievalRuntime,
    cases: Sequence[RetrievalEvaluationCase],
    *,
    k: int | None = None,
    generated_at: datetime | None = None,
    progress: ProgressCallback | None = None,
) -> RetrievalQualityReport:
    """Evaluate every case through one already-pinned managed retrieval runtime."""

    cutoff = runtime.retrieval_service.settings.final_contexts if k is None else k
    if cutoff <= 0:
        raise ValueError("k must be positive")
    if not cases:
        raise ValueError("at least one evaluation case is required")

    # Copy identity once. No repository/ACTIVE read occurs during the case loop.
    generation = runtime.generation
    collection = runtime.qdrant_collection
    results: list[RetrievalCaseResult] = []
    total = len(cases)
    for position, case in enumerate(cases, start=1):
        if progress is not None:
            progress(position, total, case)
        trace = runtime.retrieval_service.retrieve_debug(
            case.query,
            top_k=cutoff,
            access_context=case.access_context,
        )
        results.append(
            score_retrieval_case(
                case,
                trace.reranked,
                generation=generation,
                qdrant_collection=collection,
                k=cutoff,
            )
        )

    metrics = RetrievalQualityMetrics(
        k=cutoff,
        recall_at_k=fmean(result.recall_at_k for result in results),
        mrr=fmean(result.reciprocal_rank for result in results),
        hit_rate=fmean(float(result.hit_at_k) for result in results),
        missed_case_count=sum(not result.hit_at_k for result in results),
    )
    service = runtime.retrieval_service
    settings = service.settings
    timestamp = generated_at or datetime.now(UTC)
    if timestamp.tzinfo is None:
        raise ValueError("generated_at must be timezone-aware")
    return RetrievalQualityReport(
        generated_at=timestamp,
        generation=generation,
        index_state_id=runtime.index_state_id,
        qdrant_collection=collection,
        snapshot_hash=runtime.snapshot_hash,
        embedding_fingerprint=runtime.embedding_fingerprint,
        chunking_fingerprint=runtime.chunking_fingerprint,
        index_configuration_fingerprint=runtime.index_configuration_fingerprint,
        retrieval_settings=RetrievalRuntimeConfiguration(
            dense_candidates=settings.dense_candidates,
            sparse_candidates=settings.sparse_candidates,
            fused_candidates=settings.fused_candidates,
            final_contexts=settings.final_contexts,
            rrf_k=settings.rrf_k,
            hnsw_ef=settings.hnsw_ef,
            reranker_model=service.reranker.model_name,
            acl_filtering_enabled=service.access_control_enabled,
        ),
        case_count=len(results),
        metrics=metrics,
        cases=tuple(results),
    )


def score_retrieval_case(
    case: RetrievalEvaluationCase,
    retrieved: Sequence[RetrievedChunk],
    *,
    generation: int,
    qdrant_collection: str,
    k: int,
) -> RetrievalCaseResult:
    """Score safe result identities with the existing Phase 2 IR formulas."""

    if k <= 0:
        raise ValueError("k must be positive")
    identities = tuple(_identity(item) for item in retrieved[:k])
    retrieved_ids = _identity_values(identities, case.identity_kind)
    expected = set(case.expected_ids)
    first_rank = _expected_rank(retrieved_ids, expected, k)

    if case.expected_access:
        recall = recall_at_k(retrieved_ids, expected, k)
        hit = bool(hit_rate_at_k(retrieved_ids, expected, k))
        reciprocal_rank = reciprocal_rank_at_k(retrieved_ids, expected, k)
        access_expectation_met = hit
    else:
        # A negative ACL case succeeds only when the protected identity is absent.
        access_expectation_met = first_rank is None
        recall = float(access_expectation_met)
        hit = access_expectation_met
        reciprocal_rank = float(access_expectation_met)

    return RetrievalCaseResult(
        case_id=case.case_id,
        generation=generation,
        qdrant_collection=qdrant_collection,
        identity_kind=case.identity_kind,
        expected_ids=case.expected_ids,
        expected_access=case.expected_access,
        access_expectation_met=access_expectation_met,
        retrieved_identities=identities,
        hit_at_k=hit,
        recall_at_k=recall,
        reciprocal_rank=reciprocal_rank,
        expected_rank=first_rank,
        final_context_count=len(retrieved),
        scenario=case.scenario,
        category=case.category,
    )


def compare_quality_reports(
    baseline: RetrievalQualityReport,
    candidate: RetrievalQualityReport,
) -> RetrievalQualityComparison:
    """Compare equal case sets without assigning a subjective good/bad label."""

    if baseline.metrics.k != candidate.metrics.k:
        raise ValueError("quality reports must use the same K")
    baseline_cases = _cases_by_id(baseline)
    candidate_cases = _cases_by_id(candidate)
    if set(baseline_cases) != set(candidate_cases):
        raise ValueError("quality reports must contain the same case IDs")

    improved: list[str] = []
    regressed: list[str] = []
    unchanged: list[str] = []
    newly_missed: list[str] = []
    newly_recovered: list[str] = []
    for case_id in sorted(baseline_cases):
        before = baseline_cases[case_id]
        after = candidate_cases[case_id]
        _validate_comparable_case(before, after)
        before_score = _case_quality_tuple(before)
        after_score = _case_quality_tuple(after)
        if after_score > before_score:
            improved.append(case_id)
        elif after_score < before_score:
            regressed.append(case_id)
        else:
            unchanged.append(case_id)
        if before.hit_at_k and not after.hit_at_k:
            newly_missed.append(case_id)
        if not before.hit_at_k and after.hit_at_k:
            newly_recovered.append(case_id)

    return RetrievalQualityComparison(
        baseline_generation=baseline.generation,
        candidate_generation=candidate.generation,
        baseline_collection=baseline.qdrant_collection,
        candidate_collection=candidate.qdrant_collection,
        recall_delta=candidate.recall_at_k - baseline.recall_at_k,
        mrr_delta=candidate.mrr - baseline.mrr,
        hit_rate_delta=candidate.hit_rate - baseline.hit_rate,
        cases_improved=tuple(improved),
        cases_regressed=tuple(regressed),
        cases_unchanged=tuple(unchanged),
        newly_missed_cases=tuple(newly_missed),
        newly_recovered_cases=tuple(newly_recovered),
    )


def evaluate_quality_thresholds(
    comparison: RetrievalQualityComparison,
    thresholds: RetrievalQualityThresholds,
) -> RetrievalQualityGateResult:
    """Apply explicit regression limits without mutating lifecycle state."""

    violations: list[str] = []
    if comparison.recall_delta < -thresholds.max_recall_drop:
        violations.append(
            "recall drop "
            f"{-comparison.recall_delta:.6f} exceeds {thresholds.max_recall_drop:.6f}"
        )
    if comparison.mrr_delta < -thresholds.max_mrr_drop:
        violations.append(
            f"MRR drop {-comparison.mrr_delta:.6f} exceeds {thresholds.max_mrr_drop:.6f}"
        )
    return RetrievalQualityGateResult(
        status="FAIL" if violations else "PASS",
        thresholds=thresholds,
        violations=tuple(violations),
    )


def write_retrieval_quality_report(
    report: RetrievalQualityReport,
    path: str | Path,
) -> Path:
    """Write an explicitly versioned report containing no queries or raw contexts."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return output


def _case_from_phase2(
    case: EvaluationCase,
    *,
    identity_kind: Literal["chunk_id", "document_id"],
) -> RetrievalEvaluationCase:
    if identity_kind == "chunk_id":
        return RetrievalEvaluationCase(
            case_id=case.case_id,
            query=case.question,
            category=case.category,
            expected_chunk_ids=tuple(case.relevant_chunk_ids),
        )
    return RetrievalEvaluationCase(
        case_id=case.case_id,
        query=case.question,
        category=case.category,
        expected_document_ids=tuple(case.relevant_document_ids),
    )


def _identity(item: RetrievedChunk) -> RetrievedIdentity:
    source = item.chunk.metadata.get("source_id")
    return RetrievedIdentity(
        source_id=source if isinstance(source, str) and source else None,
        document_id=item.chunk.document_id,
        chunk_id=item.chunk.chunk_id,
    )


def _identity_values(
    identities: Sequence[RetrievedIdentity],
    kind: ExpectedIdentityKind,
) -> list[str]:
    values: list[str] = []
    for identity in identities:
        value = getattr(identity, kind)
        if value is not None:
            values.append(value)
    return values


def _expected_rank(retrieved: Sequence[str], expected: set[str], k: int) -> int | None:
    seen: set[str] = set()
    rank = 0
    for identity in retrieved:
        if identity in seen:
            continue
        seen.add(identity)
        rank += 1
        if rank > k:
            break
        if identity in expected:
            return rank
    return None


def _cases_by_id(report: RetrievalQualityReport) -> dict[str, RetrievalCaseResult]:
    by_id = {case.case_id: case for case in report.cases}
    if len(by_id) != len(report.cases):
        raise ValueError("quality report contains duplicate case IDs")
    return by_id


def _validate_comparable_case(
    baseline: RetrievalCaseResult,
    candidate: RetrievalCaseResult,
) -> None:
    identity = (
        baseline.identity_kind,
        baseline.expected_ids,
        baseline.expected_access,
    )
    candidate_identity = (
        candidate.identity_kind,
        candidate.expected_ids,
        candidate.expected_access,
    )
    if identity != candidate_identity:
        raise ValueError(f"case {baseline.case_id!r} has different expectations")


def _case_quality_tuple(result: RetrievalCaseResult) -> tuple[float, float, float]:
    return (float(result.hit_at_k), result.recall_at_k, result.reciprocal_rank)
