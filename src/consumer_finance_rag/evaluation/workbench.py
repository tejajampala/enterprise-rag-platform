"""Pure planning and at-most-once controls for private workbench evaluation."""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from consumer_finance_rag.costs import ModelUsageSummary
from consumer_finance_rag.evaluation.managed_rag import (
    ApprovedAnswerEvaluation,
    JudgeStageSummary,
    ManagedRAGEvaluationCase,
    ManagedRAGEvaluationDataset,
)
from consumer_finance_rag.evaluation.ragas_adapter import RagasEvaluationInput
from consumer_finance_rag.evaluation.ragas_execution_trace import RagasExecutionTraceReference
from consumer_finance_rag.models import ContextPassage, RAGResponse
from consumer_finance_rag.observability import bedrock_token_usage

EVALUATION_INPUT_CONTRACT_VERSION: Final = "evaluation-input-snapshot-v1"
RAGAS_INPUT_CONTRACT_VERSION: Final = "ragas-selected-original-chunks-v1"
RAGAS_METRIC_INPUT_FIELDS: Final[dict[str, tuple[str, ...]]] = {
    "faithfulness": ("question", "generated_answer", "ragas_contexts"),
    "context_precision": ("question", "reference_answer", "ragas_contexts"),
    "context_recall": ("question", "reference_answer", "ragas_contexts"),
    "response_groundedness": ("generated_answer", "ragas_contexts"),
}


class WorkbenchEvaluationStatus(StrEnum):
    PENDING = "PENDING"
    COMPLETED = "COMPLETED"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"
    UNAVAILABLE = "UNAVAILABLE"
    SKIPPED = "SKIPPED"
    NOT_REQUESTED = "NOT_REQUESTED"
    REFERENCE_REQUIRED = "REFERENCE_REQUIRED"


class WorkbenchEvaluationOptions(BaseModel):
    model_config = ConfigDict(frozen=True)

    mlflow_genai: bool = True
    ragas: bool = True
    save_to_mlflow: bool = True
    capture_judge_io: bool = False
    judge_io_retention_hours: int = Field(default=24, ge=1, le=168)


class BackendEvaluationState(BaseModel):
    """Display-safe state for one independently configured judge backend."""

    model_config = ConfigDict(frozen=True)

    status: WorkbenchEvaluationStatus
    reason: str
    metrics: dict[str, float] = Field(default_factory=dict)
    metric_statuses: dict[str, WorkbenchEvaluationStatus] = Field(default_factory=dict)
    metric_reasons: dict[str, str] = Field(default_factory=dict)
    run_id: str | None = None
    usage: ModelUsageSummary | None = None
    execution_trace: RagasExecutionTraceReference | None = None


class ChatEvaluationState(BaseModel):
    """Request-scoped evaluation state; never inferred from a saved batch report."""

    model_config = ConfigDict(frozen=True)

    request_id: str
    case_id: str | None
    eligible: bool
    eligibility_reason: str
    evaluation_key: str | None
    judge_model: str | None = None
    judge_region: str | None = None
    options: WorkbenchEvaluationOptions
    started_at: datetime | None = None
    completed_at: datetime | None = None
    mlflow_genai: BackendEvaluationState
    ragas: BackendEvaluationState


class EvaluationInputSnapshot(BaseModel):
    """Bounded process-local content for one exact reviewed PUBLIC request."""

    model_config = ConfigDict(frozen=True)

    format_version: Literal["evaluation-input-snapshot-v1"] = (
        EVALUATION_INPUT_CONTRACT_VERSION
    )
    ragas_contract_version: Literal["ragas-selected-original-chunks-v1"] = (
        RAGAS_INPUT_CONTRACT_VERSION
    )
    captured_at: datetime
    expires_at: datetime
    request_id: str
    case_id: str
    configuration_id: str
    trace_id: str | None
    evaluation_run_id: str | None = None
    ragas_execution_trace: RagasExecutionTraceReference | None = None
    dataset_name: str
    dataset_version: str
    dataset_privacy_classification: str
    authorization_visibility: Literal["PUBLIC"]
    review_status: Literal["reviewed"]
    generation_model: str
    generation_region: str
    generation_token_usage: dict[str, int] | None = None
    judge_model: str | None
    judge_region: str | None
    original_question: str
    generated_answer: str
    prompt_context: str
    passages: tuple[ContextPassage, ...]
    ragas_input: RagasEvaluationInput


class EvaluationInputSnapshotStore:
    """Small TTL/LRU store; it never persists or reconstructs content."""

    def __init__(
        self,
        *,
        maximum_snapshots: int = 8,
        retention: timedelta = timedelta(hours=1),
    ) -> None:
        if maximum_snapshots <= 0 or retention <= timedelta(0):
            raise ValueError("snapshot bounds must be positive")
        self.maximum_snapshots = maximum_snapshots
        self.retention = retention
        self._snapshots: OrderedDict[str, EvaluationInputSnapshot] = OrderedDict()

    def record(self, snapshot: EvaluationInputSnapshot) -> None:
        self._purge_expired(datetime.now(UTC))
        self._snapshots.pop(snapshot.request_id, None)
        self._snapshots[snapshot.request_id] = snapshot
        while len(self._snapshots) > self.maximum_snapshots:
            self._snapshots.popitem(last=False)

    def get_exact(
        self,
        request_id: str,
        *,
        case_id: str | None = None,
        configuration_id: str | None = None,
        now: datetime | None = None,
    ) -> EvaluationInputSnapshot | None:
        current = now or datetime.now(UTC)
        self._purge_expired(current)
        snapshot = self._snapshots.get(request_id)
        if snapshot is None:
            return None
        if case_id is not None and snapshot.case_id != case_id:
            return None
        if configuration_id is not None and snapshot.configuration_id != configuration_id:
            return None
        self._snapshots.move_to_end(request_id)
        return snapshot

    def attach_evaluation_run(
        self,
        request_id: str,
        *,
        case_id: str,
        configuration_id: str,
        evaluation_run_id: str | None,
        ragas_execution_trace: RagasExecutionTraceReference | None = None,
    ) -> EvaluationInputSnapshot | None:
        snapshot = self.get_exact(
            request_id,
            case_id=case_id,
            configuration_id=configuration_id,
        )
        if snapshot is None:
            return None
        updated = snapshot.model_copy(
            update={
                "evaluation_run_id": evaluation_run_id,
                "ragas_execution_trace": ragas_execution_trace,
            }
        )
        self.record(updated)
        return updated

    def _purge_expired(self, now: datetime) -> None:
        expired = [
            request_id
            for request_id, snapshot in self._snapshots.items()
            if snapshot.expires_at <= now
        ]
        for request_id in expired:
            self._snapshots.pop(request_id, None)


def capture_evaluation_input_snapshot(
    *,
    response: RAGResponse,
    request_id: str,
    case: ManagedRAGEvaluationCase,
    dataset: ManagedRAGEvaluationDataset,
    configuration_id: str,
    generation_region: str,
    judge_model: str | None,
    judge_region: str | None,
    retention: timedelta = timedelta(hours=1),
    now: datetime | None = None,
) -> EvaluationInputSnapshot:
    """Capture actual inputs only after exact reviewed-PUBLIC eligibility validation."""

    selected = next((item for item in dataset.cases if item.case_id == case.case_id), None)
    if selected != case or dataset.status != "reviewed":
        raise ValueError("case is not the exact reviewed dataset case")
    if (
        case.authorization.visibility != "PUBLIC"
        or case.authorization.groups
        or case.review_status != "reviewed"
    ):
        raise ValueError("full input inspection requires reviewed PUBLIC authorization")
    if response.question != case.question:
        raise ValueError("response does not match the reviewed evaluation question")
    if response.request_id != request_id:
        raise ValueError("response does not match the exact request ID")
    context = response.structured_context
    if context is None:
        raise ValueError("exact generation context is unavailable")
    if tuple(item.chunk.chunk_id for item in context.chunks) != tuple(
        passage.chunk_id for passage in context.passages
    ):
        raise ValueError("generation context passage identities are inconsistent")
    captured_at = now or datetime.now(UTC)
    ragas_contexts = tuple(item.chunk.text for item in response.retrieved_chunks)
    return EvaluationInputSnapshot(
        captured_at=captured_at,
        expires_at=captured_at + retention,
        request_id=request_id,
        case_id=case.case_id,
        configuration_id=configuration_id,
        trace_id=response.trace_id,
        dataset_name=dataset.name,
        dataset_version=dataset.version,
        dataset_privacy_classification=dataset.privacy_classification,
        authorization_visibility=case.authorization.visibility,
        review_status=case.review_status,
        generation_model=response.model_metadata.generation_model,
        generation_region=generation_region,
        generation_token_usage=bedrock_token_usage(response.generation_metadata),
        judge_model=judge_model,
        judge_region=judge_region,
        original_question=response.question,
        generated_answer=response.answer,
        prompt_context=context.text,
        passages=tuple(context.passages),
        ragas_input=RagasEvaluationInput(
            case_id=case.case_id,
            question=response.question,
            answer=response.answer,
            contexts=list(ragas_contexts),
            reference=case.reference_answer,
        ),
    )


class MetricPlan(BaseModel):
    model_config = ConfigDict(frozen=True)

    selected: tuple[str, ...]
    skipped: dict[str, WorkbenchEvaluationStatus]


class EvaluationPlan(BaseModel):
    model_config = ConfigDict(frozen=True)

    mlflow_genai: MetricPlan
    ragas: MetricPlan

    @property
    def logical_metric_call_count(self) -> int:
        return len(self.mlflow_genai.selected) + len(self.ragas.selected)


class EvaluationExecutionRegistry:
    """Session-owned result cache keyed by request/dataset and exact configuration."""

    def __init__(self) -> None:
        self._results: dict[str, Any] = {}

    def get(self, key: str) -> Any | None:
        return self._results.get(key)

    def record(self, key: str, result: Any) -> None:
        self._results[key] = result

    def should_execute(self, key: str, *, explicit_re_evaluate: bool = False) -> bool:
        return explicit_re_evaluate or key not in self._results


def evaluation_plan(*, has_reference: bool, has_content_trace: bool) -> EvaluationPlan:
    mlflow_selected = ["relevance_to_query", "safety"]
    mlflow_skipped: dict[str, WorkbenchEvaluationStatus] = {}
    if has_reference:
        mlflow_selected.append("correctness")
    else:
        mlflow_skipped["correctness"] = WorkbenchEvaluationStatus.REFERENCE_REQUIRED
    if has_content_trace:
        mlflow_selected.append("retrieval_groundedness")
    else:
        mlflow_skipped["retrieval_groundedness"] = WorkbenchEvaluationStatus.SKIPPED

    ragas_selected = ["faithfulness", "response_groundedness"]
    ragas_skipped: dict[str, WorkbenchEvaluationStatus] = {}
    if has_reference:
        ragas_selected.extend(["context_precision", "context_recall"])
    else:
        ragas_skipped.update(
            {
                "context_precision": WorkbenchEvaluationStatus.REFERENCE_REQUIRED,
                "context_recall": WorkbenchEvaluationStatus.REFERENCE_REQUIRED,
            }
        )
    return EvaluationPlan(
        mlflow_genai=MetricPlan(
            selected=tuple(mlflow_selected),
            skipped=mlflow_skipped,
        ),
        ragas=MetricPlan(selected=tuple(ragas_selected), skipped=ragas_skipped),
    )


def evaluation_key(
    *,
    scope_id: str,
    generation: int,
    case_ids: Sequence[str],
    generation_model: str,
    judge_model: str | None,
    judge_region: str | None = None,
    options: WorkbenchEvaluationOptions,
) -> str:
    payload = {
        "scope_id": scope_id,
        "generation": generation,
        "case_ids": list(case_ids),
        "generation_model": generation_model,
        "judge_model": judge_model,
        "judge_region": judge_region,
        "options": options.model_dump(mode="json"),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def find_eligible_case(
    question: str,
    dataset: ManagedRAGEvaluationDataset,
) -> ManagedRAGEvaluationCase | None:
    normalized = " ".join(question.split()).casefold()
    matches = [
        case
        for case in dataset.cases
        if " ".join(case.question.split()).casefold() == normalized
    ]
    return matches[0] if len(matches) == 1 else None


def proposed_chat_evaluation(
    *,
    request_id: str,
    case_id: str | None,
    evaluation_key_value: str | None,
    options: WorkbenchEvaluationOptions,
    automatic_evaluation_enabled: bool,
    evaluate_answer_selected: bool,
    has_reference: bool,
    judge_model: str | None = None,
    judge_region: str | None = None,
    now: datetime | None = None,
) -> ChatEvaluationState:
    """Describe exactly what an explicit Chat submission requested before judging."""

    eligible = case_id is not None
    if not eligible:
        eligibility_reason = (
            "Ineligible: the submitted question did not exactly match one unique reviewed "
            "PUBLIC evaluation case. No judge received this answer."
        )
    else:
        eligibility_reason = (
            "Eligible: the submitted question exactly matched one reviewed PUBLIC evaluation "
            f"case ({case_id})."
        )
    started_at = now or datetime.now(UTC)
    return ChatEvaluationState(
        request_id=request_id,
        case_id=case_id,
        eligible=eligible,
        eligibility_reason=eligibility_reason,
        evaluation_key=evaluation_key_value,
        judge_model=judge_model,
        judge_region=judge_region,
        options=options,
        started_at=(
            started_at
            if eligible and automatic_evaluation_enabled and evaluate_answer_selected
            else None
        ),
        mlflow_genai=_proposed_backend_state(
            backend="MLflow GenAI",
            enabled=options.mlflow_genai,
            eligible=eligible,
            automatic_evaluation_enabled=automatic_evaluation_enabled,
            evaluate_answer_selected=evaluate_answer_selected,
            has_reference=has_reference,
        ),
        ragas=_proposed_backend_state(
            backend="RAGAS",
            enabled=options.ragas,
            eligible=eligible,
            automatic_evaluation_enabled=automatic_evaluation_enabled,
            evaluate_answer_selected=evaluate_answer_selected,
            has_reference=has_reference,
        ),
    )


def complete_chat_evaluation(
    state: ChatEvaluationState,
    result: ApprovedAnswerEvaluation,
    *,
    now: datetime | None = None,
) -> ChatEvaluationState:
    """Attach actual adapter results to the exact request-scoped pending state."""

    if state.request_id == "" or state.case_id != result.case_id:
        raise ValueError("Evaluation result does not match the request-scoped case")
    return state.model_copy(
        update={
            "completed_at": now or datetime.now(UTC),
            "mlflow_genai": backend_state_from_summary(result.mlflow_genai),
            "ragas": backend_state_from_summary(result.ragas),
        }
    )


def fail_chat_evaluation(
    state: ChatEvaluationState,
    *,
    reason: str,
    now: datetime | None = None,
) -> ChatEvaluationState:
    """Mark only requested pending backends failed while preserving independent skips."""

    return state.model_copy(
        update={
            "completed_at": now or datetime.now(UTC),
            "mlflow_genai": _fail_pending_backend(state.mlflow_genai, reason),
            "ragas": _fail_pending_backend(state.ragas, reason),
        }
    )


def backend_state_from_summary(summary: JudgeStageSummary) -> BackendEvaluationState:
    status = display_status(summary)
    metric_statuses = {
        name: (
            WorkbenchEvaluationStatus.REFERENCE_REQUIRED
            if "reference" in reason.casefold()
            else WorkbenchEvaluationStatus.SKIPPED
        )
        for name, reason in summary.skipped_metrics.items()
    }
    metric_statuses.update(
        {
            name: (
                WorkbenchEvaluationStatus.COMPLETED
                if metric_status == "COMPLETED"
                else WorkbenchEvaluationStatus.FAILED
            )
            for name, metric_status in summary.metric_statuses.items()
        }
    )
    metric_reasons = dict(summary.skipped_metrics)
    metric_reasons.update(summary.failed_metrics)
    reason = summary.reason or {
        WorkbenchEvaluationStatus.COMPLETED: "The requested backend completed.",
        WorkbenchEvaluationStatus.PARTIAL: "Only some requested metrics completed.",
        WorkbenchEvaluationStatus.FAILED: "The requested backend failed.",
        WorkbenchEvaluationStatus.UNAVAILABLE: "The requested backend was unavailable.",
        WorkbenchEvaluationStatus.NOT_REQUESTED: "This backend was not requested.",
    }.get(status, "No additional status reason was recorded.")
    return BackendEvaluationState(
        status=status,
        reason=reason,
        metrics=summary.aggregate_metrics,
        metric_statuses=metric_statuses,
        metric_reasons=metric_reasons,
        run_id=summary.run_id,
        usage=summary.usage,
        execution_trace=summary.execution_trace,
    )


def saved_report_matches_batch(
    report_generated_at: datetime,
    current_batch_generated_at: datetime | None,
) -> bool:
    """Use exact report identity, never generation alone, to identify the current batch."""

    return (
        current_batch_generated_at is not None
        and report_generated_at == current_batch_generated_at
    )


def _proposed_backend_state(
    *,
    backend: str,
    enabled: bool,
    eligible: bool,
    automatic_evaluation_enabled: bool,
    evaluate_answer_selected: bool,
    has_reference: bool,
) -> BackendEvaluationState:
    if not eligible:
        return BackendEvaluationState(
            status=WorkbenchEvaluationStatus.SKIPPED,
            reason="The Chat answer was ineligible for the reviewed PUBLIC evaluation policy.",
        )
    if not automatic_evaluation_enabled:
        return BackendEvaluationState(
            status=WorkbenchEvaluationStatus.SKIPPED,
            reason="Automatic Chat evaluation was disabled in configured evaluation options.",
        )
    if not evaluate_answer_selected:
        return BackendEvaluationState(
            status=WorkbenchEvaluationStatus.SKIPPED,
            reason="Evaluate this answer was not selected for this Chat submission.",
        )
    if not enabled:
        return BackendEvaluationState(
            status=WorkbenchEvaluationStatus.NOT_REQUESTED,
            reason=f"{backend} was disabled for this exact evaluation configuration.",
        )
    if not has_reference:
        return BackendEvaluationState(
            status=WorkbenchEvaluationStatus.REFERENCE_REQUIRED,
            reason=(
                f"{backend} requires an approved reference for the selected metrics, but the "
                "matched case has no expected answer."
            ),
        )
    return BackendEvaluationState(
        status=WorkbenchEvaluationStatus.PENDING,
        reason=f"{backend} was requested and has not completed yet.",
    )


def _fail_pending_backend(
    backend: BackendEvaluationState,
    reason: str,
) -> BackendEvaluationState:
    if backend.status is not WorkbenchEvaluationStatus.PENDING:
        return backend
    return backend.model_copy(
        update={
            "status": WorkbenchEvaluationStatus.FAILED,
            "reason": reason,
        }
    )


def display_status(summary: JudgeStageSummary) -> WorkbenchEvaluationStatus:
    mapping = {
        "COMPLETED": WorkbenchEvaluationStatus.COMPLETED,
        "PARTIAL": WorkbenchEvaluationStatus.PARTIAL,
        "FAILED": WorkbenchEvaluationStatus.FAILED,
        "UNAVAILABLE": WorkbenchEvaluationStatus.UNAVAILABLE,
        "NOT_REQUESTED": WorkbenchEvaluationStatus.NOT_REQUESTED,
    }
    return mapping[summary.status]
