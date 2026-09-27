"""Evaluation-only adapter for MLflow GenAI datasets and LLM judges."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from numbers import Real
from typing import Any, Final

from consumer_finance_rag.config import MLflowSettings
from consumer_finance_rag.evaluation.models import (
    EvaluationCase,
    JudgeMetricValue,
    RAGCaseResult,
    SeedEvaluationSet,
)


class JudgeConfigurationError(RuntimeError):
    """Raised before any billable evaluation when no judge was explicitly selected."""


class MLflowGenAIEvaluationError(RuntimeError):
    """Raised when MLflow cannot run or persist an explicitly requested evaluation."""


@dataclass(frozen=True)
class MLflowJudgeResults:
    aggregate: dict[str, float]
    by_case: dict[str, dict[str, JudgeMetricValue]]
    run_id: str
    metric_diagnostics: dict[str, MLflowMetricDiagnostic] = field(default_factory=dict)


class MLflowMetricStatus(StrEnum):
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    MISSING = "MISSING"
    INVALID = "INVALID"


@dataclass(frozen=True)
class MLflowMetricDiagnostic:
    status: MLflowMetricStatus
    error_category: str | None = None


REGISTERED_SCORER_NAMES: Final[tuple[str, ...]] = (
    "correctness",
    "relevance_to_query",
    "safety",
    "retrieval_groundedness",
)


def require_judge_model(settings: MLflowSettings) -> str:
    """Return an explicit judge URI; never invoke MLflow's implicit OpenAI fallback."""

    if not settings.judge_model:
        raise JudgeConfigurationError(
            "LLM-as-a-judge evaluation is not configured. Set "
            "MLFLOW_GENAI_JUDGE_DEFAULT_MODEL to an explicit model URI. For Bedrock use "
            "bedrock:/<enabled-model-id>. Deterministic retrieval evaluation needs no judge."
        )
    return settings.judge_model


def evaluation_dataset_records(dataset: SeedEvaluationSet) -> list[dict[str, Any]]:
    return [
        {
            "inputs": {"question": case.question},
            "expectations": {
                "expected_facts": case.expected_facts,
                "expected_response": case.reference_answer,
            },
            "tags": {
                "case_id": case.case_id,
                "category": case.category,
                "difficulty": case.difficulty,
                "review_status": case.review_status,
                "dataset_version": dataset.version,
            },
        }
        for case in dataset.cases
    ]


def sync_evaluation_dataset(
    dataset: SeedEvaluationSet, settings: MLflowSettings
) -> tuple[str, int | None]:
    """Merge source-controlled seed cases into MLflow's SQL-backed dataset store."""

    if not settings.enabled:
        raise MLflowGenAIEvaluationError("MLflow is disabled; dataset synchronization was not run.")
    try:
        import mlflow
        from mlflow.exceptions import MlflowException

        mlflow.set_tracking_uri(settings.tracking_uri)
        experiment = mlflow.set_experiment(settings.evaluation_experiment_name)
        try:
            managed = mlflow.genai.datasets.get_dataset(name=settings.evaluation_dataset_name)
        except MlflowException as error:
            if error.error_code != "RESOURCE_DOES_NOT_EXIST":
                raise
            managed = mlflow.genai.datasets.create_dataset(
                name=settings.evaluation_dataset_name,
                experiment_id=[experiment.experiment_id],
                tags={"source": "repository_seed", "version": dataset.version},
            )
        managed.merge_records(evaluation_dataset_records(dataset))
        return managed.dataset_id, managed.version
    except Exception as error:
        raise MLflowGenAIEvaluationError(
            f"Could not sync MLflow Evaluation Dataset at {settings.tracking_uri}: {error}"
        ) from error


def _judge_rows(
    cases: Sequence[EvaluationCase], results: Sequence[RAGCaseResult]
) -> list[dict[str, Any]]:
    case_by_id = {case.case_id: case for case in cases}
    rows: list[dict[str, Any]] = []
    for result in results:
        case = case_by_id[result.case_id]
        row: dict[str, Any] = {
            "request_id": result.case_id,
            "inputs": {"question": result.question},
            "outputs": {"answer": result.answer},
            "expectations": {
                "expected_response": case.reference_answer,
            },
            "tags": {"case_id": result.case_id},
        }
        if result.trace_id:
            row["trace"] = result.trace_id
        rows.append(row)
    return rows


def _field_judge_rows(
    cases: Sequence[EvaluationCase], results: Sequence[RAGCaseResult]
) -> list[dict[str, Any]]:
    """Approved field inputs for MLflow scorers, deliberately excluding traces."""

    rows = _judge_rows(cases, results)
    for row in rows:
        row.pop("trace", None)
    return rows


def _parse_case_scores(
    frame: Any,
    scorer_names: Sequence[str] = REGISTERED_SCORER_NAMES,
) -> dict[str, dict[str, JudgeMetricValue]]:
    registered = _validated_scorer_names(scorer_names)
    parsed: dict[str, dict[str, JudgeMetricValue]] = {}
    if frame is None:
        return parsed
    for _, row in frame.iterrows():
        tags = row.get("tags") or {}
        case_id = tags.get("case_id") if isinstance(tags, Mapping) else None
        if not isinstance(case_id, str):
            case_id = str(row.get("request_id", ""))
        scores: dict[str, JudgeMetricValue] = {}
        for name in registered:
            column = f"{name}/value"
            if column not in frame.columns:
                continue
            value = row.get(column)
            if _invalid_score(value) or _row_metric_error(row, name):
                continue
            reason = row.get(f"{name}/rationale")
            scores[name] = JudgeMetricValue(
                value=None if value is None else value,
                reason=reason if isinstance(reason, str) else None,
            )
        parsed[case_id] = scores
    return parsed


def _metric_diagnostics(
    frame: Any,
    requested_metrics: Sequence[str],
) -> dict[str, MLflowMetricDiagnostic]:
    requested_metrics = _validated_scorer_names(requested_metrics)
    diagnostics: dict[str, MLflowMetricDiagnostic] = {}
    if frame is None:
        return {
            name: MLflowMetricDiagnostic(
                MLflowMetricStatus.MISSING, "MISSING_RESULT"
            )
            for name in requested_metrics
        }
    if getattr(frame, "empty", False):
        return {
            name: MLflowMetricDiagnostic(
                MLflowMetricStatus.MISSING, "MISSING_RESULT"
            )
            for name in requested_metrics
        }
    for name in requested_metrics:
        value_column = f"{name}/value"
        if value_column not in frame.columns:
            diagnostics[name] = MLflowMetricDiagnostic(
                MLflowMetricStatus.MISSING, "MISSING_RESULT"
            )
            continue
        statuses: list[MLflowMetricDiagnostic] = []
        for _, row in frame.iterrows():
            error = _row_metric_error(row, name)
            if error is not None:
                statuses.append(
                    MLflowMetricDiagnostic(
                        MLflowMetricStatus.FAILED,
                        _safe_error_category(error),
                    )
                )
            elif _invalid_score(row.get(value_column)):
                statuses.append(
                    MLflowMetricDiagnostic(
                        MLflowMetricStatus.INVALID, "INVALID_RESULT"
                    )
                )
            else:
                statuses.append(MLflowMetricDiagnostic(MLflowMetricStatus.COMPLETED))
        failures = [item for item in statuses if item.status is not MLflowMetricStatus.COMPLETED]
        diagnostics[name] = failures[0] if failures else MLflowMetricDiagnostic(
            MLflowMetricStatus.COMPLETED
        )
    return diagnostics


def _row_metric_error(row: Any, name: str) -> str | None:
    for suffix in ("error_message", "error", "error_code"):
        value = row.get(f"{name}/{suffix}")
        if isinstance(value, str) and value.strip():
            return value
    return None


def _invalid_score(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, bool):
        return False
    if isinstance(value, Real):
        return not math.isfinite(float(value))
    if isinstance(value, str):
        return not value.strip()
    return True


def _validated_scorer_names(scorer_names: Sequence[str]) -> tuple[str, ...]:
    requested = tuple(dict.fromkeys(scorer_names))
    unknown = set(requested).difference(REGISTERED_SCORER_NAMES)
    if unknown:
        raise ValueError("Only explicitly registered MLflow scorer names may be parsed")
    return requested


def _parse_aggregate_scores(
    metrics: Mapping[str, object],
    scorer_names: Sequence[str],
) -> dict[str, float]:
    parsed: dict[str, float] = {}
    for name in _validated_scorer_names(scorer_names):
        for candidate in (name, f"{name}/mean"):
            value = metrics.get(candidate)
            if (
                isinstance(value, Real)
                and not isinstance(value, bool)
                and math.isfinite(float(value))
            ):
                parsed[name] = float(value)
                break
    return parsed


def _safe_error_category(error: str) -> str:
    normalized = error.casefold()
    if "amazonbedrockprovider" in normalized and "get_endpoint_url" in normalized:
        return "PROVIDER_UNSUPPORTED"
    if "expected_response" in normalized and "expected_facts" in normalized:
        return "INVALID_EXPECTATIONS"
    if "awsidandkey" in normalized or "access key" in normalized or "credential" in normalized:
        return "AUTHENTICATION_CONFIGURATION"
    if "retriever" in normalized or "page_content" in normalized or "document text" in normalized:
        return "RETRIEVER_SCHEMA"
    if "timeout" in normalized:
        return "TIMEOUT"
    if "permission" in normalized or "accessdenied" in normalized:
        return "ACCESS_DENIED"
    return "SCORER_ERROR"


def run_mlflow_judges(
    cases: Sequence[EvaluationCase],
    results: Sequence[RAGCaseResult],
    settings: MLflowSettings,
) -> MLflowJudgeResults:
    """Run current MLflow GenAI scorers with one explicitly configured judge model."""

    judge_model = require_judge_model(settings)
    try:
        import mlflow
        import pandas as pd  # type: ignore[import-untyped]
        from mlflow.genai.scorers import (
            Correctness,
            RelevanceToQuery,
            Safety,
        )

        mlflow.set_tracking_uri(settings.tracking_uri)
        mlflow.set_experiment(settings.evaluation_experiment_name)
        rows = _field_judge_rows(cases, results)
        scorers: list[Any] = [
            Correctness(model=judge_model),
            RelevanceToQuery(model=judge_model),
            Safety(model=judge_model),  # type: ignore[no-untyped-call]
        ]
        requested_metrics = [
            "correctness",
            "relevance_to_query",
            "safety",
        ]
        evaluated = mlflow.genai.evaluate(data=pd.DataFrame(rows), scorers=scorers)
        return MLflowJudgeResults(
            aggregate=_parse_aggregate_scores(
                evaluated.metrics, requested_metrics
            ),
            by_case=_parse_case_scores(evaluated.result_df, requested_metrics),
            run_id=evaluated.run_id,
            metric_diagnostics=_metric_diagnostics(
                evaluated.result_df, requested_metrics
            ),
        )
    except JudgeConfigurationError:
        raise
    except Exception as error:
        raise MLflowGenAIEvaluationError(f"MLflow GenAI evaluation failed: {error}") from error
