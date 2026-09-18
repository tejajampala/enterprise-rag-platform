"""Evaluation-only adapter for MLflow GenAI datasets and LLM judges."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

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
                "expected_facts": case.expected_facts,
                "expected_response": case.reference_answer,
            },
            "tags": {"case_id": result.case_id},
        }
        if result.trace_id:
            row["trace"] = result.trace_id
        rows.append(row)
    return rows


def _parse_case_scores(frame: Any) -> dict[str, dict[str, JudgeMetricValue]]:
    parsed: dict[str, dict[str, JudgeMetricValue]] = {}
    if frame is None:
        return parsed
    for _, row in frame.iterrows():
        tags = row.get("tags") or {}
        case_id = tags.get("case_id") if isinstance(tags, Mapping) else None
        if not isinstance(case_id, str):
            case_id = str(row.get("request_id", ""))
        scores: dict[str, JudgeMetricValue] = {}
        for column in frame.columns:
            if not isinstance(column, str) or not column.endswith("/value"):
                continue
            name = column.removesuffix("/value")
            value = row.get(column)
            reason = row.get(f"{name}/rationale")
            scores[name] = JudgeMetricValue(
                value=None if value is None else value,
                reason=reason if isinstance(reason, str) else None,
            )
        parsed[case_id] = scores
    return parsed


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
            RetrievalGroundedness,
            Safety,
        )

        mlflow.set_tracking_uri(settings.tracking_uri)
        mlflow.set_experiment(settings.evaluation_experiment_name)
        rows = _judge_rows(cases, results)
        for row in rows:
            trace_id = row.pop("trace", None)
            if isinstance(trace_id, str):
                trace = mlflow.get_trace(trace_id, flush=True)
                if trace is not None:
                    row["trace"] = trace
        scorers: list[Any] = [
            Correctness(model=judge_model),
            RelevanceToQuery(model=judge_model),
            Safety(model=judge_model),  # type: ignore[no-untyped-call]
        ]
        if rows and all("trace" in row for row in rows):
            scorers.append(RetrievalGroundedness(model=judge_model))
        evaluated = mlflow.genai.evaluate(data=pd.DataFrame(rows), scorers=scorers)
        return MLflowJudgeResults(
            aggregate=dict(evaluated.metrics),
            by_case=_parse_case_scores(evaluated.result_df),
            run_id=evaluated.run_id,
        )
    except JudgeConfigurationError:
        raise
    except Exception as error:
        raise MLflowGenAIEvaluationError(f"MLflow GenAI evaluation failed: {error}") from error
