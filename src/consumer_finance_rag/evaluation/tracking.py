"""MLflow run logging for reproducible evaluation experiments."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Protocol

from consumer_finance_rag.config import MLflowSettings

ParameterValue = str | int | float | bool | None


class EvaluationRun(Protocol):
    @property
    def active(self) -> bool: ...

    @property
    def run_id(self) -> str | None: ...

    def log_parameters(self, parameters: Mapping[str, ParameterValue]) -> None: ...

    def log_metrics(self, metrics: Mapping[str, float]) -> None: ...

    def log_artifacts(self, paths: list[Path]) -> None: ...

    def log_tags(self, tags: Mapping[str, ParameterValue]) -> None: ...


class DisabledEvaluationRun:
    active = False
    run_id = None

    def log_parameters(self, parameters: Mapping[str, ParameterValue]) -> None:
        return None

    def log_metrics(self, metrics: Mapping[str, float]) -> None:
        return None

    def log_artifacts(self, paths: list[Path]) -> None:
        return None

    def log_tags(self, tags: Mapping[str, ParameterValue]) -> None:
        return None


class ActiveEvaluationRun:
    active = True

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id

    def log_parameters(self, parameters: Mapping[str, ParameterValue]) -> None:
        import mlflow

        logged = {
            key: "unavailable" if value is None else value for key, value in parameters.items()
        }
        mlflow.log_params(logged)

    def log_metrics(self, metrics: Mapping[str, float]) -> None:
        import mlflow

        mlflow.log_metrics(dict(metrics))

    def log_artifacts(self, paths: list[Path]) -> None:
        import mlflow

        for path in paths:
            mlflow.log_artifact(str(path), artifact_path="evaluation_reports")

    def log_tags(self, tags: Mapping[str, ParameterValue]) -> None:
        import mlflow

        logged = {key: "unavailable" if value is None else value for key, value in tags.items()}
        mlflow.set_tags(logged)


class EvaluationTrackingError(RuntimeError):
    """Raised when an explicitly enabled evaluation run cannot reach MLflow."""


@contextmanager
def evaluation_run(
    settings: MLflowSettings,
    *,
    run_name: str,
    enabled: bool = True,
    existing_run_id: str | None = None,
) -> Iterator[EvaluationRun]:
    if not settings.enabled or not enabled:
        yield DisabledEvaluationRun()
        return
    try:
        # MLflow prints an emoji-prefixed run URL at teardown, which fails on the
        # default Windows CP-1252 console even after the run has been persisted.
        os.environ.setdefault("MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT", "true")
        import mlflow

        mlflow.set_tracking_uri(settings.tracking_uri)
        mlflow.set_experiment(settings.evaluation_experiment_name)
        run_context = (
            mlflow.start_run(run_id=existing_run_id)
            if existing_run_id is not None
            else mlflow.start_run(run_name=run_name)
        )
        with run_context as active_run:
            yield ActiveEvaluationRun(active_run.info.run_id)
    except Exception as error:
        raise EvaluationTrackingError(
            f"MLflow evaluation logging failed at {settings.tracking_uri}. "
            "Start the Docker stack or pass --no-mlflow. "
            f"Details: {error}"
        ) from error


def flatten_stage_metrics(aggregate: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
    return {
        f"{stage}.{metric}": value
        for stage, metrics in aggregate.items()
        for metric, value in metrics.items()
    }


def flatten_group_metrics(
    groups: Mapping[str, Mapping[str, Mapping[str, float]]], *, prefix: str
) -> dict[str, float]:
    return {
        f"{prefix}.{group.replace(' ', '_').replace('/', '_')}.{stage}.{metric}": value
        for group, stages in groups.items()
        for stage, metrics in stages.items()
        for metric, value in metrics.items()
    }


def git_commit_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None
