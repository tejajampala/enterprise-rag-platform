from __future__ import annotations

from types import SimpleNamespace

import pytest

from consumer_finance_rag.config import MLflowSettings
from consumer_finance_rag.evaluation.mlflow_genai import (
    JudgeConfigurationError,
    require_judge_model,
)
from consumer_finance_rag.evaluation.ragas_adapter import (
    DisabledRagasEvaluator,
    build_ragas_evaluator,
)
from consumer_finance_rag.evaluation.tracking import evaluation_run
from consumer_finance_rag.observability import NoOpTracer, build_tracer


def test_mlflow_disabled_returns_noop_tracer() -> None:
    tracer = build_tracer(MLflowSettings(enabled=False))
    assert isinstance(tracer, NoOpTracer)
    assert tracer.status.active is False


def test_mlflow_disabled_evaluation_run_does_not_contact_server() -> None:
    with evaluation_run(MLflowSettings(enabled=False), run_name="offline") as run:
        assert run.active is False
        run.log_parameters({"mode": "offline"})
        run.log_metrics({"recall_at_1": 1.0})
        run.log_artifacts([])


def test_evaluation_run_can_resume_one_existing_judge_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mlflow

    calls: list[dict[str, str]] = []

    class RunContext:
        def __enter__(self) -> SimpleNamespace:
            return SimpleNamespace(info=SimpleNamespace(run_id="judge-run"))

        def __exit__(self, *_args: object) -> None:
            return None

    monkeypatch.setattr(mlflow, "set_tracking_uri", lambda _uri: None)
    monkeypatch.setattr(mlflow, "set_experiment", lambda _name: None)

    def start_run(**kwargs: str) -> RunContext:
        calls.append(kwargs)
        return RunContext()

    monkeypatch.setattr(mlflow, "start_run", start_run)

    with evaluation_run(
        MLflowSettings(enabled=True),
        run_name="managed-rag",
        existing_run_id="judge-run",
    ) as run:
        assert run.run_id == "judge-run"

    assert calls == [{"run_id": "judge-run"}]


def test_judge_configuration_never_defaults_to_openai() -> None:
    settings = MLflowSettings(judge_model=None)
    try:
        require_judge_model(settings)
    except JudgeConfigurationError as error:
        assert "bedrock:/" in str(error)
    else:
        raise AssertionError("missing judge configuration should fail before evaluation")


def test_explicit_bedrock_judge_is_preserved() -> None:
    model = "bedrock:/us.anthropic.example-v1:0"
    assert require_judge_model(MLflowSettings(judge_model=model)) == model


def test_ragas_disabled_is_offline_noop() -> None:
    evaluator = build_ragas_evaluator(enabled=False, judge_model=None, references={})
    assert isinstance(evaluator, DisabledRagasEvaluator)
    assert evaluator.evaluate([]) == {}
