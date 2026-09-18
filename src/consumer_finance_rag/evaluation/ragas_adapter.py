# mypy: disable-error-code=import-untyped
"""Removable evaluation-only adapter for RAGAS 0.4 collections and experiments."""

from __future__ import annotations

import asyncio
import importlib.util
import os
import sys
from collections.abc import Sequence
from types import ModuleType
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict

from consumer_finance_rag.evaluation.models import JudgeMetricValue, RAGCaseResult


class RagasConfigurationError(RuntimeError):
    """Raised before a paid RAGAS evaluation when no Bedrock judge is configured."""


class RagasCompatibilityError(RuntimeError):
    """Raised when the isolated RAGAS integration cannot load or execute."""


class RagasEvaluator(Protocol):
    enabled: bool

    def evaluate(
        self, results: Sequence[RAGCaseResult]
    ) -> dict[str, dict[str, JudgeMetricValue]]: ...


class DisabledRagasEvaluator:
    enabled = False

    def evaluate(self, results: Sequence[RAGCaseResult]) -> dict[str, dict[str, JudgeMetricValue]]:
        return {}


class _RagasInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    case_id: str
    question: str
    answer: str
    contexts: list[str]
    reference: str


class _RagasMetric(BaseModel):
    model_config = ConfigDict(frozen=True)

    value: float | None
    reason: str | None = None


class _RagasOutput(BaseModel):
    model_config = ConfigDict(frozen=True)

    case_id: str
    metrics: dict[str, _RagasMetric]


def _install_vertexai_import_compatibility() -> None:
    """Work around RAGAS 0.4.3 importing a module removed by LangChain Community.

    RAGAS only uses these types for legacy provider detection. This application uses
    LiteLLM/Bedrock, so unavailable sentinels are sufficient and stay evaluation-local.
    """

    module_name = "langchain_community.chat_models.vertexai"
    if module_name in sys.modules or importlib.util.find_spec(module_name) is not None:
        return

    class _UnavailableVertexAI:
        pass

    compatibility_module = ModuleType(module_name)
    compatibility_module.ChatVertexAI = _UnavailableVertexAI  # type: ignore[attr-defined]
    compatibility_module.VertexAI = _UnavailableVertexAI  # type: ignore[attr-defined]
    sys.modules[module_name] = compatibility_module


def _metric_value(result: Any) -> _RagasMetric:
    value = result.value
    numeric = float(value) if isinstance(value, int | float) else None
    reason = result.reason if isinstance(result.reason, str) else None
    return _RagasMetric(value=numeric, reason=reason)


class Ragas043Evaluator:
    """RAGAS 0.4.3 collections runner using a LiteLLM-backed Bedrock judge."""

    enabled = True

    def __init__(self, *, judge_model: str, references: dict[str, str]) -> None:
        if not judge_model:
            raise RagasConfigurationError(
                "RAGAS evaluation requires an explicit judge model. Set "
                "MLFLOW_GENAI_JUDGE_DEFAULT_MODEL=bedrock:/<enabled-model-id>."
            )
        if not judge_model.startswith("bedrock:/"):
            raise RagasConfigurationError(
                "The RAGAS adapter currently supports Bedrock judges only; use "
                "bedrock:/<enabled-model-id>."
            )
        self._model = "bedrock/" + judge_model.removeprefix("bedrock:/")
        self._references = references

    def evaluate(self, results: Sequence[RAGCaseResult]) -> dict[str, dict[str, JudgeMetricValue]]:
        try:
            return asyncio.run(self._evaluate(results))
        except RagasConfigurationError:
            raise
        except Exception as error:
            raise RagasCompatibilityError(f"RAGAS 0.4.3 evaluation failed: {error}") from error

    async def _evaluate(
        self, results: Sequence[RAGCaseResult]
    ) -> dict[str, dict[str, JudgeMetricValue]]:
        _install_vertexai_import_compatibility()
        os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "true")

        from litellm import completion
        from ragas import experiment
        from ragas.backends import InMemoryBackend
        from ragas.dataset import Dataset
        from ragas.llms import llm_factory
        from ragas.metrics.collections import (
            ContextPrecision,
            ContextRecall,
            Faithfulness,
            ResponseGroundedness,
        )

        llm = llm_factory(
            self._model,
            provider="litellm",
            client=completion,
            adapter="litellm",
        )
        metrics = {
            "faithfulness": Faithfulness(llm=llm),
            "context_precision": ContextPrecision(llm=llm),
            "context_recall": ContextRecall(llm=llm),
            "response_groundedness": ResponseGroundedness(llm=llm),
        }

        @experiment(_RagasOutput, name_prefix="consumer-finance")
        async def score(row: _RagasInput) -> _RagasOutput:
            values = {
                "faithfulness": _metric_value(
                    await metrics["faithfulness"].ascore(
                        user_input=row.question,
                        response=row.answer,
                        retrieved_contexts=row.contexts,
                    )
                ),
                "context_precision": _metric_value(
                    await metrics["context_precision"].ascore(
                        user_input=row.question,
                        reference=row.reference,
                        retrieved_contexts=row.contexts,
                    )
                ),
                "context_recall": _metric_value(
                    await metrics["context_recall"].ascore(
                        user_input=row.question,
                        reference=row.reference,
                        retrieved_contexts=row.contexts,
                    )
                ),
                "response_groundedness": _metric_value(
                    await metrics["response_groundedness"].ascore(
                        response=row.answer,
                        retrieved_contexts=row.contexts,
                    )
                ),
            }
            return _RagasOutput(case_id=row.case_id, metrics=values)

        backend = InMemoryBackend()  # type: ignore[no-untyped-call]
        data = [
            _RagasInput(
                case_id=result.case_id,
                question=result.question,
                answer=result.answer,
                contexts=result.contexts,
                reference=self._references[result.case_id],
            )
            for result in results
        ]
        dataset = Dataset(
            name="consumer-finance-rag-evaluation",
            backend=backend,
            data_model=_RagasInput,
            data=data,
        )
        completed = await score.arun(
            dataset,
            name="ragas-0.4.3",
            backend=backend,
        )
        parsed = {
            item.case_id: {
                name: JudgeMetricValue(value=value.value, reason=value.reason)
                for name, value in item.metrics.items()
            }
            for item in completed
            if isinstance(item, _RagasOutput)
        }
        if len(parsed) != len(results):
            raise RagasCompatibilityError(
                "RAGAS did not return a result for every case; inspect its experiment output."
            )
        return parsed


def build_ragas_evaluator(
    *, enabled: bool, judge_model: str | None, references: dict[str, str]
) -> RagasEvaluator:
    if not enabled:
        return DisabledRagasEvaluator()
    return Ragas043Evaluator(judge_model=judge_model or "", references=references)
