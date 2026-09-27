from __future__ import annotations

import ast
from decimal import Decimal
from pathlib import Path

import pyarrow as pa  # type: ignore[import-untyped]

from consumer_finance_rag.costs import (
    CostEstimate,
    estimate_model_usage_cost,
    load_pricing_catalog,
)
from consumer_finance_rag.evaluation.managed_rag import ManagedRAGEvaluationReport
from consumer_finance_rag.ui_dataframes import (
    DisplayColumnKind,
    arrow_safe_dataframe,
)
from consumer_finance_rag.usage_presentation import (
    UsageCostComponent,
    summarize_usage_cost,
    usage_component,
    usage_cost_dataframe,
)


def _arrow(frame: object) -> pa.Table:
    return pa.Table.from_pandas(frame, preserve_index=False)


def _component(
    *,
    name: str,
    input_tokens: int | None,
    output_tokens: int | None,
    reasoning_tokens: int | None = None,
    estimate: CostEstimate | None = None,
) -> UsageCostComponent:
    return usage_component(
        name=name,
        status="COMPLETED" if input_tokens is not None else "UNAVAILABLE",
        model_id="openai.gpt-oss-20b-1:0",
        region="ap-southeast-2",
        provider_invocations=1 if input_tokens is not None else None,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        reasoning_tokens=reasoning_tokens,
        estimate=estimate,
        pricing_version="2026-09-26",
    )


def test_arrow_boundary_with_all_numeric_token_counts() -> None:
    frame = arrow_safe_dataframe(
        [
            {"Input tokens": 100, "Output tokens": 20, "Provider calls": 1},
            {"Input tokens": 200, "Output tokens": 30, "Provider calls": 2},
        ],
        schema={
            "Input tokens": DisplayColumnKind.NULLABLE_INTEGER,
            "Output tokens": DisplayColumnKind.NULLABLE_INTEGER,
            "Provider calls": DisplayColumnKind.NULLABLE_INTEGER,
        },
    )

    table = _arrow(frame)

    assert table.column("Input tokens").to_pylist() == [100, 200]


def test_usage_table_serializes_mixed_counts_and_unavailable_values() -> None:
    view = summarize_usage_cost(
        [
            _component(name="generation", input_tokens=1_000, output_tokens=100),
            _component(name="judge", input_tokens=None, output_tokens=None),
        ],
        same_request_scope=True,
    )

    frame = usage_cost_dataframe(view)
    table = _arrow(frame)

    assert table.column("Input tokens").to_pylist() == ["1,000", "Unavailable"]


def test_usage_table_serializes_when_all_usage_is_unavailable() -> None:
    view = summarize_usage_cost(
        [
            _component(name="generation", input_tokens=None, output_tokens=None),
            _component(name="judge", input_tokens=None, output_tokens=None),
        ],
        same_request_scope=True,
    )

    table = _arrow(usage_cost_dataframe(view))

    assert table.column("Output tokens").to_pylist() == [
        "Unavailable",
        "Unavailable",
    ]


def test_missing_pricing_preserves_available_usage_and_never_displays_zero() -> None:
    component = _component(
        name="judge",
        input_tokens=14_320,
        output_tokens=3_788,
        estimate=CostEstimate(available=False, reason="No exact verified pricing rule"),
    )
    frame = usage_cost_dataframe(
        summarize_usage_cost([component], same_request_scope=True)
    )
    table = _arrow(frame)

    assert table.column("Input tokens").to_pylist() == ["14,320"]
    assert table.column("Cost").to_pylist() == ["Cost unavailable"]


def test_very_small_cost_and_reasoning_subset_serialize_without_double_counting() -> None:
    component = _component(
        name="judge",
        input_tokens=14_320,
        output_tokens=3_788,
        reasoning_tokens=2_996,
        estimate=CostEstimate(
            available=True,
            estimated_api_cost=Decimal("0.002202964"),
            currency="USD",
        ),
    )
    view = summarize_usage_cost([component], same_request_scope=True)
    table = _arrow(usage_cost_dataframe(view))

    assert view.output_tokens == 3_788
    assert view.reasoning_tokens == 2_996
    assert table.column("Cost").to_pylist() == [
        "Estimated USD $0.00220296 (0.2203¢)"
    ]


def test_historical_report_without_usage_is_arrow_safe_and_cost_unavailable() -> None:
    report = ManagedRAGEvaluationReport.model_validate_json(
        Path("reports/evaluation/paid-smoke.json").read_text(encoding="utf-8")
    )
    assert report.answer_quality.ragas.usage is None
    estimate = estimate_model_usage_cost(
        report.answer_quality.ragas.usage,
        load_pricing_catalog(),
        pricing_mode="on-demand-standard",
        as_of=report.generated_at.date(),
    )
    component = _component(
        name="historical RAGAS",
        input_tokens=None,
        output_tokens=None,
        estimate=estimate,
    )

    table = _arrow(
        usage_cost_dataframe(
            summarize_usage_cost([component], same_request_scope=True)
        )
    )

    assert table.column("Cost").to_pylist() == ["Cost unavailable"]


def test_mixed_metric_and_provider_call_records_keep_numeric_scores_nullable() -> None:
    metrics = arrow_safe_dataframe(
        [
            {"Metric": "faithfulness", "Status": "COMPLETED", "Score": 0.875},
            {"Metric": "context_recall", "Status": "UNAVAILABLE", "Score": None},
        ],
        schema={"Score": DisplayColumnKind.NULLABLE_FLOAT},
    )
    calls = arrow_safe_dataframe(
        [
            {"Metric": "faithfulness", "Provider call": "1", "Tokens": "120"},
            {
                "Metric": "context_recall",
                "Provider call": "Unavailable",
                "Tokens": "Unavailable",
            },
        ]
    )

    assert _arrow(metrics).column("Score").to_pylist() == [0.875, None]
    assert _arrow(calls).column("Tokens").to_pylist() == ["120", "Unavailable"]


def test_every_streamlit_dataframe_uses_an_arrow_validated_builder() -> None:
    for path in (
        Path("app/private_workbench.py"),
        Path("app/streamlit_app.py"),
        Path("src/consumer_finance_rag/usage_presentation.py"),
    ):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr != "dataframe":
                continue
            raise AssertionError(
                f"{path}:{node.lineno} bypasses render_arrow_dataframe"
            )

    boundary_path = Path("src/consumer_finance_rag/ui_dataframes.py")
    boundary = ast.parse(
        boundary_path.read_text(encoding="utf-8"), filename=str(boundary_path)
    )
    calls = [
        node
        for node in ast.walk(boundary)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "dataframe"
    ]
    assert len(calls) == 1
    argument = calls[0].args[0]
    assert isinstance(argument, ast.Call)
    assert isinstance(argument.func, ast.Name)
    assert argument.func.id == "arrow_safe_dataframe"
