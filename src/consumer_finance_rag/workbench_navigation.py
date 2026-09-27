"""Stable private-workbench information architecture and navigation state."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

WORKBENCH_TAB_MAP: dict[str, tuple[str, ...]] = {
    "Chat": ("Conversation", "Evidence", "Quality & Cost"),
    "Request Inspector": (
        "Overview",
        "Retrieval & Reranking",
        "Context & Prompt",
        "Generation",
        "Quality Gate",
        "Trace & Cost",
    ),
    "Evaluations": (
        "Overview",
        "Run Evaluation",
        "Results & Scores",
        "Individual Judge Calls",
        "Evaluation Inputs",
        "Compare Runs",
    ),
    "Lineage & Index": (
        "Active Index",
        "Documents & Lineage",
        "Ingestion History",
        "Generations",
    ),
    "Drift & Regression": (
        "Quality Trends",
        "Baselines",
        "Regressions",
        "Alerts",
    ),
    "Operations": (
        "Health",
        "Models & Budgets",
        "Jobs & Logs",
        "Privacy & Retention",
        "Configuration",
    ),
}


class WorkbenchSelection(BaseModel):
    """Identifiers retained across page and lazy-tab reruns."""

    model_config = ConfigDict(frozen=True)

    request_id: str | None = None
    trace_id: str | None = None
    evaluation_id: str | None = None
    case_id: str | None = None
    metric_name: str | None = None
    provider_invocation: int | None = None


def merge_selection(
    current: WorkbenchSelection,
    **updates: str | int | None,
) -> WorkbenchSelection:
    """Update explicit identifiers without clearing unrelated selections."""

    allowed = set(WorkbenchSelection.model_fields)
    invalid = set(updates) - allowed
    if invalid:
        raise ValueError(f"Unsupported workbench selection fields: {sorted(invalid)}")
    return current.model_copy(update=updates)
