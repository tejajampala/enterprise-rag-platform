from __future__ import annotations

from pathlib import Path

from consumer_finance_rag.workbench_navigation import (
    WORKBENCH_TAB_MAP,
    WorkbenchSelection,
    merge_selection,
)


def test_required_tab_map_is_stable() -> None:
    assert WORKBENCH_TAB_MAP["Chat"] == (
        "Conversation",
        "Evidence",
        "Quality & Cost",
    )
    assert len(WORKBENCH_TAB_MAP["Request Inspector"]) == 6
    assert len(WORKBENCH_TAB_MAP["Evaluations"]) == 6
    assert len(WORKBENCH_TAB_MAP["Operations"]) == 5


def test_navigation_selection_updates_one_identifier_without_losing_others() -> None:
    selected = WorkbenchSelection(
        request_id="request-1",
        trace_id="trace-1",
        evaluation_id="eval-1",
        metric_name="faithfulness",
    )

    updated = merge_selection(selected, provider_invocation=3)

    assert updated.request_id == "request-1"
    assert updated.evaluation_id == "eval-1"
    assert updated.metric_name == "faithfulness"
    assert updated.provider_invocation == 3


def test_private_tabs_are_lazy_and_public_surface_has_no_workbench_navigation() -> None:
    private = Path("app/private_workbench.py").read_text(encoding="utf-8")
    public = Path("app/streamlit_app.py").read_text(encoding="utf-8")

    assert 'on_change="rerun"' in private
    assert "if tab.open:" in private
    assert '"Get answer"' in public
    assert '"Ask another question"' in public
    assert 'st.markdown("### Answer")' in public
    assert '"### Unable to answer from available sources"' in public
    assert '"### Request could not be completed"' in public
    assert "Serving profile:" not in public
    assert "managed generation {generation}" not in public
    assert 'st.session_state.setdefault("chat_evaluate_answer", False)' in private
    assert "WORKBENCH_TAB_MAP" not in public
    assert "judge_call_debug" not in public
    assert "st.navigation" not in public
