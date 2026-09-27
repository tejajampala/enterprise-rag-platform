"""Private multipage Streamlit workbench for managed RAG engineering."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import streamlit as st

from consumer_finance_rag.config import AppSettings, MLflowSettings, load_settings
from consumer_finance_rag.costs import (
    CostEstimate,
    ModelUsageSummary,
    combine_cost_estimates,
    estimate_generation_cost,
    estimate_model_usage_cost,
    format_estimated_usd,
    load_pricing_catalog,
)
from consumer_finance_rag.evaluation.judge_call_debug import (
    ApprovedPublicDebugCase,
    JudgeCallDebugCapture,
    JudgeCallDebugConfig,
    JudgeCallDebugRecord,
    delete_judge_call_debug_artifact,
    list_judge_call_debug_artifacts,
    purge_expired_judge_call_artifacts,
)
from consumer_finance_rag.evaluation.judge_preflight import (
    JudgePreflightResult,
    preflight_bedrock_judge,
    resolve_judge_identity,
)
from consumer_finance_rag.evaluation.managed_rag import (
    CollectingRetrievalTelemetrySink,
    ManagedRAGEvaluationCase,
    ManagedRAGEvaluationDataset,
    ManagedRAGEvaluationReport,
    create_approved_public_judge_debug_capture,
    evaluate_approved_answer,
    evaluate_managed_rag,
    load_managed_rag_dataset,
    log_managed_rag_report,
    managed_evaluation_configuration_id,
    write_managed_rag_report,
)
from consumer_finance_rag.evaluation.models import RAGCaseResult
from consumer_finance_rag.evaluation.ragas_execution_trace import RagasExecutionTraceReference
from consumer_finance_rag.evaluation.tracking import evaluation_run
from consumer_finance_rag.evaluation.workbench import (
    RAGAS_METRIC_INPUT_FIELDS,
    BackendEvaluationState,
    ChatEvaluationState,
    EvaluationExecutionRegistry,
    EvaluationInputSnapshot,
    EvaluationInputSnapshotStore,
    WorkbenchEvaluationOptions,
    WorkbenchEvaluationStatus,
    capture_evaluation_input_snapshot,
    complete_chat_evaluation,
    display_status,
    evaluation_key,
    evaluation_plan,
    fail_chat_evaluation,
    find_eligible_case,
    proposed_chat_evaluation,
)
from consumer_finance_rag.factory import (
    ApplicationRuntime,
    build_application_runtime,
    build_runtime_retrieval,
    warm_up_retrieval_service,
)
from consumer_finance_rag.mlflow_inspection import (
    TraceInspection,
    TraceVisibility,
    inspect_exact_trace,
)
from consumer_finance_rag.observability import TraceContentPolicy, bedrock_token_usage
from consumer_finance_rag.operations_dashboard import (
    inspect_mlflow,
    load_drift_report_json,
)
from consumer_finance_rag.presentation import cited_only
from consumer_finance_rag.quality_policy import (
    QualityCheckResult,
    QualityDecision,
    RuntimeJudgeExecution,
)
from consumer_finance_rag.request_inspection import (
    RequestInspectionStore,
    RequestRecord,
    RequestState,
    WorkbenchAnswer,
    WorkbenchRAGService,
    WorkbenchRequestError,
    aggregate_token_usage,
)
from consumer_finance_rag.ui_dataframes import (
    DisplayColumnKind,
    format_optional_count,
    format_optional_float,
    render_arrow_dataframe,
)
from consumer_finance_rag.usage_presentation import (
    format_usd,
    render_usage_cost,
    summarize_usage_cost,
    usage_component,
)
from consumer_finance_rag.workbench_navigation import WORKBENCH_TAB_MAP

st.set_page_config(
    page_title="Private RAG Engineering Workbench",
    page_icon="🔬",
    layout="wide",
)


@st.cache_resource
def get_private_runtime() -> tuple[
    AppSettings,
    WorkbenchRAGService,
    ApplicationRuntime,
    RequestInspectionStore,
]:
    """Construct the managed runtime once; public showcase flags are irrelevant here."""

    settings = load_settings()
    records = RequestInspectionStore()
    retrieval = build_runtime_retrieval(settings, telemetry_sink=records)
    application = build_application_runtime(
        settings,
        retrieval_runtime=retrieval,
        telemetry_sink=records,
    )
    return settings, WorkbenchRAGService(settings, application, records), application, records


def _runtime() -> tuple[
    AppSettings | None,
    WorkbenchRAGService | None,
    ApplicationRuntime | None,
    RequestInspectionStore | None,
]:
    try:
        return get_private_runtime()
    except Exception as error:
        st.error(f"Startup failed ({type(error).__name__}). Check Operations diagnostics.")
        return None, None, None, None


def _evaluation_registry() -> EvaluationExecutionRegistry:
    registry = st.session_state.get("evaluation_execution_registry")
    if not isinstance(registry, EvaluationExecutionRegistry):
        registry = EvaluationExecutionRegistry()
        st.session_state.evaluation_execution_registry = registry
    return registry


def _evaluation_input_store() -> EvaluationInputSnapshotStore:
    store = st.session_state.get("evaluation_input_snapshot_store")
    if not isinstance(store, EvaluationInputSnapshotStore):
        store = EvaluationInputSnapshotStore(maximum_snapshots=8)
        st.session_state.evaluation_input_snapshot_store = store
    return store


def _approved_dataset(settings: AppSettings) -> ManagedRAGEvaluationDataset | None:
    try:
        return load_managed_rag_dataset(settings.evaluation.managed_dataset_path)
    except Exception as error:
        st.error(f"Approved evaluation dataset unavailable ({type(error).__name__}).")
        return None


def _initialize_evaluation_state(settings: AppSettings) -> None:
    defaults = {
        "workbench_auto_evaluation": (settings.private_workbench.automatic_evaluation_enabled),
        "workbench_mlflow_judges": settings.private_workbench.default_mlflow_judges,
        "workbench_ragas": settings.private_workbench.default_ragas,
        "workbench_save_mlflow": settings.private_workbench.default_save_to_mlflow,
        "workbench_capture_judge_io": False,
        "workbench_judge_io_retention_hours": (
            settings.private_workbench.judge_call_debug_retention_hours
        ),
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
    profiles = _judge_profile_choices(settings)
    configured = next(
        (
            name
            for name, identity in profiles.items()
            if identity == (settings.mlflow.judge_model, settings.mlflow.judge_region)
        ),
        next(iter(profiles)),
    )
    st.session_state.setdefault("workbench_judge_profile", configured)


def _evaluation_options() -> WorkbenchEvaluationOptions:
    return WorkbenchEvaluationOptions(
        mlflow_genai=bool(st.session_state.workbench_mlflow_judges),
        ragas=bool(st.session_state.workbench_ragas),
        save_to_mlflow=bool(st.session_state.workbench_save_mlflow),
        capture_judge_io=bool(st.session_state.workbench_capture_judge_io),
        judge_io_retention_hours=int(st.session_state.workbench_judge_io_retention_hours),
    )


def _judge_profile_choices(
    settings: AppSettings,
) -> dict[str, tuple[str | None, str | None]]:
    profiles: dict[str, tuple[str | None, str | None]] = {
        item.name: (item.configuration_uri, item.region)
        for item in settings.private_workbench.judge_profiles
    }
    configured = (settings.mlflow.judge_model, settings.mlflow.judge_region)
    if configured not in profiles.values():
        profiles = {"Configured judge": configured, **profiles}
    return profiles


def _judge_identity(settings: AppSettings) -> tuple[str | None, str | None]:
    profiles = _judge_profile_choices(settings)
    selected = st.session_state.get("workbench_judge_profile")
    return profiles.get(
        selected if isinstance(selected, str) else "",
        (settings.mlflow.judge_model, settings.mlflow.judge_region),
    )


def _judge_model_id(settings: AppSettings) -> str | None:
    return _judge_identity(settings)[0]


def _judge_region(settings: AppSettings) -> str | None:
    return _judge_identity(settings)[1]


def _effective_mlflow_settings(settings: AppSettings) -> MLflowSettings:
    return settings.mlflow.model_copy(
        update={
            "judge_model": _judge_model_id(settings),
            "judge_region": _judge_region(settings),
        }
    )


def _judge_bedrock_model_id(settings: AppSettings) -> str | None:
    judge = _judge_model_id(settings)
    return judge.removeprefix("bedrock:/") if judge and judge.startswith("bedrock:/") else judge


@st.cache_data(ttl=300)
def _cached_judge_preflight(
    judge_model: str | None,
    judge_region: str | None,
) -> JudgePreflightResult:
    from consumer_finance_rag.config import MLflowSettings

    return preflight_bedrock_judge(
        MLflowSettings(judge_model=judge_model, judge_region=judge_region)
    )


def _judge_preflight(settings: AppSettings) -> JudgePreflightResult:
    return _cached_judge_preflight(
        _judge_model_id(settings),
        _judge_region(settings),
    )


def _header(title: str, caption: str) -> None:
    st.title(title)
    st.caption(caption)


def _lazy_tabs(
    page: str,
    renderers: tuple[Callable[[], None], ...],
) -> None:
    """Render only the selected Streamlit 1.63 tab on each rerun."""

    labels = WORKBENCH_TAB_MAP[page]
    if len(labels) != len(renderers):
        raise ValueError(f"Tab renderer mismatch for {page}")
    tabs = st.tabs(labels, key=f"workbench-tabs:{page}", on_change="rerun")
    for tab, render in zip(tabs, renderers, strict=True):
        if tab.open:
            with tab:
                render()


def chat_page() -> None:
    _header("Chat", "Managed PUBLIC-only conversation with REDACT handling.")
    _lazy_tabs(
        "Chat",
        (_chat_conversation_tab, _chat_evidence_tab, _chat_quality_cost_tab),
    )


def _chat_conversation_tab() -> None:
    settings, service, runtime, _records = _runtime()
    if settings is None or service is None or runtime is None:
        return
    _initialize_evaluation_state(settings)
    dataset = _approved_dataset(settings)
    generation = runtime.retrieval.generation
    selected_policy = settings.quality_policy.for_profile(settings.quality_policy.default_profile)
    st.caption(
        f"Runtime policy: {settings.quality_policy.default_profile.value} | "
        f"{selected_policy.version} | groundedness threshold "
        f"{selected_policy.groundedness_threshold:.2f} when STRICT"
    )
    st.success(f"Ready · ACTIVE g{generation:06d} · {runtime.retrieval.qdrant_collection}")
    question = st.text_area(
        "Question",
        placeholder="Ask a consumer-finance question",
        max_chars=settings.public_showcase.max_question_characters,
    )
    eligible_case = find_eligible_case(question, dataset) if dataset is not None else None
    evaluation_enabled = bool(st.session_state.workbench_auto_evaluation)
    st.session_state.setdefault("chat_evaluate_answer", False)
    evaluate_answer = st.checkbox(
        "Evaluate this answer when it exactly matches a reviewed PUBLIC case",
        key="chat_evaluate_answer",
        disabled=(
            not evaluation_enabled
            or not (st.session_state.workbench_mlflow_judges or st.session_state.workbench_ragas)
        ),
        help=(
            "Selected only for a new explicit submission. Reruns, navigation, trace refreshes, "
            "and displaying an old answer never start evaluation."
        ),
    )
    if question.strip() and evaluate_answer and eligible_case is None:
        st.caption(
            "Evaluation ineligible: the question must exactly match one reviewed PUBLIC "
            "evaluation case. The answer request itself can still run."
        )
    elif evaluate_answer and eligible_case is not None:
        plan = evaluation_plan(
            has_reference=bool(eligible_case.reference_answer),
            has_content_trace=False,
        )
        selected_calls = (
            len(plan.mlflow_genai.selected) if st.session_state.workbench_mlflow_judges else 0
        ) + (len(plan.ragas.selected) if st.session_state.workbench_ragas else 0)
        st.info(
            f"Eligible reviewed case `{eligible_case.case_id}` · generation model "
            f"`{settings.models.generation.model or 'unavailable'}` · judge model "
            f"`{_judge_model_id(settings) or 'unavailable'}` · 1 case · "
            f"{selected_calls} logical judge metrics. Judge provider calls may be higher."
        )
        st.caption(
            f"Generation output limit: {settings.models.generation.max_tokens} tokens. "
            f"RAGAS judge output limit: {settings.mlflow.ragas_judge_max_tokens} tokens; "
            "context precision: "
            f"{settings.mlflow.ragas_context_precision_max_tokens} tokens. "
            "Actual provider token/cost usage is reported only when returned."
        )
    if st.button("Generate grounded answer", type="primary", width="stretch"):
        st.session_state.pop("workbench_answer", None)
        st.session_state.pop("workbench_failure", None)
        st.session_state.pop("chat_evaluation_key", None)
        st.session_state.pop("chat_evaluation_state", None)
        st.session_state.pop("evaluation_input_snapshot_error", None)
        try:
            if eligible_case is not None and evaluate_answer:
                st.session_state.judge_preflight = _judge_preflight(settings)
            with st.spinner("Running managed retrieval, reranking, and Bedrock generation…"):
                answer = service.answer(question)
                st.session_state.workbench_answer = answer
            assert generation is not None
            evaluation_state = _proposed_chat_state(
                settings,
                answer,
                eligible_case,
                generation,
                automatic_evaluation_enabled=evaluation_enabled,
                evaluate_answer_selected=evaluate_answer,
            )
            st.session_state.chat_evaluation_state = evaluation_state
            if (
                eligible_case is not None
                and dataset is not None
                and evaluation_state.evaluation_key is not None
            ):
                try:
                    snapshot = capture_evaluation_input_snapshot(
                        response=answer.response,
                        request_id=answer.record.request_id,
                        case=eligible_case,
                        dataset=dataset,
                        configuration_id=evaluation_state.evaluation_key,
                        generation_region=settings.aws.region,
                        judge_model=_judge_model_id(settings),
                        judge_region=_judge_region(settings),
                    )
                    _evaluation_input_store().record(snapshot)
                    st.session_state.evaluation_input_inspector_request_id = (
                        answer.response.request_id
                    )
                except ValueError:
                    st.session_state.evaluation_input_snapshot_error = "Input snapshot unavailable"
            if eligible_case is not None and _has_pending_backend(evaluation_state):
                try:
                    st.session_state.chat_evaluation_state = _evaluate_chat_answer(
                        settings,
                        answer,
                        eligible_case,
                        generation,
                        evaluation_state,
                    )
                except Exception as evaluation_error:
                    failed = fail_chat_evaluation(
                        evaluation_state,
                        reason=(
                            "Evaluation orchestration failed before all requested backends "
                            f"completed ({type(evaluation_error).__name__})."
                        ),
                    )
                    if failed.evaluation_key is not None:
                        _evaluation_registry().record(failed.evaluation_key, failed)
                    st.session_state.chat_evaluation_state = failed
                    st.warning(
                        "Answer generation completed, but approved evaluation did not "
                        f"complete ({type(evaluation_error).__name__})."
                    )
        except (ValueError, WorkbenchRequestError) as error:
            if isinstance(error, WorkbenchRequestError):
                st.session_state.workbench_failure = {
                    "request_id": error.request_id,
                    "stage": error.failure_stage,
                    "controlled_abstention": error.controlled_abstention,
                }
            if not (isinstance(error, WorkbenchRequestError) and error.controlled_abstention):
                st.error(str(error))

    failure = st.session_state.get("workbench_failure")
    if isinstance(failure, dict):
        if failure.get("controlled_abstention"):
            st.warning(
                "Unable to answer from available sources. The evidence relevance gate "
                "abstained before generation, so no candidate answer or judge call was made."
            )
        else:
            st.error(
                f"Request {failure['request_id']} failed during {failure['stage']}. "
                "No previous answer is being displayed."
            )
        if st.button("Inspect rejected request", width="stretch"):
            st.session_state.inspection_identifier = failure["request_id"]
            st.switch_page(pages[1])
    current = st.session_state.get("workbench_answer")
    if not isinstance(current, WorkbenchAnswer):
        return
    response, record = current.response, current.record
    st.subheader("Grounded answer")
    st.markdown(response.answer)
    st.caption(f"Request ID: `{record.request_id}`")
    if record.trace_id is None:
        st.caption("Trace: unavailable")
    else:
        st.info(f"MLflow trace ID: `{record.trace_id}` · Trace syncing may take a moment.")
    if st.button("Inspect this request", width="stretch"):
        st.session_state.inspection_identifier = record.request_id
        st.session_state.inspection_trace_id = record.trace_id
        st.switch_page(pages[1])
    st.caption("Open Evidence or Quality & Cost for request details; navigation is read-only.")


def _chat_evidence_tab() -> None:
    current = st.session_state.get("workbench_answer")
    if not isinstance(current, WorkbenchAnswer):
        st.info("Generate an answer in Conversation to inspect its evidence.")
        return
    response = current.response
    st.subheader("Cited evidence")
    citations = cited_only(response.answer, response.citations)
    if not citations:
        st.warning("No citations used in the answer are available.")
        return
    for citation in citations:
        with st.container(border=True):
            st.markdown(f"**[{citation.citation_id}] {citation.document_title}**")
            if citation.source_url:
                st.link_button("Open public source", citation.source_url)
            st.caption(f"Chunk `{citation.chunk_id}`")
    st.caption(
        "Citation/source validity is deterministic evidence linkage; it does not by itself "
        "establish semantic faithfulness."
    )


def _chat_quality_cost_tab() -> None:
    settings, _service, runtime, _records = _runtime()
    current = st.session_state.get("workbench_answer")
    if settings is None or runtime is None or not isinstance(current, WorkbenchAnswer):
        st.info("Generate an answer in Conversation to inspect quality and cost.")
        return
    record = current.record
    left, middle, right = st.columns(3)
    left.metric("Total latency", _duration(record.total_latency_ms))
    middle.metric("Measured bottleneck", record.bottleneck or "not recorded")
    right.metric("Generation", f"g{record.generation:06d}")
    decision = current.quality_decision
    st.subheader("Pre-response quality gate")
    status_col, outcome_col, profile_col = st.columns(3)
    status_col.metric("Gate decision", decision.status.value)
    outcome_col.metric("Delivery outcome", decision.delivery_outcome.value)
    profile_col.metric("Policy", decision.profile.value)
    st.caption(f"{decision.policy_version} · {decision.reason}")
    render_arrow_dataframe(
        st,
        [_quality_check_row(item) for item in decision.checks],
    )
    _render_runtime_request_cost(settings, record, decision)
    generation = runtime.retrieval.generation
    assert generation is not None
    _render_chat_evaluation(
        settings,
        current,
        _approved_dataset(settings),
        generation,
    )


def _proposed_chat_state(
    settings: AppSettings,
    answer: WorkbenchAnswer,
    case: ManagedRAGEvaluationCase | None,
    generation: int,
    *,
    automatic_evaluation_enabled: bool,
    evaluate_answer_selected: bool,
) -> ChatEvaluationState:
    options = _evaluation_options()
    key = (
        evaluation_key(
            scope_id=answer.record.request_id,
            generation=generation,
            case_ids=[case.case_id],
            generation_model=answer.response.model_metadata.generation_model,
            judge_model=_judge_model_id(settings),
            judge_region=_judge_region(settings),
            options=options,
        )
        if case is not None
        else None
    )
    return proposed_chat_evaluation(
        request_id=answer.record.request_id,
        case_id=case.case_id if case is not None else None,
        evaluation_key_value=key,
        options=options,
        automatic_evaluation_enabled=automatic_evaluation_enabled,
        evaluate_answer_selected=evaluate_answer_selected,
        has_reference=bool(case and case.reference_answer),
        judge_model=_judge_model_id(settings),
        judge_region=_judge_region(settings),
    )


def _has_pending_backend(state: ChatEvaluationState) -> bool:
    return any(
        backend.status is WorkbenchEvaluationStatus.PENDING
        for backend in (state.mlflow_genai, state.ragas)
    )


def _evaluate_chat_answer(
    settings: AppSettings,
    answer: WorkbenchAnswer,
    case: ManagedRAGEvaluationCase,
    generation: int,
    state: ChatEvaluationState,
    *,
    explicit_re_evaluate: bool = False,
) -> ChatEvaluationState:
    options = _evaluation_options()
    del generation
    key = state.evaluation_key
    if key is None:
        raise ValueError("Eligible Chat evaluation is missing its configuration key")
    registry = _evaluation_registry()
    if not registry.should_execute(
        key,
        explicit_re_evaluate=explicit_re_evaluate,
    ):
        existing = registry.get(key)
        if isinstance(existing, ChatEvaluationState):
            return existing
        raise RuntimeError("Evaluation registry contains an incompatible result")
    result = RAGCaseResult(
        case_id=case.case_id,
        question=case.question,
        answer=answer.response.answer,
        contexts=[item.chunk.text for item in answer.response.retrieved_chunks],
        chunk_ids=[item.chunk.chunk_id for item in answer.response.retrieved_chunks],
        token_usage=bedrock_token_usage(answer.response.generation_metadata),
        request_id=answer.record.request_id,
        # Normal Chat tracing is metadata-only; do not claim semantic trace groundedness.
        trace_id=None,
    )
    debug_capture: JudgeCallDebugCapture | None = None
    if options.capture_judge_io and options.ragas:
        judge_model = _judge_model_id(settings)
        judge_region = _judge_region(settings)
        if not settings.mlflow.enabled or not judge_model or not judge_region:
            raise RuntimeError(
                "Private judge-call capture requires active MLflow and a configured judge"
            )
        debug_capture = JudgeCallDebugCapture(
            JudgeCallDebugConfig(
                artifact_directory=settings.private_workbench.judge_call_debug_directory,
                retention_hours=options.judge_io_retention_hours,
            ),
            cases=(
                ApprovedPublicDebugCase(
                    case_id=case.case_id,
                    question=case.question,
                    review_status=case.review_status,
                    visibility=case.authorization.visibility,
                    groups=case.authorization.groups,
                ),
            ),
            evaluation_configuration_id=key,
            evaluation_run_id=None,
            model_id=judge_model.removeprefix("bedrock:/"),
            region=judge_region,
        )
    evaluated = evaluate_approved_answer(
        case,
        result,
        generation_model=answer.response.model_metadata.generation_model,
        judge_model=_judge_model_id(settings),
        with_mlflow_judges=options.mlflow_genai,
        with_ragas=options.ragas,
        mlflow_settings=_effective_mlflow_settings(settings),
        judge_preflight=(
            st.session_state.get("judge_preflight")
            if isinstance(st.session_state.get("judge_preflight"), JudgePreflightResult)
            else _judge_preflight(settings)
        ),
        evaluation_configuration_id=key,
        ragas_debug_capture=debug_capture,
    )
    completed = complete_chat_evaluation(state, evaluated)
    _evaluation_input_store().attach_evaluation_run(
        answer.record.request_id,
        case_id=case.case_id,
        configuration_id=key,
        evaluation_run_id=(
            completed.ragas.execution_trace.evaluation_run_id
            if completed.ragas.execution_trace is not None
            else completed.mlflow_genai.run_id
        ),
        ragas_execution_trace=completed.ragas.execution_trace,
    )
    registry.record(key, completed)
    st.session_state.chat_evaluation_key = key
    return completed


def _render_chat_evaluation(
    settings: AppSettings,
    answer: WorkbenchAnswer,
    dataset: ManagedRAGEvaluationDataset | None,
    generation: int,
) -> None:
    state = st.session_state.get("chat_evaluation_state")
    if not isinstance(state, ChatEvaluationState) or state.request_id != answer.record.request_id:
        st.warning(
            "Current request evaluation status is unavailable. No saved batch report is being "
            "substituted for this Chat request."
        )
        return
    st.subheader("Current request evaluation")
    _render_chat_evaluation_state(
        settings,
        state,
        generation_usage=answer.record.token_usage,
        generation_model=answer.record.model_id,
        generation_region=answer.record.region,
    )
    matching_case = (
        next(
            (case for case in dataset.cases if case.case_id == state.case_id),
            None,
        )
        if dataset is not None
        else None
    )
    snapshot = (
        _evaluation_input_store().get_exact(
            state.request_id,
            case_id=state.case_id,
            configuration_id=state.evaluation_key,
        )
        if state.case_id and state.evaluation_key
        else None
    )
    inspect_column, evaluate_column, trace_column = st.columns(3)
    if inspect_column.button(
        "Inspect evaluation inputs",
        disabled=snapshot is None,
        help=(
            "Opens the bounded process-local snapshot for this exact reviewed PUBLIC request. "
            "It never runs retrieval, generation, or judges."
        ),
    ):
        st.session_state.evaluation_input_inspector_request_id = state.request_id
        st.switch_page(pages[2])
    if snapshot is None and state.eligible:
        inspect_column.caption("Input snapshot unavailable")
    with trace_column:
        _evaluation_trace_action(
            state.ragas.execution_trace,
            request_id=state.request_id,
            key_suffix=f"chat:{state.request_id}",
        )
    if (
        evaluate_column.button(
            "Re-evaluate this answer",
            disabled=matching_case is None,
            help=(
                "Runs the selected judge backends again for this already approved PUBLIC case. "
                "It does not regenerate the answer."
            ),
        )
        and matching_case is not None
    ):
        pending = _proposed_chat_state(
            settings,
            answer,
            matching_case,
            generation,
            automatic_evaluation_enabled=True,
            evaluate_answer_selected=True,
        )
        st.session_state.chat_evaluation_state = pending
        try:
            with st.spinner("Re-running the explicitly requested judge evaluation…"):
                state = _evaluate_chat_answer(
                    settings,
                    answer,
                    matching_case,
                    generation,
                    pending,
                    explicit_re_evaluate=True,
                )
            st.session_state.chat_evaluation_state = state
        except Exception as error:
            failed = fail_chat_evaluation(
                pending,
                reason=f"Explicit re-evaluation failed ({type(error).__name__}).",
            )
            if failed.evaluation_key is not None:
                _evaluation_registry().record(failed.evaluation_key, failed)
            st.session_state.chat_evaluation_state = failed
            st.error(
                f"Evaluation status: FAILED ({type(error).__name__}). The answer remains available."
            )


def _render_chat_evaluation_state(
    settings: AppSettings,
    state: ChatEvaluationState,
    *,
    generation_usage: dict[str, int] | None = None,
    generation_model: str | None = None,
    generation_region: str | None = None,
) -> None:
    st.write(
        {
            "request_id": state.request_id,
            "case_id": state.case_id or "not matched",
            "eligible": state.eligible,
            "eligibility_reason": state.eligibility_reason,
            "evaluation_configuration_id": state.evaluation_key or "not created",
            "started_at": _timestamp(state.started_at),
            "completed_at": _timestamp(state.completed_at),
            "generation_model": settings.models.generation.model or "unavailable",
            "generation_region": settings.aws.region,
            "judge_model": state.judge_model or "unavailable",
            "judge_region": state.judge_region or "unavailable",
            "judge_bedrock_model_id": _judge_bedrock_model_id(settings) or "unavailable",
        }
    )
    preflight = st.session_state.get("judge_preflight")
    if isinstance(preflight, JudgePreflightResult):
        st.write(
            {
                "judge_metadata_preflight": preflight.metadata_status.value,
                "judge_metadata_reason": preflight.metadata_reason,
                "mlflow_provider_preflight": preflight.mlflow_genai.status.value,
                "mlflow_provider_reason": preflight.mlflow_genai.reason,
                "ragas_provider_preflight": preflight.ragas.status.value,
                "ragas_provider_reason": preflight.ragas.reason,
                "credentials_resolved": preflight.credentials_resolved,
                "credential_method": preflight.credential_method or "unresolved",
                "boto3_session_region": preflight.session_region or "unresolved",
                "invoke_permission_verified": preflight.invocation_permission_verified,
            }
        )
    rows: list[dict[str, object]] = []
    for backend_name, backend in (
        ("MLflow GenAI", state.mlflow_genai),
        ("RAGAS", state.ragas),
    ):
        rows.append(
            {
                "Backend": backend_name,
                "Status": backend.status.value,
                "Reason": backend.reason,
                "Evaluation run ID": backend.run_id or "not recorded",
                "Execution trace ID": (
                    backend.execution_trace.trace_id
                    if backend.execution_trace is not None
                    and backend.execution_trace.trace_id is not None
                    else "not captured"
                ),
                "Execution trace status": (
                    backend.execution_trace.status
                    if backend.execution_trace is not None
                    else "UNAVAILABLE"
                ),
                "Execution trace reason": (
                    backend.execution_trace.reason
                    if backend.execution_trace is not None
                    else "Evaluation execution trace not captured."
                ),
                "Metrics": ", ".join(backend.metrics) or "none",
                "Judge tokens": "unavailable",
                "Estimated cost": "Cost unavailable",
            }
        )
    render_arrow_dataframe(st, rows)
    metric_rows = _backend_metric_rows(state.mlflow_genai, "MLflow GenAI")
    metric_rows.extend(_backend_metric_rows(state.ragas, "RAGAS"))
    if metric_rows:
        render_arrow_dataframe(st, metric_rows)
    _render_cost_observability(
        generation_usage=generation_usage,
        generation_model=generation_model or settings.models.generation.model or "unavailable",
        generation_region=generation_region or settings.aws.region,
        backends=(
            ("MLflow GenAI judge", state.mlflow_genai),
            ("RAGAS judge", state.ragas),
        ),
        judge_model=(state.judge_model or "unavailable").removeprefix("bedrock:/"),
        judge_region=state.judge_region or "unavailable",
    )
    if state.mlflow_genai.run_id:
        run_url = _private_mlflow_run_url(
            settings.mlflow.tracking_uri,
            state.mlflow_genai.run_id,
        )
        if run_url:
            st.link_button("Open matching private MLflow evaluation run", run_url)
    st.caption(
        "This status and cost summary belong only to the request ID above. MLflow tracking "
        "and saving reports are not counted as paid judge calls."
    )


def _render_cost_observability(
    *,
    generation_usage: dict[str, int] | None,
    generation_model: str,
    generation_region: str,
    backends: tuple[tuple[str, BackendEvaluationState], ...],
    judge_model: str,
    judge_region: str,
) -> None:
    catalog = load_pricing_catalog()
    generation_estimate = estimate_generation_cost(
        generation_usage,
        catalog,
        provider="aws-bedrock",
        model_id=generation_model,
        region=generation_region,
        pricing_mode="on-demand-standard",
        as_of=date.today(),
    )
    components = [
        usage_component(
            name="Gemma answer generation",
            status="COMPLETED" if generation_usage is not None else "USAGE_UNAVAILABLE",
            model_id=generation_model,
            region=generation_region,
            provider_invocations=1,
            input_tokens=(generation_usage or {}).get("input_tokens"),
            output_tokens=(generation_usage or {}).get("output_tokens"),
            reasoning_tokens=None,
            estimate=generation_estimate,
            pricing_version=catalog.version,
        )
    ]
    non_billable = {
        WorkbenchEvaluationStatus.NOT_REQUESTED,
        WorkbenchEvaluationStatus.SKIPPED,
        WorkbenchEvaluationStatus.REFERENCE_REQUIRED,
    }
    for component, backend in backends:
        if backend.status in non_billable:
            components.append(
                usage_component(
                    name=component,
                    status=backend.status.value,
                    model_id=judge_model,
                    region=judge_region,
                    provider_invocations=0,
                    input_tokens=None,
                    output_tokens=None,
                    reasoning_tokens=None,
                    estimate=None,
                    pricing_version=catalog.version,
                    billable=False,
                    reason=backend.reason,
                )
            )
            continue
        estimate = estimate_model_usage_cost(
            backend.usage,
            catalog,
            pricing_mode="on-demand-standard",
            as_of=date.today(),
        )
        usage = backend.usage
        components.append(
            usage_component(
                name=component,
                status=backend.status.value,
                model_id=usage.model_id if usage is not None else judge_model,
                region=usage.region if usage is not None else judge_region,
                provider_invocations=(usage.provider_call_count if usage is not None else None),
                input_tokens=usage.input_tokens if usage is not None else None,
                output_tokens=usage.output_tokens if usage is not None else None,
                reasoning_tokens=(usage.reasoning_tokens if usage is not None else None),
                estimate=estimate,
                pricing_version=catalog.version,
            )
        )
    st.markdown("**Per-request cost and usage**")
    render_usage_cost(
        summarize_usage_cost(components, same_request_scope=True),
        streamlit=st,
    )


def _render_runtime_request_cost(
    settings: AppSettings,
    record: RequestRecord,
    decision: QualityDecision,
) -> None:
    """Show generation and the blocking judge as separate request components."""

    catalog = load_pricing_catalog()
    generation_estimate = estimate_generation_cost(
        record.token_usage,
        catalog,
        provider="aws-bedrock",
        model_id=record.model_id,
        region=record.region,
        pricing_mode="on-demand-standard",
        as_of=date.today(),
    )
    components = [
        usage_component(
            name="Gemma answer generation",
            status="COMPLETED" if record.token_usage is not None else "USAGE_UNAVAILABLE",
            model_id=record.model_id,
            region=record.region,
            provider_invocations=1,
            input_tokens=(record.token_usage or {}).get("input_tokens"),
            output_tokens=(record.token_usage or {}).get("output_tokens"),
            reasoning_tokens=None,
            estimate=generation_estimate,
            pricing_version=catalog.version,
        )
    ]
    execution = decision.judge_execution
    if decision.profile.value == "STRICT":
        check = next(
            (item for item in decision.checks if item.check == "reference_free_groundedness"),
            None,
        )
        usage = execution.usage if execution is not None else None
        estimate = _runtime_judge_estimate(execution)
        components.append(
            usage_component(
                name="Blocking GPT-OSS groundedness judge",
                status=check.status.value if check is not None else "UNAVAILABLE",
                model_id=(
                    execution.model_id
                    if execution is not None
                    else settings.quality_policy.strict.judge_model_id or "unavailable"
                ),
                region=(
                    execution.region
                    if execution is not None
                    else settings.quality_policy.strict.judge_region or "unavailable"
                ),
                provider_invocations=(usage.provider_call_count if usage is not None else 0),
                input_tokens=usage.input_tokens if usage is not None else None,
                output_tokens=usage.output_tokens if usage is not None else None,
                reasoning_tokens=usage.reasoning_tokens if usage is not None else None,
                estimate=estimate,
                pricing_version=(execution.pricing_version if execution else catalog.version),
                billable=bool(usage and usage.provider_call_count),
                reason=check.reason_category if check is not None else "JUDGE_UNAVAILABLE",
            )
        )
    st.markdown("**Generation and blocking-gate cost**")
    render_usage_cost(
        summarize_usage_cost(components, same_request_scope=True),
        streamlit=st,
    )


def _runtime_judge_estimate(
    execution: RuntimeJudgeExecution | None,
) -> CostEstimate | None:
    if execution is None or execution.usage is None:
        return None
    if execution.estimated_cost_usd is None:
        return CostEstimate(available=False, reason="Runtime judge cost unavailable")
    return CostEstimate(
        available=True,
        estimated_api_cost=execution.estimated_cost_usd,
        currency="USD",
    )


def _usage_breakdown_rows(usage: ModelUsageSummary | None) -> list[dict[str, object]]:
    if usage is None:
        return []
    rows: list[dict[str, object]] = []
    for item in usage.invocations:
        single = ModelUsageSummary(
            model_id=item.model_id,
            region=item.region,
            provider_call_count=1,
            successful_call_count=int(item.status == "COMPLETED"),
            failed_attempt_count=int(item.status == "FAILED"),
            observable_retry_count=item.retry_count,
            usage_complete=(
                item.input_tokens is not None
                and item.output_tokens is not None
                and item.total_tokens is not None
            ),
            input_tokens=item.input_tokens,
            output_tokens=item.output_tokens,
            total_tokens=item.total_tokens,
            reason="Provider usage missing for this invocation.",
            invocations=(item,),
        )
        estimate = estimate_model_usage_cost(
            single,
            load_pricing_catalog(),
            pricing_mode="on-demand-standard",
            as_of=date.today(),
        )
        rows.append(
            {
                "Case ID": item.case_id or "unattributed",
                "Metric": item.metric_name,
                "Status": item.status,
                "Model": item.model_id,
                "AWS Region": item.region,
                "Input tokens": format_optional_count(item.input_tokens),
                "Output tokens": format_optional_count(item.output_tokens),
                "Reasoning tokens": format_optional_count(item.reasoning_tokens),
                "Total tokens": format_optional_count(item.total_tokens),
                "Retries observed": format_optional_count(item.retry_count),
                "Latency ms": format_optional_float(item.latency_ms),
                "Estimated USD": format_estimated_usd(estimate),
                "Safe error": item.error_category or "",
            }
        )
    return rows


def _usage_cost_by_case_rows(
    usage: ModelUsageSummary | None,
    *,
    as_of: date,
) -> list[dict[str, object]]:
    if usage is None:
        return []
    case_ids = sorted({item.case_id or "unattributed" for item in usage.invocations})
    rows: list[dict[str, object]] = []
    for case_id in case_ids:
        calls = tuple(
            item for item in usage.invocations if (item.case_id or "unattributed") == case_id
        )
        complete = bool(calls) and all(
            item.input_tokens is not None and item.output_tokens is not None for item in calls
        )
        summary = ModelUsageSummary(
            model_id=usage.model_id,
            region=usage.region,
            provider_call_count=len(calls),
            successful_call_count=sum(item.status == "COMPLETED" for item in calls),
            failed_attempt_count=sum(item.status == "FAILED" for item in calls),
            observable_retry_count=(
                sum(item.retry_count or 0 for item in calls)
                if any(item.retry_count is not None for item in calls)
                else None
            ),
            usage_complete=complete,
            input_tokens=(sum(item.input_tokens or 0 for item in calls) if complete else None),
            output_tokens=(sum(item.output_tokens or 0 for item in calls) if complete else None),
            total_tokens=(
                sum(item.total_tokens or 0 for item in calls)
                if calls and all(item.total_tokens is not None for item in calls)
                else None
            ),
            reasoning_tokens=(
                sum(item.reasoning_tokens or 0 for item in calls)
                if calls and all(item.reasoning_tokens is not None for item in calls)
                else None
            ),
            reason=None if complete else "Provider usage is incomplete for this case.",
            invocations=calls,
        )
        estimate = estimate_model_usage_cost(
            summary,
            load_pricing_catalog(),
            pricing_mode="on-demand-standard",
            as_of=as_of,
        )
        rows.append(
            {
                "Case ID": case_id,
                "Provider calls": format_optional_count(summary.provider_call_count),
                "Metrics": ", ".join(sorted({item.metric_name for item in calls})),
                "Input tokens": format_optional_count(summary.input_tokens),
                "Output tokens": format_optional_count(summary.output_tokens),
                "Reasoning tokens": format_optional_count(summary.reasoning_tokens),
                "Estimated USD": format_estimated_usd(estimate),
            }
        )
    return rows


def _backend_metric_rows(
    backend: BackendEvaluationState,
    backend_name: str,
) -> list[dict[str, object]]:
    names = set(backend.metrics) | set(backend.metric_statuses) | set(backend.metric_reasons)
    return [
        {
            "Backend": backend_name,
            "Metric": name,
            "Status": backend.metric_statuses.get(name, WorkbenchEvaluationStatus.COMPLETED).value,
            "Value": format_optional_float(backend.metrics.get(name), digits=4),
            "Reason / safe error category": backend.metric_reasons.get(name, ""),
        }
        for name in sorted(names)
    ]


def _timestamp(value: datetime | None) -> str:
    return value.astimezone(UTC).isoformat() if value is not None else "not started"


@st.cache_data(ttl=60, show_spinner=False)
def _private_mlflow_run_url(tracking_uri: str, run_id: str) -> str | None:
    if not tracking_uri.startswith(("http://", "https://")):
        return None
    try:
        from mlflow import MlflowClient

        run = MlflowClient(tracking_uri=tracking_uri).get_run(run_id)
    except Exception:
        return None
    return f"{tracking_uri.rstrip('/')}/#/experiments/{run.info.experiment_id}/runs/{run_id}"


@dataclass(frozen=True, slots=True)
class DatasetEvaluationExecution:
    report: ManagedRAGEvaluationReport
    report_path: Path
    mlflow_run_id: str | None
    mlflow_save_status: str
    region: str
    tracking_uri: str
    completed_at: datetime
    judge_call_artifact_path: Path | None = None
    judge_call_capture_error: str | None = None


@dataclass(frozen=True, slots=True)
class DatasetEvaluationFailure:
    error_type: str


def inspector_page() -> None:
    _header(
        "Request Inspector",
        "Exact request/trace correlation, stage score provenance, latency, tokens, "
        "and trace spans.",
    )
    settings, _service, _runtime_value, records = _runtime()
    if settings is None or records is None:
        return
    completed = records.completed()
    recent = [record.request_id for record in completed]
    default = recent[0] if recent else ""
    identifier = st.text_input(
        "Exact request ID or returned trace ID",
        value=st.session_state.get("inspection_identifier", default),
    )
    st.session_state.inspection_identifier = identifier
    record = records.get_exact(identifier)
    if record is None:
        st.info("No process-local request matches that exact identifier.")
        return
    st.caption(f"Selected request `{record.request_id}`")

    def overview() -> None:
        _request_summary(record)
        _stage_sequence(record)

    def retrieval() -> None:
        _candidate_tables(record)

    def context_prompt() -> None:
        st.subheader("Context & prompt")
        st.info(
            "Raw context and prompt content are not retained in the bounded metadata-only "
            "request store. Historical unavailable content is not reconstructed."
        )
        if record.inspection is not None:
            final_chunks = [
                item
                for item in record.inspection.stages.get("reranker", ())
                if item.included_in_context
            ]
            render_arrow_dataframe(
                st,
                [
                    {
                        "chunk_id": item.chunk_id,
                        "source_title": item.source_title,
                        "inclusion_reason": item.inclusion_reason,
                    }
                    for item in final_chunks
                ],
            )

    def generation() -> None:
        st.subheader("Generation")
        current = st.session_state.get("workbench_answer")
        if isinstance(current, WorkbenchAnswer) and current.record.request_id == record.request_id:
            st.markdown(current.response.answer)
        else:
            st.info("Generated answer content is unavailable in this process-local selection.")
        st.write(
            {
                "model": record.model_id,
                "region": record.region,
                "state": record.state.value,
                "citations": len(record.citations),
                "usage": record.token_usage or "unavailable",
            }
        )

    def quality_gate() -> None:
        current = st.session_state.get("workbench_answer")
        decision = (
            current.quality_decision
            if isinstance(current, WorkbenchAnswer)
            and current.record.request_id == record.request_id
            else record.quality_decision
        )
        if decision is None:
            st.info("The runtime policy decision is unavailable for this historical request.")
            return
        st.write(
            {
                "decision": decision.status.value,
                "profile": decision.profile.value,
                "policy_version": decision.policy_version,
                "answer_delivered": decision.deliver_answer,
                "reason": decision.reason,
            }
        )
        render_arrow_dataframe(
            st,
            [_quality_check_row(item) for item in decision.checks],
        )
        _render_runtime_request_cost(settings, record, decision)
        st.caption("A policy decision is not a model quality score or compliance guarantee.")
        evaluation_state = st.session_state.get("chat_evaluation_state")
        if (
            isinstance(evaluation_state, ChatEvaluationState)
            and evaluation_state.request_id == record.request_id
        ):
            results_column, calls_column = st.columns(2)
            if results_column.button("Open corresponding evaluation result"):
                st.session_state["workbench-tabs:Evaluations"] = "Results & Scores"
                st.switch_page(pages[2])
            if calls_column.button("Open corresponding private judge calls"):
                st.session_state["workbench-tabs:Evaluations"] = "Individual Judge Calls"
                st.switch_page(pages[2])

    def trace_cost() -> None:
        _cost_panel(settings, record, records)
        if record.quality_decision is not None:
            _render_runtime_request_cost(settings, record, record.quality_decision)
        _trace_panel(settings, record)
        state = st.session_state.get("chat_evaluation_state")
        if isinstance(state, ChatEvaluationState) and state.request_id == record.request_id:
            _evaluation_trace_action(
                state.ragas.execution_trace,
                request_id=record.request_id,
                key_suffix="request-inspector-current",
            )
        _render_selected_evaluation_trace(settings)

    _lazy_tabs(
        "Request Inspector",
        (overview, retrieval, context_prompt, generation, quality_gate, trace_cost),
    )


def _request_summary(record: RequestRecord) -> None:
    columns = st.columns(4)
    columns[0].metric("State", record.state.value)
    columns[1].metric("Total latency", _duration(record.total_latency_ms))
    columns[2].metric("Bottleneck", record.bottleneck or "not recorded")
    columns[3].metric("Generation", f"g{record.generation:06d}")
    st.code(record.request_id, language=None)
    st.caption("Request ID")
    st.code(record.trace_id or "unavailable", language=None)
    st.caption("Exact returned MLflow trace ID")
    if record.metrics is not None:
        st.write(
            {
                "ACL": "PUBLIC-only" if record.metrics.acl_filter_applied else "not applied",
                "PII policy": record.metrics.pii_policy,
                "rejected candidates": record.metrics.rejected_candidate_count,
            }
        )


def _stage_sequence(record: RequestRecord) -> None:
    st.subheader("End-to-end sequence")
    rows: list[dict[str, object]] = [
        {
            "Stage": stage,
            "Measured duration": _duration(duration),
            "Status": _stage_status(record, stage, duration),
        }
        for stage, duration in record.stage_timings_ms.items()
        if stage not in {"retrieval", "total"}
    ]
    render_arrow_dataframe(st, rows)
    st.caption(
        "A missing stage duration means not recorded; it does not prove the stage did not run. "
        "ACL checks occur after each retrieval/reranking boundary and are not timed separately."
    )


def _stage_status(
    record: RequestRecord,
    stage: str,
    duration: float | None,
) -> str:
    if record.state is RequestState.FAILED and record.failure_stage == stage:
        return "failed"
    if duration is not None:
        return "completed"
    if stage == "ACTIVE generation pin":
        return "completed"
    if stage == "ACL_validation" and record.metrics is not None:
        return "completed" if record.metrics.acl_filter_applied else "failed"
    if stage == "answer_and_citations" and record.state is RequestState.COMPLETED:
        return "completed"
    if stage == "MLflow_sync":
        return "pending" if record.trace_id is not None else "unavailable"
    return "not recorded"


def _candidate_tables(record: RequestRecord) -> None:
    st.subheader("Candidate rankings and score provenance")
    st.caption(
        "Dense similarity, BM25, reciprocal-rank-fusion, and reranker scores have different "
        "scales. Compare ranks within a stage; do not compare raw values across methods."
    )
    if record.inspection is None:
        st.info("Candidate inspection was not recorded for this request.")
        return
    labels = {
        "dense": "Dense HNSW",
        "sparse": "Sparse BM25",
        "rrf": "Reciprocal rank fusion",
        "reranker": "Qwen reranker / final context",
    }
    for stage, candidates in record.inspection.stages.items():
        st.markdown(f"#### {labels[stage]}")
        render_arrow_dataframe(
            st,
            [candidate.model_dump() for candidate in candidates],
        )


def _cost_panel(
    settings: AppSettings,
    record: RequestRecord,
    records: RequestInspectionStore,
) -> None:
    st.subheader("Tokens and estimated API cost")
    usage = record.token_usage
    if usage is None:
        st.info("Provider token usage unavailable.")
    else:
        st.write(usage)
    estimate = estimate_generation_cost(
        usage,
        load_pricing_catalog(),
        provider="aws-bedrock",
        model_id=record.model_id,
        region=record.region,
        pricing_mode="on-demand-standard",
        as_of=date.today(),
    )
    if not estimate.available or estimate.estimated_api_cost is None:
        st.info("Cost unavailable")
    else:
        st.metric(
            "Estimated generation API cost",
            f"{estimate.currency} ${estimate.estimated_api_cost:.8f}",
        )
        st.caption(
            "Estimate from Bedrock-reported tokens and the exact versioned pricing rule. "
            "This is not an AWS bill and excludes unverified local compute cost."
        )
    st.caption(
        f"Available answer-generation aggregate: {aggregate_token_usage(records.completed())}"
    )
    available_costs = []
    for available_record in records.completed():
        available_estimate = estimate_generation_cost(
            available_record.token_usage,
            load_pricing_catalog(),
            provider="aws-bedrock",
            model_id=available_record.model_id,
            region=available_record.region,
            pricing_mode="on-demand-standard",
            as_of=date.today(),
        )
        if available_estimate.available and available_estimate.estimated_api_cost is not None:
            available_costs.append(available_estimate.estimated_api_cost)
    if available_costs:
        st.caption(
            "Estimated API cost aggregate over uniquely keyed available requests: "
            f"USD ${sum(available_costs, Decimal(0)):.8f}"
        )
    st.caption("Judge/evaluation usage is accounted for separately from ordinary answers.")
    state = st.session_state.get("chat_evaluation_state")
    if isinstance(state, ChatEvaluationState) and state.request_id == record.request_id:
        _render_cost_observability(
            generation_usage=record.token_usage,
            generation_model=record.model_id,
            generation_region=record.region,
            backends=(
                ("MLflow GenAI judge", state.mlflow_genai),
                ("RAGAS judge", state.ragas),
            ),
            judge_model=_judge_bedrock_model_id(settings) or "unavailable",
            judge_region=_judge_region(settings) or "unavailable",
        )


def _trace_panel(settings: AppSettings, record: RequestRecord) -> None:
    _read_only_trace_panel(
        settings,
        title="MLflow generation trace",
        trace_id=record.trace_id,
        correlation_id=record.request_id,
        cache_key=f"trace:{record.request_id}",
    )


def _read_only_trace_panel(
    settings: AppSettings,
    *,
    title: str,
    trace_id: str | None,
    correlation_id: str,
    cache_key: str,
) -> None:
    st.subheader(title)
    if trace_id is not None and settings.mlflow.tracking_uri.startswith(("http://", "https://")):
        st.link_button(
            "Open exact request trace in MLflow",
            f"{settings.mlflow.tracking_uri.rstrip('/')}/#/traces/{trace_id}",
        )
    refresh = st.button("Refresh trace", type="primary", key=f"refresh:{cache_key}")
    if refresh or cache_key not in st.session_state:
        st.session_state[cache_key] = inspect_exact_trace(
            settings.mlflow,
            trace_id=trace_id,
            expected_request_id=correlation_id,
        )
    inspection = st.session_state[cache_key]
    if not isinstance(inspection, TraceInspection):
        return
    if inspection.visibility is TraceVisibility.PENDING:
        st.info(inspection.message)
    elif inspection.visibility is TraceVisibility.UNAVAILABLE:
        st.warning(inspection.message)
    elif inspection.visibility is TraceVisibility.FAILED:
        st.error(inspection.message)
    else:
        st.success(inspection.message)
    if inspection.root_duration_ms is not None:
        st.metric("MLflow root duration", _duration(inspection.root_duration_ms))
    if inspection.token_usage is not None:
        st.caption(f"MLflow token metadata: {inspection.token_usage}")
    if inspection.cost_metadata:
        st.caption(f"MLflow cost metadata: {inspection.cost_metadata}")
    if inspection.spans:
        render_arrow_dataframe(
            st,
            [span.model_dump() for span in inspection.spans],
        )
    elif inspection.visibility is TraceVisibility.COMPLETED:
        st.info("No spans were recorded for this exact trace.")
    st.caption(
        "Absent spans are shown as not recorded. A generated answer is not evidence that every "
        "expected span has finished syncing."
    )


def _evaluation_trace_action(
    reference: RagasExecutionTraceReference | None,
    *,
    request_id: str | None,
    key_suffix: str,
) -> None:
    if reference is None or reference.trace_id is None:
        st.info(
            reference.reason
            if reference is not None and reference.reason
            else "Evaluation execution trace not captured."
        )
        return
    if reference.client_request_id is None:
        st.info(
            "Evaluation execution trace not captured with a verifiable bounded client "
            "request ID. Historical execution is not replayed."
        )
        return
    if reference.status == "EXPORT_FAILED":
        st.error(reference.reason or "Evaluation execution trace export failed.")
    elif reference.status == "HIERARCHY_INCOMPLETE":
        st.warning(
            reference.reason
            or "The evaluation trace persisted without its complete execution hierarchy."
        )
    elif reference.status != "CONFIRMED":
        st.info(reference.reason or "Trace syncing: persistence is not confirmed.")
    diagnostics = reference.diagnostics
    st.caption(
        "RAGAS trace diagnostics · "
        f"metrics {diagnostics.metric_invocations_completed}/"
        f"{diagnostics.metric_invocations_started} · "
        f"LiteLLM completions {diagnostics.litellm_completion_invocations_completed}/"
        f"{diagnostics.litellm_completion_invocations_started} · "
        f"callbacks success/failure {diagnostics.success_callbacks_received}/"
        f"{diagnostics.failure_callbacks_received} · "
        f"provider spans attempted/created/persisted "
        f"{diagnostics.provider_spans_attempted}/"
        f"{diagnostics.provider_spans_created}/"
        f"{diagnostics.provider_spans_persisted}"
    )
    if diagnostics.missing_span_reason_categories:
        st.caption(
            "Safe trace reason categories: " + ", ".join(diagnostics.missing_span_reason_categories)
        )
    if st.button("View evaluation trace", key=f"view-evaluation-trace:{key_suffix}"):
        st.session_state.evaluation_trace_target = {
            "trace_id": reference.trace_id,
            "configuration_id": reference.configuration_id,
            "evaluation_run_id": reference.evaluation_run_id,
            "client_request_id": reference.client_request_id,
            "request_id": request_id,
            "diagnostics": diagnostics.model_dump(mode="json"),
        }
        st.switch_page(pages[1])


def _render_selected_evaluation_trace(settings: AppSettings) -> None:
    target = st.session_state.get("evaluation_trace_target")
    if not isinstance(target, dict):
        return
    trace_id = target.get("trace_id")
    configuration_id = target.get("configuration_id")
    client_request_id = target.get("client_request_id")
    if (
        not isinstance(trace_id, str)
        or not isinstance(configuration_id, str)
        or not isinstance(client_request_id, str)
    ):
        return
    st.divider()
    st.caption(
        "Separate RAGAS execution trace. It is correlated to the originating RAG request "
        "through exact request and original trace IDs recorded as metadata-only attributes."
    )
    st.write(
        {
            "ragas_evaluation_trace_id": trace_id,
            "evaluation_configuration_id": configuration_id,
            "evaluation_client_request_id": client_request_id,
            "mlflow_evaluation_run_id": target.get("evaluation_run_id") or "not recorded",
            "originating_request_id": target.get("request_id") or "not recorded",
        }
    )
    diagnostics = target.get("diagnostics")
    if isinstance(diagnostics, dict):
        st.caption("Metadata-only execution diagnostics")
        st.json(diagnostics)
    _read_only_trace_panel(
        settings,
        title="MLflow RAGAS evaluation execution trace",
        trace_id=trace_id,
        correlation_id=client_request_id,
        cache_key=f"evaluation-trace:{trace_id}",
    )


def _render_evaluation_input_inspector(
    dataset: ManagedRAGEvaluationDataset,
) -> None:
    st.subheader("Evaluation Input Inspector")
    st.caption(
        "Read-only data flow: reviewed PUBLIC case -> retrieval -> ContextBuilder -> Gemma "
        "answer -> versioned RAGAS inputs. Opening this section never invokes a model or "
        "retriever."
    )
    requested_id = st.session_state.get("evaluation_input_inspector_request_id")
    state = st.session_state.get("chat_evaluation_state")
    exact_state = (
        state
        if isinstance(state, ChatEvaluationState)
        and isinstance(requested_id, str)
        and state.request_id == requested_id
        else None
    )
    snapshot = (
        _evaluation_input_store().get_exact(
            requested_id,
            case_id=exact_state.case_id,
            configuration_id=exact_state.evaluation_key,
        )
        if isinstance(requested_id, str)
        and exact_state is not None
        and exact_state.case_id is not None
        and exact_state.evaluation_key is not None
        else None
    )
    selected_case_id = snapshot.case_id if snapshot is not None else dataset.cases[0].case_id
    case_ids = [case.case_id for case in dataset.cases]
    selected_case_id = st.selectbox(
        "Reviewed dataset case",
        case_ids,
        index=case_ids.index(selected_case_id),
        key="evaluation_input_dataset_case",
    )
    case = next(item for item in dataset.cases if item.case_id == selected_case_id)
    st.markdown("#### Reviewed dataset input")
    st.write(
        {
            "case_id": case.case_id,
            "dataset": f"{dataset.name} v{dataset.version}",
            "dataset_format": dataset.format_version,
            "privacy_classification": dataset.privacy_classification,
            "authorization": case.authorization.visibility,
            "review_status": case.review_status,
            "reviewed_question": case.question,
            "reference_answer": case.reference_answer,
            "expected_facts": list(case.expected_facts),
            "expected_source_ids": list(case.expected_source_ids),
            "expected_document_ids": list(case.expected_document_ids),
            "expected_chunk_ids": list(case.expected_chunk_ids),
        }
    )
    st.info(
        "The reference answer and expected facts are human-reviewed evaluation material, "
        "not independent external verification."
    )
    if snapshot is None or snapshot.case_id != case.case_id:
        st.warning("Input snapshot unavailable")
        st.caption(
            "Reviewed fixture content is available above. Historical reports contain metadata "
            "and scores only; missing answers and passages are not reconstructed or regenerated."
        )
        return
    _render_live_evaluation_snapshot(snapshot, exact_state)


def _render_live_evaluation_snapshot(
    snapshot: EvaluationInputSnapshot,
    state: ChatEvaluationState | None,
) -> None:
    st.markdown("#### Current live request input")
    st.write(
        {
            "input_contract": snapshot.format_version,
            "ragas_contract": snapshot.ragas_contract_version,
            "request_id": snapshot.request_id,
            "case_id": snapshot.case_id,
            "configuration_id": snapshot.configuration_id,
            "trace_id": snapshot.trace_id or "not recorded",
            "evaluation_run_id": snapshot.evaluation_run_id or "not recorded",
            "ragas_evaluation_trace_id": (
                snapshot.ragas_execution_trace.trace_id
                if snapshot.ragas_execution_trace is not None
                and snapshot.ragas_execution_trace.trace_id is not None
                else "not captured"
            ),
            "ragas_evaluation_trace_status": (
                snapshot.ragas_execution_trace.status
                if snapshot.ragas_execution_trace is not None
                else "UNAVAILABLE"
            ),
            "ragas_evaluation_client_request_id": (
                snapshot.ragas_execution_trace.client_request_id
                if snapshot.ragas_execution_trace is not None
                else "not recorded"
            ),
            "captured_at": snapshot.captured_at.isoformat(),
            "expires_at": snapshot.expires_at.isoformat(),
            "generation_model": snapshot.generation_model,
            "generation_region": snapshot.generation_region,
            "judge_model": snapshot.judge_model or "unavailable",
            "judge_region": snapshot.judge_region or "unavailable",
        }
    )
    if state is not None:
        _render_cost_observability(
            generation_usage=snapshot.generation_token_usage,
            generation_model=snapshot.generation_model,
            generation_region=snapshot.generation_region,
            backends=(
                ("MLflow GenAI judge", state.mlflow_genai),
                ("RAGAS judge", state.ragas),
            ),
            judge_model=(snapshot.judge_model or "unavailable").removeprefix("bedrock:/"),
            judge_region=snapshot.judge_region or "unavailable",
        )
        usage_rows = _usage_breakdown_rows(state.ragas.usage)
        if usage_rows:
            st.markdown("**Per-RAGAS-metric provider usage**")
            render_arrow_dataframe(st, usage_rows)
        elif state.ragas.status not in {
            WorkbenchEvaluationStatus.NOT_REQUESTED,
            WorkbenchEvaluationStatus.SKIPPED,
        }:
            st.info("RAGAS judge usage breakdown unavailable; no historical values are inferred.")
        _evaluation_trace_action(
            snapshot.ragas_execution_trace or state.ragas.execution_trace,
            request_id=snapshot.request_id,
            key_suffix=f"input-inspector:{snapshot.request_id}",
        )
    st.markdown("**Original reviewed question**")
    st.write(snapshot.original_question)
    st.markdown("**Generated Gemma output**")
    st.write(snapshot.generated_answer)
    st.markdown("**Selected evidence passages, in generation order**")
    for passage in snapshot.passages:
        mismatch = passage.original_text not in passage.prompt_section_text
        label = (
            f"{passage.order}. {passage.citation_id} | {passage.document_title} | "
            f"{passage.chunk_id}"
        )
        with st.expander(label):
            st.write(
                {
                    "entered_generation_context": True,
                    "source_id": passage.source_id or "not recorded",
                    "document_id": passage.document_id,
                    "chunk_id": passage.chunk_id,
                    "citation_id": passage.citation_id,
                    "truncated_by_context_builder": passage.truncated,
                    "original_and_prompt_text_differ": mismatch,
                }
            )
            st.markdown("**Original retrieved evidence**")
            st.text(passage.original_text)
            st.markdown("**Prompt-formatted evidence supplied to Gemma**")
            st.text(passage.prompt_section_text)
            if passage.truncated:
                st.warning(
                    "ContextBuilder truncated this final passage. RAGAS currently receives the "
                    "full original selected chunk under its v1 input contract."
                )
    with st.expander("Exact prompt context supplied to Gemma"):
        st.code(snapshot.prompt_context, language=None)
    with st.expander("Actual contexts supplied to RAGAS"):
        for index, context in enumerate(snapshot.ragas_input.contexts, start=1):
            st.markdown(f"**RAGAS context {index}**")
            st.text(context)
    st.markdown("**Human-reviewed reference answer**")
    st.write(snapshot.ragas_input.reference)
    st.markdown("#### Metric-by-metric input matrix")
    ragas_state = state.ragas if state is not None else None
    metric_rows: list[dict[str, object]] = []
    for metric, fields in RAGAS_METRIC_INPUT_FIELDS.items():
        metric_rows.append(
            {
                "Metric": metric,
                "Input fields": " + ".join(fields),
                "Score": (
                    format_optional_float(ragas_state.metrics.get(metric), digits=4)
                    if ragas_state is not None
                    else "Unavailable"
                ),
                "Backend status": (
                    ragas_state.metric_statuses.get(metric, ragas_state.status).value
                    if ragas_state is not None
                    else "UNAVAILABLE"
                ),
                "Judge model": snapshot.judge_model or "unavailable",
                "Safe error category": (
                    ragas_state.metric_reasons.get(metric, "") if ragas_state is not None else ""
                ),
            }
        )
    render_arrow_dataframe(st, metric_rows)
    for metric, fields in RAGAS_METRIC_INPUT_FIELDS.items():
        with st.expander(f"View inputs | {metric}"):
            for field in fields:
                st.markdown(f"**{field}**")
                if field == "question":
                    st.write(snapshot.ragas_input.question)
                elif field == "generated_answer":
                    st.write(snapshot.ragas_input.answer)
                elif field == "reference_answer":
                    st.write(snapshot.ragas_input.reference)
                else:
                    for context in snapshot.ragas_input.contexts:
                        st.text(context)
    st.caption(
        "These metrics assess the listed inputs under their configured rubrics. A score does "
        "not establish absolute factual correctness. Full content remains process-local and is "
        "not added to routine MLflow reports or diagnostic JSON."
    )


def _render_individual_judge_calls(
    latest_batch: object,
    saved_reports: list[tuple[Path, ManagedRAGEvaluationReport]],
    *,
    artifact_directory: Path,
) -> None:
    """Read one explicitly captured local artifact; this function never runs a judge."""

    st.subheader("Individual Judge Calls")
    st.caption(
        "Private reviewed-PUBLIC debug artifacts only. Displayed messages and responses are "
        "actual captured provider-boundary values after deterministic PII redaction and size "
        "limits; nothing is inferred or reconstructed."
    )
    artifacts = list_judge_call_debug_artifacts(artifact_directory)
    targets: dict[str, tuple[str | None, str | None]] = {}
    if isinstance(latest_batch, DatasetEvaluationExecution):
        trace = latest_batch.report.answer_quality.ragas.execution_trace
        targets["Latest completed batch"] = (
            trace.trace_id if trace is not None else None,
            trace.evaluation_run_id if trace is not None else latest_batch.mlflow_run_id,
        )
    chat_state = st.session_state.get("chat_evaluation_state")
    if isinstance(chat_state, ChatEvaluationState):
        trace = chat_state.ragas.execution_trace
        targets["Current Chat request evaluation"] = (
            trace.trace_id if trace is not None else None,
            trace.evaluation_run_id if trace is not None else chat_state.mlflow_genai.run_id,
        )
    for path, report in saved_reports:
        trace = report.answer_quality.ragas.execution_trace
        targets[f"Saved report · {path.name} · {report.generated_at.isoformat()}"] = (
            trace.trace_id if trace is not None else None,
            trace.evaluation_run_id if trace is not None else None,
        )
    for _path, artifact in artifacts:
        label = (
            f"Private capture · {artifact.created_at.isoformat()} · "
            f"{artifact.evaluation_trace_id or artifact.artifact_id}"
        )
        targets.setdefault(
            label,
            (artifact.evaluation_trace_id, artifact.evaluation_run_id),
        )
    if not targets:
        st.info("Individual judge-call content was not captured for this run.")
        return

    selection = st.selectbox("Evaluation", list(targets), key="judge_call_evaluation")
    trace_id, run_id = targets[selection]
    matching = [
        (path, artifact)
        for path, artifact in artifacts
        if (trace_id is not None and artifact.evaluation_trace_id == trace_id)
        or (run_id is not None and artifact.evaluation_run_id == run_id)
    ]
    if not matching:
        st.info("Individual judge-call content was not captured for this run.")
        return
    path, artifact = matching[0]
    st.write(
        {
            "evaluation_trace_id": artifact.evaluation_trace_id or "not recorded",
            "evaluation_run_id": artifact.evaluation_run_id or "not recorded",
            "evaluation_configuration_id": artifact.evaluation_configuration_id,
            "model_id": artifact.model_id,
            "region": artifact.region,
            "created_at": artifact.created_at.isoformat(),
            "expires_at": artifact.expires_at.isoformat(),
            "local_artifact": str(path),
            "content_policy": artifact.content_policy,
        }
    )
    hierarchy = [
        {
            "Case": call.case_id,
            "Metric": call.metric_name,
            "Provider call": call.sequence_within_metric,
            "Provider span ID": call.provider_span_id or "not recorded",
            "Status": call.call_status,
            "Validation": call.instructor.validation_status,
        }
        for call in artifact.calls
    ]
    if hierarchy:
        render_arrow_dataframe(st, hierarchy)
    metric_keys = sorted({(call.case_id, call.metric_name) for call in artifact.calls})
    if not metric_keys:
        st.info("The artifact contains no provider calls.")
        return
    metric_label = st.selectbox(
        "Metric",
        [f"{case_id} · {metric_name}" for case_id, metric_name in metric_keys],
        key=f"judge_call_metric:{artifact.artifact_id}",
    )
    selected_metric = metric_keys[
        [f"{case_id} · {metric_name}" for case_id, metric_name in metric_keys].index(metric_label)
    ]
    calls = [call for call in artifact.calls if (call.case_id, call.metric_name) == selected_metric]
    call_labels = [
        f"Provider call {call.sequence_within_metric} · {call.call_id}" for call in calls
    ]
    call_label = st.selectbox(
        "Individual provider call",
        call_labels,
        key=f"judge_call:{artifact.artifact_id}:{metric_label}",
    )
    call = calls[call_labels.index(call_label)]
    _render_individual_judge_call(call)

    delete_col, purge_col = st.columns(2)
    if delete_col.button(
        "Delete selected private artifact",
        key=f"delete_judge_artifact:{artifact.artifact_id}",
    ):
        delete_judge_call_debug_artifact(
            path,
            root=artifact_directory,
        )
        st.success("Private judge-call artifact deleted.")
        st.rerun()
    if purge_col.button("Purge expired private artifacts"):
        removed = purge_expired_judge_call_artifacts(artifact_directory)
        st.success(f"Removed {len(removed)} expired private artifact(s).")
        st.rerun()


def _render_individual_judge_call(call: JudgeCallDebugRecord) -> None:
    st.markdown("**INPUT — actual captured request (redacted/bounded)**")
    st.json(
        {
            "messages": list(call.request.messages),
            "structured_output_schema": call.request.structured_output_schema,
            "tools": call.request.tools,
            "parameters": call.request.parameters,
            "truncated": call.request.truncated,
        }
    )
    st.markdown("**OUTPUT — actual captured provider response (redacted/bounded)**")
    st.json(
        {
            "content": call.response.content,
            "tool_calls": call.response.tool_calls,
            "finish_reason": call.response.finish_reason,
            "response_type": call.response.response_type,
            "truncated": call.response.truncated,
            "instructor_parsed_result": call.instructor.parsed_response,
            "schema_validation": call.instructor.validation_status,
            "safe_error_category": call.instructor.error_category,
        }
    )
    st.markdown("**EXECUTION**")
    st.write(
        {
            "call_id": call.call_id,
            "metric_span_id": call.metric_span_id or "not recorded",
            "provider_span_id": call.provider_span_id or "not recorded",
            "sequence_within_metric": call.sequence_within_metric,
            "observed_retry_count": call.retry_count,
            "input_tokens": call.response.input_tokens,
            "output_tokens": call.response.output_tokens,
            "reasoning_tokens": call.response.reasoning_tokens,
            "total_tokens": call.response.total_tokens,
            "duration_ms": call.duration_ms,
            "estimated_usd": call.estimated_cost_usd or "Cost unavailable",
            "provider_request_id": call.response.provider_request_id or "not recorded",
            "model_id": call.model_id,
            "region": call.region,
            "effective_output_token_limit": call.effective_output_token_limit,
            "call_status": call.call_status,
            "safe_error_category": call.error_category,
        }
    )
    if call.metric_name == "context_precision":
        if call.context_chunk_ids:
            st.write(
                {
                    "context_judgment_chunk_ids": list(call.context_chunk_ids),
                    "association_source": call.context_correlation,
                }
            )
        else:
            st.info(
                "No unique context/chunk association was available from the actual outgoing "
                "request; none is inferred from call order."
            )


def _evaluation_run_tab() -> None:
    settings, _service, runtime, _records = _runtime()
    if settings is None or runtime is None:
        return
    _initialize_evaluation_state(settings)
    dataset = _approved_dataset(settings)
    if dataset is None:
        return
    st.subheader("Configured evaluation options")
    profiles = _judge_profile_choices(settings)
    selected_profile = st.selectbox(
        "Development judge",
        list(profiles),
        key="workbench_judge_profile",
        help=(
            "GPT-OSS 20B in Sydney is the development default. Llama 3 70B remains "
            "selectable for explicit historical comparison; saved reports are unchanged."
        ),
    )
    selected_uri, selected_region = profiles[selected_profile]
    selected_bedrock_id = selected_uri.removeprefix("bedrock:/") if selected_uri else "unavailable"
    try:
        selected_litellm_route = resolve_judge_identity(
            selected_uri or "", selected_region or ""
        ).litellm_model
    except ValueError:
        selected_litellm_route = "unavailable"
    st.caption(
        f"Configuration URI: `{selected_uri or 'unavailable'}` | LiteLLM: "
        f"`{selected_litellm_route}` | AWS modelId: `{selected_bedrock_id}` | "
        f"Region: `{selected_region or 'unavailable'}`"
    )
    st.checkbox(
        "Automatically evaluate a new eligible Chat submission",
        key="workbench_auto_evaluation",
        help=(
            "This only applies when Generate is explicitly selected and the question exactly "
            "matches the reviewed PUBLIC dataset. It never runs on rerender or navigation."
        ),
    )
    columns = st.columns(3)
    columns[0].checkbox("MLflow GenAI judges", key="workbench_mlflow_judges")
    columns[1].checkbox("RAGAS", key="workbench_ragas")
    columns[2].checkbox("Save evaluation to MLflow", key="workbench_save_mlflow")
    st.checkbox(
        "Capture exact RAGAS judge I/O in a private local artifact",
        key="workbench_capture_judge_io",
        disabled=not bool(st.session_state.workbench_ragas),
        help=(
            "Explicit reviewed-PUBLIC debug mode. Content is redacted, bounded, stored "
            "owner-only outside routine reports, and never added to MLflow spans."
        ),
    )
    st.number_input(
        "Private judge-call artifact retention (hours)",
        min_value=1,
        max_value=168,
        step=1,
        key="workbench_judge_io_retention_hours",
        disabled=not bool(st.session_state.workbench_capture_judge_io),
    )
    options = _evaluation_options()

    maximum = min(len(dataset.cases), settings.private_workbench.max_evaluation_cases)
    case_count = int(
        st.number_input(
            "Reviewed PUBLIC cases",
            min_value=1,
            max_value=maximum,
            value=maximum,
            step=1,
        )
    )
    selected_cases = dataset.cases[:case_count]
    plan = evaluation_plan(has_reference=True, has_content_trace=True)
    selected_metric_count = (len(plan.mlflow_genai.selected) if options.mlflow_genai else 0) + (
        len(plan.ragas.selected) if options.ragas else 0
    )
    st.subheader("Proposed paid batch execution")
    st.write(
        {
            "generation_model": settings.models.generation.model or "unavailable",
            "generation_region": settings.aws.region,
            "judge_model_for_both_backends": _judge_model_id(settings) or "unavailable",
            "judge_bedrock_model_id": _judge_bedrock_model_id(settings) or "unavailable",
            "judge_litellm_model": (
                f"bedrock/{_judge_bedrock_model_id(settings)}"
                if _judge_bedrock_model_id(settings)
                else "unavailable"
            ),
            "judge_region": _judge_region(settings) or "unavailable",
            "cases": case_count,
            "generation_calls": case_count,
            "logical_judge_metrics": selected_metric_count * case_count,
            "mlflow_metrics": list(plan.mlflow_genai.selected) if options.mlflow_genai else [],
            "ragas_metrics": list(plan.ragas.selected) if options.ragas else [],
            "generation_max_output_tokens_per_case": settings.models.generation.max_tokens,
            "ragas_default_max_output_tokens_per_call": (settings.mlflow.ragas_judge_max_tokens),
            "ragas_context_precision_max_output_tokens_per_call": (
                settings.mlflow.ragas_context_precision_max_tokens
            ),
            "estimated_call_range": (
                "unavailable; RAGAS metrics may invoke the judge multiple times"
            ),
            "estimated_cost_range": (
                "unavailable; judge token volume is not defensibly known pre-run"
            ),
            "case_limit": settings.private_workbench.max_evaluation_cases,
            "monetary_budget_limit": "not configured",
        }
    )
    st.caption(
        "Logical metric count is not a guaranteed provider-call count; a metric may use multiple "
        "model invocations. The higher context-precision ceiling can increase output-token cost; "
        "only provider-returned usage is priced after execution. Only the reviewed PUBLIC dataset "
        "is submitted."
    )
    if _judge_model_id(settings) is None and (options.mlflow_genai or options.ragas):
        st.warning(
            "Judge model unavailable. Generation may still complete; requested judge backends "
            "will report UNAVAILABLE rather than failing generated answers."
        )

    generation = runtime.retrieval.generation
    assert generation is not None
    key = evaluation_key(
        scope_id=f"{dataset.name}:{dataset.version}",
        generation=generation,
        case_ids=[case.case_id for case in selected_cases],
        generation_model=settings.models.generation.model or "unavailable",
        judge_model=_judge_model_id(settings),
        judge_region=_judge_region(settings),
        options=options,
    )
    registry = _evaluation_registry()
    existing = registry.get(key)
    run = st.button("Run approved evaluation", type="primary")
    rerun = st.button("Re-evaluate selected configuration", disabled=existing is None)
    if run and not registry.should_execute(key):
        st.info(
            "This request/configuration already has a result. Use Re-evaluate for another "
            "explicit paid execution."
        )
    if (run and registry.should_execute(key)) or rerun:
        with st.spinner("Running the explicitly selected approved evaluation…"):
            execution = _execute_dataset_evaluation(
                settings,
                runtime,
                dataset,
                selected_cases,
                options,
                evaluation_configuration_id=key,
            )
        registry.record(key, execution)
        if isinstance(execution, DatasetEvaluationExecution):
            st.session_state.latest_batch_execution = execution
        existing = execution
    if isinstance(existing, DatasetEvaluationFailure):
        st.error(
            f"Evaluation failed ({existing.error_type}). A successful Chat answer is unaffected."
        )


def evaluations_page() -> None:
    _header(
        "Evaluations",
        "Reviewed-dataset execution, results, judge calls, inputs, and comparisons.",
    )
    _lazy_tabs(
        "Evaluations",
        (
            _evaluation_overview_tab,
            _evaluation_run_tab,
            _evaluation_results_tab,
            _evaluation_judge_calls_tab,
            _evaluation_inputs_tab,
            _evaluation_compare_tab,
        ),
    )


def _evaluation_context() -> tuple[
    AppSettings | None,
    ManagedRAGEvaluationDataset | None,
    list[tuple[Path, ManagedRAGEvaluationReport]],
]:
    settings, _service, _runtime_value, _records = _runtime()
    if settings is None:
        return None, None, []
    _initialize_evaluation_state(settings)
    return (
        settings,
        _approved_dataset(settings),
        _saved_reports(settings.mlflow.report_directory),
    )


def _evaluation_overview_tab() -> None:
    settings, dataset, saved = _evaluation_context()
    if settings is None or dataset is None:
        return
    latest = st.session_state.get("latest_batch_execution")
    reports = [report for _path, report in saved]
    status_counts: dict[str, int] = {}
    for report in reports:
        status = display_status(report.answer_quality.ragas).value
        status_counts[status] = status_counts.get(status, 0) + 1
    left, middle, right, fourth = st.columns(4)
    left.metric("Reviewed dataset", f"{dataset.name} v{dataset.version}")
    middle.metric("Saved runs", len(reports))
    right.metric("Judge model", _judge_model_id(settings) or "unavailable")
    fourth.metric(
        "Latest in session",
        "yes" if isinstance(latest, DatasetEvaluationExecution) else "no",
    )
    st.subheader("Status breakdown")
    st.write(status_counts or {"NO_SAVED_RUNS": 0})
    if reports:
        report = reports[0]
        st.subheader("Latest saved scores")
        _render_saved_report(report)
    else:
        st.info("No saved managed evaluation reports are available.")


def _evaluation_results_tab() -> None:
    settings, _dataset, saved = _evaluation_context()
    if settings is None:
        return
    _settings, _service, _runtime_value, records = _runtime()
    chat_state = st.session_state.get("chat_evaluation_state")
    latest = st.session_state.get("latest_batch_execution")
    choices: dict[str, ManagedRAGEvaluationReport] = {
        f"{path.name} · {report.generated_at.isoformat()}": report for path, report in saved
    }
    if isinstance(latest, DatasetEvaluationExecution):
        choices = {"Latest completed in this session": latest.report, **choices}
    if not choices and not isinstance(chat_state, ChatEvaluationState):
        st.info("No evaluation result is available.")
        return
    labels = list(choices)
    if isinstance(chat_state, ChatEvaluationState):
        labels.insert(0, "Current Chat request")
    selected = st.selectbox(
        "Evaluation run",
        labels,
        key="evaluation_results_selected_run",
    )
    if selected == "Current Chat request" and isinstance(chat_state, ChatEvaluationState):
        record = records.get_exact(chat_state.request_id) if records is not None else None
        _render_chat_evaluation_state(
            settings,
            chat_state,
            generation_usage=record.token_usage if record is not None else None,
            generation_model=record.model_id if record is not None else None,
            generation_region=record.region if record is not None else None,
        )
        return
    report = choices[selected]
    _render_saved_report(report)
    if (
        report.answer_cases
        and report.answer_cases[0].request_id
        and st.button("Open first request in Request Inspector")
    ):
        st.session_state.inspection_identifier = report.answer_cases[0].request_id
        st.switch_page(pages[1])


def _evaluation_judge_calls_tab() -> None:
    settings, _dataset, saved = _evaluation_context()
    if settings is None:
        return
    latest = st.session_state.get("latest_batch_execution")
    _render_individual_judge_calls(
        latest if isinstance(latest, DatasetEvaluationExecution) else None,
        saved,
        artifact_directory=settings.private_workbench.judge_call_debug_directory,
    )


def _evaluation_inputs_tab() -> None:
    _settings, dataset, _saved = _evaluation_context()
    if dataset is None:
        return
    _render_evaluation_input_inspector(dataset)


def _evaluation_compare_tab() -> None:
    _settings, _dataset, saved = _evaluation_context()
    if len(saved) < 2:
        st.info("At least two saved runs are required for comparison.")
        return
    choices = {f"{path.name} · {report.generated_at.isoformat()}": report for path, report in saved}
    labels = list(choices)
    baseline_label = st.selectbox(
        "Baseline",
        labels,
        index=min(1, len(labels) - 1),
        key="evaluation_compare_baseline",
    )
    candidate_label = st.selectbox(
        "Candidate",
        labels,
        index=0,
        key="evaluation_compare_candidate",
    )
    baseline = choices[baseline_label]
    candidate = choices[candidate_label]
    compatible = (
        baseline.dataset_name == candidate.dataset_name
        and baseline.dataset_version == candidate.dataset_version
        and baseline.case_count == candidate.case_count
    )
    if not compatible:
        st.warning(
            "Dataset name/version or case count differs. Metric deltas are shown as "
            "incompatible and must not be interpreted as regression evidence."
        )
    baseline_metrics = _report_metrics(baseline)
    candidate_metrics = _report_metrics(candidate)
    rows = []
    for name in sorted(set(baseline_metrics) | set(candidate_metrics)):
        before = baseline_metrics.get(name)
        after = candidate_metrics.get(name)
        rows.append(
            {
                "Metric": name,
                "Baseline": before,
                "Candidate": after,
                "Change": (
                    after - before
                    if compatible and before is not None and after is not None
                    else None
                ),
                "Comparison status": (
                    "Comparable"
                    if compatible and before is not None and after is not None
                    else "Incompatible or unavailable"
                ),
            }
        )
    render_arrow_dataframe(
        st,
        rows,
        schema={
            "Baseline": DisplayColumnKind.NULLABLE_FLOAT,
            "Candidate": DisplayColumnKind.NULLABLE_FLOAT,
            "Change": DisplayColumnKind.NULLABLE_FLOAT,
        },
        column_config={
            "Baseline": st.column_config.NumberColumn(format="%.4f"),
            "Candidate": st.column_config.NumberColumn(format="%.4f"),
            "Change": st.column_config.NumberColumn(format="%+.4f"),
        },
    )
    st.write(
        {
            "baseline_generation_latency_ms": baseline.answer_quality.generation.mean_latency_ms,
            "candidate_generation_latency_ms": candidate.answer_quality.generation.mean_latency_ms,
            "baseline_retrieval_latency_ms": baseline.retrieval_latency.mean_total_ms,
            "candidate_retrieval_latency_ms": candidate.retrieval_latency.mean_total_ms,
            "cost_comparison": (
                "See each selected report; unavailable usage is never shown as zero."
            ),
        }
    )


def _report_metrics(report: ManagedRAGEvaluationReport) -> dict[str, float]:
    values: dict[str, float] = {}
    for backend, summary in (
        ("mlflow", report.answer_quality.mlflow_genai),
        ("ragas", report.answer_quality.ragas),
    ):
        values.update(
            {f"{backend}.{name}": value for name, value in summary.aggregate_metrics.items()}
        )
    return values


def _execute_dataset_evaluation(
    settings: AppSettings,
    runtime: ApplicationRuntime,
    dataset: ManagedRAGEvaluationDataset,
    selected_cases: tuple[ManagedRAGEvaluationCase, ...],
    options: WorkbenchEvaluationOptions,
    *,
    evaluation_configuration_id: str,
) -> DatasetEvaluationExecution | DatasetEvaluationFailure:
    try:
        judge_preflight = (
            _judge_preflight(settings) if options.mlflow_genai or options.ragas else None
        )
        telemetry = CollectingRetrievalTelemetrySink()
        evaluation_application = build_application_runtime(
            settings,
            retrieval_runtime=runtime.retrieval,
            telemetry_sink=telemetry,
            trace_content_policy=TraceContentPolicy.APPROVED_PUBLIC_EVALUATION,
        )
        managed_runtime = evaluation_application.retrieval.managed_runtime
        if managed_runtime is None:
            raise RuntimeError("Managed evaluation runtime identity is unavailable")
        debug_capture: JudgeCallDebugCapture | None = None
        if options.capture_judge_io:
            if not options.ragas or not settings.mlflow.enabled:
                raise RuntimeError(
                    "Private judge-call capture requires RAGAS and active MLflow tracing"
                )
            judge_model = _judge_model_id(settings)
            judge_region = _judge_region(settings)
            if not judge_model or not judge_region:
                raise RuntimeError("Private judge-call capture requires a configured judge")
            expected_configuration_id = managed_evaluation_configuration_id(
                dataset=dataset,
                cases=selected_cases,
                generation=managed_runtime.generation,
                judge_model=judge_model,
                judge_region=judge_region,
            )
            # The workbench at-most-once key includes UI options, while the execution
            # trace configuration identifies only the judge execution contract.
            evaluation_configuration_id = expected_configuration_id
            debug_capture = create_approved_public_judge_debug_capture(
                JudgeCallDebugConfig(
                    artifact_directory=(settings.private_workbench.judge_call_debug_directory),
                    retention_hours=options.judge_io_retention_hours,
                ),
                dataset=dataset,
                cases=selected_cases,
                evaluation_configuration_id=evaluation_configuration_id,
                evaluation_run_id=None,
                judge_model=judge_model,
                judge_region=judge_region,
            )
        report = evaluate_managed_rag(
            managed_runtime,
            dataset,
            configured_corpus_key=settings.managed_ingestion.corpus_key,
            selected_cases=selected_cases,
            pipeline=evaluation_application.pipeline,
            generation_model=settings.models.generation.model,
            generation_region=settings.aws.region,
            judge_model=_judge_model_id(settings),
            with_mlflow_judges=options.mlflow_genai,
            with_ragas=options.ragas,
            mlflow_settings=_effective_mlflow_settings(settings),
            telemetry_sink=telemetry,
            judge_preflight=judge_preflight,
            evaluation_configuration_id=evaluation_configuration_id,
            ragas_debug_capture=debug_capture,
        )
        report_path = write_managed_rag_report(
            report,
            settings.mlflow.report_directory
            / f"managed_rag_g{managed_runtime.generation:06d}.json",
        )
        run_id: str | None = None
        save_status = "SKIPPED"
        if options.save_to_mlflow:
            if settings.mlflow.enabled:
                with evaluation_run(
                    settings.mlflow,
                    run_name=f"managed-rag-g{managed_runtime.generation:06d}",
                    existing_run_id=report.answer_quality.mlflow_genai.run_id,
                ) as active_run:
                    log_managed_rag_report(report, report_path, active_run)
                    run_id = active_run.run_id
                save_status = "COMPLETED" if run_id is not None else "UNAVAILABLE"
            else:
                save_status = "UNAVAILABLE"
        return DatasetEvaluationExecution(
            report=report,
            report_path=report_path,
            mlflow_run_id=run_id,
            mlflow_save_status=save_status,
            region=settings.aws.region,
            tracking_uri=settings.mlflow.tracking_uri,
            completed_at=datetime.now(UTC),
            judge_call_artifact_path=(
                debug_capture.persisted_path if debug_capture is not None else None
            ),
            judge_call_capture_error=(
                debug_capture.failure_category if debug_capture is not None else None
            ),
        )
    except Exception as error:
        return DatasetEvaluationFailure(error_type=type(error).__name__)


def _render_dataset_execution(execution: DatasetEvaluationExecution) -> None:
    report = execution.report
    ragas_usage = report.answer_quality.ragas.usage
    mlflow_usage = report.answer_quality.mlflow_genai.usage
    st.caption(
        f"Batch evaluation ID: {report.generated_at.isoformat()} · completed at "
        f"{_timestamp(execution.completed_at)}"
    )
    rows: list[dict[str, object]] = [
        {
            "Execution": "Gemma answer generation",
            "Status": report.answer_quality.generation.status,
            "Model": report.models.generation_model or "unavailable",
            "AWS Region": execution.region,
            "Cases": report.answer_quality.generation.generated_case_count,
            "Input tokens": format_optional_count(
                report.answer_quality.generation.token_usage.input_tokens
            ),
            "Output tokens": format_optional_count(
                report.answer_quality.generation.token_usage.output_tokens
            ),
            "Estimated cost": _generation_cost_label(report, region=execution.region),
            "Run ID": "not applicable",
        },
        {
            "Execution": "MLflow GenAI evaluation",
            "Status": display_status(report.answer_quality.mlflow_genai).value,
            "Model": report.models.judge_model or "unavailable",
            "AWS Region": report.models.judge_region or "unavailable",
            "Cases": report.case_count,
            "Input tokens": format_optional_count(
                mlflow_usage.input_tokens if mlflow_usage else None
            ),
            "Output tokens": format_optional_count(
                mlflow_usage.output_tokens if mlflow_usage else None
            ),
            "Estimated cost": _judge_cost_label(mlflow_usage),
            "Run ID": report.answer_quality.mlflow_genai.run_id or "not recorded",
        },
        {
            "Execution": "RAGAS evaluation",
            "Status": display_status(report.answer_quality.ragas).value,
            "Model": report.models.judge_model or "unavailable",
            "AWS Region": report.models.judge_region or "unavailable",
            "Cases": report.case_count,
            "Input tokens": format_optional_count(
                ragas_usage.input_tokens if ragas_usage else None
            ),
            "Output tokens": format_optional_count(
                ragas_usage.output_tokens if ragas_usage else None
            ),
            "Estimated cost": _judge_cost_label(ragas_usage),
            "Run ID": "not recorded",
            "Execution trace ID": (
                report.answer_quality.ragas.execution_trace.trace_id
                if report.answer_quality.ragas.execution_trace is not None
                and report.answer_quality.ragas.execution_trace.trace_id is not None
                else "not captured"
            ),
            "Execution trace status": (
                report.answer_quality.ragas.execution_trace.status
                if report.answer_quality.ragas.execution_trace is not None
                else "UNAVAILABLE"
            ),
        },
        {
            "Execution": "Save safe report to MLflow",
            "Status": execution.mlflow_save_status,
            "Model": "not applicable",
            "AWS Region": "not applicable",
            "Cases": report.case_count,
            "Input tokens": "Not applicable",
            "Output tokens": "Not applicable",
            "Estimated cost": "Not applicable",
            "Run ID": execution.mlflow_run_id or "not recorded",
        },
    ]
    render_arrow_dataframe(st, rows)
    _render_report_cost_total(report, generation_region=execution.region)
    usage_rows = _usage_breakdown_rows(ragas_usage)
    if usage_rows:
        st.markdown("**RAGAS estimated cost by case**")
        render_arrow_dataframe(
            st,
            _usage_cost_by_case_rows(ragas_usage, as_of=report.generated_at.date()),
        )
        st.markdown("**RAGAS provider usage by case and metric**")
        render_arrow_dataframe(st, usage_rows)
    st.caption(f"Safe local report: `{execution.report_path}`")
    if execution.judge_call_artifact_path is not None:
        st.success(
            "Private judge-call content captured separately: "
            f"`{execution.judge_call_artifact_path}`"
        )
    elif execution.judge_call_capture_error is not None:
        st.warning(
            "RAGAS evaluation completed independently, but private judge-call capture "
            f"failed ({execution.judge_call_capture_error})."
        )
    else:
        st.caption("Individual judge-call content was not captured for this run.")
    matching_run_id = report.answer_quality.mlflow_genai.run_id or execution.mlflow_run_id
    if matching_run_id:
        run_url = _private_mlflow_run_url(execution.tracking_uri, matching_run_id)
        if run_url:
            st.link_button("Open matching private MLflow evaluation run", run_url)
    _evaluation_trace_action(
        report.answer_quality.ragas.execution_trace,
        request_id=(report.answer_cases[0].request_id if report.answer_cases else None),
        key_suffix=f"batch:{report.generated_at.isoformat()}",
    )
    _render_saved_report(report)


def _saved_reports(directory: Path) -> list[tuple[Path, ManagedRAGEvaluationReport]]:
    reports: list[tuple[Path, ManagedRAGEvaluationReport]] = []
    for path in sorted(directory.glob("managed_rag_g*.json"), reverse=True):
        try:
            report = ManagedRAGEvaluationReport.model_validate_json(
                path.read_text(encoding="utf-8")
            )
        except Exception:
            continue
        reports.append((path, report))
    return reports


def _generation_cost_label(
    report: ManagedRAGEvaluationReport,
    *,
    region: str,
) -> str:
    usage = report.answer_quality.generation.token_usage
    if usage.input_tokens is None or usage.output_tokens is None:
        return "Cost unavailable"
    estimate = estimate_generation_cost(
        {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "total_tokens": usage.total_tokens or usage.input_tokens + usage.output_tokens,
        },
        load_pricing_catalog(),
        provider="aws-bedrock",
        model_id=report.models.generation_model or "",
        region=region,
        pricing_mode="on-demand-standard",
        as_of=date.today(),
    )
    return format_estimated_usd(estimate)


def _generation_report_cost(
    report: ManagedRAGEvaluationReport,
    *,
    region: str,
) -> CostEstimate:
    usage = report.answer_quality.generation.token_usage
    return estimate_generation_cost(
        (
            {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "total_tokens": usage.total_tokens
                or (usage.input_tokens or 0) + (usage.output_tokens or 0),
            }
            if usage.input_tokens is not None and usage.output_tokens is not None
            else None
        ),
        load_pricing_catalog(),
        provider="aws-bedrock",
        model_id=report.models.generation_model or "",
        region=region,
        pricing_mode="on-demand-standard",
        as_of=report.generated_at.date(),
    )


def _judge_cost_label(usage: ModelUsageSummary | None) -> str:
    estimate = estimate_model_usage_cost(
        usage,
        load_pricing_catalog(),
        pricing_mode="on-demand-standard",
        as_of=date.today(),
    )
    return format_estimated_usd(estimate)


def _render_report_cost_total(
    report: ManagedRAGEvaluationReport,
    *,
    generation_region: str,
) -> None:
    estimates = {
        "Gemma answer generation": _generation_report_cost(report, region=generation_region)
    }
    for name, summary in (
        ("MLflow GenAI judge", report.answer_quality.mlflow_genai),
        ("RAGAS judge", report.answer_quality.ragas),
    ):
        if summary.status != "NOT_REQUESTED":
            estimates[name] = estimate_model_usage_cost(
                summary.usage,
                load_pricing_catalog(),
                pricing_mode="on-demand-standard",
                as_of=report.generated_at.date(),
            )
    combined = combine_cost_estimates(estimates)
    st.write(
        {
            "total_known_estimated_usd": f"${combined.known_estimated_cost:.8f}",
            "complete_total": combined.complete,
            "cost_unavailable_for": list(combined.unavailable_components),
            "currency": "USD",
            "pricing_mode": "on-demand-standard",
            "estimate_disclaimer": "Estimated USD; not an AWS invoice charge.",
        }
    )


def _render_saved_report(report: ManagedRAGEvaluationReport) -> None:
    st.info(
        "Historical report: metadata and scores only. Full question, generated answer, prompt "
        "context, and retrieved passages are unavailable and are not reconstructed."
    )
    st.write(
        {
            "dataset": f"{report.dataset_name} v{report.dataset_version}",
            "privacy": report.dataset_privacy_classification,
            "generation": f"g{report.generation:06d}",
            "cases": report.case_count,
            "generation_model": report.models.generation_model,
            "generation_region": report.models.generation_region or "unavailable",
            "judge_model": report.models.judge_model,
            "judge_region": report.models.judge_region or "unavailable",
            "generation_status": report.answer_quality.generation.status,
            "mlflow_genai_status": display_status(report.answer_quality.mlflow_genai).value,
            "ragas_status": display_status(report.answer_quality.ragas).value,
        }
    )
    metrics: list[dict[str, object]] = []
    for backend, summary in (
        ("MLflow GenAI", report.answer_quality.mlflow_genai),
        ("RAGAS", report.answer_quality.ragas),
    ):
        for name, value in summary.aggregate_metrics.items():
            metrics.append(
                {
                    "Backend": backend,
                    "Metric": name,
                    "Status": "COMPLETED",
                    "Score": value,
                    "Reason": "",
                }
            )
        for name, reason in summary.skipped_metrics.items():
            metrics.append(
                {
                    "Backend": backend,
                    "Metric": name,
                    "Status": "SKIPPED",
                    "Score": None,
                    "Reason": reason,
                }
            )
        for name, status in summary.metric_statuses.items():
            if status != "COMPLETED":
                metrics.append(
                    {
                        "Backend": backend,
                        "Metric": name,
                        "Status": status,
                        "Score": None,
                        "Reason": summary.failed_metrics.get(name, "SCORER_ERROR"),
                    }
                )
    if metrics:
        render_arrow_dataframe(
            st,
            metrics,
            schema={"Score": DisplayColumnKind.NULLABLE_FLOAT},
            column_config={
                "Score": st.column_config.NumberColumn(format="%.4f"),
            },
        )
    st.write(
        {
            "mlflow_evaluation_run_id": (
                report.answer_quality.mlflow_genai.run_id or "not recorded"
            ),
            "mlflow_reason": report.answer_quality.mlflow_genai.reason,
            "ragas_reason": report.answer_quality.ragas.reason,
            "ragas_evaluation_trace_id": (
                report.answer_quality.ragas.execution_trace.trace_id
                if report.answer_quality.ragas.execution_trace is not None
                and report.answer_quality.ragas.execution_trace.trace_id is not None
                else "not captured"
            ),
            "ragas_evaluation_trace_status": (
                report.answer_quality.ragas.execution_trace.status
                if report.answer_quality.ragas.execution_trace is not None
                else "UNAVAILABLE"
            ),
            "ragas_evaluation_client_request_id": (
                report.answer_quality.ragas.execution_trace.client_request_id
                if report.answer_quality.ragas.execution_trace is not None
                else "not recorded"
            ),
            "report_timestamp": report.generated_at.isoformat(),
        }
    )
    _render_report_cost_total(
        report,
        generation_region=report.models.generation_region or "unavailable",
    )
    ragas_usage_rows = _usage_breakdown_rows(report.answer_quality.ragas.usage)
    if ragas_usage_rows:
        render_arrow_dataframe(
            st,
            _usage_cost_by_case_rows(
                report.answer_quality.ragas.usage,
                as_of=report.generated_at.date(),
            ),
        )
        render_arrow_dataframe(st, ragas_usage_rows)
    elif report.answer_quality.ragas.status != "NOT_REQUESTED":
        st.info(
            "Historical judge usage and cost unavailable; the report is not replayed to fill it."
        )
    _evaluation_trace_action(
        report.answer_quality.ragas.execution_trace,
        request_id=(report.answer_cases[0].request_id if report.answer_cases else None),
        key_suffix=f"saved:{report.generated_at.isoformat()}",
    )
    st.caption(
        "Groundedness measures support from retrieved evidence; answer quality includes "
        "correctness/relevance against the reviewed case. Missing references are reported as "
        "REFERENCE_REQUIRED, never filled with fabricated values."
    )


def lineage_page() -> None:
    _header("Lineage & Index", "Read-only lineage and managed index state.")
    _lazy_tabs(
        "Lineage & Index",
        (
            _lineage_active_tab,
            _lineage_documents_tab,
            _lineage_ingestion_tab,
            _lineage_generations_tab,
        ),
    )


def _lineage_active_tab() -> None:
    settings, _service, runtime, _records = _runtime()
    if settings is None or runtime is None:
        return
    generation = runtime.retrieval.generation
    left, middle, right = st.columns(3)
    left.metric("Active generation", f"g{generation:06d}" if generation else "Unavailable")
    middle.metric("Managed index", "ACTIVE")
    right.metric("Collection", runtime.retrieval.qdrant_collection)
    render_arrow_dataframe(
        st,
        [
            {"Setting": "Index state ID", "Value": str(runtime.retrieval.index_state_id)},
            {"Setting": "Dense vector", "Value": settings.managed_index.dense_vector_name},
            {"Setting": "Sparse vector", "Value": settings.managed_index.sparse_vector_name},
            {"Setting": "HNSW m", "Value": str(settings.managed_index.hnsw_m)},
            {
                "Setting": "HNSW ef_construct",
                "Value": str(settings.managed_index.hnsw_ef_construct),
            },
        ],
    )
    with st.expander("Advanced runtime metadata"):
        st.json(dict(runtime.retrieval.display_metadata()))
    st.caption("Publication, restoration, cleanup, deletion, and index mutation remain CLI-only.")


def _lineage_documents_tab() -> None:
    st.info(
        "Document text is not copied into this UI. Document/version/chunk lineage remains "
        "authoritative in PostgreSQL and the managed generation metadata."
    )


def _lineage_ingestion_tab() -> None:
    st.info(
        "Ingestion history is read-only here. Index creation, publication, restoration, and "
        "cleanup remain explicit CLI operations."
    )


def _lineage_generations_tab() -> None:
    _lineage_active_tab()
    st.caption("Only the pinned ACTIVE generation is loaded by the serving runtime.")


def drift_page() -> None:
    _header("Drift & Regression", "Read-only quality, baseline, regression, and alert review.")
    _lazy_tabs(
        "Drift & Regression",
        (
            _drift_quality_tab,
            _drift_baselines_tab,
            _drift_regressions_tab,
            _drift_alerts_tab,
        ),
    )


def _drift_quality_tab() -> None:
    uploaded = st.file_uploader("Managed drift assessment JSON", type=["json"])
    if uploaded is None:
        st.info("Upload a genuine report produced by scripts/assess_managed_drift.py.")
        return
    try:
        report = load_drift_report_json(uploaded.getvalue())
    except Exception:
        st.error("The supplied file is not a valid managed drift assessment.")
        return
    st.session_state.managed_drift_report = report
    payload = report.model_dump(mode="json")
    st.success("Validated managed drift assessment loaded.")
    render_arrow_dataframe(
        st,
        [
            {"Field": key, "Value": str(value)}
            for key, value in payload.items()
            if not isinstance(value, (dict, list))
        ],
    )
    with st.expander("Raw metadata"):
        st.json(payload)


def _drift_baselines_tab() -> None:
    report = st.session_state.get("managed_drift_report")
    if report is None:
        st.info("Upload a genuine managed drift report in Quality Trends first.")
        return
    payload = report.model_dump(mode="json")
    render_arrow_dataframe(
        st,
        [
            {"Field": key, "Value": str(value)}
            for key, value in payload.items()
            if not isinstance(value, (dict, list))
        ],
    )
    with st.expander("Raw metadata"):
        st.json(payload)


def _drift_regressions_tab() -> None:
    st.info(
        "Regression decisions use the configured, versioned drift thresholds and genuine "
        "baseline/candidate reports. No missing result is synthesized."
    )


def _drift_alerts_tab() -> None:
    st.info(
        "No alert dispatcher is active in this local deployment. Drift assessment remains "
        "an explicit offline workflow."
    )


def operations_page() -> None:
    _header("Operations", "Health, models, budgets, jobs, privacy, and configuration.")
    _lazy_tabs(
        "Operations",
        (
            _operations_health_tab,
            _operations_models_tab,
            _operations_jobs_tab,
            _operations_privacy_tab,
            _operations_configuration_tab,
        ),
    )


def _operations_health_tab() -> None:
    settings, _service, runtime, records = _runtime()
    if settings is None or runtime is None:
        st.error("Managed runtime readiness failed.")
        return
    st.success("Managed runtime readiness passed.")
    render_arrow_dataframe(
        st,
        [
            {
                "Component": "Runtime",
                "Identity": settings.runtime.profile,
                "Location / device": "Local process",
            },
            {
                "Component": "Embedding",
                "Identity": settings.models.embedding.model,
                "Location / device": settings.models.embedding.device,
            },
            {
                "Component": "Reranker",
                "Identity": settings.models.reranker.model,
                "Location / device": settings.models.reranker.device,
            },
            {
                "Component": "Generation",
                "Identity": settings.models.generation.model,
                "Location / device": settings.aws.region,
            },
        ],
    )
    st.info(
        "Before changing models or retrieval quality, investigate one-time warm-up, batched "
        "candidate scoring, configured device, fused candidate count, and passage length."
    )
    st.subheader("Bounded process-local estimated cost summary")
    today = datetime.now(UTC).date()
    daily_records = (
        [item for item in records.completed() if item.created_at.date() == today]
        if records is not None
        else []
    )
    known_daily = Decimal(0)
    unavailable_daily = 0
    for item in daily_records:
        estimate = estimate_generation_cost(
            item.token_usage,
            load_pricing_catalog(),
            provider="aws-bedrock",
            model_id=item.model_id,
            region=item.region,
            pricing_mode="on-demand-standard",
            as_of=today,
        )
        if estimate.available and estimate.estimated_api_cost is not None:
            known_daily += estimate.estimated_api_cost
        else:
            unavailable_daily += 1
    cost_left, cost_middle, cost_right = st.columns(3)
    cost_left.metric("Requests today", len(daily_records))
    cost_middle.metric("Known generation estimate", format_usd(known_daily))
    cost_right.metric("Cost unavailable", unavailable_daily)
    st.caption("Process-local bounded request store only; not a complete AWS invoice.")
    latest_batch = st.session_state.get("latest_batch_execution")
    if isinstance(latest_batch, DatasetEvaluationExecution):
        st.markdown("**Latest evaluation run cost summary**")
        _render_report_cost_total(
            latest_batch.report,
            generation_region=latest_batch.region,
        )
    else:
        st.caption("No completed evaluation run exists in this session.")
    if st.button("Warm embedding and Qwen reranker"):
        messages: list[str] = []
        try:
            warm_up_retrieval_service(
                runtime.retrieval.retrieval_service,
                progress=messages.append,
            )
            st.success("Warm-up completed.")
        except Exception as error:
            st.error(f"Warm-up failed ({type(error).__name__}).")
        for message in messages:
            st.caption(message)
    if st.button("Check MLflow"):
        status = inspect_mlflow(settings.mlflow)
        (st.success if status.available else st.warning)(status.message)


def _operations_models_tab() -> None:
    settings, _service, _runtime_value, records = _runtime()
    if settings is None:
        return
    render_arrow_dataframe(
        st,
        [
            {
                "Role": "Embedding",
                "Model": settings.models.embedding.model,
                "Region / device": settings.models.embedding.device,
            },
            {
                "Role": "Reranker",
                "Model": settings.models.reranker.model,
                "Region / device": settings.models.reranker.device,
            },
            {
                "Role": "Generation",
                "Model": settings.models.generation.model,
                "Region / device": settings.aws.region,
            },
            {
                "Role": "Runtime policy",
                "Model": settings.quality_policy.default_profile.value,
                "Region / device": "Configured",
            },
        ],
    )
    budget_left, request_right = st.columns(2)
    budget_left.metric(
        "Monitoring daily budget",
        format_usd(Decimal(str(settings.online_monitoring.daily_budget_usd))),
    )
    request_right.metric(
        "Process-local completed requests", len(records.completed()) if records else 0
    )
    st.caption("Model identities and budgets come from validated configuration, not UI constants.")


def _operations_jobs_tab() -> None:
    st.info(
        "No background worker is started by this UI. Evaluation, index mutation, and drift "
        "assessment run only through their explicit actions or documented CLI commands."
    )


def _operations_privacy_tab() -> None:
    settings, _service, _runtime_value, _records = _runtime()
    if settings is None:
        return
    retention = settings.private_workbench.judge_call_debug_retention_hours
    render_arrow_dataframe(
        st,
        [
            {
                "Control": "Ordinary trace content",
                "Effective value": TraceContentPolicy.METADATA_ONLY.value,
            },
            {"Control": "Judge I/O default", "Effective value": "Disabled"},
            {
                "Control": "Judge I/O retention",
                "Effective value": f"{retention} hours",
            },
            {"Control": "Eligible debug content", "Effective value": "Reviewed PUBLIC cases only"},
        ],
    )
    with st.expander("Advanced retention details"):
        st.code(str(settings.private_workbench.judge_call_debug_directory))
    st.caption("Expired artifacts remain deletable from Individual Judge Calls.")


def _operations_configuration_tab() -> None:
    settings, _service, runtime, _records = _runtime()
    if settings is None or runtime is None:
        return
    payload = {
        "runtime_profile": settings.runtime.profile,
        "generation": runtime.retrieval.generation,
        "qdrant_collection": runtime.retrieval.qdrant_collection,
        "quality_policy": settings.quality_policy.model_dump(mode="json"),
        "online_monitoring": settings.online_monitoring.model_dump(mode="json"),
        "mlflow_tracking_uri": settings.mlflow.tracking_uri,
    }
    st.info("Effective non-secret configuration is available below for operator review.")
    with st.expander("Advanced configuration"):
        st.json(payload)


def _duration(value: float | None) -> str:
    return "not recorded" if value is None else f"{value / 1000:.2f} s"


def _quality_check_row(check: QualityCheckResult) -> dict[str, object]:
    return {
        "Check class": (
            "STRICT semantic judge"
            if check.check == "reference_free_groundedness"
            else "STANDARD deterministic"
        ),
        **check.model_dump(mode="json"),
    }


pages: list[Any] = [
    st.Page(chat_page, title="Chat", icon="💬", default=True),
    st.Page(inspector_page, title="Request Inspector", icon="🔎"),
    st.Page(evaluations_page, title="Evaluations", icon="🧪"),
    st.Page(lineage_page, title="Lineage & Index", icon="🧬"),
    st.Page(drift_page, title="Drift & Regression", icon="📉"),
    st.Page(operations_page, title="Operations", icon="🛠️"),
]
st.sidebar.caption("Private operator surface · use loopback or authenticated private access only.")
st.navigation(pages).run()
