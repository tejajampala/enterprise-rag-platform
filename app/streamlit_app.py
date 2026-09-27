"""Public, managed-only Streamlit showcase for Consumer Finance Intelligence."""

from __future__ import annotations

import logging
from datetime import date
from enum import StrEnum
from time import monotonic
from uuid import uuid4

import streamlit as st

from consumer_finance_rag.config import AppSettings, load_settings
from consumer_finance_rag.costs import (
    CostEstimate,
    combine_cost_estimates,
    estimate_generation_cost,
    load_pricing_catalog,
)
from consumer_finance_rag.evaluation.managed_rag import (
    JudgeStageSummary,
    ManagedRAGEvaluationReport,
)
from consumer_finance_rag.evaluation.reporting import load_report
from consumer_finance_rag.factory import (
    ApplicationRuntime,
    build_application_runtime,
    build_runtime_retrieval,
)
from consumer_finance_rag.observability import NoOpTracer, TraceContentPolicy
from consumer_finance_rag.presentation import cited_only
from consumer_finance_rag.public_showcase import (
    PublicAnswer,
    PublicInferenceGate,
    PublicRAGService,
    PublicRequestError,
    RequestTelemetryCollector,
    load_managed_showcase_report,
)
from consumer_finance_rag.usage_presentation import format_usd

st.set_page_config(
    page_title="Consumer Finance Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.set_option("client.showErrorDetails", "none")

st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(145deg, #f7fafc 0%, #edf5f3 100%); }
    .block-container { max-width: 860px; padding-top: 2rem; }
    .hero {
        padding: 2.2rem; border-radius: 22px; color: #f8fafc;
        background: linear-gradient(125deg, #12372a 0%, #1f6650 58%, #267f68 100%);
        box-shadow: 0 18px 45px rgba(18,55,42,.18); margin-bottom: 1.25rem;
    }
    .hero h1 { margin: 0 0 .45rem 0; font-size: clamp(2rem, 5vw, 3.6rem); }
    .hero p { max-width: 760px; font-size: 1.08rem; margin-bottom: 0; color: #e7f4ef; }
    .eyebrow { letter-spacing: .12em; text-transform: uppercase; font-weight: 700;
               font-size: .78rem; color: #bce8d7; }
    .safe-card { padding: 1rem 1.1rem; border: 1px solid #cbded7; border-radius: 14px;
                 background: rgba(255,255,255,.78); min-height: 126px; }
    .safe-card h3 { margin-top: 0; color: #174a3a; }
    .status-pill { display: inline-block; padding: .2rem .65rem; border-radius: 999px;
                   background: #dff3ea; color: #12543f; font-weight: 700; font-size: .82rem; }
    .muted { color: #4b635a; }
    @media (max-width: 700px) {
      .block-container { padding: 1rem .75rem; }
      .hero { padding: 1.4rem; border-radius: 16px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

SUGGESTIONS = (
    "What should I compare when choosing a credit card?",
    "How can I dispute an error on my credit report?",
    "When should I receive a mortgage Closing Disclosure?",
    "What information must a debt collector give me?",
)

LOGGER = logging.getLogger(__name__)


class PublicRequestState(StrEnum):
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    ABSTAINED = "ABSTAINED"
    ERROR = "ERROR"


def _request_state() -> PublicRequestState:
    raw = st.session_state.setdefault("public_request_state", PublicRequestState.IDLE.value)
    try:
        return PublicRequestState(str(raw))
    except ValueError:
        st.session_state.public_request_state = PublicRequestState.IDLE.value
        return PublicRequestState.IDLE


def _transition_request(
    state: PublicRequestState,
    *,
    request_id: str | None = None,
) -> None:
    previous = _request_state()
    st.session_state.public_request_state = state.value
    started = st.session_state.get("public_request_started_at")
    elapsed_ms = max(0.0, (monotonic() - started) * 1000.0) if isinstance(started, float) else None
    LOGGER.info(
        "public_request_transition submission_id=%s request_id=%s from=%s to=%s elapsed_ms=%s",
        st.session_state.get("public_submission_id", "unavailable"),
        request_id or "unavailable",
        previous.value,
        state.value,
        f"{elapsed_ms:.1f}" if elapsed_ms is not None else "unavailable",
    )


def _disclaimer() -> None:
    st.caption(
        "Educational demonstration only. Answers come from public sources and are not legal "
        "or financial advice."
    )


@st.cache_resource
def get_public_runtime() -> tuple[PublicRAGService, ApplicationRuntime]:
    """Build one process-wide managed runtime and one process-wide billing gate."""

    settings = load_settings()
    telemetry = RequestTelemetryCollector()
    retrieval = build_runtime_retrieval(
        settings,
        tracer=NoOpTracer(),
        telemetry_sink=telemetry,
    )
    application = build_application_runtime(
        settings,
        retrieval_runtime=retrieval,
        telemetry_sink=telemetry,
        trace_content_policy=TraceContentPolicy.METADATA_ONLY,
    )
    service = PublicRAGService(
        settings,
        application,
        telemetry,
        PublicInferenceGate(settings.public_showcase),
    )
    return service, application


def _startup(settings: AppSettings) -> tuple[PublicRAGService | None, ApplicationRuntime | None]:
    if not settings.public_showcase.enabled:
        return None, None
    try:
        return get_public_runtime()
    except Exception:
        return None, None


def _hero(*, compact: bool = False) -> None:
    st.title("Consumer Finance Assistant")
    if not compact:
        st.write(
            "Ask about U.S. consumer-finance topics and get an answer based on reviewed "
            "public information. Sources are shown with each supported answer."
        )


def _overview() -> None:
    st.subheader("Why this project exists")
    st.write(
        "A production-shaped retrieval-augmented generation system that makes lineage, "
        "authorization, privacy, observability, and evaluation visible—not just the final answer."
    )
    columns = st.columns(3)
    cards = (
        (
            "Evidence first",
            "Dense HNSW and sparse BM25 candidates are fused, reranked, and "
            "converted into cited context.",
        ),
        (
            "Governed by default",
            "Anonymous requests can retrieve PUBLIC payloads only. Restricted groups "
            "are never accepted.",
        ),
        (
            "Operationally inspectable",
            "Every request carries its own ID, pinned generation, stage timings, and "
            "optional MLflow trace ID.",
        ),
    )
    for column, (title, body) in zip(columns, cards, strict=True):
        column.markdown(
            f'<div class="safe-card"><h3>{title}</h3><p>{body}</p></div>',
            unsafe_allow_html=True,
        )
    st.info(
        "Educational demonstration only. Answers are generated from public source material and "
        "are not legal or financial advice."
    )


def _question_picker() -> None:
    st.markdown("#### Try a guided question")
    columns = st.columns(2)
    for index, suggestion in enumerate(SUGGESTIONS):
        if columns[index % 2].button(
            suggestion,
            key=f"suggestion-{index}",
            width="stretch",
        ):
            st.session_state.public_question = suggestion


def _answer_panel(answer: PublicAnswer, settings: AppSettings) -> None:
    st.markdown("### Answer")
    st.markdown(answer.answer)
    st.markdown("#### Sources")
    citations = cited_only(answer.answer, answer.citations)
    for citation in citations:
        with st.container(border=True):
            st.markdown(f"**[{citation.citation_id}] {citation.title}**")
            if citation.source_url:
                st.link_button("Open source", citation.source_url)
    if not citations:
        st.caption("No source citation was explicitly referenced in the generated answer.")
    cost = _public_cost_summary(answer, settings)
    usage = answer.token_usage or {}
    st.caption(
        f"Estimated request API cost: {cost['estimated_request_api_cost']} · "
        f"Input tokens: {usage.get('input_tokens', 'Unavailable')} · "
        f"Output tokens: {usage.get('output_tokens', 'Unavailable')}"
    )
    with st.expander("About this answer"):
        st.caption(
            "The answer used PUBLIC-only retrieval, deterministic privacy handling, and a "
            "pre-delivery quality policy. Costs are estimates, not billed AWS totals."
        )
        st.markdown(
            f"- Generation estimate: {cost['estimated_generation_api_cost']}\n"
            f"- Blocking judge estimate: {cost['estimated_blocking_judge_api_cost']}\n"
            f"- Quality decision: {answer.quality_decision.status.value}"
        )


def _public_cost_summary(answer: PublicAnswer, settings: AppSettings) -> dict[str, str]:
    generation = estimate_generation_cost(
        answer.token_usage,
        load_pricing_catalog(),
        provider="aws-bedrock",
        model_id=answer.models.generation_model,
        region=settings.aws.region,
        pricing_mode="on-demand-standard",
        as_of=date.today(),
    )
    components: dict[str, CostEstimate | None] = {"generation": generation}
    quality = answer.quality_decision
    if quality.blocking_judge_invoked:
        judge = (
            CostEstimate(
                available=True,
                estimated_api_cost=quality.estimated_judge_cost_usd,
                currency="USD",
            )
            if quality.estimated_judge_cost_usd is not None
            else CostEstimate(available=False, reason="Runtime judge cost unavailable")
        )
        components["blocking groundedness judge"] = judge
        judge_label = format_usd(judge.estimated_api_cost)
    else:
        judge_label = "Not applicable"
    combined = combine_cost_estimates(components)
    has_known_cost = any(
        estimate is not None
        and estimate.available
        and estimate.estimated_api_cost is not None
        and estimate.currency == "USD"
        for estimate in components.values()
    )
    total_label = (
        format_usd(combined.known_estimated_cost)
        if combined.complete
        else f"Known subtotal {format_usd(combined.known_estimated_cost)}; total unavailable"
        if has_known_cost
        else "Cost unavailable"
    )
    return {
        "estimated_generation_api_cost": format_usd(generation.estimated_api_cost),
        "estimated_blocking_judge_api_cost": judge_label,
        "estimated_request_api_cost": total_label,
    }


def _ask(
    settings: AppSettings,
    service: PublicRAGService | None,
    runtime: ApplicationRuntime | None,
) -> None:
    state = _request_state()
    current = st.session_state.get("public_answer")
    if state in {
        PublicRequestState.COMPLETED,
        PublicRequestState.ABSTAINED,
        PublicRequestState.ERROR,
    }:
        _hero(compact=True)
        _disclaimer()
        submitted = st.session_state.get("public_submitted_question")
        if isinstance(submitted, str):
            st.caption("Your question")
            st.markdown(f"**{submitted}**")
        if state is PublicRequestState.COMPLETED and isinstance(current, PublicAnswer):
            _answer_panel(current, settings)
        elif state is PublicRequestState.ABSTAINED:
            st.markdown("### Unable to answer from available sources")
            st.write(
                "The reviewed public material retrieved for this question was not "
                "relevant enough to support an answer. No candidate answer was shown."
            )
        else:
            st.markdown("### Request could not be completed")
            st.write("Please try again later or ask another consumer-finance question.")
        if st.button("Ask another question", type="primary", width="stretch"):
            _reset_public_request()
            st.rerun()
        return

    if state is PublicRequestState.RUNNING:
        _hero(compact=True)
        _disclaimer()
        submitted = st.session_state.get("public_submitted_question")
        if isinstance(submitted, str):
            st.caption("Your question")
            st.markdown(f"**{submitted}**")
        st.info(
            "Request in progress: searching reviewed sources and applying the configured "
            "quality policy."
        )
        _run_public_request(service)
        return

    _hero()
    _disclaimer()
    _question_picker()
    question = st.text_area(
        "Ask a consumer-finance question",
        key="public_question",
        placeholder="For example: How do I dispute a credit-report error?",
        height=110,
        max_chars=settings.public_showcase.max_question_characters,
    )
    remaining = settings.public_showcase.max_question_characters - len(question)
    st.caption(f"{remaining} characters remaining · anonymous PUBLIC-only request")
    st.warning(
        "Do not enter account numbers, Social Security numbers, or other sensitive personal "
        "information."
    )

    ready = service is not None and runtime is not None
    if not settings.public_showcase.enabled:
        st.warning("The public showcase is disabled by the operator.")
    elif not settings.public_showcase.inference_enabled:
        st.warning("Answer generation is paused by the operator kill switch.")
    elif not ready:
        st.error("The managed serving runtime is unavailable. No request was sent to Bedrock.")

    if st.button(
        "Get answer",
        type="primary",
        width="stretch",
        disabled=not ready or not settings.public_showcase.inference_enabled,
    ):
        if _request_state() is not PublicRequestState.IDLE:
            return
        _clear_public_result()
        st.session_state.public_submitted_question = question.strip()
        st.session_state.public_submission_id = str(uuid4())
        st.session_state.public_request_started_at = monotonic()
        st.session_state.public_invocation_started = False
        _transition_request(PublicRequestState.RUNNING)
        st.rerun()


def _run_public_request(service: PublicRAGService | None) -> None:
    """Run one accepted submission once, then force one terminal-state rerun."""

    if bool(st.session_state.get("public_invocation_started")):
        st.session_state.public_failure = {
            "code": "request_interrupted",
            "message": "The in-progress request was interrupted.",
        }
        _transition_request(PublicRequestState.ERROR)
        st.rerun()
        return

    st.session_state.public_invocation_started = True
    submitted = st.session_state.get("public_submitted_question")
    try:
        if service is None or not isinstance(submitted, str):
            raise RuntimeError("Public serving runtime is unavailable")
        with st.spinner("Searching reviewed sources and preparing an answer…"):
            answer = service.answer(submitted)
        st.session_state.public_answer = answer
        st.session_state.public_result_request_id = answer.trace.request_id
        _transition_request(
            PublicRequestState.COMPLETED,
            request_id=answer.trace.request_id,
        )
    except PublicRequestError as error:
        st.session_state.public_failure = {
            "code": error.code,
            "message": error.public_message,
        }
        if error.request_id is not None:
            st.session_state.public_result_request_id = error.request_id
        terminal = (
            PublicRequestState.ABSTAINED
            if error.code == "insufficient_relevant_evidence"
            else PublicRequestState.ERROR
        )
        _transition_request(terminal, request_id=error.request_id)
    except Exception:
        st.session_state.public_failure = {"code": "request_failed"}
        _transition_request(PublicRequestState.ERROR)
    st.rerun()


def _clear_public_result() -> None:
    for key in (
        "public_answer",
        "public_failure",
        "public_result_request_id",
    ):
        st.session_state.pop(key, None)


def _reset_public_request() -> None:
    _clear_public_result()
    for key in (
        "public_submitted_question",
        "public_submission_id",
        "public_request_started_at",
        "public_invocation_started",
    ):
        st.session_state.pop(key, None)
    st.session_state.public_request_state = PublicRequestState.IDLE.value
    st.session_state.public_question = ""


def _architecture() -> None:
    st.subheader("Compact serving architecture")
    st.graphviz_chart(
        """
        digraph serving {
          rankdir=LR; bgcolor="transparent";
          node [shape=box style="rounded,filled" fillcolor="#f7fffb" color="#267f68"];
          Visitor -> Streamlit -> "PublicRAGService" -> "ManagedRetrievalRuntime";
          "ManagedRetrievalRuntime" -> PostgreSQL [label="pin ACTIVE"];
          "ManagedRetrievalRuntime" -> Qdrant [label="PUBLIC filter"];
          Qdrant -> "PII REDACT" -> Bedrock -> Citations;
          "PublicRAGService" -> MLflow [style=dashed label="metadata only"];
        }
        """,
        width="stretch",
    )
    st.caption(
        "PostgreSQL owns immutable lineage and the ACTIVE generation pointer. Qdrant serves the "
        "pinned PUBLIC generation. The public tunnel exposes only this Streamlit process."
    )


def _metric_row(values: tuple[tuple[str, float | int], ...]) -> None:
    columns = st.columns(len(values))
    for column, (label, value) in zip(columns, values, strict=True):
        rendered = f"{value:.3f}" if isinstance(value, float) else str(value)
        column.metric(label, rendered)


def _judge_section(label: str, summary: JudgeStageSummary) -> None:
    st.markdown(f"#### {label}")
    st.markdown(f'<span class="status-pill">{summary.status}</span>', unsafe_allow_html=True)
    if summary.status == "COMPLETED" and summary.aggregate_metrics:
        _metric_row(
            tuple(
                (name.replace("_", " ").title(), value)
                for name, value in summary.aggregate_metrics.items()
            )
        )
    else:
        status_messages = {
            "NOT_REQUESTED": "This judge was not explicitly requested for the saved report.",
            "UNAVAILABLE": "This judge was unavailable for the saved report.",
            "FAILED": "This judge did not complete for the saved report.",
        }
        st.caption(status_messages.get(summary.status, "No judge result is available."))
    for metric in summary.skipped_metrics:
        st.caption(f"{metric}: skipped in this report")


def _managed_evaluation(report: ManagedRAGEvaluationReport | None) -> None:
    st.subheader("Managed answer evaluation")
    st.caption(
        "Loaded from a previously generated metadata-only report. Opening this tab never runs "
        "Bedrock, MLflow judges, or RAGAS."
    )
    if report is None:
        st.info(
            "No managed report is available. Run `uv run python scripts/evaluate_managed_rag.py "
            "--current` for retrieval-only evaluation. Generation and judges require "
            "explicit flags."
        )
        return

    st.markdown(
        f"**{report.dataset_name} v{report.dataset_version}** · "
        f"generation `g{report.generation:06d}` · {report.case_count} reviewed PUBLIC cases"
    )
    st.markdown("#### Deterministic retrieval metrics")
    _metric_row(
        (
            (f"Recall@{report.retrieval_quality.k}", report.retrieval_quality.recall_at_k),
            ("MRR", report.retrieval_quality.mrr),
            ("Hit rate", report.retrieval_quality.hit_rate),
            ("Missed cases", report.retrieval_quality.missed_case_count),
        )
    )

    generation = report.answer_quality.generation
    st.markdown("#### Generated-answer execution")
    st.markdown(f'<span class="status-pill">{generation.status}</span>', unsafe_allow_html=True)
    if generation.status == "COMPLETED":
        values: list[tuple[str, float | int]] = [
            ("Generated cases", generation.generated_case_count)
        ]
        if generation.mean_latency_ms is not None:
            values.append(("Mean generation latency (ms)", generation.mean_latency_ms))
        if generation.token_usage.total_tokens is not None:
            values.append(("Total tokens", generation.token_usage.total_tokens))
        _metric_row(tuple(values))
    else:
        st.caption("Answer generation was not run or did not complete for this saved report.")

    left, right = st.columns(2)
    with left:
        _judge_section("MLflow GenAI judges", report.answer_quality.mlflow_genai)
    with right:
        _judge_section("RAGAS metrics", report.answer_quality.ragas)

    st.markdown("#### Request-level operational telemetry")
    st.caption(
        "This is not an evaluation score. It is generated per visitor request and appears only "
        "after an answer in the Ask tab."
    )


def _phase2_evaluation(settings: AppSettings) -> None:
    with st.expander("Phase 2 evaluation reports (separate legacy/evaluation profile)"):
        retrieval = load_report(settings.mlflow.report_directory / "retrieval_summary.json")
        rag = load_report(settings.mlflow.report_directory / "rag_summary.json")
        if retrieval is None:
            st.info("Phase 2 deterministic retrieval report not available.")
        else:
            st.write(
                f"Dataset: {retrieval.get('dataset_name')} v{retrieval.get('dataset_version')} · "
                f"{retrieval.get('case_count')} cases"
            )
        if rag is None:
            st.info("Phase 2 generated-answer report not available.")
        else:
            st.caption("Phase 2 generated-answer report is available locally to the demo process.")


def _evaluation(
    settings: AppSettings,
    runtime: ApplicationRuntime | None,
) -> None:
    generation = runtime.retrieval.generation if runtime is not None else None
    report = load_managed_showcase_report(
        settings.mlflow.report_directory,
        generation=generation,
    )
    _managed_evaluation(report)
    _phase2_evaluation(settings)
    st.info(
        "A request trace explains one execution. An evaluation experiment compares a fixed set "
        "of cases. Neither is a substitute for production monitoring or human review."
    )


def _limitations(settings: AppSettings) -> None:
    st.subheader("Safety and operating limits")
    st.markdown(
        f"""
        - Anonymous access is forced to an empty identity context: no principals or groups.
        - Qdrant receives a PUBLIC-only filter and returned payloads are validated again.
        - Retrieved context and the displayed answer pass through deterministic REDACT handling.
        - Questions are capped at **{settings.public_showcase.max_question_characters} characters**.
        - The process admits at most
          **{settings.public_showcase.max_concurrent_requests} concurrent**
          Bedrock requests and **{settings.public_showcase.max_requests_per_window} requests per
          {settings.public_showcase.rate_window_seconds}-second window**.
        - The process-wide limiter resets on restart; it is a bounded demo control, not a
          distributed production quota service.
        - MLflow, PostgreSQL, Qdrant, and the operations dashboard remain bound to localhost.
        - Cloudflare Quick Tunnels are temporary development/demo endpoints, not production hosting.
        """
    )


def main() -> None:
    try:
        settings = load_settings()
    except Exception:
        st.error(
            "The showcase configuration is unavailable. The operator can inspect the private "
            "preflight output for details."
        )
        st.stop()
        return

    service, runtime = _startup(settings)
    _ask(settings, service, runtime)


if __name__ == "__main__":
    main()
