"""Public, managed-only Streamlit showcase for Consumer Finance Intelligence."""

from __future__ import annotations

import pandas as pd  # type: ignore[import-untyped]
import streamlit as st

from consumer_finance_rag.config import AppSettings, load_settings
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
from consumer_finance_rag.public_showcase import (
    PublicAnswer,
    PublicInferenceGate,
    PublicRAGService,
    PublicRequestError,
    RequestTelemetryCollector,
    load_managed_showcase_report,
)

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
    .block-container { max-width: 1180px; padding-top: 2rem; }
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


def _hero(settings: AppSettings, application: ApplicationRuntime | None) -> None:
    runtime = application.retrieval if application is not None else None
    runtime_metadata = dict(runtime.display_metadata()) if runtime is not None else {}
    generation = runtime_metadata.get("Generation", "unavailable")
    st.markdown(
        f"""
        <section class="hero">
          <div class="eyebrow">Managed hybrid RAG · local public showcase</div>
          <h1>Consumer Finance Intelligence</h1>
          <p>Evidence-grounded answers over an immutable managed index, with anonymous
          PUBLIC-only retrieval, deterministic PII redaction, request-bound observability,
          and source citations. Serving generation: <strong>{generation}</strong>.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    cols = st.columns(4)
    values = (
        ("Serving profile", runtime_metadata.get("Retrieval profile", settings.runtime.profile)),
        ("Corpus", settings.public_showcase.approved_corpus_key),
        ("Generation", generation),
        ("Inference", "enabled" if settings.public_showcase.inference_enabled else "paused"),
    )
    for column, (label, value) in zip(cols, values, strict=True):
        column.metric(label, value)


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


def _answer_panel(answer: PublicAnswer) -> None:
    st.markdown("### Grounded answer")
    st.markdown(answer.answer)
    st.markdown("#### Sources")
    for citation in answer.citations:
        if citation.source_url:
            st.markdown(
                f"**[{citation.citation_id}]** [{citation.title}]({citation.source_url})  "
                f"\n`{citation.chunk_id}`"
            )
        else:
            st.markdown(
                f"**[{citation.citation_id}]** {citation.title}  \n`{citation.chunk_id}`"
            )

    with st.expander("Retrieval-stage timeline", expanded=True):
        if answer.trace.stages:
            frame = pd.DataFrame(
                [
                    {
                        "Stage": stage.stage,
                        "Latency (ms)": stage.latency_ms,
                        "Candidates": stage.candidate_count,
                    }
                    for stage in answer.trace.stages
                ]
            )
            st.dataframe(frame, hide_index=True, width="stretch")
        else:
            st.info("Request telemetry was unavailable; no timing is inferred.")

    with st.expander("Sanitized request trace"):
        st.caption(
            "This allowlist is request-specific. It contains no question, context text, "
            "answer text, principal, group, or matched PII value."
        )
        left, right = st.columns(2)
        left.code(answer.trace.request_id, language=None)
        left.caption("Request ID")
        right.code(answer.trace.mlflow_trace_id or "unavailable", language=None)
        right.caption("MLflow trace ID (operator UI remains private)")
        st.json(answer.trace.model_dump(mode="json"), expanded=False)

    with st.expander("Models and immutable serving identity"):
        st.markdown(f"- **Generation:** `g{answer.trace.generation:06d}`")
        st.markdown(f"- **Index state:** `{answer.trace.index_state_id}`")
        st.markdown(f"- **Collection:** `{answer.trace.qdrant_collection}`")
        st.markdown(f"- **Embedding:** `{answer.models.embedding_model}`")
        st.markdown(f"- **Reranker:** `{answer.models.reranker_model}`")
        st.markdown(f"- **Bedrock generation:** `{answer.models.generation_model}`")
        if answer.token_usage is None:
            st.caption("Provider token usage unavailable.")
        else:
            st.caption(
                "Token usage: "
                f"{answer.token_usage['input_tokens']} input · "
                f"{answer.token_usage['output_tokens']} output · "
                f"{answer.token_usage['total_tokens']} total"
            )


def _ask(
    settings: AppSettings,
    service: PublicRAGService | None,
    runtime: ApplicationRuntime | None,
) -> None:
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

    ready = service is not None and runtime is not None
    if not settings.public_showcase.enabled:
        st.warning("The public showcase is disabled by the operator.")
    elif not settings.public_showcase.inference_enabled:
        st.warning("Answer generation is paused by the operator kill switch.")
    elif not ready:
        st.error(
            "The managed serving runtime is unavailable. No request was sent to Bedrock."
        )

    if st.button(
        "Generate grounded answer",
        type="primary",
        width="stretch",
        disabled=not ready or not settings.public_showcase.inference_enabled,
    ):
        try:
            with st.spinner("Searching the pinned managed generation and generating an answer…"):
                st.session_state.public_answer = service.answer(question) if service else None
        except PublicRequestError as error:
            st.error(error.public_message)
        except Exception:
            st.error("The request could not be completed. Please try again later.")

    current = st.session_state.get("public_answer")
    if isinstance(current, PublicAnswer):
        _answer_panel(current)


def _architecture() -> None:
    st.subheader("Serving architecture")
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
    st.subheader("Control plane and immutable generations")
    st.graphviz_chart(
        """
        digraph lifecycle {
          rankdir=LR; bgcolor="transparent";
          node [shape=box style="rounded,filled" fillcolor="#f7fffb" color="#267f68"];
          Source -> Version -> Chunks -> "Derived ACL/PII" -> "Build generation";
          "Build generation" -> "Qdrant gN" -> ACTIVE;
          ACTIVE -> "Pinned runtime";
          "Old runtime" -> "Qdrant gN-1" [style=dashed];
        }
        """,
        width="stretch",
    )
    st.caption(
        "PostgreSQL is the lineage/control plane. Qdrant is the immutable serving data plane. "
        "The public tunnel exposes only this Streamlit process."
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
        st.caption(
            "Answer generation was not run or did not complete for this saved report."
        )

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


try:
    settings = load_settings()
except Exception:
    st.error(
        "The showcase configuration is unavailable. The operator can inspect the private "
        "preflight output for details."
    )
    st.stop()

service, runtime = _startup(settings)
_hero(settings, runtime)

overview_tab, ask_tab, architecture_tab, evaluation_tab, limitations_tab = st.tabs(
    ["Overview", "Ask", "Architecture", "Evaluation", "Safety & limits"]
)
with overview_tab:
    _overview()
with ask_tab:
    _ask(settings, service, runtime)
with architecture_tab:
    _architecture()
with evaluation_tab:
    _evaluation(settings, runtime)
with limitations_tab:
    _limitations(settings)
