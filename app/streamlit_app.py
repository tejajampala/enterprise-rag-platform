"""Streamlit portfolio interface for Consumer Finance Intelligence."""

from __future__ import annotations

from typing import Any

import pandas as pd  # type: ignore[import-untyped]
import streamlit as st

from consumer_finance_rag.bedrock import BedrockConfigurationError
from consumer_finance_rag.config import AppSettings, load_settings
from consumer_finance_rag.evaluation.reporting import load_report
from consumer_finance_rag.factory import (
    ApplicationRuntime,
    RetrievalRuntime,
    build_application_runtime,
    build_runtime_retrieval,
)
from consumer_finance_rag.lineage.database import DatabaseConfigurationError
from consumer_finance_rag.lineage.managed_retrieval import ManagedRetrievalError
from consumer_finance_rag.observability import NoOpTracer

st.set_page_config(
    page_title="Consumer Finance Intelligence",
    page_icon="🏦",
    layout="wide",
)

st.title("Consumer Finance Intelligence")
st.caption("An evidence-grounded RAG system using public U.S. consumer-finance information.")

SUGGESTIONS = [
    "What should I compare when choosing a credit card?",
    "How can I dispute an error on my credit report?",
    "What documents should I review before closing on a mortgage?",
    "What information must a debt collector give me?",
]


@st.cache_resource
def get_retrieval_runtime() -> RetrievalRuntime:
    return build_runtime_retrieval(load_settings(), tracer=NoOpTracer())


@st.cache_resource
def get_application_runtime() -> ApplicationRuntime:
    return build_application_runtime(
        load_settings(),
        retrieval_runtime=get_retrieval_runtime(),
    )


def _display_runtime_profile(
    settings: AppSettings,
    runtime: RetrievalRuntime | None,
) -> None:
    metadata = (
        runtime.display_metadata()
        if runtime is not None
        else (("Retrieval profile", settings.runtime.profile),)
    )
    st.caption(" · ".join(f"{label}: `{value}`" for label, value in metadata))


def _display_chat(settings: AppSettings, runtime: RetrievalRuntime | None) -> None:
    with st.expander("Suggested questions", expanded=True):
        for suggestion in SUGGESTIONS:
            st.markdown(f"- {suggestion}")

    question = st.text_area(
        "Ask a consumer-finance question",
        placeholder="For example: How do I dispute a credit-report error?",
        height=100,
    )

    if not st.button("Ask", type="primary", use_container_width=True):
        return
    if not question.strip():
        st.warning("Enter a question first.")
        return
    if not settings.models.generation.model:
        st.error(
            "AWS Bedrock generation is not configured. Set `BEDROCK_GENERATION_MODEL` "
            "to a model ID enabled in your AWS account, then restart the app."
        )
        return
    if runtime is None:
        st.error(
            f"The `{settings.runtime.profile}` retrieval profile did not start. "
            "Resolve the startup error shown above, then restart the app."
        )
        return

    try:
        with st.spinner("Retrieving and grounding an answer…"):
            response = get_application_runtime().pipeline.answer(question.strip())
        st.subheader("Answer")
        st.markdown(response.answer)
        st.subheader("Sources")
        for citation in response.citations:
            title = citation.document_title
            chunk = f"`{citation.chunk_id}`"
            if citation.source_url:
                st.markdown(
                    f"- **[{citation.citation_id}]** [{title}]({citation.source_url}) — {chunk}"
                )
            else:
                st.markdown(f"- **[{citation.citation_id}]** {title} — {chunk}")

        with st.expander("How this answer was generated"):
            retrieval = response.retrieval_metadata
            models = response.model_metadata
            st.markdown(f"**Retrieval strategy:** {retrieval.strategy}")
            st.markdown(f"**Dense candidates:** {retrieval.dense_candidates}")
            st.markdown(f"**Sparse candidates:** {retrieval.sparse_candidates}")
            st.markdown(f"**Candidates reranked:** {retrieval.reranked}")
            st.markdown(f"**Final contexts:** {retrieval.final_contexts}")
            st.markdown(f"**Embedding model:** `{models.embedding_model}`")
            st.markdown(f"**Reranker model:** `{models.reranker_model}`")
            st.markdown(f"**Generation model:** `{models.generation_model}`")
    except BedrockConfigurationError as error:
        st.error(str(error))
    except Exception as error:
        st.error(
            "The answer could not be generated. Confirm Qdrant is running, the corpus "
            "is indexed, and AWS credentials/model access are configured. "
            f"Details: {error}"
        )


def _retrieval_table(report: dict[str, Any]) -> pd.DataFrame | None:
    aggregate = report.get("aggregate")
    k_values = report.get("k_values")
    if not isinstance(aggregate, dict) or not isinstance(k_values, list) or not k_values:
        return None
    k = max(int(value) for value in k_values)
    rows: list[dict[str, str | float]] = []
    for stage in ("dense", "sparse", "fused", "reranked"):
        metrics = aggregate.get(stage)
        if not isinstance(metrics, dict):
            continue
        row: dict[str, str | float] = {"Stage": stage}
        for label, key in (
            (f"Recall@{k}", f"recall_at_{k}"),
            (f"MRR@{k}", f"mrr_at_{k}"),
            (f"NDCG@{k}", f"ndcg_at_{k}"),
            (f"MAP@{k}", f"map_at_{k}"),
        ):
            value = metrics.get(key)
            if isinstance(value, int | float):
                row[label] = float(value)
        rows.append(row)
    return pd.DataFrame(rows) if rows else None


def _display_evaluation(settings: AppSettings) -> None:
    report_dir = settings.mlflow.report_directory
    retrieval = load_report(report_dir / "retrieval_summary.json")
    rag = load_report(report_dir / "rag_summary.json")

    st.subheader("Retrieval Quality")
    if retrieval is None:
        st.info("Not run. Execute `uv run python scripts/evaluate_retrieval.py`.")
    else:
        table = _retrieval_table(retrieval)
        if table is None:
            st.info("The retrieval report does not contain stage metrics.")
        else:
            st.dataframe(table, hide_index=True, use_container_width=True)
        st.caption(
            f"Seed set: {retrieval.get('dataset_name')} v{retrieval.get('dataset_version')} — "
            f"{retrieval.get('case_count')} cases ({retrieval.get('dataset_status')})."
        )

    st.subheader("RAG Quality")
    quality = rag.get("rag") if isinstance(rag, dict) else None
    labels = {
        "correctness": "Correctness",
        "faithfulness": "Faithfulness",
        "retrieval_groundedness": "Retrieval groundedness",
        "response_groundedness": "Response groundedness",
        "relevance": "Relevance",
        "safety": "Safety",
    }
    columns = st.columns(3)
    for index, (key, label) in enumerate(labels.items()):
        value = quality.get(key) if isinstance(quality, dict) else None
        columns[index % 3].metric(label, "not run" if value is None else f"{float(value):.3f}")

    st.subheader("Operations")
    operations = rag.get("operations") if isinstance(rag, dict) else None
    operational_labels = {
        "trace_count": "Trace count",
        "mean_latency_ms": "Mean request latency (ms)",
        "input_tokens": "Input tokens",
        "output_tokens": "Output tokens",
        "total_tokens": "Total tokens",
        "cost_usd": "Cost (USD)",
    }
    operation_columns = st.columns(3)
    for index, (key, label) in enumerate(operational_labels.items()):
        value = operations.get(key) if isinstance(operations, dict) else None
        operation_columns[index % 3].metric(label, "unavailable" if value is None else value)
    st.markdown(
        f"Open the [local MLflow UI]({settings.mlflow.tracking_uri}) for Trace Explorer, "
        "span timing, run comparisons, and artifacts."
    )
    st.caption("Streamlit reads generated reports; it does not execute evaluation jobs.")


settings = load_settings()
runtime: RetrievalRuntime | None
try:
    runtime = get_retrieval_runtime()
except (DatabaseConfigurationError, ManagedRetrievalError, FileNotFoundError) as error:
    runtime = None
    st.error(
        f"Retrieval startup failed for `{settings.runtime.profile}` profile: {error}"
    )
except Exception as error:
    runtime = None
    st.error(
        f"The `{settings.runtime.profile}` retrieval profile could not start. "
        f"Details: {error}"
    )

_display_runtime_profile(settings, runtime)
chat_tab, evaluation_tab = st.tabs(["Ask", "Evaluation & Observability"])
with chat_tab:
    _display_chat(settings, runtime)
with evaluation_tab:
    _display_evaluation(settings)
