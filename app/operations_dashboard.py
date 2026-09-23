"""Read-only Streamlit dashboard for Phase 3 managed RAG operations."""

from __future__ import annotations

from contextlib import suppress

import streamlit as st
from qdrant_client import QdrantClient
from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from consumer_finance_rag.config import AppSettings, load_settings
from consumer_finance_rag.lineage.database import (
    create_database_engine,
    require_database_url,
)
from consumer_finance_rag.lineage.index_operations import (
    ManagedIndexOperationsService,
)
from consumer_finance_rag.operations_dashboard import (
    ComponentHealth,
    MLflowDashboardStatus,
    OperationsSnapshot,
    cleanup_plan_rows,
    collect_operations_snapshot,
    compare_quality_for_dashboard,
    comparison_summary,
    generation_history_rows,
    inspect_mlflow,
    load_quality_report_json,
    probe_component,
    quality_summary,
    restoration_provenance_rows,
    safe_failure,
    safe_identifier,
    validation_rows,
    validation_summary,
)
from consumer_finance_rag.retrieval_quality import RetrievalQualityThresholds

st.set_page_config(
    page_title="Managed RAG Operations",
    page_icon="🔎",
    layout="wide",
)


def _read_platform(
    settings: AppSettings,
    *,
    deep_validation: bool,
) -> tuple[ComponentHealth, ComponentHealth, OperationsSnapshot | None]:
    """Open short-lived read connections only after an explicit UI action."""

    engine: Engine | None = None
    client: QdrantClient | None = None
    postgres = ComponentHealth("PostgreSQL", False, "PostgreSQL has not been checked.")
    qdrant = ComponentHealth("Qdrant", False, "Qdrant has not been checked.")
    try:
        try:
            require_database_url(settings.database)
            engine = create_database_engine(settings.database)
            def probe_postgres() -> None:
                assert engine is not None
                with engine.connect() as connection:
                    connection.execute(text("SELECT 1"))

            postgres = probe_component("PostgreSQL", probe_postgres)
        except Exception as exc:
            postgres = ComponentHealth(
                "PostgreSQL",
                False,
                safe_failure("PostgreSQL", exc),
            )

        try:
            client = QdrantClient(url=settings.qdrant.url)
            qdrant = probe_component("Qdrant", client.get_collections)
        except Exception as exc:
            qdrant = ComponentHealth("Qdrant", False, safe_failure("Qdrant", exc))

        if not postgres.available or not qdrant.available or engine is None or client is None:
            return postgres, qdrant, None

        sessions = sessionmaker(bind=engine, class_=Session, expire_on_commit=False)
        service = ManagedIndexOperationsService(
            session_factory=sessions,
            qdrant_client=client,
            corpus_key=settings.managed_ingestion.corpus_key,
            settings=settings.managed_index,
        )
        snapshot = collect_operations_snapshot(
            service,
            corpus_key=settings.managed_ingestion.corpus_key,
            deep_validation=deep_validation,
            point_counter=lambda collection: int(
                client.count(collection_name=collection, exact=True).count
            ),
        )
        return postgres, qdrant, snapshot
    finally:
        if client is not None:
            with suppress(Exception):
                client.close()
        if engine is not None:
            with suppress(Exception):
                engine.dispose()


def _show_component(health: ComponentHealth) -> None:
    if health.component == "Not checked":
        st.info(health.message)
    elif health.available:
        st.success(health.message)
    else:
        st.error(health.message)


def _display_platform_health(
    settings: AppSettings,
    postgres: ComponentHealth,
    qdrant: ComponentHealth,
    snapshot: OperationsSnapshot | None,
) -> None:
    st.header("Platform health")
    connection_columns = st.columns(2)
    with connection_columns[0]:
        _show_component(postgres)
    with connection_columns[1]:
        _show_component(qdrant)

    active = snapshot.active if snapshot is not None else None
    validation = snapshot.validation if snapshot is not None else None
    point_count = snapshot.active_point_count if snapshot is not None else None
    summary = validation_summary(validation) if validation is not None else None
    metrics = st.columns(7)
    metrics[0].metric(
        "Managed corpus",
        safe_identifier(settings.managed_ingestion.corpus_key),
    )
    metrics[1].metric(
        "ACTIVE generation",
        "Unavailable" if active is None else f"g{active.generation:06d}",
    )
    metrics[2].metric(
        "ACTIVE collection",
        "Unavailable" if active is None else safe_identifier(active.qdrant_collection),
    )
    metrics[3].metric(
        "Documents",
        "Unavailable" if active is None else active.document_count,
    )
    metrics[4].metric(
        "Chunks / physical points",
        (
            "Unavailable"
            if active is None
            else (
                f"{active.chunk_count} / "
                f"{point_count if point_count is not None else 'Unavailable'}"
            )
        ),
    )
    metrics[5].metric(
        "ACTIVE validation",
        "Unavailable" if summary is None else summary["active_validation"],
    )
    metrics[6].metric(
        "Historical audit",
        "Unavailable" if summary is None else summary["historical_audit"],
    )
    if snapshot is not None:
        for error in (
            snapshot.status_error,
            snapshot.validation_error,
            snapshot.point_count_error,
        ):
            if error is not None:
                st.error(error)


def _display_generation_history(snapshot: OperationsSnapshot | None) -> None:
    st.header("Generation history")
    if snapshot is None or not snapshot.statuses:
        st.info("Generation history is unavailable until both data stores are reachable.")
        return
    st.dataframe(
        list(generation_history_rows(snapshot.statuses)),
        hide_index=True,
        use_container_width=True,
    )
    st.caption(
        "FAILED details are intentionally withheld; use secured operator logs or the CLI for "
        "diagnosis. ACTIVE is marked with a star."
    )


def _display_validation(snapshot: OperationsSnapshot | None) -> None:
    st.header("Generation integrity")
    if snapshot is None or snapshot.validation is None:
        st.info("No validation result is available.")
        return
    report = snapshot.validation
    if report.overall_status.value == "PASS":
        st.success("ACTIVE validation passed; historical audit has no findings.")
    elif report.overall_status.value == "PASS_WITH_WARNINGS":
        st.warning("ACTIVE validation passed; historical audit has warnings.")
    else:
        st.error("ACTIVE managed generation validation failed.")
    if report.issues:
        st.error("One or more corpus-level invariants failed.")
    st.dataframe(
        list(validation_rows(report)),
        hide_index=True,
        use_container_width=True,
    )


def _display_cleanup_and_restoration(snapshot: OperationsSnapshot | None) -> None:
    st.header("Cleanup and restoration visibility")
    if snapshot is None or snapshot.cleanup_plan is None:
        message = snapshot.cleanup_error if snapshot is not None else None
        st.info(message or "No cleanup plan is available.")
    else:
        plan = snapshot.cleanup_plan
        st.caption(
            f"Read-only plan from {plan.planned_at.isoformat()}; "
            f"{len(plan.eligible)} physical collections are cleanup-eligible."
        )
        st.dataframe(
            list(cleanup_plan_rows(plan)),
            hide_index=True,
            use_container_width=True,
        )
    provenance = (
        restoration_provenance_rows(snapshot.statuses) if snapshot is not None else ()
    )
    st.subheader("Restoration provenance")
    if provenance:
        st.dataframe(list(provenance), hide_index=True, use_container_width=True)
    else:
        st.info("No restored generation provenance is recorded.")
    st.caption(
        "Cleanup execution, restoration, publication, deletion, and status changes remain "
        "operator CLI actions and are not available here."
    )


def _display_quality() -> None:
    st.header("Retrieval quality")
    uploaded = st.file_uploader(
        "Upload one or two safe retrieval-quality-report-v1 JSON files",
        type=["json"],
        accept_multiple_files=True,
    )
    if not uploaded:
        st.info("Upload an existing safe report; the dashboard never runs retrieval evaluation.")
        return
    if len(uploaded) > 2:
        st.error("Provide at most two reports.")
        return
    reports = []
    for item in uploaded:
        try:
            reports.append(load_quality_report_json(item.getvalue()))
        except Exception:
            st.error("A supplied file is not a valid retrieval-quality-report-v1 report.")
            return
    for position, report in enumerate(reports, start=1):
        st.subheader(f"Report {position}")
        st.dataframe(
            [quality_summary(report)],
            hide_index=True,
            use_container_width=True,
        )
    if len(reports) != 2:
        return

    st.subheader("Generation comparison")
    threshold_columns = st.columns(2)
    max_recall_drop = threshold_columns[0].number_input(
        "Maximum Recall@K drop",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.01,
    )
    max_mrr_drop = threshold_columns[1].number_input(
        "Maximum MRR drop",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.01,
    )
    try:
        view = compare_quality_for_dashboard(
            reports[0],
            reports[1],
            RetrievalQualityThresholds(
                max_recall_drop=float(max_recall_drop),
                max_mrr_drop=float(max_mrr_drop),
            ),
        )
    except Exception:
        st.error("The two reports are not comparable; verify their case sets and K.")
        return
    st.dataframe(
        [comparison_summary(view)],
        hide_index=True,
        use_container_width=True,
    )
    comparison = view.comparison
    case_columns = st.columns(4)
    case_columns[0].write(
        {"Improved": [safe_identifier(item) for item in comparison.cases_improved]}
    )
    case_columns[1].write(
        {"Regressed": [safe_identifier(item) for item in comparison.cases_regressed]}
    )
    case_columns[2].write(
        {
            "Newly missed": [
                safe_identifier(item) for item in comparison.newly_missed_cases
            ]
        }
    )
    case_columns[3].write(
        {
            "Newly recovered": [
                safe_identifier(item) for item in comparison.newly_recovered_cases
            ]
        }
    )


def _display_mlflow(settings: AppSettings) -> None:
    st.header("MLflow")
    st.caption("This check is read-only and never creates an MLflow run.")
    if not settings.mlflow.enabled:
        st.info("MLflow is disabled by configuration.")
        return
    if st.button("Check MLflow availability"):
        st.session_state["operations_mlflow"] = inspect_mlflow(settings.mlflow)
    status: MLflowDashboardStatus | None = st.session_state.get("operations_mlflow")
    if status is None:
        st.info("MLflow has not been checked.")
    elif status.available:
        st.success(status.message)
        if status.link is not None:
            st.link_button("Open retrieval-quality experiment", status.link)
    else:
        st.warning(status.message)


settings = load_settings()

st.title("Managed RAG Operations")
st.warning(
    "Read-only operator view. Run locally or behind appropriate external authentication; "
    "this is not an authenticated production administration console."
)
st.caption(
    "No query text, source/chunk text, payloads, PII values, principals, access groups, "
    "credentials, or connection strings are displayed."
)

action_columns = st.columns(2)
refresh = action_columns[0].button("Refresh + bounded validate", type="primary")
deep_validate = action_columns[1].button("Deep validate (explicit Qdrant scan)")
if deep_validate:
    st.warning("Deep validation may scan every point in retained managed collections.")
if refresh or deep_validate:
    with st.spinner("Reading managed lineage and collection metadata…"):
        st.session_state["operations_platform"] = _read_platform(
            settings,
            deep_validation=deep_validate,
        )

platform_state: tuple[
    ComponentHealth,
    ComponentHealth,
    OperationsSnapshot | None,
] | None = st.session_state.get("operations_platform")
if platform_state is None:
    unchecked = ComponentHealth(
        "Not checked",
        False,
        "Select Refresh + bounded validate to connect lazily.",
    )
    postgres_health, qdrant_health, operations_snapshot = unchecked, unchecked, None
else:
    postgres_health, qdrant_health, operations_snapshot = platform_state

_display_platform_health(
    settings,
    postgres_health,
    qdrant_health,
    operations_snapshot,
)
_display_generation_history(operations_snapshot)
_display_validation(operations_snapshot)
_display_cleanup_and_restoration(operations_snapshot)
_display_quality()
_display_mlflow(settings)
