from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from consumer_finance_rag.config import MLflowSettings
from consumer_finance_rag.costs import estimate_generation_cost, load_pricing_catalog
from consumer_finance_rag.mlflow_inspection import TraceVisibility, inspect_exact_trace
from consumer_finance_rag.models import Chunk, RetrievalMetadata, RetrievedChunk
from consumer_finance_rag.presentation import cited_only
from consumer_finance_rag.public_showcase import PublicCitation
from consumer_finance_rag.request_inspection import (
    RequestInspectionStore,
    measured_bottleneck,
)
from consumer_finance_rag.retrieval import RetrievalTrace


def _candidate(
    chunk_id: str,
    *,
    score: float,
    dense_score: float | None = None,
    sparse_score: float | None = None,
    rrf_score: float | None = None,
    reranker_score: float | None = None,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(
            chunk_id=chunk_id,
            document_id="document",
            document_version="v1",
            text="private text must not be copied into inspection",
            ordinal=0,
            chunk_hash="a" * 64,
            metadata={"title": "Safe title", "access_visibility": "PUBLIC"},
        ),
        score=score,
        dense_rank=1,
        sparse_rank=2,
        dense_score=dense_score,
        sparse_score=sparse_score,
        rrf_score=rrf_score,
        rerank_score=reranker_score,
    )


def test_private_inspection_preserves_stage_specific_scores_without_text() -> None:
    store = RequestInspectionStore()
    store.begin(request_id="request-1", generation=4, model_id="model", region="us-east-1")
    candidate = _candidate(
        "chunk-1",
        score=0.03,
        dense_score=0.81,
        sparse_score=4.2,
        rrf_score=0.03,
        reranker_score=0.94,
    )
    trace = RetrievalTrace(
        dense=[candidate],
        sparse=[candidate],
        fused=[candidate],
        reranked=[candidate],
        metadata=RetrievalMetadata(
            dense_candidates=1,
            sparse_candidates=1,
            fused_candidates=1,
            reranked=1,
            final_contexts=1,
        ),
    )

    store.emit_inspection("request-1", trace)
    record = store.get_exact("request-1")
    assert record is not None
    inspection = record.inspection
    assert inspection is not None

    row = inspection.stages["reranker"][0]
    assert row.dense_score == 0.81
    assert row.bm25_score == 4.2
    assert row.rrf_score == 0.03
    assert row.reranker_score == 0.94
    assert row.included_in_context is True
    assert "private text" not in inspection.model_dump_json()


def test_citations_are_limited_to_ids_used_in_answer() -> None:
    citations = (
        PublicCitation(citation_id="C1", title="One", chunk_id="one", source_url=None),
        PublicCitation(citation_id="C2", title="Two", chunk_id="two", source_url=None),
    )

    assert [item.citation_id for item in cited_only("Supported [C2].", citations)] == ["C2"]


def test_exact_pricing_and_unavailable_fallback() -> None:
    catalog = load_pricing_catalog()
    usage = {"input_tokens": 928, "output_tokens": 75, "total_tokens": 1003}

    estimate = estimate_generation_cost(
        usage,
        catalog,
        provider="aws-bedrock",
        model_id="google.gemma-3-4b-it",
        region="us-east-1",
        pricing_mode="on-demand-standard",
        as_of=date(2026, 9, 25),
    )
    unavailable = estimate_generation_cost(
        usage,
        catalog,
        provider="aws-bedrock",
        model_id="different-model",
        region="us-east-1",
        pricing_mode="on-demand-standard",
        as_of=date(2026, 9, 25),
    )

    assert estimate.available is True
    assert str(estimate.estimated_api_cost) == "0.00004312"
    assert unavailable.available is False
    assert unavailable.estimated_api_cost is None


class _TraceClient:
    def __init__(self, trace: Any = None, error: Exception | None = None) -> None:
        self.trace = trace
        self.error = error
        self.seen: list[str] = []

    def get_trace(self, trace_id: str, display: bool = False) -> Any:
        assert display is False
        self.seen.append(trace_id)
        if self.error is not None:
            raise self.error
        return self.trace


def test_trace_reader_uses_exact_id_and_rejects_cross_request_fallback() -> None:
    trace = SimpleNamespace(
        info=SimpleNamespace(
            client_request_id="another-request",
            state="OK",
            execution_duration=10,
        ),
        data=SimpleNamespace(spans=[]),
    )
    client = _TraceClient(trace)

    result = inspect_exact_trace(
        MLflowSettings(),
        trace_id="tr-exact",
        expected_request_id="expected-request",
        client=client,
    )

    assert client.seen == ["tr-exact"]
    assert result.visibility is TraceVisibility.FAILED
    assert "substituted" in result.message


def test_missing_exact_trace_is_pending_not_generation_failure() -> None:
    client = _TraceClient(error=RuntimeError("RESOURCE_DOES_NOT_EXIST"))

    result = inspect_exact_trace(
        MLflowSettings(),
        trace_id="tr-syncing",
        expected_request_id="request-1",
        client=client,
    )

    assert result.visibility is TraceVisibility.PENDING
    assert "Trace syncing" in result.message


def test_completed_trace_keeps_actual_hierarchy_status_duration_and_usage() -> None:
    span = SimpleNamespace(
        span_id="span-child",
        parent_id="span-root",
        name="bedrock_generation",
        start_time_ns=1_000_000,
        end_time_ns=662_000_000,
        status=SimpleNamespace(status_code=SimpleNamespace(value="OK")),
        attributes={
            "mlflow.chat.tokenUsage": {
                "input_tokens": 928,
                "output_tokens": 75,
                "total_tokens": 1003,
            }
        },
    )
    trace = SimpleNamespace(
        info=SimpleNamespace(
            client_request_id="request-1",
            state="OK",
            execution_duration=31_720,
            trace_metadata={"estimated_cost": "0.00004312"},
        ),
        data=SimpleNamespace(spans=[span]),
    )

    result = inspect_exact_trace(
        MLflowSettings(),
        trace_id="tr-exact",
        expected_request_id="request-1",
        client=_TraceClient(trace),
    )

    assert result.visibility is TraceVisibility.COMPLETED
    assert result.root_duration_ms == 31_720
    assert result.spans[0].parent_span_id == "span-root"
    assert result.spans[0].duration_ms == 661
    assert result.spans[0].status == "OK"
    assert result.token_usage == {
        "input_tokens": 928,
        "output_tokens": 75,
        "total_tokens": 1003,
    }
    assert result.cost_metadata == {"estimated_cost": "0.00004312"}


def test_trace_without_returned_id_is_unavailable() -> None:
    result = inspect_exact_trace(
        MLflowSettings(),
        trace_id=None,
        expected_request_id="request-1",
    )

    assert result.visibility is TraceVisibility.UNAVAILABLE


def test_navigation_and_public_private_isolation() -> None:
    private_source = Path("app/private_workbench.py").read_text(encoding="utf-8")
    public_source = Path("app/streamlit_app.py").read_text(encoding="utf-8")

    for title in (
        "Chat",
        "Request Inspector",
        "Evaluations",
        "Lineage & Index",
        "Drift & Regression",
        "Operations",
    ):
        assert f'title="{title}"' in private_source
    assert "st.navigation" in private_source
    assert "public_showcase.enabled" not in private_source
    assert "Request Inspector" not in public_source
    assert "inspect_exact_trace" not in public_source
    assert "Evaluation Input Inspector" in private_source
    assert "Evaluation Input Inspector" not in public_source
    assert "EvaluationInputSnapshot" not in public_source

    inspector = private_source[
        private_source.index("def _render_evaluation_input_inspector") :
        private_source.index("def evaluations_page")
    ]
    for paid_or_retrieval_call in (
        "service.answer(",
        "evaluate_approved_answer(",
        "evaluate_managed_rag(",
        "build_application_runtime(",
        ".retrieve(",
    ):
        assert paid_or_retrieval_call not in inspector


def test_real_observation_fixture_has_no_answer_or_private_span_content() -> None:
    payload = json.loads(
        Path("data/fixtures/live_request_observation.json").read_text(encoding="utf-8")
    )

    assert payload["generation"] == 4
    assert payload["reranker_seconds"] == 25.84
    assert payload["model_id"] == "google.gemma-3-4b-it"
    assert payload["input_tokens"] == 928
    assert payload["recorded_span_count"] == 8
    assert "answer" not in payload
    assert "spans" not in payload


def test_bottleneck_uses_leaf_stages_not_overlapping_total() -> None:
    assert measured_bottleneck(
        {"total": 31_720.0, "retrieval": 31_000.0, "reranker": 25_840.0, "dense": 2_100.0}
    ) == "reranker"
