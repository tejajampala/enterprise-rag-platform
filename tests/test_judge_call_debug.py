"""Offline security and lifecycle tests for private judge-call artifacts."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest
from litellm import ModelResponse

from consumer_finance_rag.costs import ModelInvocationUsage
from consumer_finance_rag.evaluation.judge_call_debug import (
    ApprovedPublicDebugCase,
    JudgeCallDebugCapture,
    JudgeCallDebugConfig,
    JudgeCallDebugError,
    delete_judge_call_debug_artifact,
    load_judge_call_debug_artifact,
    purge_expired_judge_call_artifacts,
)
from consumer_finance_rag.evaluation.managed_rag import ManagedRAGEvaluationReport


def _capture(
    tmp_path: Path,
    *,
    now: datetime | None = None,
    groups: tuple[str, ...] = (),
) -> JudgeCallDebugCapture:
    capture = JudgeCallDebugCapture(
        JudgeCallDebugConfig(artifact_directory=tmp_path, retention_hours=1),
        cases=(
            ApprovedPublicDebugCase(
                case_id="public-case",
                question="Reviewed PUBLIC question?",
                review_status="reviewed",
                visibility="PUBLIC",
                groups=groups,
            ),
        ),
        evaluation_configuration_id="configuration-a",
        evaluation_run_id="run-a",
        model_id="openai.gpt-oss-20b-1:0",
        region="ap-southeast-2",
        now=now,
    )
    capture.bind_case_inputs(
        case_id="public-case",
        question="Reviewed PUBLIC question?",
        contexts=("Approved context with user@example.com",),
        chunk_ids=("public-doc:chunk-1",),
    )
    return capture


def _invocation() -> ModelInvocationUsage:
    return ModelInvocationUsage(
        backend="ragas",
        metric_name="context_precision",
        case_id="public-case",
        status="COMPLETED",
        model_id="openai.gpt-oss-20b-1:0",
        region="ap-southeast-2",
        input_tokens=100,
        output_tokens=20,
        total_tokens=120,
        reasoning_tokens=8,
        retry_count=1,
        latency_ms=12.5,
        provider_request_id="provider-request-a",
        finish_reason="stop",
        effective_output_token_limit=2048,
        response_type="ModelResponse",
        completion_returned_normally=True,
        estimated_cost_usd=Decimal("0.00001339"),
    )


def test_capture_redacts_content_filters_secrets_and_preserves_response(
    tmp_path: Path,
) -> None:
    capture = _capture(tmp_path)
    response = ModelResponse(
        id="provider-request-a",
        model="bedrock/converse/openai.gpt-oss-20b-1:0",
        choices=[
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": '{"reason":"contact user@example.com","verdict":1}',
                    "reasoning_content": "hidden chain of thought",
                },
                "finish_reason": "stop",
            }
        ],
        usage={
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
        },
    )
    before = response.model_dump(mode="json")
    call_id = capture.capture_invocation(
        invocation=_invocation(),
        request_kwargs={
            "model": "bedrock/converse/openai.gpt-oss-20b-1:0",
            "messages": [
                {
                    "role": "user",
                    "content": "Approved context with user@example.com",
                }
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "ContextPrecisionOutput"},
            },
            "max_tokens": 2048,
            "aws_region_name": "ap-southeast-2",
            "api_key": "must-not-appear",
            "headers": {"authorization": "must-not-appear"},
        },
        response=response,
        evaluation_trace_id="tr-evaluation",
        metric_span_id="metric-span",
        provider_span_id="provider-span",
        sequence_within_metric=1,
    )
    capture.record_instructor_result(call_id, {"reason": "ok", "verdict": 1})
    output = capture.persist()

    assert output is not None
    assert response.model_dump(mode="json") == before
    artifact = load_judge_call_debug_artifact(output)
    record = artifact.calls[0]
    assert record.call_id == "provider-span"
    assert record.provider_span_id == "provider-span"
    assert record.metric_span_id == "metric-span"
    assert record.context_chunk_ids == ("public-doc:chunk-1",)
    assert record.instructor.validation_status == "PASSED"
    rendered = output.read_text(encoding="utf-8")
    assert "user@example.com" not in rendered
    assert "must-not-appear" not in rendered
    assert "hidden chain of thought" not in rendered
    assert "reasoning_content" not in rendered
    assert "input_tokens" in rendered
    assert "reasoning_tokens" in rendered


def test_private_capture_rejects_group_restricted_case(tmp_path: Path) -> None:
    with pytest.raises(JudgeCallDebugError, match="group-free"):
        _capture(tmp_path, groups=("private-analysts",))


def test_private_artifact_expiry_and_exact_deletion(tmp_path: Path) -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    capture = _capture(tmp_path, now=now)
    capture.capture_invocation(
        invocation=_invocation(),
        request_kwargs={"messages": [], "max_tokens": 2048},
        response={},
        evaluation_trace_id="tr-evaluation",
        metric_span_id="metric-span",
        provider_span_id="provider-span",
        sequence_within_metric=1,
    )
    output = capture.persist()
    assert output is not None and output.exists()

    removed = purge_expired_judge_call_artifacts(
        tmp_path,
        now=now + timedelta(hours=2),
    )
    assert removed == (output,)
    assert not output.exists()

    outside = tmp_path.parent / "outside.private.json"
    outside.write_text("{}", encoding="utf-8")
    with pytest.raises(JudgeCallDebugError, match="outside"):
        delete_judge_call_debug_artifact(outside, root=tmp_path)
    assert outside.exists()
    outside.unlink()


def test_public_application_has_no_judge_call_inspector() -> None:
    source = Path("app/streamlit_app.py").read_text(encoding="utf-8")
    assert "JudgeCallDebug" not in source
    assert "Individual Judge Calls" not in source
    assert "capture-judge-io" not in source


def test_safe_report_schema_cannot_embed_content_bearing_artifacts() -> None:
    fields = ManagedRAGEvaluationReport.model_fields
    assert "judge_call_debug" not in fields
    assert "judge_call_artifact" not in fields
    assert "artifact_path" not in fields
