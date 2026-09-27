"""Persist and exactly verify a metadata-only MLflow trace without model calls."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from uuid import uuid4

from consumer_finance_rag.config import load_settings
from consumer_finance_rag.observability import MLflowTracer, trace_id_from_span


def main(argv: Sequence[str] | None = None) -> int:
    if argv:
        raise SystemExit("This smoke test accepts no arguments")
    settings = load_settings()
    client_request_id = str(uuid4())
    configuration_id = hashlib.sha256(client_request_id.encode()).hexdigest()
    tracer = MLflowTracer(settings.mlflow)
    with tracer.span(
        "ragas_evaluation",
        span_type="CHAIN",
        attributes={
            "evaluation.kind": "offline_trace_export_smoke",
            "evaluation.client_request_id": client_request_id,
            "evaluation.configuration_id": configuration_id,
            "trace.content_policy": "metadata_only",
        },
    ) as root:
        trace_id = trace_id_from_span(root)
        with tracer.span(
            "trace_export_smoke_child",
            span_type="CHAIN",
            attributes={
                "evaluation.client_request_id": client_request_id,
                "evaluation.configuration_id": configuration_id,
                "trace.content_policy": "metadata_only",
            },
        ) as child:
            child.set_outputs({"status": "COMPLETED"})
        root.set_outputs({"status": "COMPLETED"})
    if trace_id is None:
        print("Trace export: UNAVAILABLE (no local trace ID)")
        return 3
    persistence = tracer.verify_trace(
        trace_id,
        expected_client_request_id=client_request_id,
        expected_span_count=2,
    )
    print(f"Trace ID: {trace_id}")
    print(f"Evaluation client request ID: {client_request_id}")
    print(f"Configuration ID: {configuration_id}")
    print(f"Persistence: {persistence.status}")
    print(f"Persisted spans: {persistence.persisted_span_count or 0}")
    print(persistence.reason)
    if persistence.status == "CONFIRMED":
        return 0
    if persistence.status == "PENDING":
        return 2
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
