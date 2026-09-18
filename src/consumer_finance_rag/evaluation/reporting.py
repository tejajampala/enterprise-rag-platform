"""Machine-readable reports consumed by Streamlit and future CI."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from consumer_finance_rag.evaluation.models import (
    RAGEvaluationReport,
    RetrievalEvaluationReport,
)


def write_retrieval_report(
    report: RetrievalEvaluationReport, directory: str | Path
) -> tuple[Path, Path]:
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "retrieval_summary.json"
    csv_path = output / "retrieval_cases.csv"
    json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")

    rows: list[dict[str, str | float]] = []
    for case in report.cases:
        for stage, result in case.stages.items():
            row: dict[str, str | float] = {
                "case_id": case.case_id,
                "category": case.category,
                "difficulty": case.difficulty,
                "stage": stage,
                "retrieved_ids": json.dumps(result.retrieved_ids),
            }
            row.update(result.metrics)
            rows.append(row)
    fieldnames = list(rows[0]) if rows else ["case_id", "category", "difficulty", "stage"]
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def write_rag_report(report: RAGEvaluationReport, directory: str | Path) -> Path:
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "rag_summary.json"
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_report(path: str | Path) -> dict[str, object] | None:
    report_path = Path(path)
    if not report_path.is_file():
        return None
    data = json.loads(report_path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else None
