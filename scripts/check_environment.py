"""Run read-only configuration preflight and selected-profile readiness checks."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from consumer_finance_rag.config import load_settings
from consumer_finance_rag.environment_check import (
    CheckStatus,
    EnvironmentCheck,
    EnvironmentReport,
    format_environment_report,
    run_environment_check,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/config.yaml"),
        help="Application YAML configuration path (default: config/config.yaml).",
    )
    parser.add_argument(
        "--managed-smoke",
        action="store_true",
        help=(
            "Run explicit deep ACTIVE payload validation; requires the managed profile "
            "and never initializes application models."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        settings = load_settings(args.config)
    except Exception as exc:
        report = EnvironmentReport(
            profile="unavailable",
            checks=(
                EnvironmentCheck(
                    "configuration",
                    CheckStatus.FAIL,
                    "Application configuration is invalid "
                    f"({type(exc).__name__}); details withheld.",
                ),
            ),
        )
        print(format_environment_report(report))
        return 2
    if args.managed_smoke and settings.runtime.profile != "managed":
        print(
            "--managed-smoke requires RAG__RUNTIME__PROFILE=managed.",
            file=sys.stderr,
        )
        return 2
    report = run_environment_check(
        settings,
        deep_managed=bool(args.managed_smoke),
        alembic_config_path=Path("alembic.ini"),
    )
    print(format_environment_report(report))
    return 0 if report.ready else 3


if __name__ == "__main__":
    raise SystemExit(main())
