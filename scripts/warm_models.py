"""Warm the configured retrieval models through the production provider factory."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from consumer_finance_rag.config import load_settings
from consumer_finance_rag.factory import build_retrieval_service, warm_up_retrieval_service
from consumer_finance_rag.observability import NoOpTracer


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    args = parser.parse_args(argv)
    service = build_retrieval_service(load_settings(args.config), tracer=NoOpTracer())
    warm_up_retrieval_service(service, progress=lambda message: print(message, flush=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
