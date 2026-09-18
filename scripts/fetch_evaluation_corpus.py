"""Fetch a bounded, cached CFPB evaluation corpus from the versioned manifest."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from consumer_finance_rag.config import load_settings
from consumer_finance_rag.corpus_expansion import (
    acquire_manifest_corpus,
    load_corpus_manifest,
    write_source_records,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args(argv)
    settings = load_settings(args.config)
    corpus = settings.corpus_expansion
    manifest = load_corpus_manifest(corpus.manifest_path)
    records = acquire_manifest_corpus(
        manifest,
        cache_directory=corpus.cache_directory,
        timeout=corpus.request_timeout_seconds,
        refresh=args.refresh,
        progress=lambda position, total, url: print(
            f"Acquiring CFPB document {position}/{total}: {url}", flush=True
        ),
    )
    output = write_source_records(records, corpus.output_path)
    print(f"Fetched {len(records)} CFPB source documents to {output}.")
    print("Downloaded pages are cached and ignored by Git; the manifest is source-controlled.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
