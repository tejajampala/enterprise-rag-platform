"""Stable content identities used for lineage and incremental indexing."""

from __future__ import annotations

import hashlib


def sha256_text(text: str) -> str:
    """Hash UTF-8 text after normalizing only newline representation."""

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
