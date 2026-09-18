"""Typed persistence inputs for the Phase 3 lineage repository."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ChunkPersistenceInput:
    """Precomputed chunk values ready for lineage persistence."""

    chunking_fingerprint: str
    compatibility_chunk_id: str
    ordinal: int
    chunk_hash: str
    text: str
    metadata: Mapping[str, Any]
