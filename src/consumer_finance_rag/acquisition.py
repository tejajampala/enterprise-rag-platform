"""Source acquisition, deliberately separate from document normalization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict


class SourceRecord(BaseModel):
    """Raw material and provenance as supplied by an acquisition source."""

    model_config = ConfigDict(frozen=True)

    source_id: str
    title: str
    source_url: str | None
    category: str
    source_version: str
    text: str


class DocumentSource(Protocol):
    def acquire(self) -> list[SourceRecord]:
        """Return source records without normalizing their content."""


class JsonFixtureSource:
    """Offline, reproducible development corpus stored as JSON."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def acquire(self) -> list[SourceRecord]:
        data = json.loads(self._path.read_text(encoding="utf-8"))
        return [SourceRecord.model_validate(item) for item in data]
