"""Convert acquired records into normalized, versioned documents."""

from __future__ import annotations

import re

from consumer_finance_rag.acquisition import SourceRecord
from consumer_finance_rag.hashing import sha256_text
from consumer_finance_rag.models import Document


def normalize_text(text: str) -> str:
    """Normalize newlines and whitespace while preserving paragraph boundaries."""

    paragraphs = re.split(r"\n\s*\n", text.replace("\r\n", "\n").replace("\r", "\n"))
    cleaned = [re.sub(r"[ \t]+", " ", paragraph).strip() for paragraph in paragraphs]
    return "\n\n".join(paragraph for paragraph in cleaned if paragraph)


def normalize_record(record: SourceRecord) -> Document:
    text = normalize_text(record.text)
    content_hash = sha256_text(text)
    document_id = f"{record.source_id}:{content_hash[:16]}"
    return Document(
        document_id=document_id,
        source_id=record.source_id,
        title=record.title,
        source_url=record.source_url,
        category=record.category,
        text=text,
        source_version=record.source_version,
        content_hash=content_hash,
        access_policy=record.access_policy,
    )


def normalize_records(records: list[SourceRecord]) -> list[Document]:
    return [normalize_record(record) for record in records]
