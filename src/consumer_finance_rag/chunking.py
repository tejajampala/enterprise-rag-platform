"""A single paragraph-aware, recursive chunking strategy."""

from __future__ import annotations

import re

from consumer_finance_rag.hashing import sha256_text
from consumer_finance_rag.models import Chunk, Document


class ParagraphChunker:
    """Pack paragraphs to a target size, recursively splitting oversized blocks."""

    def __init__(self, chunk_size: int, overlap: int) -> None:
        if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
            raise ValueError("require chunk_size > overlap >= 0")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, document: Document) -> list[Chunk]:
        pieces: list[str] = []
        for paragraph in re.split(r"\n\s*\n", document.text):
            paragraph = paragraph.strip()
            if paragraph:
                pieces.extend(self._split_oversized(paragraph))

        texts = self._pack(pieces)
        metadata = {
            "source_id": document.source_id,
            "title": document.title,
            "source_url": document.source_url,
            "category": document.category,
            "content_hash": document.content_hash,
            "access_visibility": document.access_policy.visibility.value,
            "allowed_groups": list(document.access_policy.canonical_groups),
        }
        chunks: list[Chunk] = []
        for ordinal, text in enumerate(texts):
            chunk_hash = sha256_text(text)
            chunks.append(
                Chunk(
                    chunk_id=f"{document.document_id}:{ordinal}:{chunk_hash[:12]}",
                    document_id=document.document_id,
                    document_version=document.source_version,
                    text=text,
                    ordinal=ordinal,
                    chunk_hash=chunk_hash,
                    metadata=metadata,
                )
            )
        return chunks

    def _split_oversized(self, text: str) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text]
        sentences = re.split(r"(?<=[.!?])\s+", text)
        if len(sentences) > 1:
            units: list[str] = []
            for sentence in sentences:
                if len(sentence) <= self.chunk_size:
                    units.append(sentence)
                else:
                    units.extend(self._hard_split(sentence))
            return self._pack(units, add_overlap=False)
        return self._hard_split(text)

    def _hard_split(self, text: str) -> list[str]:
        step = self.chunk_size - self.overlap
        return [
            text[start : start + self.chunk_size].strip() for start in range(0, len(text), step)
        ]

    def _pack(self, pieces: list[str], *, add_overlap: bool = True) -> list[str]:
        packed: list[str] = []
        current = ""
        for piece in pieces:
            candidate = f"{current}\n\n{piece}" if current else piece
            if current and len(candidate) > self.chunk_size:
                packed.append(current)
                prefix = self._overlap_tail(current) if add_overlap else ""
                overlapped = f"{prefix}\n\n{piece}".strip() if prefix else piece
                current = overlapped if len(overlapped) <= self.chunk_size else piece
            else:
                current = candidate
        if current:
            packed.append(current)
        return packed

    def _overlap_tail(self, text: str) -> str:
        if self.overlap == 0:
            return ""
        tail = text[-self.overlap :]
        first_space = tail.find(" ")
        return tail[first_space + 1 :] if first_space >= 0 else tail
