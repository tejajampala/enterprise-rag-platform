"""Deterministic local PII classification and fail-closed redaction."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from consumer_finance_rag.hashing import sha256_text

DETERMINISTIC_PII_VERSION = "deterministic-pii-v1"


class PIIType(StrEnum):
    """Conservative PII categories supported by the first local detector."""

    EMAIL = "EMAIL"
    PHONE = "PHONE"
    AU_TFN = "AU_TFN"
    CREDIT_CARD = "CREDIT_CARD"


class PIIHandlingPolicy(StrEnum):
    """Representation permitted after authorized retrieval and reranking."""

    REDACT = "REDACT"
    ALLOW = "ALLOW"


class PIIMatch(BaseModel):
    """Safe structural match metadata; the matched value is intentionally absent."""

    model_config = ConfigDict(frozen=True)

    pii_type: PIIType
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    detector_version: str

    @model_validator(mode="after")
    def validate_offsets(self) -> PIIMatch:
        if self.end <= self.start:
            raise ValueError("PII match end must be greater than start")
        if not self.detector_version.strip():
            raise ValueError("PII detector version cannot be empty")
        return self


class DocumentPIISummary(BaseModel):
    """Safe derived classification attached to a document version or chunk."""

    model_config = ConfigDict(frozen=True)

    detector_version: str
    classification_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    pii_types: tuple[PIIType, ...] = ()
    match_count: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_summary(self) -> DocumentPIISummary:
        if not self.detector_version.strip():
            raise ValueError("PII detector version cannot be empty")
        if tuple(sorted(set(self.pii_types), key=lambda item: item.value)) != self.pii_types:
            raise ValueError("PII types must be unique and canonically sorted")
        if bool(self.pii_types) != bool(self.match_count):
            raise ValueError("PII types and match count must agree")
        return self

    @property
    def contains_pii(self) -> bool:
        return self.match_count > 0


class PIIRedactionError(RuntimeError):
    """Raised when supported PII remains after a requested redaction."""


_EMAIL_PATTERN = re.compile(
    r"(?<![\w.+-])[A-Z0-9!#$%&'*+/=?^_`{|}~-]+"
    r"(?:\.[A-Z0-9!#$%&'*+/=?^_`{|}~-]+)*@"
    r"[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?"
    r"(?:\.[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?)+(?![\w.-])",
    re.IGNORECASE,
)
_AU_TFN_PATTERN = re.compile(
    r"(?<!\d)(?<!\d[ -])(?:\d[ -]?){8}\d(?!\d)(?![ -]\d)"
)
_CREDIT_CARD_PATTERN = re.compile(
    r"(?<!\d)(?<!\d[ -])(?:\d[ -]?){12,18}\d(?!\d)(?![ -]\d)"
)
_PHONE_PATTERN = re.compile(
    r"(?<![\w])(?:"
    r"\+61(?:[\s().-]*\d){9}"
    r"|0[23478](?:[\s().-]*\d){8}"
    r"|\+[1-9]\d{0,2}(?:[\s().-]*\d){7,12}"
    r")(?!\d)"
)
_DIGITS = re.compile(r"\d")
_MATCH_PRIORITY = {
    PIIType.CREDIT_CARD: 0,
    PIIType.AU_TFN: 1,
    PIIType.EMAIL: 2,
    PIIType.PHONE: 3,
}


class DeterministicPIIDetector:
    """Compiled-regex detector with checksum validation and no external calls."""

    def __init__(self, version: str = DETERMINISTIC_PII_VERSION) -> None:
        normalized = version.strip()
        if not normalized:
            raise ValueError("PII detector version cannot be empty")
        self.version = normalized

    def detect(self, text: str) -> tuple[PIIMatch, ...]:
        candidates: list[PIIMatch] = []
        candidates.extend(self._regex_matches(text, _EMAIL_PATTERN, PIIType.EMAIL))
        candidates.extend(
            self._validated_numeric_matches(
                text,
                _AU_TFN_PATTERN,
                PIIType.AU_TFN,
                _valid_au_tfn,
            )
        )
        candidates.extend(
            self._validated_numeric_matches(
                text,
                _CREDIT_CARD_PATTERN,
                PIIType.CREDIT_CARD,
                _valid_credit_card,
            )
        )
        candidates.extend(self._regex_matches(text, _PHONE_PATTERN, PIIType.PHONE))
        return resolve_pii_matches(candidates)

    def classify(self, text: str) -> DocumentPIISummary:
        matches = self.detect(text)
        pii_types = tuple(
            sorted(
                {match.pii_type for match in matches},
                key=lambda item: item.value,
            )
        )
        payload = {
            "detector_version": self.version,
            "matches": [
                {
                    "end": match.end,
                    "pii_type": match.pii_type.value,
                    "start": match.start,
                }
                for match in matches
            ],
            "text_hash": sha256_text(text),
        }
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return DocumentPIISummary(
            detector_version=self.version,
            classification_hash=hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
            pii_types=pii_types,
            match_count=len(matches),
        )

    def redact(self, text: str) -> str:
        matches = self.detect(text)
        if not matches:
            return text
        parts: list[str] = []
        cursor = 0
        for match in matches:
            parts.append(text[cursor : match.start])
            parts.append(f"[REDACTED:{match.pii_type.value}]")
            cursor = match.end
        parts.append(text[cursor:])
        redacted = "".join(parts)
        remaining = self.detect(redacted)
        if remaining:
            remaining_types = ", ".join(
                sorted({match.pii_type.value for match in remaining})
            )
            raise PIIRedactionError(
                "PII redaction failed closed; supported categories remain: "
                f"{remaining_types}"
            )
        return redacted

    def _regex_matches(
        self,
        text: str,
        pattern: re.Pattern[str],
        pii_type: PIIType,
    ) -> list[PIIMatch]:
        return [
            PIIMatch(
                pii_type=pii_type,
                start=match.start(),
                end=match.end(),
                detector_version=self.version,
            )
            for match in pattern.finditer(text)
        ]

    def _validated_numeric_matches(
        self,
        text: str,
        pattern: re.Pattern[str],
        pii_type: PIIType,
        validator: Callable[[str], bool],
    ) -> list[PIIMatch]:
        matches: list[PIIMatch] = []
        for match in pattern.finditer(text):
            digits = "".join(_DIGITS.findall(match.group(0)))
            if validator(digits):
                matches.append(
                    PIIMatch(
                        pii_type=pii_type,
                        start=match.start(),
                        end=match.end(),
                        detector_version=self.version,
                    )
                )
        return matches


def empty_pii_summary(
    detector_version: str = DETERMINISTIC_PII_VERSION,
) -> DocumentPIISummary:
    """Return a deterministic empty summary for compatibility constructors."""

    return DeterministicPIIDetector(detector_version).classify("")


def pii_summary_payload(summary: DocumentPIISummary) -> dict[str, object]:
    """Return safe coarse metadata for a managed Qdrant payload."""

    return {
        "contains_pii": summary.contains_pii,
        "pii_types": [pii_type.value for pii_type in summary.pii_types],
        "pii_detector_version": summary.detector_version,
    }


def _valid_au_tfn(digits: str) -> bool:
    if len(digits) != 9 or len(set(digits)) == 1:
        return False
    weights = (1, 4, 3, 7, 5, 8, 6, 9, 10)
    return sum(int(digit) * weight for digit, weight in zip(digits, weights, strict=True)) % 11 == 0


def _valid_credit_card(digits: str) -> bool:
    if not 13 <= len(digits) <= 19 or len(set(digits)) == 1:
        return False
    total = 0
    parity = len(digits) % 2
    for index, character in enumerate(digits):
        value = int(character)
        if index % 2 == parity:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


def resolve_pii_matches(candidates: list[PIIMatch]) -> tuple[PIIMatch, ...]:
    """Deduplicate overlapping candidates with stable conservative precedence."""

    selected: list[PIIMatch] = []
    for candidate in sorted(
        candidates,
        key=lambda item: (
            _MATCH_PRIORITY[item.pii_type],
            item.start,
            -(item.end - item.start),
            item.end,
        ),
    ):
        if any(
            candidate.start < existing.end and existing.start < candidate.end
            for existing in selected
        ):
            continue
        selected.append(candidate)
    return tuple(
        sorted(
            selected,
            key=lambda item: (item.start, item.end, item.pii_type.value),
        )
    )
