"""Immutable access-policy types and deterministic Qdrant authorization filters."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from qdrant_client import models


class AccessVisibility(StrEnum):
    """Supported document visibility modes for managed retrieval."""

    PUBLIC = "PUBLIC"
    RESTRICTED = "RESTRICTED"


def normalize_group_identifier(value: str) -> str:
    """Canonicalize a caller-supplied group identifier or reject ambiguity."""

    if not isinstance(value, str):
        raise TypeError("group identifiers must be strings")
    normalized = value.strip().casefold()
    if not normalized:
        raise ValueError("group identifiers cannot be empty")
    return normalized


def _normalized_groups(value: object) -> frozenset[str]:
    if value is None:
        return frozenset()
    if isinstance(value, (str, bytes)):
        raise TypeError("groups must be a collection of identifiers, not a string")
    if not isinstance(value, Iterable):
        raise TypeError("groups must be an iterable of string identifiers")
    return frozenset(normalize_group_identifier(item) for item in value)


class DocumentAccessPolicy(BaseModel):
    """Access policy preserved with one immutable document version."""

    model_config = ConfigDict(frozen=True)

    visibility: AccessVisibility = AccessVisibility.PUBLIC
    allowed_groups: frozenset[str] = Field(default_factory=frozenset)

    @field_validator("allowed_groups", mode="before")
    @classmethod
    def canonicalize_groups(cls, value: object) -> frozenset[str]:
        return _normalized_groups(value)

    @model_validator(mode="after")
    def validate_visibility_contract(self) -> DocumentAccessPolicy:
        if self.visibility is AccessVisibility.PUBLIC and self.allowed_groups:
            raise ValueError("PUBLIC access policy cannot contain allowed groups")
        if self.visibility is AccessVisibility.RESTRICTED and not self.allowed_groups:
            raise ValueError("RESTRICTED access policy requires at least one allowed group")
        return self

    @property
    def canonical_groups(self) -> tuple[str, ...]:
        """Return stable ordering for persistence, hashing, and Qdrant payloads."""

        return tuple(sorted(self.allowed_groups))


class AccessContext(BaseModel):
    """Request-scoped identity facts resolved by a future authentication layer."""

    model_config = ConfigDict(frozen=True)

    principal_id: str | None = None
    groups: frozenset[str] = Field(default_factory=frozenset)

    @field_validator("principal_id", mode="before")
    @classmethod
    def normalize_principal(cls, value: object) -> object:
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError("principal_id must be a string")
        normalized = value.strip()
        if not normalized:
            raise ValueError("principal_id cannot be empty")
        return normalized

    @field_validator("groups", mode="before")
    @classmethod
    def canonicalize_groups(cls, value: object) -> frozenset[str]:
        return _normalized_groups(value)


def public_access_policy() -> DocumentAccessPolicy:
    """Create the backward-compatible default policy."""

    return DocumentAccessPolicy()


def access_policy_payload(policy: DocumentAccessPolicy) -> dict[str, object]:
    """Return the stable Qdrant/storage representation of a validated policy."""

    return {
        "access_visibility": policy.visibility.value,
        "allowed_groups": list(policy.canonical_groups),
    }


def access_policy_from_payload(payload: Mapping[str, Any]) -> DocumentAccessPolicy:
    """Require and validate ACL fields from a managed Qdrant payload."""

    if "access_visibility" not in payload or "allowed_groups" not in payload:
        raise ValueError("managed payload is missing required access-control fields")
    return DocumentAccessPolicy(
        visibility=payload["access_visibility"],
        allowed_groups=payload["allowed_groups"],
    )


def is_access_allowed(
    policy: DocumentAccessPolicy,
    context: AccessContext,
) -> bool:
    """Evaluate the same policy semantics used by the Qdrant filter."""

    return policy.visibility is AccessVisibility.PUBLIC or bool(
        policy.allowed_groups & context.groups
    )


def build_qdrant_access_filter(context: AccessContext) -> models.Filter:
    """Build PUBLIC OR matching-RESTRICTED authorization for pre-filtered search."""

    public = models.FieldCondition(
        key="access_visibility",
        match=models.MatchValue(value=AccessVisibility.PUBLIC.value),
    )
    if not context.groups:
        return models.Filter(must=[public])

    restricted = models.Filter(
        must=[
            models.FieldCondition(
                key="access_visibility",
                match=models.MatchValue(value=AccessVisibility.RESTRICTED.value),
            ),
            models.FieldCondition(
                key="allowed_groups",
                match=models.MatchAny(any=list(sorted(context.groups))),
            ),
        ]
    )
    return models.Filter(should=[public, restricted])
