"""Pure BM25 state planning for a deterministic active chunk manifest."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

from pydantic import ValidationError

from consumer_finance_rag.lineage.fingerprints import (
    ManagedIndexConfiguration,
    managed_index_configuration_payload_fingerprint,
)
from consumer_finance_rag.lineage.index_types import (
    ActiveChunkManifest,
    BM25GenerationPlan,
    BM25PlanAction,
    PreviousBM25GenerationState,
)
from consumer_finance_rag.sparse import (
    BM25SparseEncoder,
    BM25State,
    bm25_corpus_hash,
    bm25_state_from_json_dict,
    bm25_state_hash,
)


class BM25GenerationPlanningError(ValueError):
    """Base exception for an invalid current BM25 planning request."""


class EmptyBM25CorpusError(BM25GenerationPlanningError):
    """Raised because the existing BM25 state contract cannot represent zero chunks."""


class InvalidBM25ManifestError(BM25GenerationPlanningError):
    """Raised when manifest counts or chunking identity are internally inconsistent."""


class CorruptActiveBM25GenerationError(BM25GenerationPlanningError):
    """Raised when an ACTIVE generation's persisted BM25 facts are untrustworthy."""


@dataclass(frozen=True, slots=True)
class _BM25Compatibility:
    k1: float
    b: float
    implementation_version: str
    tokenizer_version: str
    chunking_fingerprint: str


def plan_bm25_generation(
    manifest: ActiveChunkManifest,
    configuration: ManagedIndexConfiguration,
    previous: PreviousBM25GenerationState | None = None,
) -> BM25GenerationPlan:
    """Return verified reusable state or refit once over every manifest chunk text."""

    _validate_current_inputs(manifest, configuration)
    texts = tuple(chunk.text for chunk in manifest.chunks)
    current_corpus_hash = bm25_corpus_hash(texts)
    ordered_chunk_ids = tuple(
        chunk.compatibility_chunk_id for chunk in manifest.chunks
    )

    if previous is None:
        return _refit(
            manifest,
            configuration,
            texts,
            current_corpus_hash,
            ordered_chunk_ids,
            reason="No previous ACTIVE index generation exists",
        )

    previous_state, previous_config = _validate_previous_active_generation(previous)
    incompatibility = _reuse_incompatibility_reason(
        manifest=manifest,
        configuration=configuration,
        current_corpus_hash=current_corpus_hash,
        previous=previous,
        previous_state=previous_state,
        previous_config=previous_config,
    )
    if incompatibility is not None:
        return _refit(
            manifest,
            configuration,
            texts,
            current_corpus_hash,
            ordered_chunk_ids,
            reason=incompatibility,
        )

    return BM25GenerationPlan(
        action=BM25PlanAction.REUSE,
        corpus_hash=current_corpus_hash,
        state=previous_state,
        state_hash=bm25_state_hash(previous_state),
        logical_document_count=manifest.document_count,
        chunk_count=manifest.chunk_count,
        bm25_k1=configuration.bm25_k1,
        bm25_b=configuration.bm25_b,
        reason="Ordered chunk texts and BM25 compatibility settings are unchanged",
        ordered_chunk_ids=ordered_chunk_ids,
    )


def _refit(
    manifest: ActiveChunkManifest,
    configuration: ManagedIndexConfiguration,
    texts: tuple[str, ...],
    current_corpus_hash: str,
    ordered_chunk_ids: tuple[str, ...],
    *,
    reason: str,
) -> BM25GenerationPlan:
    encoder = BM25SparseEncoder.fit(
        texts,
        k1=configuration.bm25_k1,
        b=configuration.bm25_b,
    )
    state = encoder.state
    if state.document_count != manifest.chunk_count:
        raise BM25GenerationPlanningError(
            "Refitted BM25 document count does not equal manifest chunk count"
        )
    if state.corpus_hash != current_corpus_hash:
        raise BM25GenerationPlanningError(
            "Refitted BM25 corpus hash does not match the ordered manifest texts"
        )
    if state.k1 != configuration.bm25_k1 or state.b != configuration.bm25_b:
        raise BM25GenerationPlanningError(
            "Refitted BM25 state does not preserve requested k1/b"
        )
    return BM25GenerationPlan(
        action=BM25PlanAction.REFIT,
        corpus_hash=current_corpus_hash,
        state=state,
        state_hash=bm25_state_hash(state),
        logical_document_count=manifest.document_count,
        chunk_count=manifest.chunk_count,
        bm25_k1=configuration.bm25_k1,
        bm25_b=configuration.bm25_b,
        reason=reason,
        ordered_chunk_ids=ordered_chunk_ids,
    )


def _validate_current_inputs(
    manifest: ActiveChunkManifest,
    configuration: ManagedIndexConfiguration,
) -> None:
    if manifest.document_count != len(manifest.documents):
        raise InvalidBM25ManifestError(
            "Manifest document_count does not equal its logical document tuple length"
        )
    if manifest.chunk_count != len(manifest.chunks):
        raise InvalidBM25ManifestError(
            "Manifest chunk_count does not equal its chunk tuple length"
        )
    if not manifest.chunks:
        raise EmptyBM25CorpusError("Cannot prepare BM25 state for an empty chunk corpus")
    if configuration.chunking_fingerprint != manifest.chunking_fingerprint:
        raise InvalidBM25ManifestError(
            "Managed index configuration chunking fingerprint does not match the manifest"
        )
    if any(
        chunk.chunking_fingerprint != manifest.chunking_fingerprint
        for chunk in manifest.chunks
    ):
        raise InvalidBM25ManifestError(
            "Manifest contains a chunk from a different chunking derivation"
        )
    if configuration.bm25_k1 <= 0 or not 0 <= configuration.bm25_b <= 1:
        raise BM25GenerationPlanningError("BM25 requires k1 > 0 and 0 <= b <= 1")


def _validate_previous_active_generation(
    previous: PreviousBM25GenerationState,
) -> tuple[BM25State, _BM25Compatibility]:
    missing = [
        label
        for label, value in (
            ("bm25_state", previous.bm25_state),
            ("bm25_state_hash", previous.bm25_state_hash),
            ("bm25_corpus_hash", previous.bm25_corpus_hash),
            ("index_configuration", previous.index_configuration),
            (
                "index_configuration_fingerprint",
                previous.index_configuration_fingerprint,
            ),
        )
        if value is None
    ]
    if missing:
        raise CorruptActiveBM25GenerationError(
            "ACTIVE index generation is missing required BM25 fields: "
            + ", ".join(missing)
        )

    state_payload = cast(Mapping[str, Any], previous.bm25_state)
    persisted_state_hash = cast(str, previous.bm25_state_hash)
    persisted_corpus_hash = cast(str, previous.bm25_corpus_hash)
    configuration_payload = cast(Mapping[str, Any], previous.index_configuration)
    persisted_configuration_hash = cast(
        str, previous.index_configuration_fingerprint
    )

    try:
        state = bm25_state_from_json_dict(state_payload)
    except (ValidationError, TypeError, ValueError) as exc:
        raise CorruptActiveBM25GenerationError(
            "ACTIVE index generation contains an invalid BM25 state"
        ) from exc
    calculated_state_hash = bm25_state_hash(state)
    if calculated_state_hash != persisted_state_hash:
        raise CorruptActiveBM25GenerationError(
            "ACTIVE index generation BM25 state hash validation failed"
        )
    if state.corpus_hash != persisted_corpus_hash:
        raise CorruptActiveBM25GenerationError(
            "ACTIVE index generation BM25 corpus hash disagrees with its state"
        )
    if state.document_count != previous.chunk_count:
        raise CorruptActiveBM25GenerationError(
            "ACTIVE index generation chunk count disagrees with BM25 document_count"
        )

    calculated_configuration_hash = managed_index_configuration_payload_fingerprint(
        configuration_payload
    )
    if calculated_configuration_hash != persisted_configuration_hash:
        raise CorruptActiveBM25GenerationError(
            "ACTIVE index generation configuration fingerprint validation failed"
        )
    compatibility = _parse_previous_compatibility(configuration_payload)
    if state.k1 != compatibility.k1 or state.b != compatibility.b:
        raise CorruptActiveBM25GenerationError(
            "ACTIVE index generation BM25 state disagrees with its configuration"
        )
    if previous.chunking_fingerprint != compatibility.chunking_fingerprint:
        raise CorruptActiveBM25GenerationError(
            "ACTIVE index generation chunking fingerprint disagrees with its configuration"
        )
    return state, compatibility


def _parse_previous_compatibility(
    configuration: Mapping[str, Any],
) -> _BM25Compatibility:
    try:
        bm25 = _require_mapping(configuration["bm25"], "bm25")
        return _BM25Compatibility(
            k1=_require_number(bm25["k1"], "bm25.k1"),
            b=_require_number(bm25["b"], "bm25.b"),
            implementation_version=_require_string(
                bm25["implementation_version"], "bm25.implementation_version"
            ),
            tokenizer_version=_require_string(
                bm25["tokenizer_version"], "bm25.tokenizer_version"
            ),
            chunking_fingerprint=_require_string(
                configuration["chunking_fingerprint"], "chunking_fingerprint"
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise CorruptActiveBM25GenerationError(
            "ACTIVE index generation has an invalid BM25 configuration payload"
        ) from exc


def _reuse_incompatibility_reason(
    *,
    manifest: ActiveChunkManifest,
    configuration: ManagedIndexConfiguration,
    current_corpus_hash: str,
    previous: PreviousBM25GenerationState,
    previous_state: BM25State,
    previous_config: _BM25Compatibility,
) -> str | None:
    if previous_config.k1 != configuration.bm25_k1:
        return "BM25 k1 changed"
    if previous_config.b != configuration.bm25_b:
        return "BM25 b changed"
    if (
        previous_config.implementation_version
        != configuration.bm25_implementation_version
    ):
        return "BM25 implementation version changed"
    if previous_config.tokenizer_version != configuration.bm25_tokenizer_version:
        return "BM25 tokenizer version changed"
    if previous_config.chunking_fingerprint != manifest.chunking_fingerprint:
        return "Chunking fingerprint changed"
    if previous.bm25_corpus_hash != current_corpus_hash:
        return "Ordered active chunk text corpus changed"
    if previous_state.document_count != manifest.chunk_count:
        return "Active chunk count changed"
    return None


def _require_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _require_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    return float(value)


def _require_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value
