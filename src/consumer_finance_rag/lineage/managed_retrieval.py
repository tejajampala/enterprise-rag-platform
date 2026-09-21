"""Resolve one immutable ACTIVE managed generation into the existing retriever."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast
from uuid import UUID

from pydantic import ValidationError
from qdrant_client import QdrantClient

from consumer_finance_rag.config import AppSettings, EmbeddingModelSettings
from consumer_finance_rag.lineage.fingerprints import (
    PAYLOAD_SCHEMA_VERSION,
    ManagedIndexConfiguration,
    embedding_fingerprint,
    managed_index_configuration_identity,
    managed_index_configuration_payload_fingerprint,
)
from consumer_finance_rag.lineage.managed_vector_store import (
    ManagedQdrantError,
    ManagedQdrantStore,
    managed_generation_collection_name,
)
from consumer_finance_rag.lineage.models import IndexState
from consumer_finance_rag.lineage.repository import (
    InvalidLineageStateError,
    LineageRepository,
)
from consumer_finance_rag.observability import Tracer
from consumer_finance_rag.providers import (
    EmbeddingProvider,
    HuggingFaceEmbeddingProvider,
    HuggingFaceRerankerProvider,
    RerankerProvider,
)
from consumer_finance_rag.retrieval import HybridRetrievalService
from consumer_finance_rag.sparse import (
    BM25SparseEncoder,
    bm25_state_from_json_dict,
    bm25_state_hash,
)
from consumer_finance_rag.vector_store import QdrantHybridStore


class ManagedRetrievalError(RuntimeError):
    """Base error for safe managed-runtime resolution."""


class NoActiveManagedIndexError(ManagedRetrievalError):
    """Raised when the configured corpus has no ACTIVE generation."""


class CorruptActiveManagedIndexError(ManagedRetrievalError):
    """Raised when persisted ACTIVE metadata is incomplete or inconsistent."""


class ManagedEmbeddingCompatibilityError(ManagedRetrievalError):
    """Raised when query embeddings are incompatible with stored dense vectors."""


class ManagedCollectionUnavailableError(ManagedRetrievalError):
    """Raised when bounded physical Qdrant validation fails."""


@dataclass(frozen=True, slots=True)
class ManagedRetrievalRuntime:
    """Generation-pinned retrieval runtime detached from SQLAlchemy state."""

    index_state_id: UUID
    generation: int
    qdrant_collection: str
    snapshot_hash: str
    embedding_fingerprint: str
    chunking_fingerprint: str
    index_configuration_fingerprint: str
    document_count: int
    chunk_count: int
    retrieval_service: HybridRetrievalService


def build_managed_retrieval_runtime(
    settings: AppSettings,
    repository: LineageRepository,
    *,
    qdrant_client: QdrantClient | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    reranker_provider: RerankerProvider | None = None,
    tracer: Tracer | None = None,
) -> ManagedRetrievalRuntime:
    """Pin the configured corpus's single ACTIVE generation for this process."""

    corpus = repository.get_corpus_by_key(settings.managed_ingestion.corpus_key)
    if corpus is None:
        raise NoActiveManagedIndexError(
            f"Managed corpus {settings.managed_ingestion.corpus_key!r} is not initialized"
        )
    try:
        active = repository.get_active_index_state(corpus.corpus_id)
    except InvalidLineageStateError as exc:
        raise CorruptActiveManagedIndexError(str(exc)) from exc
    if active is None:
        raise NoActiveManagedIndexError(
            f"Managed corpus {corpus.corpus_key!r} has no ACTIVE index generation"
        )
    if active.status != "ACTIVE":
        raise CorruptActiveManagedIndexError(
            f"Resolved managed index {active.index_state_id} is {active.status}, not ACTIVE"
        )

    configuration, sparse_encoder = _validate_active_metadata(active)
    try:
        expected_collection = managed_generation_collection_name(
            corpus.qdrant_collection_prefix,
            active.generation,
        )
    except ManagedQdrantError as exc:
        raise CorruptActiveManagedIndexError(
            f"Managed corpus generation naming is invalid: {exc}"
        ) from exc
    if active.qdrant_collection != expected_collection:
        raise CorruptActiveManagedIndexError(
            "ACTIVE Qdrant collection does not match its managed corpus prefix/generation: "
            f"{active.qdrant_collection!r} != {expected_collection!r}"
        )

    embedding = embedding_provider or HuggingFaceEmbeddingProvider(
        settings.models.embedding.model,
        settings.models.embedding.device,
        settings.models.embedding.trust_remote_code,
        settings.models.embedding.query_prompt_name,
    )
    _validate_embedding_compatibility(
        active,
        configuration,
        settings.models.embedding,
        embedding,
    )
    reranker = reranker_provider or HuggingFaceRerankerProvider(
        settings.models.reranker.model,
        settings.models.reranker.device,
        settings.models.reranker.trust_remote_code,
    )
    client = qdrant_client or QdrantClient(url=settings.qdrant.url)
    _validate_physical_collection(
        client,
        managed_prefix=corpus.qdrant_collection_prefix,
        active=active,
        configuration=configuration,
    )

    managed_qdrant_settings = settings.qdrant.model_copy(
        update={
            "collection_name": active.qdrant_collection,
            "dense_vector_name": configuration.dense_vector_name,
            "sparse_vector_name": configuration.sparse_vector_name,
            "hnsw_m": configuration.hnsw_m,
            "hnsw_ef_construct": configuration.hnsw_ef_construct,
        }
    )
    store = QdrantHybridStore(
        managed_qdrant_settings,
        client,
        query_hnsw_ef=settings.retrieval.hnsw_ef,
    )
    retrieval = HybridRetrievalService(
        embedding_provider=embedding,
        reranker=reranker,
        store=store,
        sparse_encoder=sparse_encoder,
        settings=settings.retrieval,
        tracer=tracer,
        collection_name=active.qdrant_collection,
        access_control_enabled=True,
    )
    return ManagedRetrievalRuntime(
        index_state_id=active.index_state_id,
        generation=active.generation,
        qdrant_collection=active.qdrant_collection,
        snapshot_hash=active.snapshot_hash,
        embedding_fingerprint=active.embedding_fingerprint,
        chunking_fingerprint=active.chunking_fingerprint,
        index_configuration_fingerprint=cast(
            str, active.index_configuration_fingerprint
        ),
        document_count=active.document_count,
        chunk_count=active.chunk_count,
        retrieval_service=retrieval,
    )


def _validate_active_metadata(
    active: IndexState,
) -> tuple[ManagedIndexConfiguration, BM25SparseEncoder]:
    required_strings = {
        "qdrant_collection": active.qdrant_collection,
        "bm25_state_hash": active.bm25_state_hash,
        "bm25_corpus_hash": active.bm25_corpus_hash,
        "embedding_fingerprint": active.embedding_fingerprint,
        "chunking_fingerprint": active.chunking_fingerprint,
        "index_configuration_fingerprint": active.index_configuration_fingerprint,
        "snapshot_hash": active.snapshot_hash,
    }
    missing = [name for name, value in required_strings.items() if not value]
    if active.bm25_state is None:
        missing.append("bm25_state")
    if active.index_configuration is None:
        missing.append("index_configuration")
    if active.embedding_dimension is None:
        missing.append("embedding_dimension")
    if missing:
        raise CorruptActiveManagedIndexError(
            "ACTIVE managed index is missing required metadata: "
            + ", ".join(sorted(missing))
        )
    if active.generation <= 0:
        raise CorruptActiveManagedIndexError("ACTIVE index generation must be positive")
    if active.embedding_dimension is None or active.embedding_dimension <= 0:
        raise CorruptActiveManagedIndexError(
            "ACTIVE embedding_dimension must be positive"
        )
    if active.document_count < 0 or active.chunk_count < 0:
        raise CorruptActiveManagedIndexError(
            "ACTIVE document_count and chunk_count cannot be negative"
        )

    raw_configuration = cast(dict[str, Any], active.index_configuration)
    calculated_configuration_hash = managed_index_configuration_payload_fingerprint(
        cast(Mapping[str, object], raw_configuration)
    )
    if calculated_configuration_hash != active.index_configuration_fingerprint:
        raise CorruptActiveManagedIndexError(
            "ACTIVE index configuration fingerprint does not match its payload"
        )
    configuration = _parse_configuration(raw_configuration)
    if configuration.payload_schema_version != PAYLOAD_SCHEMA_VERSION:
        raise CorruptActiveManagedIndexError(
            "ACTIVE managed index uses an unsupported payload schema; publish a new generation"
        )
    canonical_identity = managed_index_configuration_identity(configuration)
    if canonical_identity.payload != raw_configuration:
        raise CorruptActiveManagedIndexError(
            "ACTIVE index configuration is not the canonical managed configuration shape"
        )
    if canonical_identity.fingerprint != active.index_configuration_fingerprint:
        raise CorruptActiveManagedIndexError(
            "ACTIVE canonical index configuration fingerprint is inconsistent"
        )
    if configuration.embedding_fingerprint != active.embedding_fingerprint:
        raise CorruptActiveManagedIndexError(
            "ACTIVE embedding fingerprint disagrees with index configuration"
        )
    if configuration.chunking_fingerprint != active.chunking_fingerprint:
        raise CorruptActiveManagedIndexError(
            "ACTIVE chunking fingerprint disagrees with index configuration"
        )

    try:
        state = bm25_state_from_json_dict(cast(Mapping[str, Any], active.bm25_state))
    except (ValidationError, TypeError, ValueError) as exc:
        raise CorruptActiveManagedIndexError(
            f"ACTIVE BM25 state is invalid: {exc}"
        ) from exc
    if bm25_state_hash(state) != active.bm25_state_hash:
        raise CorruptActiveManagedIndexError(
            "ACTIVE BM25 state hash does not match its canonical state"
        )
    if state.corpus_hash != active.bm25_corpus_hash:
        raise CorruptActiveManagedIndexError(
            "ACTIVE BM25 corpus hash disagrees with persisted BM25 metadata"
        )
    if state.document_count != active.chunk_count:
        raise CorruptActiveManagedIndexError(
            "ACTIVE BM25 document_count does not match index chunk_count"
        )
    if state.k1 != configuration.bm25_k1 or state.b != configuration.bm25_b:
        raise CorruptActiveManagedIndexError(
            "ACTIVE BM25 k1/b do not match the managed index configuration"
        )
    return configuration, BM25SparseEncoder(state)


def _parse_configuration(payload: Mapping[str, Any]) -> ManagedIndexConfiguration:
    try:
        bm25 = _mapping(payload, "bm25")
        dense = _mapping(payload, "dense")
        sparse = _mapping(payload, "sparse")
        return ManagedIndexConfiguration(
            embedding_fingerprint=_string(payload, "embedding_fingerprint"),
            chunking_fingerprint=_string(payload, "chunking_fingerprint"),
            bm25_k1=_number(bm25, "k1"),
            bm25_b=_number(bm25, "b"),
            dense_vector_name=_string(dense, "vector_name"),
            sparse_vector_name=_string(sparse, "vector_name"),
            distance_metric=_string(dense, "distance_metric"),
            hnsw_m=_integer(dense, "hnsw_m"),
            hnsw_ef_construct=_integer(dense, "hnsw_ef_construct"),
            sparse_modifier=_string(sparse, "modifier"),
            sparse_on_disk=_boolean(sparse, "on_disk"),
            managed_index_format_version=_string(
                payload, "managed_index_format_version"
            ),
            bm25_implementation_version=_string(
                bm25, "implementation_version"
            ),
            bm25_tokenizer_version=_string(bm25, "tokenizer_version"),
            point_id_format_version=_string(payload, "point_id_format_version"),
            payload_schema_version=_string(payload, "payload_schema_version"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise CorruptActiveManagedIndexError(
            f"ACTIVE index configuration is invalid: {exc}"
        ) from exc


def _validate_embedding_compatibility(
    active: IndexState,
    configuration: ManagedIndexConfiguration,
    embedding_settings: EmbeddingModelSettings,
    provider: EmbeddingProvider,
) -> None:
    if provider.model_name != embedding_settings.model:
        raise ManagedEmbeddingCompatibilityError(
            "Embedding provider model does not match configured embedding model"
        )
    runtime_fingerprint = embedding_fingerprint(
        provider=embedding_settings.provider,
        model=embedding_settings.model,
        trust_remote_code=embedding_settings.trust_remote_code,
        device=embedding_settings.device,
    )
    if runtime_fingerprint != active.embedding_fingerprint:
        raise ManagedEmbeddingCompatibilityError(
            "Configured query embedding provider is incompatible with the ACTIVE "
            "document-vector fingerprint"
        )
    if runtime_fingerprint != configuration.embedding_fingerprint:
        raise ManagedEmbeddingCompatibilityError(
            "Configured embedding fingerprint is inconsistent with ACTIVE configuration"
        )
    exposed_dimension = getattr(provider, "dimension", None)
    if exposed_dimension is not None:
        if isinstance(exposed_dimension, bool) or not isinstance(exposed_dimension, int):
            raise ManagedEmbeddingCompatibilityError(
                "Embedding provider dimension must be a positive integer when exposed"
            )
        if exposed_dimension != active.embedding_dimension:
            raise ManagedEmbeddingCompatibilityError(
                "Embedding provider dimension does not match ACTIVE embedding_dimension"
            )


def _validate_physical_collection(
    client: QdrantClient,
    *,
    managed_prefix: str,
    active: IndexState,
    configuration: ManagedIndexConfiguration,
) -> None:
    try:
        inspector = ManagedQdrantStore(
            client,
            managed_prefix=managed_prefix,
            configuration=configuration,
            embedding_dimension=cast(int, active.embedding_dimension),
        )
    except ManagedQdrantError as exc:
        raise CorruptActiveManagedIndexError(
            f"ACTIVE managed Qdrant configuration is unsupported: {exc}"
        ) from exc
    try:
        metadata = inspector.inspect_collection(active.qdrant_collection)
        point_count = inspector.count_points(active.qdrant_collection)
    except Exception as exc:
        raise ManagedCollectionUnavailableError(
            f"ACTIVE managed Qdrant collection is unavailable: {exc}"
        ) from exc

    issues: list[str] = []
    if configuration.dense_vector_name not in metadata.dense_vector_names:
        issues.append(f"missing dense vector {configuration.dense_vector_name!r}")
    if configuration.sparse_vector_name not in metadata.sparse_vector_names:
        issues.append(f"missing sparse vector {configuration.sparse_vector_name!r}")
    if metadata.dense_dimension != active.embedding_dimension:
        issues.append(
            f"dense dimension {metadata.dense_dimension} != {active.embedding_dimension}"
        )
    if metadata.distance_metric != configuration.distance_metric.lower():
        issues.append(
            f"distance {metadata.distance_metric!r} != "
            f"{configuration.distance_metric.lower()!r}"
        )
    if point_count != active.chunk_count:
        issues.append(f"point count {point_count} != {active.chunk_count}")
    if issues:
        raise ManagedCollectionUnavailableError(
            "ACTIVE managed Qdrant validation failed: " + "; ".join(issues)
        )


def _mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload[key]
    if not isinstance(value, Mapping):
        raise TypeError(f"{key} must be an object")
    return cast(Mapping[str, Any], value)


def _string(payload: Mapping[str, Any], key: str) -> str:
    value = payload[key]
    if not isinstance(value, str) or not value:
        raise TypeError(f"{key} must be a non-empty string")
    return value


def _number(payload: Mapping[str, Any], key: str) -> float:
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{key} must be numeric")
    return float(value)


def _integer(payload: Mapping[str, Any], key: str) -> int:
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{key} must be an integer")
    return value


def _boolean(payload: Mapping[str, Any], key: str) -> bool:
    value = payload[key]
    if not isinstance(value, bool):
        raise TypeError(f"{key} must be a boolean")
    return value
