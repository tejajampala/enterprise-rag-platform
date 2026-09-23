from __future__ import annotations

import ast
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from consumer_finance_rag.hashing import sha256_text
from consumer_finance_rag.lineage.bm25_generation import (
    CorruptActiveBM25GenerationError,
    plan_bm25_generation,
)
from consumer_finance_rag.lineage.chunk_materialization import (
    active_chunk_manifest_hash,
)
from consumer_finance_rag.lineage.fingerprints import (
    BM25_IMPLEMENTATION_VERSION,
    BM25_TOKENIZER_VERSION,
    ManagedIndexConfiguration,
    managed_index_configuration_payload_fingerprint,
)
from consumer_finance_rag.lineage.index_types import (
    ActiveChunk,
    ActiveChunkManifest,
    ActiveDocumentVersion,
    BM25GenerationPlan,
    BM25PlanAction,
    PreviousBM25GenerationState,
)
from consumer_finance_rag.sparse import (
    BM25SparseEncoder,
    bm25_corpus_hash,
    bm25_state_hash,
    bm25_state_to_json_dict,
)

CHUNKING_FINGERPRINT = "c" * 64


def managed_configuration(**changes: object) -> ManagedIndexConfiguration:
    values: dict[str, object] = {
        "embedding_fingerprint": "e" * 64,
        "chunking_fingerprint": CHUNKING_FINGERPRINT,
        "bm25_k1": 1.2,
        "bm25_b": 0.75,
        "dense_vector_name": "dense_hnsw",
        "sparse_vector_name": "sparse_bm25",
        "distance_metric": "cosine",
        "hnsw_m": 16,
        "hnsw_ef_construct": 100,
        "sparse_modifier": "none",
        "sparse_on_disk": False,
        "managed_index_format_version": "managed-hybrid-index-v1",
    }
    values.update(changes)
    return ManagedIndexConfiguration(**values)  # type: ignore[arg-type]


def active_manifest(
    texts: tuple[str, ...] = ("alpha credit card", "beta mortgage", "gamma fraud"),
    *,
    logical_document_count: int = 1,
    title: str = "Consumer guidance",
    category: str = "credit-cards",
    chunking_fingerprint: str = CHUNKING_FINGERPRINT,
    document_pks: tuple[UUID, ...] | None = None,
    chunk_pks: tuple[UUID, ...] | None = None,
) -> ActiveChunkManifest:
    document_pks = document_pks or tuple(
        uuid4() for _ in range(logical_document_count)
    )
    document_version_ids = tuple(uuid4() for _ in range(logical_document_count))
    chunk_pks = chunk_pks or tuple(uuid4() for _ in texts)
    documents: list[ActiveDocumentVersion] = []
    for index in range(logical_document_count):
        source_key = f"https://example.test/source-{index}"
        content_hash = sha256_text(
            "\n".join(
                text
                for ordinal, text in enumerate(texts)
                if ordinal % logical_document_count == index
            )
        )
        documents.append(
            ActiveDocumentVersion(
                document_pk=document_pks[index],
                document_version_id=document_version_ids[index],
                version_number=2 if title != "Consumer guidance" else 1,
                source_key=source_key,
                source_id=f"source-{index}",
                source_url=source_key,
                title=title,
                category=category,
                source_version="2026-09-19",
                normalized_text="source text",
                content_hash=content_hash,
                metadata_hash=sha256_text(f"{title}:{category}"),
                version_hash=sha256_text(f"{content_hash}:{title}:{category}"),
                compatibility_document_id=f"document-{index}-{content_hash[:12]}",
            )
        )
    chunks: list[ActiveChunk] = []
    for ordinal, text in enumerate(texts):
        document_index = ordinal % logical_document_count
        document = documents[document_index]
        chunk_hash = sha256_text(text)
        chunks.append(
            ActiveChunk(
                document_pk=document.document_pk,
                document_version_id=document.document_version_id,
                version_number=document.version_number,
                source_key=document.source_key,
                source_id=document.source_id,
                source_url=document.source_url,
                title=document.title,
                category=document.category,
                source_version=document.source_version,
                content_hash=document.content_hash,
                compatibility_document_id=document.compatibility_document_id,
                chunk_pk=chunk_pks[ordinal],
                compatibility_chunk_id=f"{document.compatibility_document_id}:{ordinal}",
                ordinal=ordinal,
                chunk_hash=chunk_hash,
                text=text,
                chunking_fingerprint=chunking_fingerprint,
            )
        )
    document_tuple = tuple(documents)
    chunk_tuple = tuple(chunks)
    source_snapshot_hash = sha256_text(
        "|".join(document.version_hash for document in document_tuple)
    )
    manifest_hash = active_chunk_manifest_hash(
        corpus_key="consumer-finance-managed",
        source_snapshot_hash=source_snapshot_hash,
        chunking_fingerprint=chunking_fingerprint,
        documents=document_tuple,
        chunks=chunk_tuple,
    )
    return ActiveChunkManifest(
        corpus_id=uuid4(),
        corpus_key="consumer-finance-managed",
        source_snapshot_hash=source_snapshot_hash,
        chunking_fingerprint=chunking_fingerprint,
        documents=document_tuple,
        chunks=chunk_tuple,
        document_count=len(document_tuple),
        chunk_count=len(chunk_tuple),
        manifest_hash=manifest_hash,
    )


def previous_generation(
    plan: BM25GenerationPlan,
    configuration: ManagedIndexConfiguration,
) -> PreviousBM25GenerationState:
    payload = configuration.canonical_payload()
    return PreviousBM25GenerationState(
        index_state_id=uuid4(),
        bm25_state=bm25_state_to_json_dict(plan.state),
        bm25_state_hash=plan.state_hash,
        bm25_corpus_hash=plan.corpus_hash,
        index_configuration=payload,
        index_configuration_fingerprint=(
            managed_index_configuration_payload_fingerprint(payload)
        ),
        chunking_fingerprint=configuration.chunking_fingerprint,
        chunk_count=plan.chunk_count,
    )


def fitted_baseline(
    manifest: ActiveChunkManifest | None = None,
    configuration: ManagedIndexConfiguration | None = None,
) -> tuple[ActiveChunkManifest, ManagedIndexConfiguration, BM25GenerationPlan]:
    selected_manifest = manifest or active_manifest()
    selected_configuration = configuration or managed_configuration()
    plan = plan_bm25_generation(selected_manifest, selected_configuration)
    assert plan.action is BM25PlanAction.REFIT
    return selected_manifest, selected_configuration, plan


def test_first_generation_refits_all_active_chunk_texts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = active_manifest(logical_document_count=1)
    configuration = managed_configuration()
    observed: list[tuple[str, ...]] = []
    original_fit = BM25SparseEncoder.fit

    def recording_fit(
        cls: type[BM25SparseEncoder],
        texts: Sequence[str],
        *,
        k1: float,
        b: float,
    ) -> BM25SparseEncoder:
        del cls
        observed.append(tuple(texts))
        return original_fit(texts, k1=k1, b=b)

    monkeypatch.setattr(BM25SparseEncoder, "fit", classmethod(recording_fit))

    plan = plan_bm25_generation(manifest, configuration)

    assert plan.action is BM25PlanAction.REFIT
    assert observed == [tuple(chunk.text for chunk in manifest.chunks)]
    assert plan.logical_document_count == manifest.document_count == 1
    assert plan.chunk_count == manifest.chunk_count == 3
    assert plan.state.document_count == manifest.chunk_count
    assert plan.state.document_count != manifest.document_count


def test_refit_hashes_are_deterministic_and_use_existing_corpus_semantics() -> None:
    manifest = active_manifest()
    configuration = managed_configuration()

    first = plan_bm25_generation(manifest, configuration)
    second = plan_bm25_generation(manifest, configuration)

    expected_corpus_hash = bm25_corpus_hash(
        tuple(chunk.text for chunk in manifest.chunks)
    )
    assert first.corpus_hash == second.corpus_hash == expected_corpus_hash
    assert first.state_hash == second.state_hash == bm25_state_hash(first.state)
    assert first.corpus_hash != manifest.manifest_hash


def test_valid_previous_generation_reuses_without_fitting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, configuration, baseline = fitted_baseline()
    previous = previous_generation(baseline, configuration)

    def unexpected_fit(*_args: object, **_kwargs: object) -> BM25SparseEncoder:
        raise AssertionError("REUSE must not fit BM25")

    monkeypatch.setattr(BM25SparseEncoder, "fit", unexpected_fit)

    plan = plan_bm25_generation(manifest, configuration, previous)

    assert plan.action is BM25PlanAction.REUSE
    assert plan.state == baseline.state
    assert plan.state_hash == baseline.state_hash


@pytest.mark.parametrize(
    ("metadata_field", "value"),
    [
        ("title", "Updated citation title"),
        ("category", "payments-fraud"),
    ],
)
def test_metadata_only_change_reuses_bm25(
    metadata_field: str,
    value: str,
) -> None:
    original, configuration, baseline = fitted_baseline()
    changed = (
        active_manifest(title=value)
        if metadata_field == "title"
        else active_manifest(category=value)
    )

    plan = plan_bm25_generation(
        changed,
        configuration,
        previous_generation(baseline, configuration),
    )

    assert changed.manifest_hash != original.manifest_hash
    assert plan.corpus_hash == baseline.corpus_hash
    assert plan.action is BM25PlanAction.REUSE


@pytest.mark.parametrize(
    "current_config",
    [
        managed_configuration(embedding_fingerprint="f" * 64),
        managed_configuration(hnsw_m=32),
        managed_configuration(hnsw_ef_construct=200),
    ],
    ids=["embedding", "hnsw-m", "hnsw-ef-construct"],
)
def test_dense_only_configuration_changes_reuse_bm25(
    current_config: ManagedIndexConfiguration,
) -> None:
    manifest, previous_config, baseline = fitted_baseline()

    plan = plan_bm25_generation(
        manifest,
        current_config,
        previous_generation(baseline, previous_config),
    )

    assert plan.action is BM25PlanAction.REUSE


@pytest.mark.parametrize(
    ("current_config", "reason"),
    [
        (managed_configuration(bm25_k1=1.6), "k1"),
        (managed_configuration(bm25_b=0.5), "b changed"),
        (
            managed_configuration(bm25_implementation_version="client-bm25-v2"),
            "implementation",
        ),
        (
            managed_configuration(bm25_tokenizer_version="tokenizer-v2"),
            "tokenizer",
        ),
    ],
)
def test_bm25_compatibility_changes_refit(
    current_config: ManagedIndexConfiguration,
    reason: str,
) -> None:
    manifest, previous_config, baseline = fitted_baseline()

    plan = plan_bm25_generation(
        manifest,
        current_config,
        previous_generation(baseline, previous_config),
    )

    assert plan.action is BM25PlanAction.REFIT
    assert reason in plan.reason
    assert plan.state.k1 == current_config.bm25_k1
    assert plan.state.b == current_config.bm25_b


@pytest.mark.parametrize(
    "current_manifest",
    [
        active_manifest(
            ("alpha credit card", "beta mortgage", "gamma fraud", "new document"),
            logical_document_count=2,
        ),
        active_manifest(("alpha credit card", "beta mortgage")),
        active_manifest(("changed credit text", "beta mortgage", "gamma fraud")),
        active_manifest(("beta mortgage", "alpha credit card", "gamma fraud")),
    ],
    ids=["document-added", "chunk-removed", "text-changed", "order-changed"],
)
def test_corpus_membership_text_or_order_changes_refit(
    current_manifest: ActiveChunkManifest,
) -> None:
    original, configuration, baseline = fitted_baseline()

    plan = plan_bm25_generation(
        current_manifest,
        configuration,
        previous_generation(baseline, configuration),
    )

    assert bm25_corpus_hash(
        tuple(chunk.text for chunk in current_manifest.chunks)
    ) != bm25_corpus_hash(tuple(chunk.text for chunk in original.chunks))
    assert plan.action is BM25PlanAction.REFIT
    assert plan.corpus_hash == bm25_corpus_hash(
        tuple(chunk.text for chunk in current_manifest.chunks)
    )


def test_logical_document_deletion_refits_bm25() -> None:
    original = active_manifest(
        ("first document text", "second document text"),
        logical_document_count=2,
    )
    configuration = managed_configuration()
    baseline = plan_bm25_generation(original, configuration)
    current = active_manifest(("first document text",), logical_document_count=1)

    plan = plan_bm25_generation(
        current,
        configuration,
        previous_generation(baseline, configuration),
    )

    assert current.document_count == original.document_count - 1
    assert plan.action is BM25PlanAction.REFIT


def test_changed_chunking_fingerprint_refits_even_for_same_texts() -> None:
    manifest, previous_config, baseline = fitted_baseline()
    changed_fingerprint = "d" * 64
    changed_manifest = replace(
        manifest,
        chunking_fingerprint=changed_fingerprint,
        chunks=tuple(
            replace(chunk, chunking_fingerprint=changed_fingerprint)
            for chunk in manifest.chunks
        ),
    )
    current_config = replace(
        previous_config,
        chunking_fingerprint=changed_fingerprint,
    )

    plan = plan_bm25_generation(
        changed_manifest,
        current_config,
        previous_generation(baseline, previous_config),
    )

    assert plan.action is BM25PlanAction.REFIT
    assert "Chunking fingerprint" in plan.reason


def test_missing_bm25_state_on_active_generation_is_corruption() -> None:
    manifest, configuration, baseline = fitted_baseline()
    previous = replace(
        previous_generation(baseline, configuration),
        bm25_state=None,
    )

    with pytest.raises(CorruptActiveBM25GenerationError, match="bm25_state"):
        plan_bm25_generation(manifest, configuration, previous)


def test_invalid_bm25_state_hash_is_corruption() -> None:
    manifest, configuration, baseline = fitted_baseline()
    previous = replace(
        previous_generation(baseline, configuration),
        bm25_state_hash="f" * 64,
    )

    with pytest.raises(CorruptActiveBM25GenerationError, match="state hash"):
        plan_bm25_generation(manifest, configuration, previous)


def test_inconsistent_bm25_corpus_hash_is_corruption() -> None:
    manifest, configuration, baseline = fitted_baseline()
    previous = replace(
        previous_generation(baseline, configuration),
        bm25_corpus_hash="f" * 64,
    )

    with pytest.raises(CorruptActiveBM25GenerationError, match="corpus hash"):
        plan_bm25_generation(manifest, configuration, previous)


def test_previous_chunk_count_inconsistent_with_state_is_corruption() -> None:
    manifest, configuration, baseline = fitted_baseline()
    previous = replace(
        previous_generation(baseline, configuration),
        chunk_count=baseline.chunk_count + 1,
    )

    with pytest.raises(CorruptActiveBM25GenerationError, match="chunk count"):
        plan_bm25_generation(manifest, configuration, previous)


def test_planner_is_independent_of_database_uuid_values() -> None:
    original, configuration, baseline = fitted_baseline()
    remapped = active_manifest(
        tuple(chunk.text for chunk in original.chunks),
        document_pks=tuple(uuid4() for _ in original.documents),
        chunk_pks=tuple(uuid4() for _ in original.chunks),
    )

    plan = plan_bm25_generation(
        remapped,
        configuration,
        previous_generation(baseline, configuration),
    )

    assert plan.action is BM25PlanAction.REUSE
    assert plan.corpus_hash == baseline.corpus_hash
    assert plan.state_hash == baseline.state_hash


def test_bm25_identity_constants_match_default_managed_configuration() -> None:
    configuration = managed_configuration()

    assert configuration.bm25_implementation_version == BM25_IMPLEMENTATION_VERSION
    assert configuration.bm25_tokenizer_version == BM25_TOKENIZER_VERSION


def test_bm25_planner_has_no_qdrant_dense_or_database_dependencies() -> None:
    path = Path("src/consumer_finance_rag/lineage/bm25_generation.py")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    forbidden = {
        "qdrant_client",
        "sqlalchemy",
        "consumer_finance_rag.providers",
        "consumer_finance_rag.vector_store",
        "consumer_finance_rag.lineage.models",
        "consumer_finance_rag.lineage.repository",
    }
    assert not {
        name
        for name in imported
        if any(name == root or name.startswith(f"{root}.") for root in forbidden)
    }
