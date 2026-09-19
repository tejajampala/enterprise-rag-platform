from __future__ import annotations

import ast
import math
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from uuid import UUID, uuid4, uuid5

import pytest

from consumer_finance_rag.hashing import sha256_text
from consumer_finance_rag.lineage.bm25_generation import plan_bm25_generation
from consumer_finance_rag.lineage.fingerprints import ManagedIndexConfiguration
from consumer_finance_rag.lineage.index_types import (
    ActiveChunk,
    ActiveChunkManifest,
    ActiveDocumentVersion,
    BM25GenerationPlan,
    BM25PlanAction,
    DenseGenerationPlan,
    DensePlanAction,
    PreparedDenseChunk,
    PreparedIndexIdentity,
    PreparedManagedGeneration,
    managed_point_id,
)
from consumer_finance_rag.lineage.point_preparation import (
    ManagedPointPreparationError,
    PlanAlignmentError,
    prepare_managed_generation,
)
from consumer_finance_rag.sparse import (
    BM25SparseEncoder,
    BM25State,
    SparseVectorData,
    bm25_state_hash,
)

NAMESPACE = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
CHUNKING_FINGERPRINT = "c" * 64
EMBEDDING_FINGERPRINT = "e" * 64


def uid(label: str) -> UUID:
    return uuid5(NAMESPACE, label)


def configuration(
    *, chunking_fingerprint: str = CHUNKING_FINGERPRINT
) -> ManagedIndexConfiguration:
    return ManagedIndexConfiguration(
        embedding_fingerprint=EMBEDDING_FINGERPRINT,
        chunking_fingerprint=chunking_fingerprint,
        bm25_k1=1.2,
        bm25_b=0.75,
        dense_vector_name="dense_hnsw",
        sparse_vector_name="sparse_bm25",
        distance_metric="cosine",
        hnsw_m=16,
        hnsw_ef_construct=100,
        sparse_modifier="none",
        sparse_on_disk=False,
        managed_index_format_version="managed-hybrid-index-v1",
    )


def active_manifest(
    texts: tuple[str, ...] = (
        "credit card billing dispute",
        "mortgage payment assistance",
        "fraudulent bank transfer",
        "debt collection validation",
        "credit report correction",
    ),
    *,
    document_count: int = 3,
    title: str = "Current consumer guidance",
    chunking_fingerprint: str = CHUNKING_FINGERPRINT,
    chunk_pks: tuple[UUID, ...] | None = None,
) -> ActiveChunkManifest:
    document_pks = tuple(uid(f"document-{index}") for index in range(document_count))
    version_pks = tuple(uid(f"version-{title}-{index}") for index in range(document_count))
    chunks_by_document = tuple(
        tuple(text for offset, text in enumerate(texts) if offset % document_count == index)
        for index in range(document_count)
    )
    documents = tuple(
        ActiveDocumentVersion(
            document_pk=document_pks[index],
            document_version_id=version_pks[index],
            version_number=2,
            source_key=f"cfpb:source-{index}",
            source_id=f"source-{index}",
            source_url=f"https://consumerfinance.gov/source-{index}",
            title=title,
            category="credit-cards" if index == 0 else "mortgages",
            source_version="2026-09-19",
            normalized_text="\n".join(chunks_by_document[index]),
            content_hash=sha256_text("\n".join(chunks_by_document[index])),
            metadata_hash=sha256_text(f"metadata:{title}:{index}"),
            version_hash=sha256_text(f"version:{title}:{index}"),
            compatibility_document_id=f"document-{index}",
        )
        for index in range(document_count)
    )
    selected_chunk_pks = chunk_pks or tuple(
        uid(f"chunk-{title}-{index}") for index in range(len(texts))
    )
    chunks = tuple(
        ActiveChunk(
            document_pk=documents[index % document_count].document_pk,
            document_version_id=(
                documents[index % document_count].document_version_id
            ),
            version_number=documents[index % document_count].version_number,
            source_key=documents[index % document_count].source_key,
            source_id=documents[index % document_count].source_id,
            source_url=documents[index % document_count].source_url,
            title=documents[index % document_count].title,
            category=documents[index % document_count].category,
            source_version=documents[index % document_count].source_version,
            content_hash=documents[index % document_count].content_hash,
            compatibility_document_id=(
                documents[index % document_count].compatibility_document_id
            ),
            chunk_pk=selected_chunk_pks[index],
            compatibility_chunk_id=f"chunk-{index}",
            ordinal=index,
            chunk_hash=sha256_text(text),
            text=text,
            chunking_fingerprint=chunking_fingerprint,
        )
        for index, text in enumerate(texts)
    )
    return ActiveChunkManifest(
        corpus_id=uid("corpus"),
        corpus_key="consumer-finance-managed",
        source_snapshot_hash=sha256_text("snapshot"),
        chunking_fingerprint=chunking_fingerprint,
        documents=documents,
        chunks=chunks,
        document_count=len(documents),
        chunk_count=len(chunks),
        manifest_hash=sha256_text("manifest:" + ":".join(texts)),
    )


def dense_plan(
    manifest: ActiveChunkManifest,
    *,
    reused_positions: frozenset[int] = frozenset(),
    dimension: int = 3,
) -> DenseGenerationPlan:
    entries: list[PreparedDenseChunk] = []
    for index, chunk in enumerate(manifest.chunks):
        reused = index in reused_positions
        entries.append(
            PreparedDenseChunk(
                lineage_chunk_pk=chunk.chunk_pk,
                compatibility_chunk_id=chunk.compatibility_chunk_id,
                chunk_hash=chunk.chunk_hash,
                action=(DensePlanAction.REUSED if reused else DensePlanAction.COMPUTED),
                dense_vector=tuple(float(index + offset + 1) for offset in range(dimension)),
                dense_source_chunk_pk=uid(f"old-chunk-{index}") if reused else None,
                dense_source_point_id=uid(f"old-point-{index}") if reused else None,
                reason="test",
            )
        )
    return DenseGenerationPlan(
        embedding_fingerprint=EMBEDDING_FINGERPRINT,
        embedding_dimension=dimension,
        chunks=tuple(entries),
    )


def bm25_plan(
    manifest: ActiveChunkManifest,
    *,
    action: BM25PlanAction = BM25PlanAction.REFIT,
) -> BM25GenerationPlan:
    plan = plan_bm25_generation(
        manifest,
        configuration(chunking_fingerprint=manifest.chunking_fingerprint),
    )
    return replace(plan, action=action)


def identity(
    *,
    chunking_fingerprint: str = CHUNKING_FINGERPRINT,
    embedding_dimension: int = 3,
) -> PreparedIndexIdentity:
    return PreparedIndexIdentity(
        ingestion_run_id=uid("ingestion-run"),
        index_state_id=uid("index-state"),
        generation=7,
        chunking_fingerprint=chunking_fingerprint,
        embedding_fingerprint=EMBEDDING_FINGERPRINT,
        embedding_dimension=embedding_dimension,
    )


def prepared(
    manifest: ActiveChunkManifest | None = None,
    *,
    reused_positions: frozenset[int] = frozenset(),
    action: BM25PlanAction = BM25PlanAction.REFIT,
) -> PreparedManagedGeneration:
    selected = manifest or active_manifest()
    return prepare_managed_generation(
        selected,
        dense_plan(selected, reused_positions=reused_positions),
        bm25_plan(selected, action=action),
        identity(chunking_fingerprint=selected.chunking_fingerprint),
    )


def test_complete_first_generation_preserves_order_counts_and_dense_values() -> None:
    manifest = active_manifest(document_count=3)
    dense = dense_plan(manifest)

    result = prepare_managed_generation(
        manifest,
        dense,
        bm25_plan(manifest),
        identity(),
    )

    assert result.document_count == 3
    assert result.chunk_count == 5
    assert len(result.points) == 5
    assert len(result.memberships) == 5
    assert [point.chunk.chunk_pk for point in result.points] == [
        chunk.chunk_pk for chunk in manifest.chunks
    ]
    assert [point.dense_vector for point in result.points] == [
        entry.dense_vector for entry in dense.chunks
    ]
    assert all(
        point.dense_vector is entry.dense_vector
        for point, entry in zip(result.points, dense.chunks, strict=True)
    )
    assert result.reused_dense_count == 0
    assert result.computed_dense_count == 5


def test_dense_reused_and_computed_provenance_is_preserved() -> None:
    manifest = active_manifest(texts=("alpha credit", "beta mortgage"), document_count=2)
    dense = dense_plan(manifest, reused_positions=frozenset({0}))

    result = prepare_managed_generation(
        manifest, dense, bm25_plan(manifest), identity()
    )

    assert result.reused_dense_count == 1
    assert result.computed_dense_count == 1
    assert result.memberships[0].dense_action is DensePlanAction.REUSED
    assert (
        result.memberships[0].dense_source_chunk_pk
        == dense.chunks[0].dense_source_chunk_pk
    )
    assert result.memberships[1].dense_action is DensePlanAction.COMPUTED
    assert result.memberships[1].dense_source_chunk_pk is None


@pytest.mark.parametrize("action", [BM25PlanAction.REUSE, BM25PlanAction.REFIT])
def test_selected_bm25_state_encodes_every_chunk_once_without_refitting(
    action: BM25PlanAction,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = active_manifest()
    sparse_plan = bm25_plan(manifest, action=action)
    expected = tuple(
        BM25SparseEncoder(sparse_plan.state).encode_document(chunk.text)
        for chunk in manifest.chunks
    )
    encoded_texts: list[str] = []
    original_encode = BM25SparseEncoder.encode_document

    def record_encode(self: BM25SparseEncoder, text: str) -> SparseVectorData:
        encoded_texts.append(text)
        return original_encode(self, text)

    monkeypatch.setattr(BM25SparseEncoder, "encode_document", record_encode)
    monkeypatch.setattr(
        BM25SparseEncoder,
        "fit",
        classmethod(lambda _cls, *_args, **_kwargs: pytest.fail("fit called")),
    )

    result = prepare_managed_generation(
        manifest, dense_plan(manifest), sparse_plan, identity()
    )

    assert encoded_texts == [chunk.text for chunk in manifest.chunks]
    assert [point.sparse_indices for point in result.points] == [
        tuple(vector.indices) for vector in expected
    ]
    assert [point.sparse_values for point in result.points] == [
        tuple(vector.values) for vector in expected
    ]
    assert result.bm25_action is action


def test_sparse_vectors_are_deterministic_and_use_selected_vocabulary() -> None:
    manifest = active_manifest(texts=("alpha beta alpha", "beta gamma"), document_count=2)
    sparse_plan = bm25_plan(manifest)

    first = prepare_managed_generation(
        manifest, dense_plan(manifest), sparse_plan, identity()
    )
    second = prepare_managed_generation(
        manifest, dense_plan(manifest), sparse_plan, identity()
    )

    valid_indices = set(sparse_plan.state.vocabulary.values())
    assert first.points == second.points
    assert all(set(point.sparse_indices) <= valid_indices for point in first.points)
    assert first.points[0].sparse_indices == tuple(
        sorted(
            {
                sparse_plan.state.vocabulary["alpha"],
                sparse_plan.state.vocabulary["beta"],
            }
        )
    )


@pytest.mark.parametrize(
    "bad_state",
    [
        BM25State.model_construct(
            vocabulary={"alpha": 2},
            document_frequencies={"alpha": 1},
            document_count=1,
            average_document_length=1.0,
            corpus_hash="a" * 64,
            k1=1.2,
            b=0.75,
        ),
        BM25State.model_construct(
            vocabulary={"alpha": 0},
            document_frequencies={"beta": 1},
            document_count=1,
            average_document_length=1.0,
            corpus_hash="a" * 64,
            k1=1.2,
            b=0.75,
        ),
        BM25State.model_construct(
            vocabulary={"alpha": 0},
            document_frequencies={"alpha": 2},
            document_count=1,
            average_document_length=1.0,
            corpus_hash="a" * 64,
            k1=1.2,
            b=0.75,
        ),
    ],
)
def test_malformed_bm25_state_is_rejected(bad_state: BM25State) -> None:
    manifest = active_manifest(texts=("alpha",), document_count=1)
    plan = replace(
        bm25_plan(manifest),
        state=bad_state,
        state_hash=bm25_state_hash(bad_state),
    )

    with pytest.raises(ManagedPointPreparationError, match="BM25"):
        prepare_managed_generation(
            manifest, dense_plan(manifest), plan, identity()
        )


def test_bm25_chunk_and_state_document_count_mismatches_are_rejected() -> None:
    manifest = active_manifest()
    plan = bm25_plan(manifest)
    with pytest.raises(PlanAlignmentError, match="chunk count"):
        prepare_managed_generation(
            manifest,
            dense_plan(manifest),
            replace(plan, chunk_count=plan.chunk_count + 1),
            identity(),
        )

    bad_state = plan.state.model_copy(update={"document_count": plan.chunk_count + 1})
    with pytest.raises(PlanAlignmentError, match="state document count"):
        prepare_managed_generation(
            manifest,
            dense_plan(manifest),
            replace(plan, state=bad_state, state_hash=bm25_state_hash(bad_state)),
            identity(),
        )


@pytest.mark.parametrize(
    "change",
    ["state_hash", "ordered_ids", "logical_documents", "parameters"],
)
def test_bm25_internal_plan_alignment_is_validated(change: str) -> None:
    manifest = active_manifest()
    plan = bm25_plan(manifest)
    if change == "state_hash":
        plan = replace(plan, state_hash="f" * 64)
    elif change == "ordered_ids":
        plan = replace(plan, ordered_chunk_ids=tuple(reversed(plan.ordered_chunk_ids)))
    elif change == "logical_documents":
        plan = replace(plan, logical_document_count=plan.logical_document_count + 1)
    else:
        plan = replace(plan, bm25_k1=plan.bm25_k1 + 0.1)

    with pytest.raises(PlanAlignmentError):
        prepare_managed_generation(
            manifest, dense_plan(manifest), plan, identity()
        )


def test_sparse_encoder_output_is_validated_against_selected_vocabulary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = active_manifest()
    plan = bm25_plan(manifest)
    monkeypatch.setattr(
        BM25SparseEncoder,
        "encode_document",
        lambda _self, _text: SparseVectorData(indices=[999], values=[1.0]),
    )

    with pytest.raises(ManagedPointPreparationError, match="vocabulary"):
        prepare_managed_generation(
            manifest, dense_plan(manifest), plan, identity()
        )


@pytest.mark.parametrize("target", ["plan", "state"])
def test_bm25_corpus_hash_mismatch_is_rejected(target: str) -> None:
    manifest = active_manifest()
    plan = bm25_plan(manifest)
    if target == "plan":
        plan = replace(plan, corpus_hash="f" * 64)
    else:
        state = plan.state.model_copy(update={"corpus_hash": "f" * 64})
        plan = replace(plan, state=state, state_hash=bm25_state_hash(state))

    with pytest.raises(PlanAlignmentError, match="corpus hash"):
        prepare_managed_generation(
            manifest, dense_plan(manifest), plan, identity()
        )


@pytest.mark.parametrize("change", ["missing", "extra", "reordered"])
def test_dense_plan_cardinality_and_order_must_match_manifest(change: str) -> None:
    manifest = active_manifest()
    dense = dense_plan(manifest)
    if change == "missing":
        dense = replace(dense, chunks=dense.chunks[:-1])
    elif change == "extra":
        dense = replace(dense, chunks=(*dense.chunks, dense.chunks[-1]))
    else:
        dense = replace(dense, chunks=(dense.chunks[1], dense.chunks[0], *dense.chunks[2:]))

    with pytest.raises(PlanAlignmentError, match=r"exactly one|lineage chunk"):
        prepare_managed_generation(
            manifest, dense, bm25_plan(manifest), identity()
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("lineage_chunk_pk", uuid4(), "lineage chunk"),
        ("compatibility_chunk_id", "wrong", "compatibility chunk ID"),
        ("chunk_hash", "f" * 64, "chunk hash"),
    ],
)
def test_dense_business_identity_mismatch_is_rejected(
    field: str, value: object, message: str
) -> None:
    manifest = active_manifest()
    dense = dense_plan(manifest)
    original = dense.chunks[0]
    if field == "lineage_chunk_pk":
        changed = replace(original, lineage_chunk_pk=value)  # type: ignore[arg-type]
    elif field == "compatibility_chunk_id":
        changed = replace(original, compatibility_chunk_id=value)  # type: ignore[arg-type]
    else:
        changed = replace(original, chunk_hash=value)  # type: ignore[arg-type]
    dense = replace(
        dense,
        chunks=(changed, *dense.chunks[1:]),
    )

    with pytest.raises(PlanAlignmentError, match=message):
        prepare_managed_generation(
            manifest, dense, bm25_plan(manifest), identity()
        )


@pytest.mark.parametrize(
    "vector",
    [(1.0, 2.0), (1.0, 2.0, math.nan), (1.0, 2.0, math.inf)],
)
def test_dense_dimension_and_finite_values_are_validated(
    vector: tuple[float, ...]
) -> None:
    manifest = active_manifest()
    dense = dense_plan(manifest)
    dense = replace(
        dense,
        chunks=(replace(dense.chunks[0], dense_vector=vector), *dense.chunks[1:]),
    )

    with pytest.raises((PlanAlignmentError, ManagedPointPreparationError)):
        prepare_managed_generation(
            manifest, dense, bm25_plan(manifest), identity()
        )


@pytest.mark.parametrize(
    "changed_identity",
    [
        replace(identity(), embedding_dimension=4),
        replace(identity(), embedding_fingerprint="f" * 64),
        replace(identity(), chunking_fingerprint="f" * 64),
    ],
)
def test_prepared_index_identity_must_match_manifest_and_dense_plan(
    changed_identity: PreparedIndexIdentity,
) -> None:
    manifest = active_manifest()

    with pytest.raises(PlanAlignmentError):
        prepare_managed_generation(
            manifest,
            dense_plan(manifest),
            bm25_plan(manifest),
            changed_identity,
        )


@pytest.mark.parametrize(
    ("action", "source"),
    [
        (DensePlanAction.REUSED, None),
        (DensePlanAction.COMPUTED, uid("unexpected-source")),
    ],
)
def test_dense_provenance_contract_is_validated(
    action: DensePlanAction, source: UUID | None
) -> None:
    manifest = active_manifest()
    dense = dense_plan(manifest)
    dense = replace(
        dense,
        chunks=(
            replace(
                dense.chunks[0],
                action=action,
                dense_source_chunk_pk=source,
            ),
            *dense.chunks[1:],
        ),
    )

    with pytest.raises(PlanAlignmentError, match="provenance"):
        prepare_managed_generation(
            manifest, dense, bm25_plan(manifest), identity()
        )


def test_point_ids_payloads_and_results_are_deterministic_and_complete() -> None:
    manifest = active_manifest()
    first = prepared(manifest)
    second = prepared(manifest)
    legacy_fields = {
        "document_id",
        "chunk_id",
        "document_version",
        "title",
        "category",
        "source_url",
        "source_id",
        "chunk_hash",
        "ordinal",
        "text",
    }
    lineage_fields = {
        "content_hash",
        "source_version",
        "source_key",
        "lineage_document_pk",
        "lineage_document_version_id",
        "lineage_chunk_pk",
        "lineage_version_number",
        "ingestion_run_id",
        "index_state_id",
        "index_generation",
        "chunking_fingerprint",
        "point_id_format_version",
        "payload_schema_version",
    }

    assert first == second
    assert all(isinstance(point.payload, MappingProxyType) for point in first.points)
    assert [point.qdrant_point_id for point in first.points] == [
        managed_point_id(chunk.chunk_pk) for chunk in manifest.chunks
    ]
    assert [point.qdrant_point_id for point in first.points] == [
        membership.qdrant_point_id for membership in first.memberships
    ]
    assert all(
        legacy_fields | lineage_fields <= set(point.payload)
        for point in first.points
    )


def test_metadata_only_version_uses_current_payload_and_new_point_identity() -> None:
    old_chunk_pk = uid("old-lineage-chunk")
    new_chunk_pk = uid("new-lineage-chunk")
    manifest = active_manifest(
        texts=("identical consumer evidence",),
        document_count=1,
        title="New title",
        chunk_pks=(new_chunk_pk,),
    )
    dense = dense_plan(manifest, reused_positions=frozenset({0}))
    dense = replace(
        dense,
        chunks=(replace(dense.chunks[0], dense_source_chunk_pk=old_chunk_pk),),
    )

    result = prepare_managed_generation(
        manifest,
        dense,
        bm25_plan(manifest, action=BM25PlanAction.REUSE),
        identity(),
    )
    point = result.points[0]

    assert point.payload["title"] == "New title"
    assert point.payload["lineage_document_version_id"] == str(
        manifest.chunks[0].document_version_id
    )
    assert point.payload["lineage_chunk_pk"] == str(new_chunk_pk)
    assert point.qdrant_point_id == managed_point_id(new_chunk_pk)
    assert point.qdrant_point_id != managed_point_id(old_chunk_pk)
    assert point.dense_vector is dense.chunks[0].dense_vector
    assert result.memberships[0].dense_source_chunk_pk == old_chunk_pk


def test_chunking_change_prepares_mixed_dense_actions_with_refit_bm25() -> None:
    new_fingerprint = "n" * 64
    manifest = active_manifest(chunking_fingerprint=new_fingerprint)
    result = prepare_managed_generation(
        manifest,
        dense_plan(manifest, reused_positions=frozenset({0, 2})),
        bm25_plan(manifest, action=BM25PlanAction.REFIT),
        identity(chunking_fingerprint=new_fingerprint),
    )

    assert result.chunk_count == manifest.chunk_count == len(result.points)
    assert result.reused_dense_count == 2
    assert result.computed_dense_count == 3
    assert result.bm25_action is BM25PlanAction.REFIT
    assert all(
        point.payload["chunking_fingerprint"] == new_fingerprint
        for point in result.points
    )


def test_empty_manifest_is_rejected() -> None:
    manifest = active_manifest(texts=(), document_count=1)
    empty_dense = dense_plan(manifest)
    state = BM25State(
        vocabulary={"placeholder": 0},
        document_frequencies={"placeholder": 1},
        document_count=1,
        average_document_length=1.0,
        corpus_hash=sha256_text("placeholder"),
    )
    sparse_plan = BM25GenerationPlan(
        action=BM25PlanAction.REFIT,
        corpus_hash=state.corpus_hash,
        state=state,
        state_hash=bm25_state_hash(state),
        logical_document_count=1,
        chunk_count=0,
        bm25_k1=state.k1,
        bm25_b=state.b,
        reason="test",
        ordered_chunk_ids=(),
    )

    with pytest.raises(ManagedPointPreparationError, match="empty"):
        prepare_managed_generation(manifest, empty_dense, sparse_plan, identity())


def test_point_preparation_has_no_external_service_or_model_dependencies() -> None:
    path = Path("src/consumer_finance_rag/lineage/point_preparation.py")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported: set[str] = set()
    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    forbidden = {
        "qdrant_client",
        "sqlalchemy",
        "consumer_finance_rag.providers",
        "consumer_finance_rag.lineage.repository",
        "consumer_finance_rag.lineage.managed_vector_store",
    }
    assert not {
        name
        for name in imported
        if any(name == root or name.startswith(f"{root}.") for root in forbidden)
    }
    assert "fit" not in attributes
    assert not {
        "embed_documents",
        "create_generation_collection",
        "upsert_points",
        "retrieve_dense_vectors",
        "create_building_index_state",
        "record_index_state_chunk",
    } & attributes
