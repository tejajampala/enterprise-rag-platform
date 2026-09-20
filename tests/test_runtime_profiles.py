from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from sqlalchemy import create_engine

from consumer_finance_rag.config import DatabaseSettings, RuntimeSettings, load_settings
from consumer_finance_rag.factory import (
    RetrievalRuntime,
    RuntimeProfile,
    build_application_runtime,
    build_pipeline,
    build_runtime_retrieval,
)
from consumer_finance_rag.lineage.database import DatabaseConfigurationError
from consumer_finance_rag.lineage.managed_retrieval import (
    CorruptActiveManagedIndexError,
    ManagedEmbeddingCompatibilityError,
    ManagedRetrievalRuntime,
    NoActiveManagedIndexError,
)
from consumer_finance_rag.models import RetrievedChunk
from consumer_finance_rag.observability import NoOpTracer
from consumer_finance_rag.retrieval import HybridRetrievalService
from consumer_finance_rag.sparse import BM25SparseEncoder, BM25State


class FakeEmbedding:
    model_name = "fake-embedding"

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        raise AssertionError(f"runtime construction must not embed documents: {texts!r}")

    def embed_query(self, text: str) -> list[float]:
        del text
        return [1.0]


class FakeReranker:
    model_name = "fake-reranker"

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        del query
        return list(candidates[:top_k])


class FakeStore:
    def search_dense(self, vector: list[float], limit: int) -> list[RetrievedChunk]:
        del vector, limit
        return []

    def search_sparse(
        self,
        query: str,
        encoder: BM25SparseEncoder,
        limit: int,
    ) -> list[RetrievedChunk]:
        del query, encoder, limit
        return []


def fake_service(collection: str) -> HybridRetrievalService:
    state = BM25State(
        vocabulary={"test": 0},
        document_frequencies={"test": 1},
        document_count=1,
        average_document_length=1.0,
        corpus_hash="a" * 64,
    )
    return HybridRetrievalService(
        embedding_provider=FakeEmbedding(),
        reranker=FakeReranker(),
        store=FakeStore(),
        sparse_encoder=BM25SparseEncoder(state),
        settings=load_settings("config/config.yaml").retrieval,
        tracer=NoOpTracer(),
        collection_name=collection,
    )


def profile_settings(
    profile: RuntimeProfile, *, database_url: str | None = None
) -> Any:
    base = load_settings("config/config.yaml")
    return base.model_copy(
        update={
            "runtime": RuntimeSettings(profile=profile),
            "database": DatabaseSettings(url=database_url),
            "mlflow": base.mlflow.model_copy(update={"enabled": False}),
        }
    )


def test_runtime_profile_defaults_to_fixture_and_supports_environment_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
models:
  embedding: {provider: huggingface, model: local-embedding}
  reranker: {provider: huggingface, model: local-reranker}
  generation: {provider: bedrock, model: null}
""",
        encoding="utf-8",
    )

    assert load_settings(config_path).runtime.profile == "fixture"

    monkeypatch.setenv("RAG__RUNTIME__PROFILE", "managed")

    assert load_settings(config_path).runtime.profile == "managed"


def test_fixture_profile_uses_existing_collection_and_state_without_database(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = profile_settings("fixture")
    expected = fake_service(settings.qdrant.collection_name)
    calls: list[tuple[str, Path | str]] = []

    def legacy_builder(
        received: Any,
        *,
        bm25_state_path: Path | str = "data/index/bm25_state.json",
        tracer: Any = None,
    ) -> HybridRetrievalService:
        del tracer
        calls.append((received.qdrant.collection_name, bm25_state_path))
        return expected

    monkeypatch.setattr(
        "consumer_finance_rag.factory.build_retrieval_service", legacy_builder
    )
    monkeypatch.setattr(
        "consumer_finance_rag.factory.create_database_engine",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("fixture profile must not require lineage DB")
        ),
    )

    runtime = build_runtime_retrieval(settings, tracer=NoOpTracer())

    assert runtime.profile == "fixture"
    assert runtime.retrieval_service is expected
    assert runtime.qdrant_collection == "consumer_finance_phase1"
    assert runtime.generation is None
    assert runtime.index_state_id is None
    assert calls == [("consumer_finance_phase1", "data/index/bm25_state.json")]


def test_evaluation_profile_uses_dedicated_collection_and_bm25_without_database(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = profile_settings("evaluation")
    expected = fake_service(settings.evaluation.qdrant_collection_name)
    calls: list[tuple[str, Path | str]] = []

    def legacy_builder(
        received: Any,
        *,
        bm25_state_path: Path | str = "data/index/bm25_state.json",
        tracer: Any = None,
    ) -> HybridRetrievalService:
        del tracer
        calls.append((received.qdrant.collection_name, bm25_state_path))
        return expected

    monkeypatch.setattr(
        "consumer_finance_rag.factory.build_retrieval_service", legacy_builder
    )
    monkeypatch.setattr(
        "consumer_finance_rag.factory.create_database_engine",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("evaluation profile must not require lineage DB")
        ),
    )

    runtime = build_runtime_retrieval(settings, tracer=NoOpTracer())

    assert runtime.profile == "evaluation"
    assert runtime.retrieval_service is expected
    assert runtime.qdrant_collection == "consumer_finance_evaluation"
    assert runtime.generation is None
    assert runtime.index_state_id is None
    assert calls == [
        (
            "consumer_finance_evaluation",
            settings.evaluation.bm25_state_path,
        )
    ]


def test_managed_profile_requires_database_before_resolver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = profile_settings("managed")
    monkeypatch.setattr(
        "consumer_finance_rag.factory.build_managed_retrieval_runtime",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("resolver must not run without DB configuration")
        ),
    )

    with pytest.raises(DatabaseConfigurationError, match="RAG__DATABASE__URL"):
        build_runtime_retrieval(settings, tracer=NoOpTracer())


def test_managed_profile_delegates_resolution_and_surfaces_generation_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = profile_settings("managed", database_url="sqlite+pysqlite:///:memory:")
    service = fake_service("consumer_finance_managed_g000007")
    state_id = uuid4()
    managed = ManagedRetrievalRuntime(
        index_state_id=state_id,
        generation=7,
        qdrant_collection="consumer_finance_managed_g000007",
        snapshot_hash="a" * 64,
        embedding_fingerprint="b" * 64,
        chunking_fingerprint="c" * 64,
        index_configuration_fingerprint="d" * 64,
        document_count=3,
        chunk_count=8,
        retrieval_service=service,
    )
    calls: list[str] = []
    engine = create_engine("sqlite+pysqlite:///:memory:")
    monkeypatch.setattr(
        "consumer_finance_rag.factory.create_database_engine", lambda _settings: engine
    )

    def managed_builder(*args: Any, **kwargs: Any) -> ManagedRetrievalRuntime:
        calls.append(type(args[1]).__name__)
        assert kwargs["tracer"] is not None
        return managed

    monkeypatch.setattr(
        "consumer_finance_rag.factory.build_managed_retrieval_runtime",
        managed_builder,
    )
    monkeypatch.setattr(
        "consumer_finance_rag.factory.build_retrieval_service",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("managed profile must not fall back to static retrieval")
        ),
    )

    runtime = build_runtime_retrieval(settings, tracer=NoOpTracer())

    assert calls == ["LineageRepository"]
    assert runtime.retrieval_service is service
    assert runtime.generation == 7
    assert runtime.index_state_id == state_id
    assert runtime.qdrant_collection == "consumer_finance_managed_g000007"


@pytest.mark.parametrize(
    "error",
    [
        NoActiveManagedIndexError("publish a managed index first"),
        CorruptActiveManagedIndexError("corrupt ACTIVE generation"),
        ManagedEmbeddingCompatibilityError("embedding mismatch"),
    ],
)
def test_managed_errors_propagate_without_fixture_fallback(
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
) -> None:
    settings = profile_settings("managed", database_url="sqlite+pysqlite:///:memory:")
    engine = create_engine("sqlite+pysqlite:///:memory:")
    monkeypatch.setattr(
        "consumer_finance_rag.factory.create_database_engine", lambda _settings: engine
    )
    monkeypatch.setattr(
        "consumer_finance_rag.factory.build_managed_retrieval_runtime",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(error),
    )
    monkeypatch.setattr(
        "consumer_finance_rag.factory.build_retrieval_service",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("managed failures must never fall back")
        ),
    )

    with pytest.raises(type(error), match=str(error)):
        build_runtime_retrieval(settings, tracer=NoOpTracer())


def test_runtime_display_metadata_only_shows_generation_for_managed_profile() -> None:
    service = fake_service("fixture")
    fixture = RetrievalRuntime("fixture", service, "consumer_finance_phase1")
    evaluation = RetrievalRuntime(
        "evaluation", service, "consumer_finance_evaluation"
    )
    managed = RetrievalRuntime(
        "managed",
        service,
        "consumer_finance_managed_g000003",
        generation=3,
        index_state_id=uuid4(),
    )

    assert fixture.display_metadata() == (("Retrieval profile", "fixture"),)
    assert evaluation.display_metadata() == (("Retrieval profile", "evaluation"),)
    assert managed.display_metadata() == (
        ("Retrieval profile", "managed"),
        ("Generation", "g000003"),
        ("Collection", "consumer_finance_managed_g000003"),
    )


def test_application_runtime_injects_retrieval_without_pipeline_backend_knowledge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = profile_settings("managed", database_url="sqlite+pysqlite:///:memory:")
    service = fake_service("consumer_finance_managed_g000002")
    retrieval = RetrievalRuntime(
        "managed",
        service,
        "consumer_finance_managed_g000002",
        generation=2,
        index_state_id=uuid4(),
    )
    monkeypatch.setattr(
        "consumer_finance_rag.factory.build_tracer", lambda _settings: NoOpTracer()
    )
    monkeypatch.setattr(
        "consumer_finance_rag.factory.build_runtime_retrieval",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("a pinned runtime must not be resolved twice")
        ),
    )

    runtime = build_application_runtime(settings, retrieval_runtime=retrieval)

    assert runtime.retrieval.profile == retrieval.profile
    assert runtime.retrieval.generation == retrieval.generation
    assert runtime.retrieval.qdrant_collection == retrieval.qdrant_collection
    assert runtime.pipeline.retrieval.store is service.store
    assert runtime.pipeline.retrieval.sparse_encoder is service.sparse_encoder
    assert not hasattr(runtime.pipeline, "repository")
    assert not hasattr(runtime.pipeline, "index_state")


def test_legacy_build_pipeline_does_not_dispatch_runtime_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = profile_settings("managed")
    service = fake_service("legacy-static")
    monkeypatch.setattr(
        "consumer_finance_rag.factory.build_tracer", lambda _settings: NoOpTracer()
    )
    monkeypatch.setattr(
        "consumer_finance_rag.factory.build_retrieval_service",
        lambda *_args, **_kwargs: service,
    )
    monkeypatch.setattr(
        "consumer_finance_rag.factory.build_runtime_retrieval",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("legacy build_pipeline must retain existing semantics")
        ),
    )

    pipeline = build_pipeline(settings)

    assert pipeline.retrieval is service


def test_streamlit_renders_factory_metadata_instead_of_fabricated_generation() -> None:
    source = Path("app/streamlit_app.py").read_text(encoding="utf-8")

    assert "runtime.display_metadata()" in source
    assert "settings.runtime.profile" in source
    assert "build_application_runtime" in source
    assert "get_pipeline" not in source
    assert "g000001" not in source
