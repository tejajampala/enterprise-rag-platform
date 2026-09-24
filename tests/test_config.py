from __future__ import annotations

from pathlib import Path

from pytest import MonkeyPatch

from consumer_finance_rag.config import load_settings

CONFIG = """
models:
  embedding: {provider: huggingface, model: local-embedding}
  reranker: {provider: huggingface, model: local-reranker}
  generation: {provider: bedrock, vendor: test, model: null}
aws: {region: us-west-2}
qdrant: {collection_name: test-collection}
chunking: {chunk_size: 500, overlap: 50}
retrieval: {dense_candidates: 9, sparse_candidates: 8, fused_candidates: 7, final_contexts: 4}
context: {max_characters: 2000}
"""


def test_configuration_loads_yaml_and_standard_environment_aliases(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(CONFIG, encoding="utf-8")
    monkeypatch.setenv("BEDROCK_GENERATION_MODEL", "test.bedrock-model-v1")
    monkeypatch.setenv("AWS_REGION", "ap-southeast-2")
    monkeypatch.setenv("RAG__MODELS__EMBEDDING__QUERY_PROMPT_NAME", "custom-query")
    monkeypatch.setenv("RAG__QDRANT__COLLECTION_NAME", "environment-collection")
    monkeypatch.setenv("MLFLOW_GENAI_JUDGE_DEFAULT_MODEL", "bedrock:/test.judge-model-v1")

    settings = load_settings(path)

    assert settings.models.embedding.model == "local-embedding"
    assert settings.models.embedding.query_prompt_name == "custom-query"
    assert settings.models.generation.model == "test.bedrock-model-v1"
    assert settings.aws.region == "ap-southeast-2"
    assert settings.mlflow.judge_model == "bedrock:/test.judge-model-v1"
    assert settings.aws.connect_timeout_seconds == 5
    assert settings.aws.read_timeout_seconds == 120
    assert settings.aws.retry_mode == "standard"
    assert settings.aws.total_max_attempts == 3
    assert settings.aws.tcp_keepalive is True
    assert settings.retrieval.dense_candidates == 9
    assert settings.retrieval.hnsw_ef is None
    assert settings.qdrant.collection_name == "environment-collection"
    assert settings.evaluation.qdrant_collection_name == "consumer_finance_evaluation"
    assert settings.evaluation.bm25_state_path == Path("data/index/evaluation_bm25_state.json")


def test_application_database_is_optional(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(CONFIG, encoding="utf-8")
    monkeypatch.setenv("RAG__DATABASE__URL", "")

    settings = load_settings(path)

    assert settings.database.url is None


def test_application_database_url_uses_nested_environment_override(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(f"{CONFIG}\ndatabase: {{url: null}}\n", encoding="utf-8")
    expected = "postgresql+psycopg://lineage:secret@localhost:5433/lineage"
    monkeypatch.setenv("RAG__DATABASE__URL", expected)

    settings = load_settings(path)

    assert settings.database.url == expected


def test_managed_ingestion_settings_have_defaults_and_environment_overrides(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(CONFIG, encoding="utf-8")

    defaults = load_settings(path)

    assert defaults.managed_ingestion.corpus_key == "consumer-finance-managed"
    assert defaults.managed_ingestion.qdrant_collection_prefix == "consumer_finance_managed"

    monkeypatch.setenv("RAG__MANAGED_INGESTION__CORPUS_KEY", "custom-managed")
    monkeypatch.setenv(
        "RAG__MANAGED_INGESTION__QDRANT_COLLECTION_PREFIX", "custom_prefix"
    )

    overridden = load_settings(path)

    assert overridden.managed_ingestion.corpus_key == "custom-managed"
    assert overridden.managed_ingestion.qdrant_collection_prefix == "custom_prefix"


def test_managed_index_settings_are_typed_and_environment_configurable(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(CONFIG, encoding="utf-8")

    defaults = load_settings(path)

    assert defaults.managed_index.format_version == "managed-hybrid-index-v1"
    assert defaults.managed_index.dense_vector_name == "dense_hnsw"
    assert defaults.managed_index.sparse_vector_name == "sparse_bm25"
    assert defaults.managed_index.distance_metric == "cosine"
    assert defaults.managed_index.hnsw_m == 16
    assert defaults.managed_index.hnsw_ef_construct == 100
    assert defaults.managed_index.bm25_k1 == 1.2
    assert defaults.managed_index.bm25_b == 0.75
    assert defaults.managed_index.sparse_modifier == "none"
    assert defaults.managed_index.sparse_on_disk is False
    assert defaults.managed_index.retained_retired_generations == 2
    assert defaults.managed_index.retired_cleanup_min_age_hours == 24
    assert defaults.managed_index.dense_reuse_read_max_attempts == 3
    assert defaults.managed_index.dense_reuse_read_backoff_seconds == 0.25

    monkeypatch.setenv("RAG__MANAGED_INDEX__HNSW_M", "32")
    monkeypatch.setenv("RAG__MANAGED_INDEX__BM25_K1", "1.5")
    monkeypatch.setenv("RAG__MANAGED_INDEX__RETAINED_RETIRED_GENERATIONS", "3")
    monkeypatch.setenv("RAG__MANAGED_INDEX__RETIRED_CLEANUP_MIN_AGE_HOURS", "48")
    monkeypatch.setenv("RAG__MANAGED_INDEX__DENSE_REUSE_READ_MAX_ATTEMPTS", "4")
    monkeypatch.setenv(
        "RAG__MANAGED_INDEX__DENSE_REUSE_READ_BACKOFF_SECONDS",
        "0.5",
    )

    overridden = load_settings(path)

    assert overridden.managed_index.hnsw_m == 32
    assert overridden.managed_index.bm25_k1 == 1.5
    assert overridden.managed_index.retained_retired_generations == 3
    assert overridden.managed_index.retired_cleanup_min_age_hours == 48
    assert overridden.managed_index.dense_reuse_read_max_attempts == 4
    assert overridden.managed_index.dense_reuse_read_backoff_seconds == 0.5


def test_query_hnsw_ef_is_optional_and_environment_configurable(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(CONFIG, encoding="utf-8")

    assert load_settings(path).retrieval.hnsw_ef is None

    monkeypatch.setenv("RAG__RETRIEVAL__HNSW_EF", "80")

    assert load_settings(path).retrieval.hnsw_ef == 80


def test_public_showcase_is_fail_closed_and_environment_configurable(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(CONFIG, encoding="utf-8")

    defaults = load_settings(path).public_showcase
    assert defaults.enabled is False
    assert defaults.inference_enabled is False
    assert defaults.approved_corpus_key == "consumer-finance-managed"

    monkeypatch.setenv("RAG__PUBLIC_SHOWCASE__ENABLED", "true")
    monkeypatch.setenv("RAG__PUBLIC_SHOWCASE__INFERENCE_ENABLED", "true")
    monkeypatch.setenv("RAG__PUBLIC_SHOWCASE__MAX_REQUESTS_PER_WINDOW", "7")
    monkeypatch.setenv("RAG__PUBLIC_SHOWCASE__MAX_CONCURRENT_REQUESTS", "1")

    overridden = load_settings(path).public_showcase
    assert overridden.enabled is True
    assert overridden.inference_enabled is True
    assert overridden.max_requests_per_window == 7
    assert overridden.max_concurrent_requests == 1


def test_drift_thresholds_are_typed_and_environment_configurable(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(CONFIG, encoding="utf-8")

    defaults = load_settings(path).drift_monitoring
    assert defaults.max_recall_drop == 0.0
    assert defaults.max_mrr_drop == 0.0
    assert defaults.minimum_operational_samples == 20

    monkeypatch.setenv("RAG__DRIFT_MONITORING__MAX_RECALL_DROP", "0.05")
    monkeypatch.setenv("RAG__DRIFT_MONITORING__MINIMUM_OPERATIONAL_SAMPLES", "30")
    monkeypatch.setenv(
        "RAG__DRIFT_MONITORING__MAX_MEAN_LATENCY_INCREASE_RATIO", "0.4"
    )

    overridden = load_settings(path).drift_monitoring
    assert overridden.max_recall_drop == 0.05
    assert overridden.minimum_operational_samples == 30
    assert overridden.max_mean_latency_increase_ratio == 0.4
