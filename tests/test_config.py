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
