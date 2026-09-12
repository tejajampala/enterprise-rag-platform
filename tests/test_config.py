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

    settings = load_settings(path)

    assert settings.models.embedding.model == "local-embedding"
    assert settings.models.embedding.query_prompt_name == "custom-query"
    assert settings.models.generation.model == "test.bedrock-model-v1"
    assert settings.aws.region == "ap-southeast-2"
    assert settings.aws.connect_timeout_seconds == 5
    assert settings.aws.read_timeout_seconds == 120
    assert settings.aws.retry_mode == "standard"
    assert settings.aws.total_max_attempts == 3
    assert settings.aws.tcp_keepalive is True
    assert settings.retrieval.dense_candidates == 9
    assert settings.qdrant.collection_name == "environment-collection"
