"""Application configuration loaded from YAML with environment overrides."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict


class HuggingFaceModelSettings(BaseModel):
    provider: Literal["huggingface"] = "huggingface"
    model: str
    device: str = "cpu"
    trust_remote_code: bool = True


class EmbeddingModelSettings(HuggingFaceModelSettings):
    query_prompt_name: str | None = "query"


class GenerationModelSettings(BaseModel):
    provider: Literal["bedrock"] = "bedrock"
    vendor: str | None = None
    model: str | None = None
    max_tokens: int = Field(default=800, gt=0)
    temperature: float = Field(default=0.1, ge=0, le=1)
    top_p: float = Field(default=0.9, gt=0, le=1)


class ModelsSettings(BaseModel):
    embedding: EmbeddingModelSettings
    reranker: HuggingFaceModelSettings
    generation: GenerationModelSettings


class AWSSettings(BaseModel):
    region: str = "us-east-1"
    connect_timeout_seconds: float = Field(default=5.0, gt=0)
    read_timeout_seconds: float = Field(default=120.0, gt=0)
    retry_mode: Literal["standard", "adaptive"] = "standard"
    total_max_attempts: int = Field(default=3, ge=1)
    tcp_keepalive: bool = True


class DatabaseSettings(BaseModel):
    """Optional Phase 3 application database configuration."""

    url: str | None = None

    @field_validator("url", mode="before")
    @classmethod
    def blank_url_is_unconfigured(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value


class RuntimeSettings(BaseModel):
    """Explicit retrieval backend selected for the application process."""

    profile: Literal["fixture", "evaluation", "managed"] = "fixture"


class ManagedIngestionSettings(BaseModel):
    """Identity for the independently managed Phase 3 corpus."""

    corpus_key: str = "consumer-finance-managed"
    qdrant_collection_prefix: str = "consumer_finance_managed"


class ManagedIndexSettings(BaseModel):
    """Build identity and conservative operations settings for managed generations."""

    format_version: str = "managed-hybrid-index-v1"
    dense_vector_name: str = "dense_hnsw"
    sparse_vector_name: str = "sparse_bm25"
    distance_metric: Literal["cosine"] = "cosine"
    hnsw_m: int = Field(default=16, gt=0)
    hnsw_ef_construct: int = Field(default=100, gt=0)
    bm25_k1: float = Field(default=1.2, gt=0)
    bm25_b: float = Field(default=0.75, ge=0, le=1)
    sparse_modifier: Literal["none"] = "none"
    sparse_on_disk: bool = False
    retained_retired_generations: int = Field(default=2, ge=0)
    retired_cleanup_min_age_hours: float = Field(default=24.0, ge=0)
    dense_reuse_read_max_attempts: int = Field(default=3, ge=1, le=10)
    dense_reuse_read_backoff_seconds: float = Field(default=0.25, ge=0, le=10)


class QdrantSettings(BaseModel):
    url: str = "http://localhost:6333"
    collection_name: str = "consumer_finance_phase1"
    dense_vector_name: str = "dense_hnsw"
    sparse_vector_name: str = "sparse_bm25"
    hnsw_m: int = Field(default=16, gt=0)
    hnsw_ef_construct: int = Field(default=100, gt=0)


class ChunkingSettings(BaseModel):
    chunk_size: int = Field(default=900, gt=0)
    overlap: int = Field(default=120, ge=0)

    @model_validator(mode="after")
    def overlap_is_smaller_than_chunk(self) -> ChunkingSettings:
        if self.overlap >= self.chunk_size:
            raise ValueError("chunk overlap must be smaller than chunk size")
        return self


class RetrievalSettings(BaseModel):
    # Phase 2 will tune these depths with MRR, NDCG, and Recall.
    dense_candidates: int = Field(default=20, gt=0)
    sparse_candidates: int = Field(default=20, gt=0)
    fused_candidates: int = Field(default=12, gt=0)
    final_contexts: int = Field(default=5, gt=0)
    rrf_k: int = Field(default=60, gt=0)
    hnsw_ef: int | None = Field(default=None, gt=0)


class ContextSettings(BaseModel):
    max_characters: int = Field(default=12000, gt=0)


class MLflowSettings(BaseModel):
    enabled: bool = True
    tracking_uri: str = "http://localhost:5000"
    experiment_name: str = "consumer-finance-rag"
    evaluation_experiment_name: str = "consumer-finance-rag-evaluation"
    evaluation_dataset_name: str = "consumer-finance-seed-evaluation-v1"
    judge_model: str | None = None
    report_directory: Path = Path("reports/evaluation")


class EvaluationSettings(BaseModel):
    dataset_path: Path = Path("data/evaluation/seed_cases.json")
    dataset_name: str = "consumer-finance-seed-evaluation"
    dataset_version: str = "1.0"
    qdrant_collection_name: str = "consumer_finance_evaluation"
    bm25_state_path: Path = Path("data/index/evaluation_bm25_state.json")
    k_values: list[int] = Field(default_factory=lambda: [1, 3, 5, 10], min_length=1)

    @model_validator(mode="after")
    def k_values_are_positive_and_unique(self) -> EvaluationSettings:
        if any(value <= 0 for value in self.k_values):
            raise ValueError("evaluation K values must be positive")
        if len(set(self.k_values)) != len(self.k_values):
            raise ValueError("evaluation K values must be unique")
        return self


class CorpusExpansionSettings(BaseModel):
    manifest_path: Path = Path("data/corpus_manifest.json")
    cache_directory: Path = Path("data/evaluation_corpus/cache")
    output_path: Path = Path("data/evaluation_corpus/cfpb_evaluation_corpus.json")
    request_timeout_seconds: float = Field(default=30.0, gt=0)


class AppSettings(BaseSettings):
    """Validated settings; nested values can use the ``RAG__`` environment prefix."""

    model_config = SettingsConfigDict(
        env_prefix="RAG__",
        env_nested_delimiter="__",
        env_file=".env",
        extra="ignore",
    )

    models: ModelsSettings
    aws: AWSSettings = AWSSettings()
    database: DatabaseSettings = DatabaseSettings()
    runtime: RuntimeSettings = RuntimeSettings()
    managed_ingestion: ManagedIngestionSettings = ManagedIngestionSettings()
    managed_index: ManagedIndexSettings = ManagedIndexSettings()
    qdrant: QdrantSettings = QdrantSettings()
    chunking: ChunkingSettings = ChunkingSettings()
    retrieval: RetrievalSettings = RetrievalSettings()
    context: ContextSettings = ContextSettings()
    mlflow: MLflowSettings = MLflowSettings()
    evaluation: EvaluationSettings = EvaluationSettings()
    corpus_expansion: CorpusExpansionSettings = CorpusExpansionSettings()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Let deployment environment values override versioned YAML defaults."""

        return env_settings, dotenv_settings, init_settings, file_secret_settings


class BedrockEnvironment(BaseSettings):
    """Conventional AWS/Bedrock aliases accepted in addition to nested settings."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    bedrock_generation_model: str | None = None
    aws_region: str | None = None
    mlflow_tracking_uri: str | None = None
    mlflow_genai_judge_default_model: str | None = None


def load_settings(path: str | Path = "config/config.yaml") -> AppSettings:
    """Load versioned defaults, then apply environment-based deployment choices."""

    config_path = Path(path)
    raw: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    settings = AppSettings(**raw)
    env = BedrockEnvironment()

    updates: dict[str, Any] = {}
    if env.bedrock_generation_model:
        updates["models"] = settings.models.model_copy(
            update={
                "generation": settings.models.generation.model_copy(
                    update={"model": env.bedrock_generation_model}
                )
            }
        )
    if env.aws_region:
        updates["aws"] = settings.aws.model_copy(update={"region": env.aws_region})
    mlflow_updates: dict[str, Any] = {}
    if env.mlflow_tracking_uri:
        mlflow_updates["tracking_uri"] = env.mlflow_tracking_uri
    if env.mlflow_genai_judge_default_model:
        mlflow_updates["judge_model"] = env.mlflow_genai_judge_default_model
    if mlflow_updates:
        updates["mlflow"] = settings.mlflow.model_copy(update=mlflow_updates)
    return settings.model_copy(update=updates)
