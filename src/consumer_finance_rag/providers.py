"""Configurable model provider interfaces and initial implementations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from consumer_finance_rag.models import RetrievedChunk


class EmbeddingProvider(Protocol):
    @property
    def model_name(self) -> str: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class RerankerProvider(Protocol):
    @property
    def model_name(self) -> str: ...

    def rerank(
        self, query: str, candidates: Sequence[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]: ...


class LLMResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    model_id: str
    response_metadata: dict[str, Any] = Field(default_factory=dict)


class LLMProvider(Protocol):
    @property
    def model_name(self) -> str: ...

    def generate(self, system_prompt: str, user_prompt: str) -> LLMResult: ...


class HuggingFaceEmbeddingProvider:
    """Sentence Transformers embedding adapter, loaded lazily."""

    def __init__(
        self,
        model: str,
        device: str,
        trust_remote_code: bool,
        query_prompt_name: str | None = "query",
    ) -> None:
        self._model_name = model
        self._device = device
        self._trust_remote_code = trust_remote_code
        self._query_prompt_name = query_prompt_name
        self._encoder: Any | None = None

    @property
    def model_name(self) -> str:
        return self._model_name

    def _load(self) -> Any:
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(
                self._model_name,
                device=self._device,
                trust_remote_code=self._trust_remote_code,
            )
        return self._encoder

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self._load().encode_document(
            list(texts), normalize_embeddings=True, show_progress_bar=False
        )
        return [[float(value) for value in vector] for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        prompt_options = (
            {"prompt_name": self._query_prompt_name}
            if self._query_prompt_name is not None
            else {"prompt": ""}
        )
        vector = self._load().encode_query(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
            **prompt_options,
        )
        return [float(value) for value in vector]


class HuggingFaceRerankerProvider:
    """Cross-encoder reranker adapter, loaded lazily."""

    def __init__(self, model: str, device: str, trust_remote_code: bool) -> None:
        self._model_name = model
        self._device = device
        self._trust_remote_code = trust_remote_code
        self._reranker: Any | None = None

    @property
    def model_name(self) -> str:
        return self._model_name

    def _load(self) -> Any:
        if self._reranker is None:
            from sentence_transformers import CrossEncoder

            self._reranker = CrossEncoder(
                self._model_name,
                device=self._device,
                trust_remote_code=self._trust_remote_code,
            )
        return self._reranker

    def rerank(
        self, query: str, candidates: Sequence[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        if not candidates:
            return []
        pairs = [(query, item.chunk.text) for item in candidates]
        scores = self._load().predict(pairs)
        rescored = [
            item.model_copy(update={"rerank_score": float(score)})
            for item, score in zip(candidates, scores, strict=True)
        ]
        return sorted(
            rescored,
            key=lambda item: item.rerank_score if item.rerank_score is not None else float("-inf"),
            reverse=True,
        )[:top_k]
