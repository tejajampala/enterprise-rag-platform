from __future__ import annotations

from typing import Any

from consumer_finance_rag.providers import HuggingFaceEmbeddingProvider


class FakeEmbeddingModel:
    def __init__(self) -> None:
        self.query_call: tuple[str, dict[str, Any]] | None = None
        self.document_call: tuple[list[str], dict[str, Any]] | None = None

    def encode_query(self, text: str, **kwargs: Any) -> list[float]:
        self.query_call = (text, kwargs)
        return [0.6, 0.8]

    def encode_document(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        self.document_call = (texts, kwargs)
        return [[1, 0], [0, 1]]


def test_embed_query_uses_configured_query_prompt_and_normalization() -> None:
    encoder = FakeEmbeddingModel()
    provider = HuggingFaceEmbeddingProvider(
        "Qwen/Qwen3-Embedding-0.6B",
        "cpu",
        True,
        query_prompt_name="query",
    )
    provider._encoder = encoder

    vector = provider.embed_query("How do I dispute an error?")

    assert vector == [0.6, 0.8]
    assert encoder.query_call == (
        "How do I dispute an error?",
        {
            "normalize_embeddings": True,
            "show_progress_bar": False,
            "prompt_name": "query",
        },
    )
    assert encoder.document_call is None


def test_embed_documents_uses_document_route_without_query_prompt() -> None:
    encoder = FakeEmbeddingModel()
    provider = HuggingFaceEmbeddingProvider(
        "Qwen/Qwen3-Embedding-0.6B",
        "cpu",
        True,
        query_prompt_name="query",
    )
    provider._encoder = encoder

    vectors = provider.embed_documents(["First passage", "Second passage"])

    assert vectors == [[1.0, 0.0], [0.0, 1.0]]
    assert encoder.document_call == (
        ["First passage", "Second passage"],
        {"normalize_embeddings": True, "show_progress_bar": False},
    )
    assert encoder.query_call is None
