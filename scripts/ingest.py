"""Build the local hybrid Qdrant index from the reproducible fixture corpus."""

from pathlib import Path

from consumer_finance_rag.acquisition import JsonFixtureSource
from consumer_finance_rag.chunking import ParagraphChunker
from consumer_finance_rag.config import load_settings
from consumer_finance_rag.ingestion import index_chunks, prepare_chunks
from consumer_finance_rag.providers import HuggingFaceEmbeddingProvider
from consumer_finance_rag.vector_store import QdrantHybridStore


def main() -> None:
    settings = load_settings()
    source = JsonFixtureSource(Path("data/fixtures/cfpb_consumer_finance.json"))
    chunker = ParagraphChunker(settings.chunking.chunk_size, settings.chunking.overlap)
    chunks = prepare_chunks(source, chunker)
    model = settings.models.embedding
    embedding = HuggingFaceEmbeddingProvider(model.model, model.device, model.trust_remote_code)
    index_chunks(
        chunks,
        embedding_provider=embedding,
        store=QdrantHybridStore(settings.qdrant),
        bm25_state_path="data/index/bm25_state.json",
    )
    print(
        f"Indexed {len(chunks)} chunks into Qdrant collection '{settings.qdrant.collection_name}'."
    )


if __name__ == "__main__":
    main()
