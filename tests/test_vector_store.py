from __future__ import annotations

from qdrant_client import QdrantClient, models

from consumer_finance_rag.config import QdrantSettings
from consumer_finance_rag.models import Chunk
from consumer_finance_rag.sparse import BM25SparseEncoder
from consumer_finance_rag.vector_store import QdrantHybridStore


def test_qdrant_collection_supports_dense_hnsw_and_sparse_bm25_queries() -> None:
    client = QdrantClient(":memory:")
    store = QdrantHybridStore(QdrantSettings(collection_name="hybrid-test"), client=client)
    chunks = [
        Chunk(
            chunk_id="mortgage-chunk",
            document_id="mortgage-doc",
            document_version="v1",
            text="A borrower receives a closing disclosure before mortgage closing.",
            ordinal=0,
            chunk_hash="1" * 64,
            metadata={
                "source_id": "source-1",
                "title": "Mortgage source",
                "source_url": "https://example.test/mortgage",
                "category": "mortgages",
            },
        ),
        Chunk(
            chunk_id="card-chunk",
            document_id="card-doc",
            document_version="v1",
            text="A promotional credit card rate can end after a stated period.",
            ordinal=0,
            chunk_hash="2" * 64,
            metadata={
                "source_id": "source-2",
                "title": "Card source",
                "source_url": "https://example.test/card",
                "category": "credit cards",
            },
        ),
    ]
    encoder = BM25SparseEncoder.fit([chunk.text for chunk in chunks])
    store.recreate_collection(dense_size=2)
    collection = client.get_collection("hybrid-test")
    dense_config = collection.config.params.vectors
    sparse_config = collection.config.params.sparse_vectors
    assert isinstance(dense_config, dict)
    assert dense_config["dense_hnsw"].size == 2
    assert dense_config["dense_hnsw"].distance == models.Distance.COSINE
    assert sparse_config is not None
    assert sparse_config["sparse_bm25"].modifier == models.Modifier.NONE
    store.upsert(chunks, [[1.0, 0.0], [0.0, 1.0]], encoder)

    dense = store.search_dense([1.0, 0.0], limit=2)
    sparse = store.search_sparse("mortgage closing", encoder, limit=2)

    assert dense[0].chunk.chunk_id == "mortgage-chunk"
    assert dense[0].dense_rank == 1
    assert sparse[0].chunk.chunk_id == "mortgage-chunk"
    assert sparse[0].sparse_rank == 1
