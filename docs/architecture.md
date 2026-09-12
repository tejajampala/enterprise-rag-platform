# Phase 1 architecture notes

## Boundaries

- `acquisition.py` returns raw `SourceRecord` values. The initial adapter reads the offline
  fixture; future live acquisition can implement the same protocol.
- `normalization.py` creates immutable `Document` values and stable SHA-256 content hashes.
- `chunking.py` implements one paragraph-aware strategy and carries provenance into every
  immutable `Chunk`.
- `providers.py` defines embedding, reranking, and LLM protocols. Hugging Face model
  adapters load lazily; `bedrock.py` owns the external generation call.
- `sparse.py` owns fitted BM25 corpus statistics. Its persisted state must match the
  indexed collection.
- `vector_store.py` explicitly creates/searches Qdrant's HNSW dense vector and sparse BM25
  vector. Payloads include citation data and fields suitable for later authorization
  filters.
- `retrieval.py` performs both searches, deterministic RRF deduplication, and reranking.
- `context.py` preserves rank, removes duplicate chunk IDs, assigns citation IDs, and
  enforces a maximum prompt context size.
- `pipeline.py` is the plain-Python application service joining retrieval and generation.

## Lineage and identifiers

Normalized text is SHA-256 hashed. A document ID combines its stable source ID and the
first 16 hash characters. A chunk ID combines document ID, ordinal, and a chunk-hash
prefix; the full document and chunk hashes remain in the model/payload. Qdrant point IDs
are deterministic UUIDv5 values derived from chunk IDs because Qdrant accepts integers or
UUIDs as point IDs.

Changing normalization, source text, or chunk boundaries therefore produces visible new
identities. Phase 1 rebuilds its development collection. Phase 2 can use these identities
for incremental diffing and version-aware lineage.

## Retrieval decisions

Dense vectors use cosine distance and named HNSW configuration. Sparse document vectors
encode standard BM25 IDF, term-frequency saturation, and length normalization; sparse
query vectors contain query term frequency. Qdrant's collection IDF modifier is explicitly
disabled, so its sparse dot product applies the client-computed IDF exactly once. Unknown
query terms are ignored safely.

RRF intentionally uses ranks, not raw dense/sparse scores. The fused list retains each
path's rank for inspection. The reranker returns only the requested final context depth.
No values are described as benchmark-optimal before Phase 2 evaluation.

## Security and failure behavior

The generation prompt treats retrieved content as untrusted evidence to reduce indirect
prompt-injection risk. No credentials are accepted from model configuration files. Missing
Bedrock model configuration fails before a boto3 client is created. Bedrock service and
client failures retain actionable error codes, messages, and request IDs without exposing
request contents. The UI also handles a missing local BM25 state file and surfaces
operational Qdrant/AWS errors without fabricating an answer.
