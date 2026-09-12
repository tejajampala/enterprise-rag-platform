# Consumer Finance Intelligence

Consumer Finance Intelligence is an evidence-grounded retrieval-augmented generation
(RAG) application over a small public U.S. consumer-finance corpus. Phase 1 provides a
runnable vertical slice from offline source acquisition through cited AWS Bedrock answers
and a Streamlit interface.

The development corpus covers credit cards, credit reporting, mortgages, bank accounts,
debt collection, and payments/fraud. It is a clearly labelled local fixture paraphrased
from linked CFPB consumer-information pages, so tests and ingestion do not require network
access. This project provides general information, not legal or financial advice.

## Phase 1 architecture

```text
CFPB-derived fixture
  -> source acquisition -> normalization -> paragraph-aware chunks
  -> Hugging Face dense embeddings ----------------------+
  -> corpus-fitted BM25 sparse vectors ------------------+-> one Qdrant collection

question -> Hugging Face query embedding -> Qdrant HNSW/ANN dense search --+
         -> BM25 query vector -----------> Qdrant sparse search -----------+
         -> Reciprocal Rank Fusion -> Hugging Face reranker
         -> bounded citation-aware context -> AWS Bedrock Converse API
         -> grounded answer + citations -> Streamlit
```

The implementation is plain, typed Python. Provider protocols isolate embeddings,
reranking, and generation, while `config/config.yaml` and environment variables select
models. There is no orchestration framework or second retrieval stack.

### Why embeddings are stored in Qdrant

Embeddings turn semantically similar passages and questions into nearby vectors. Storing
them beside citation payloads in Qdrant lets semantic search return the exact versioned
chunk, title, category, source URL, and lineage hashes needed to build grounded evidence.
The same collection also stores BM25 sparse vectors, avoiding synchronization and
operational complexity across separate search engines.

### HNSW/ANN dense retrieval

Qdrant indexes the named `dense_hnsw` vector using Hierarchical Navigable Small World
(HNSW), an approximate nearest-neighbor (ANN) graph. ANN explores a small, promising
portion of the graph rather than comparing the query with every vector. The collection's
HNSW construction values and the search `hnsw_ef` are visible in code/config rather than
hidden behind a generic retrieval helper.

### BM25 sparse retrieval

BM25 is lexical retrieval: it rewards query terms found in a passage, discounts terms
common across the corpus, saturates repeated occurrences, and normalizes for passage
length. In Phase 1, ingestion fits deterministic corpus statistics and stores explicit
BM25-weighted sparse vectors in Qdrant's named sparse index. The saved vocabulary/state at
`data/index/bm25_state.json` is needed to encode future queries consistently.

### Reciprocal Rank Fusion

Reciprocal Rank Fusion (RRF) combines dense and sparse rankings without treating their
incompatible raw scores as comparable. Each candidate receives `1 / (k + rank)` from each
list in which it occurs. Candidates are deduplicated by chunk ID, summed, and sorted before
reranking.

### Reranking

The configured Hugging Face cross-encoder evaluates the question and each fused candidate
together. This costs more per passage than retrieval but is applied only to a small fused
set, improving final ordering before context construction. Candidate depths are defaults,
not claimed optima; Phase 2 will tune them with labelled queries and MRR, NDCG, and Recall.

### AWS Bedrock

Bedrock is the inference gateway, not the RAG implementation. `BedrockLLMProvider` uses
the Bedrock Runtime Converse API, standard AWS credential resolution, and a model ID from
configuration. Its client has explicit, configurable connect/read timeouts, standard retries,
and TCP keepalive. The pipeline has no Anthropic-specific or OpenAI-specific request logic.
The prompt requires evidence-only answers, explicit insufficiency, citation IDs, and
treating retrieved content as untrusted data rather than instructions.

## Run locally

Prerequisites: Python 3.12, `uv`, Docker with Compose, enough memory/disk for the configured
Hugging Face models, and (for generation) AWS credentials with access to a Bedrock model.

```bash
uv sync
docker compose up -d
uv run python scripts/ingest.py
uv run streamlit run app/streamlit_app.py
```

The first ingestion downloads the configured Hugging Face embedding model. The first query
downloads the reranker. CPU inference works for development but can be slow; change
`device` and model names in configuration for another supported environment.

For Bedrock, copy `.env.example` to `.env` and set:

```dotenv
BEDROCK_GENERATION_MODEL=your-enabled-bedrock-model-id
AWS_REGION=us-east-1
```

Do not put AWS access keys in the repository. Boto3 uses its normal credential chain
(environment, AWS profile, workload identity, or instance/task role). If the model ID is
unset, the UI displays a setup message instead of attempting an AWS call.

Configuration defaults live in [`config/config.yaml`](config/config.yaml). Deployments can
use conventional `BEDROCK_GENERATION_MODEL` and `AWS_REGION` variables. Other settings can
be overridden with Pydantic's nested `RAG__...` names where supported, for example
`RAG__QDRANT__COLLECTION_NAME`. Bedrock network settings are under `aws` and can be
overridden with names such as `RAG__AWS__READ_TIMEOUT_SECONDS` and
`RAG__AWS__TOTAL_MAX_ATTEMPTS`.

### Inspect retrieval from the developer CLI

After Qdrant is running and ingestion has completed, pass a question to the debugging
CLI. It invokes the same `HybridRetrievalService` used by the application and does not call
Bedrock:

```bash
uv run python scripts/debug_retrieval.py "How can I dispute a credit report error?"
```

The output contains four ordered sections: dense HNSW/ANN results, sparse BM25 results,
the RRF-fused candidate set, and final reranked results. Each row includes its score, chunk
ID, source title/category, dense and sparse ranks, and a text preview.

To override only the number of final reranked results:

```bash
uv run python scripts/debug_retrieval.py "What is a Closing Disclosure?" --top-k 3
```

Run `uv run python scripts/debug_retrieval.py --help` for configuration and BM25-state
path options. The first execution may download the configured Hugging Face models. AWS
credentials and `BEDROCK_GENERATION_MODEL` are not required for this retrieval-only tool.

## Development checks

```bash
uv run ruff check .
uv run mypy .
uv run pytest
docker compose config
```

Unit tests mock external systems and never require AWS credentials or a running Qdrant.

## Phase 2

Phase 2 will add labelled evaluation data and real MLflow 3.16+/RAGAS 0.4+ evaluation and
tracing. It will measure and tune candidate depths and retrieval/generation quality rather
than reporting synthetic metrics. PostgreSQL-backed lineage, token/latency tracing, and
incremental indexing are also intentionally deferred. Authentication, production
deployment, Supabase, Redis, Kafka, Kubernetes, OpenSearch, and Airflow are outside this
phase.

More implementation detail is in [`docs/architecture.md`](docs/architecture.md).
