# Consumer Finance Intelligence

Consumer Finance Intelligence is an evidence-grounded RAG application over public U.S.
consumer-finance information. The plain-Python pipeline retrieves CFPB-derived evidence,
generates a cited answer through AWS Bedrock, and now records traces and repeatable quality
experiments in MLflow.

The small committed fixture covers credit cards, credit reporting, mortgages, bank accounts,
debt collection, and payments/fraud. It is paraphrased from the linked CFPB pages and supports
offline development and tests; it is not legal or financial advice. A separate manifest-driven
workflow can fetch a larger official CFPB evaluation corpus without committing downloads.

## Phase 2 architecture

```text
CFPB fixture / bounded CFPB manifest acquisition
  -> normalization -> paragraph-aware chunks
  -> Hugging Face embeddings + BM25 sparse vectors -> one Qdrant collection

question -> Qdrant dense HNSW/ANN search ----+
         -> Qdrant sparse BM25 search -------+-> RRF -> HF reranker
         -> citation-aware context -> Bedrock Converse -> grounded answer
         -> Streamlit
                  |
                  +-> MLflow trace (request and stage spans)
                  +-> deterministic dense/BM25/RRF/reranked metrics
                  +-> optional MLflow GenAI judges
                  +-> optional RAGAS 0.4.3 metrics
                  +-> JSON/CSV regression-ready reports

Docker: Qdrant | PostgreSQL (MLflow backend) | MLflow Tracking Server
```

Provider protocols keep embeddings, reranking, and generation configurable. Qdrant remains the
only retrieval engine, Bedrock remains only the generation gateway, MLflow is the observability
and experiment system, and Streamlit remains presentation. No orchestration framework is used.

### Retrieval foundations

Embeddings are stored with citation payloads in Qdrant so semantic search returns the exact
chunk, title, category, source URL, and hashes needed for grounding. The same collection stores
the sparse representation, avoiding a duplicate search stack.

The named `dense_hnsw` vector uses Hierarchical Navigable Small World (HNSW), an approximate
nearest-neighbor (ANN) graph. It searches promising graph neighborhoods instead of exhaustively
comparing every vector. HNSW construction and query settings remain visible in configuration and
`vector_store.py`.

BM25 rewards matching terms while accounting for corpus rarity, term-frequency saturation, and
document length. Ingestion fits deterministic BM25 statistics, stores sparse vectors in Qdrant,
and saves the query encoder state at `data/index/bm25_state.json`.

Reciprocal Rank Fusion (RRF) combines dense and lexical rankings without comparing incompatible
raw scores. Each occurrence contributes `1 / (rrf_k + rank)`; candidates are deduplicated by
chunk ID and sorted before reranking. The Hugging Face cross-encoder then evaluates each fused
question/passage pair and selects the final context.

Candidate depths and `rrf_k` are experimental parameters, not claimed optima. Phase 2 measures
each retrieval stage with Recall@K, Precision@K, Hit Rate@K, MRR@K, NDCG@K, and MAP@K.

### Bedrock and model separation

`BedrockLLMProvider` uses the vendor-neutral Bedrock Runtime Converse API, standard AWS credential
resolution, and a configured model ID. It records returned input, output, and total token counts
on the generation span. MLflow may calculate cost when it recognizes the provider/model; the
application otherwise reports cost as unavailable and never invents a zero.

The generation model and evaluation judge are independent. A judge must be explicitly set using
`MLFLOW_GENAI_JUDGE_DEFAULT_MODEL`; Bedrock uses `bedrock:/<model-id>`. No OpenAI judge fallback is
allowed by application code.

## MLflow observability and evaluation

An MLflow **trace** represents one production RAG request. Its nested **spans** represent the
embedding, dense HNSW search, sparse BM25 search, RRF, reranking, context construction, and
Bedrock call. Span duration supplies stage latency without a second timing subsystem. Compact IDs,
ranks, models, retrieval parameters, token usage, stop reason, and AWS request ID are recorded;
dense vectors and unnecessary full payloads are not.

An MLflow **run** represents an evaluation experiment. It records model/config parameters,
aggregate metrics, Git SHA, and report artifacts. Traces explain individual executions; runs make
whole experiments comparable. PostgreSQL is the MLflow backend because MLflow Evaluation
Datasets require a database-backed tracking server. It is not the application's document-lineage
database; that remains Phase 3.

Deterministic retrieval evaluation needs no LLM and compares dense, sparse/BM25, RRF fused, and
reranked output from the existing `retrieve_debug()` path. Generation evaluation is separate and
can use references/evidence with optional LLM judges:

- MLflow GenAI: `Correctness`, `RelevanceToQuery`, `Safety`, and trace-based
  `RetrievalGroundedness` when traces are available.
- RAGAS 0.4.3 collections: `Faithfulness`, `ContextPrecision`, `ContextRecall`, and
  `ResponseGroundedness`, executed through its experiment API with `MetricResult.value` and
  `MetricResult.reason` retained.

Groundedness asks whether claims follow from retrieved evidence; correctness asks whether the
answer matches expected facts. An answer can be grounded in an irrelevant passage yet incorrect
for the question, so both dimensions matter.

### RAGAS dependency boundary

RAGAS 0.4.3 declares LangChain, LangGraph, and LangSmith packages as mandatory transitive
dependencies. They are installed by `uv` solely for RAGAS. Application retrieval, generation,
tracing, orchestration, and UI do not import or use them; RAGAS is isolated in
`evaluation/ragas_adapter.py` and can be removed without changing `RAGPipeline`. A static unit test
enforces this core-module boundary. RAGAS 0.4.3 also references a Vertex AI compatibility module
removed from the resolved LangChain Community release, so the adapter supplies a local sentinel
for legacy provider detection; evaluation still uses only LiteLLM/Bedrock.

## Run locally

Prerequisites are Python 3.12, `uv`, Docker Compose, and enough memory/disk for the configured
Hugging Face models. Copy `.env.example` to `.env`, replace the local MLflow database password,
and optionally configure a Bedrock generation model:

```dotenv
MLFLOW_DB_USER=mlflow
MLFLOW_DB_PASSWORD=choose-a-local-password
MLFLOW_DB_NAME=mlflow
BEDROCK_GENERATION_MODEL=your-enabled-bedrock-model-id
AWS_REGION=us-east-1
```

Then run:

```bash
uv sync
docker compose up -d
uv run python scripts/ingest.py
uv run streamlit run app/streamlit_app.py
```

Open Streamlit at its displayed URL and MLflow at <http://localhost:5000>. Boto3 uses the standard
AWS credential chain; never commit access keys. If Bedrock is not configured, the UI displays an
actionable message rather than calling AWS.

### Debug retrieval

The developer CLI calls the production retrieval service but never Bedrock:

```bash
uv run python scripts/debug_retrieval.py "How can I dispute a credit report error?"
uv run python scripts/debug_retrieval.py "What is a Closing Disclosure?" --top-k 3
```

It prints dense HNSW/ANN, sparse BM25, RRF-fused, and final reranked sections with chunk IDs,
scores, ranks, source metadata, and previews.

### Validate and evaluate retrieval

```bash
uv run python scripts/validate_evaluation_dataset.py
uv run python scripts/evaluate_retrieval.py --max-cases 5 --no-mlflow
uv run python scripts/evaluate_retrieval.py --k 1,3,5,10
```

The first command confirms that all qrels match the selected normalized documents/chunks. The
second is a smoke run over five cases and the third evaluates the complete seed set. Both are
deterministic and offline apart from locally running Qdrant/model caches. The MLflow-enabled run
records corpus counts, model/index/retrieval parameters, stage metrics, category/difficulty
breakdowns, and JSON/CSV artifacts. CLI overrides such as `--dense-candidates`,
`--sparse-candidates`, `--fused-candidates`,
`--final-contexts`, and `--rrf-k` affect only that process and never rewrite configuration.

The CLI prints distinct embedding-load and reranker-load phases, warms each configured model once,
then reports every case as it enters the unchanged `retrieve_debug()` production path. To download
models and prove one tiny inference separately from evaluation, reuse the same factory with:

```bash
uv run python scripts/warm_models.py
```

The committed 36-case **seed evaluation set** is marked `review_required`; it is not a production
golden dataset. Its current qrels deliberately concentrate on the small set of stable fixture
chunks so every reference remains reviewable and reproducible after real distractors are added.
Future human review should expand relevant document/chunk judgments across the larger corpus;
the acquisition workflow does not fabricate qrels. To sync the seed set to MLflow's SQL-backed
Evaluation Dataset store:

```bash
uv run python scripts/sync_evaluation_dataset.py
```

### Evaluate generated answers safely

Retrieval-only mode makes no Bedrock generation or judge calls:

```bash
uv run python scripts/evaluate_rag.py --retrieval-only --no-mlflow
```

Running generation invokes the configured Bedrock model, so use `--limit` for a smoke test:

```bash
uv run python scripts/evaluate_rag.py --limit 2
```

LLM judges are never enabled by default. Set an explicitly enabled Bedrock judge, then opt in:

```dotenv
MLFLOW_GENAI_JUDGE_DEFAULT_MODEL=bedrock:/your-enabled-judge-model-id
```

```bash
uv run python scripts/evaluate_rag.py --limit 2 --with-mlflow-judges
uv run python scripts/evaluate_rag.py --limit 2 --with-ragas
```

Both commands can incur AWS charges. RAGAS uses LiteLLM only inside its adapter. Normal tests,
retrieval evaluation, and Streamlit report display do not execute judges.

### Controlled corpus expansion

The source-controlled `data/corpus_manifest.json` selects official CFPB Ask CFPB URLs, categories,
and a hard document limit. Downloads and parsed output are cached outside Git:

```bash
uv run python scripts/fetch_evaluation_corpus.py
uv run python scripts/fetch_evaluation_corpus.py --refresh
uv run python scripts/ingest.py --require-expanded
```

Acquisition is separate from normalization. It is intended to create hundreds of real,
overlapping documents when CFPB network access is available; it never manufactures duplicates and
is never called by unit tests. The manifest currently bounds acquisition at 400 pages. Ingestion
combines distinct downloaded records with the original six fixture documents (fixture URLs take
precedence), prints document/chunk/category counts, and rebuilds the one existing Qdrant hybrid
collection. Downloaded pages and parsed corpus JSON remain ignored by Git.

## Development checks

```bash
uv run ruff check .
uv run mypy .
uv run pytest
docker compose config
```

Tests mock external systems and require no Qdrant server, MLflow server, internet, AWS credentials,
or paid calls.

## Offline and online evaluation

Offline evaluation runs a fixed corpus and reviewable cases before release, making experiments
repeatable. Production/online evaluation observes real traffic and feedback under privacy and
operational controls. Phase 2 implements offline experiments and production-path traces but does
not claim production monitoring, quality gains, or release thresholds.

Phase 3 correctly retains document lineage/versioning, incremental indexing, ACLs, PII handling,
application PostgreSQL schema, production deployment, and formal CI regression gates. Kafka,
Airflow, Kubernetes, Supabase, OpenSearch, and additional orchestration frameworks remain out of
scope.

See [architecture details](docs/architecture.md) and the
[Phase 2 interview notes](docs/phase2-interview-notes.md). The first expanded-corpus run and its
non-significant sanity analysis are recorded in
[Phase 2 deterministic validation](docs/phase2-validation.md).
