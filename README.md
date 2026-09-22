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

### Optional Phase 3 application database foundation

Phase 3 adds an optional SQLAlchemy/Alembic foundation for a separate application lineage
database. The `lineage-postgres` Docker service uses its own credentials, database, localhost-only
port, and volume; it does not reuse MLflow's PostgreSQL database. Set `RAG__DATABASE__URL` only
when running Phase 3 database operations. Fixture ingestion, deterministic evaluation, retrieval,
and normal application startup remain database-independent. The lineage schema, repository,
pure change planner, and first transactional lineage write path are present. This write path records
source observations and immutable versions. The explicit managed publisher creates immutable
Qdrant/BM25 generations, and the managed runtime resolves one ACTIVE generation.

### Retrieval foundations

Embeddings are stored with citation payloads in Qdrant so semantic search returns the exact
chunk, title, category, source URL, and hashes needed for grounding. The same collection stores
the sparse representation, avoiding a duplicate search stack.

The named `dense_hnsw` vector uses Hierarchical Navigable Small World (HNSW), an approximate
nearest-neighbor (ANN) graph. It searches promising graph neighborhoods instead of exhaustively
comparing every vector. HNSW construction and query settings remain visible in configuration and
`vector_store.py`.

BM25 rewards matching terms while accounting for corpus rarity, term-frequency saturation, and
document length. Ingestion fits deterministic BM25 statistics and stores sparse vectors in
Qdrant. Fixture mode uses `data/index/bm25_state.json`; evaluation mode uses the isolated,
corpus-hashed `data/index/evaluation_bm25_state.json`.

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
Bedrock call. Retrieval spans record safe counts, ranks, scores, model/configuration identity, and
stage structure. Raw queries, chunk/document text, titles, source URLs, principals, and access-group
names are omitted. Bedrock token usage, stop reason, and AWS request ID remain available without
recording prompts.

Phase 3 managed retrieval also emits one immutable `RetrievalMetrics` event when each request
finishes or fails. It includes the runtime-pinned `index_state_id`, generation, collection, profile,
stage latencies, candidate/final counts, ACL mode and rejection count, caller group count, PII
policy and aggregate PII type/redaction counts, and a bounded failure stage. It never contains raw
query or chunk text, matched PII values, principal identity, group names, credentials, or exception
messages. Fixture and evaluation traces use their real profile/collection but leave generation and
index-state fields null.

`RetrievalTelemetrySink` defaults to an in-process no-op, so no external monitoring service is
required. `StructuredLoggingRetrievalTelemetrySink` is an opt-in Python logging implementation
that emits a JSON allowlist. `MLflowRetrievalTelemetrySink` can append the same aggregate-safe
request data to a caller-supplied active MLflow run; it never creates a run per query, and MLflow
failures degrade to a no-op. Prometheus, OpenTelemetry, Datadog, CloudWatch, and other remote
backends remain deferred.

Managed retrieval quality is measured separately with deterministic golden cases. `RetrievalMetrics`
is operational telemetry for one request; `RetrievalQualityReport` is a controlled, offline report
for one generation-pinned `ManagedRetrievalRuntime`. The report scores stable source, document, or
chunk identity with Recall@K, hit rate, expected rank, and MRR. It includes generation/index
fingerprints and retrieval settings, but excludes queries, context text, payload metadata,
principals, and access-group names. Explicit ACL-negative cases pass when protected content stays
inaccessible, and REDACT-mode text changes do not affect identity scoring.

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
uv run python scripts/ingest.py --mode fixture
uv run streamlit run app/streamlit_app.py
```

Open Streamlit at its displayed URL and MLflow at <http://localhost:5000>. Boto3 uses the standard
AWS credential chain; never commit access keys. If Bedrock is not configured, the UI displays an
actionable message rather than calling AWS.

### Phase 3 managed-ingestion planning and lineage apply

After configuring `RAG__DATABASE__URL` and applying the application migration with
`uv run alembic upgrade head`, inspect a deterministic plan without writing it:

```bash
uv run python scripts/ingest_incremental.py --mode fixture
uv run python scripts/ingest_incremental.py --mode evaluation --authoritative-snapshot --details
```

Persist exactly that plan to the application lineage database by adding `--apply`:

```bash
uv run python scripts/ingest_incremental.py --mode fixture --apply
uv run python scripts/ingest_incremental.py --mode evaluation --authoritative-snapshot --apply
```

Fixture mode uses only the six stable fixture documents. Evaluation mode requires the existing
downloaded expanded corpus and combines it with the fixture using the production corpus assembler.
Without `--apply`, the command remains strictly read-only. With `--apply`, it writes only managed
corpus registration, ingestion runs, logical documents, immutable document versions/chunks, and
per-run classifications. The authoritative flag is required before absence can be marked deleted.
Deleted sources are retained with all history; reappearance is rejected clearly until restoration
semantics are designed.

An apply creates and commits its `RUNNING` audit row first. All document/version/chunk/classification
mutations and the final `SUCCEEDED` or `NOOP` transition then share one transaction. If that
transaction fails, it is rolled back and the run is marked `FAILED` in a separate transaction.
Concurrent-apply locking is deferred. Neither mode updates Qdrant, embeddings, or BM25, so an apply
must not be interpreted as a retrieval-index update.

Publish the persisted lineage into an immutable managed hybrid-index generation as a separate,
explicit operator step:

```bash
uv run python scripts/ingest_incremental.py --mode evaluation --authoritative-snapshot --apply
uv run python scripts/index_managed.py --publish
```

The first command acquires/assembles the configured corpus and persists source lineage. The second
command does not reacquire sources: it resolves the latest successful/no-op lineage run,
transactionally materializes the configured active chunk derivation, commits that transaction,
and then delegates publication to `ManagedIndexService`. Running `scripts/index_managed.py`
without `--publish` is a cheap, read-only PostgreSQL status check; `--details` adds persisted
identity/count information without contacting Qdrant or loading the embedding model.

Publication creates a new immutable managed Qdrant generation only when lineage or index
configuration requires it. Repeating publication with unchanged lineage/configuration returns
`NO_INDEX_CHANGE`.

### Operate managed generations

PostgreSQL remains the control-plane authority and retains every generation, membership, source,
version, chunk, ACL, and PII history row. Operators can inspect and validate that history against
the physical Qdrant data plane without loading application models:

```bash
uv run python scripts/manage_index_generations.py --status
uv run python scripts/manage_index_generations.py --validate
uv run python scripts/manage_index_generations.py --validate --deep
uv run python scripts/manage_index_generations.py --cleanup-plan
```

Normal validation is bounded to persisted metadata, hashes, collection configuration, exact point
counts, and membership counts. `--deep` is intentionally explicit because it scrolls every point
to verify generation/lineage identities, point IDs, payload schema, ACL fields, PII fields,
chunking identity, and vector presence.

Cleanup planning is read-only. Physical deletion requires both flags:

```bash
uv run python scripts/manage_index_generations.py --cleanup --confirm
```

Without `--confirm`, `--cleanup` only prints the plan. ACTIVE, READY, and BUILDING generations are
never eligible. The newest `managed_index.retained_retired_generations` RETIRED generations are
protected, and older RETIRED generations must exceed
`managed_index.retired_cleanup_min_age_hours`. Defaults are two retained generations and 24 hours.
FAILED physical collections may be cleaned conservatively after a locked authoritative status
re-check; missing FAILED collections are normal. Canonically named orphan collections are reported
but not deleted. Phase 1 and evaluation collections are never considered managed cleanup targets.

The minimum age is a grace period for generation-pinned older processes, not proof that no process
still uses the collection and not a distributed runtime lease. Each deletion re-locks/re-reads the
corpus and recomputes eligibility to avoid deleting from a stale plan. Cleanup removes only an
eligible Qdrant collection; PostgreSQL history remains immutable. Deleting a RETIRED physical
collection removes simple restoration capability for that generation, although its PostgreSQL
audit history remains. Retention and minimum-age policy remain operator choices.

Restore a historical, still-present RETIRED generation with the same operations CLI. The first
command is a mutation-free plan; the second explicitly confirms collection creation and the
traffic switch:

```bash
uv run python scripts/manage_index_generations.py --restore-generation 1
uv run python scripts/manage_index_generations.py --restore-generation 1 --confirm
```

Restoration never performs `RETIRED -> ACTIVE`. It validates the historical metadata, membership,
BM25 state/hash, collection shape, every point, ACL/PII payload, and current query-embedding
compatibility, then creates a new monotonic generation and immutable collection. Dense vectors,
sparse vectors, text, ACL/PII metadata, and the historical BM25/configuration identity are copied;
no embedding, reranker, LLM, PII detector, source acquisition, or BM25 fit runs. The copied point
payload retains historical content/provenance and changes only `index_generation` and
`index_state_id` for the new physical generation.

The source must be `RETIRED`; `BUILDING`, `READY`, `FAILED`, and `ACTIVE` sources are rejected. Its
canonical physical collection must still exist, so restoration from cleaned PostgreSQL-only
history is future rebuild-from-history work. The final corpus-locked transaction aborts if ACTIVE
changed after planning, retires only the expected ACTIVE generation, and activates the READY
candidate. The new `index_states.restored_from_index_state_id` self-reference and
`restoration_reason` record audit provenance. Source lineage, ingestion runs, document versions,
chunks, ACL policy, and PII classification are not rolled back or recreated; consequently source
lineage may describe newer content than the restored serving index.

### Select the runtime retrieval profile

The application profile is explicit and defaults to the existing Phase 1 fixture path. Set one of
these values in `.env`, then start Streamlit normally:

```dotenv
RAG__RUNTIME__PROFILE=fixture
```

```dotenv
RAG__RUNTIME__PROFILE=evaluation
```

```dotenv
RAG__RUNTIME__PROFILE=managed
```

`fixture` uses `consumer_finance_phase1` and `data/index/bm25_state.json`. `evaluation` uses
`consumer_finance_evaluation` and `data/index/evaluation_bm25_state.json`. These static profiles
remain independent of the application lineage database.

`managed` requires `RAG__DATABASE__URL`, completed application migrations, persisted lineage, and
an ACTIVE generation published by `scripts/index_managed.py --publish`. PostgreSQL resolves the
Qdrant collection and matching persisted BM25 state together. Streamlit displays the pinned
generation and collection. It never silently falls back to fixture when managed startup fails.
If another generation becomes ACTIVE while the process is running, the current process remains
pinned to its original collection/BM25 pair; restart the application to resolve the new ACTIVE
generation. Automatic polling and hot reload are not implemented.

### Managed retrieval access control

Managed document versions carry an immutable `DocumentAccessPolicy`. Existing sources default to
`PUBLIC`; a restricted source names one or more normalized groups. Authentication is deliberately
outside this milestone: the caller supplies an already-resolved, request-scoped `AccessContext`.
For example:

```python
from consumer_finance_rag.access import AccessContext

result = managed_runtime.retrieval_service.retrieve(
    "How should this fraud case be handled?",
    access_context=AccessContext(
        principal_id="user-123",
        groups=frozenset({"fraud-analysts", "au-employees"}),
    ),
)
```

The same Qdrant authorization filter is applied before both dense and sparse candidate search.
Missing access context is fail-safe and means public-only; it never means unrestricted. Phase 1
fixture and Phase 2 evaluation profiles remain unfiltered and accept their legacy payloads.

An ACL-only source change creates a new immutable document version and managed generation because
the Qdrant payload changed. When ordered chunk text is identical, the publisher reuses the BM25
state and all eligible dense vectors. Old generation-pinned runtimes retain the historical ACL
payload; newly constructed runtimes resolve the newly ACTIVE generation. This is group-based
authorization metadata, not user/group storage or authentication.

### Managed PII classification and redaction

Managed indexing also applies a local, deterministic PII detector for `EMAIL`, practical AU and
international `PHONE` values, checksum-valid Australian `AU_TFN` values, and Luhn-valid
`CREDIT_CARD` values. It uses no external model, network service, or cloud DLP API. Detection
results are derived governance metadata: PostgreSQL stores only detector version, classification
hash, canonical PII types, and match count for each immutable document version. Raw detected
values are not copied into classification metadata or logs, and original lineage/chunk text is
never rewritten.

Managed retrieval defaults to `REDACT`. ACL authorization is applied first, original authorized
text is used internally for retrieval and reranking, and only then are final contexts transformed:

```python
from consumer_finance_rag.access import AccessContext
from consumer_finance_rag.pii import PIIHandlingPolicy

result = managed_runtime.retrieval_service.retrieve(
    "How should this fraud case be handled?",
    access_context=AccessContext(groups=frozenset({"fraud-analysts"})),
    pii_policy=PIIHandlingPolicy.REDACT,  # managed default when omitted
)
```

Redactions use typed placeholders such as `[REDACTED:EMAIL]` and
`[REDACTED:CREDIT_CARD]`. A post-redaction local scan fails closed if a supported value remains.
`ALLOW` must be requested explicitly and is independent of ACL group membership. Fixture and
evaluation profiles retain their existing unredacted defaults.

Detector identity is part of the managed index configuration, but not source-document identity.
Changing detector versions persists another derived classification and publishes a new immutable
generation while retaining the same source version. If text is unchanged, BM25 state and eligible
dense vectors are reused. Managed payload schema v3 contains only safe coarse fields:
`contains_pii`, sorted `pii_types`, and `pii_detector_version`.

### Validate the complete Phase 3 lifecycle offline

`tests/test_phase3_end_to_end.py` is the deterministic managed-lifecycle regression test. It uses
SQLite, `QdrantClient(":memory:")`, small deterministic model doubles, and the real planner,
lineage apply service, paragraph chunker, BM25 encoder, managed publisher, Qdrant adapter, managed
runtime resolver, and hybrid retriever. No PostgreSQL server, Qdrant server, model download,
network access, AWS, MLflow judge, or RAGAS call is required.

```bash
uv run pytest tests/test_phase3_end_to_end.py -q
```

The test publishes `g000001`, queries it, changes exactly one source through the transactional
lineage path, publishes `g000002`, and proves dense reuse plus fresh computation, independent BM25
refits, `g000001` retirement, `g000002` activation, process-lifetime runtime pinning, and a final
`NO_INDEX_CHANGE`. Both immutable physical collections remain present. This validates explicit
runtime reconstruction after publication; it does not implement or claim hot reload.

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
uv run python scripts/fetch_evaluation_corpus.py
uv run python scripts/ingest.py --mode evaluation
uv run python scripts/validate_evaluation_dataset.py
uv run python scripts/evaluate_retrieval.py --max-cases 5 --no-mlflow
uv run python scripts/evaluate_retrieval.py --k 1,3,5,10
```

The validator confirms that all qrels match the assembled corpus, the dedicated BM25 state was
fitted from exactly those chunks, and live Qdrant chunk IDs/hashes match the same snapshot. The
smoke command evaluates five cases and the final command evaluates the complete seed set. Both are
deterministic and offline apart from locally running Qdrant/model caches. The MLflow-enabled run
records corpus counts, model/index/retrieval parameters, stage metrics, category/difficulty
breakdowns, and JSON/CSV artifacts. CLI overrides such as `--dense-candidates`,
`--sparse-candidates`, `--fused-candidates`,
`--final-contexts`, and `--rrf-k` affect only that process and never rewrite configuration.

To evaluate the currently ACTIVE managed generation through a runtime pinned once at startup:

```bash
uv run python scripts/evaluate_managed_retrieval.py --current
uv run python scripts/evaluate_managed_retrieval.py --current --log-mlflow
uv run python scripts/evaluate_managed_retrieval.py --current \
  --output artifacts/retrieval_quality_current.json
```

This command reuses the Phase 2 seed cases and metric formulas, does not call Bedrock or an LLM
judge, and does not require MLflow unless `--log-mlflow` is explicitly passed. The opt-in mode uses
the configured evaluation experiment, logs safe generation/configuration parameters, aggregate
quality metrics, and the versioned JSON artifact, then prints the MLflow run ID. Optional JSON uses
`retrieval-quality-report-v1` and contains only generation/configuration identity, aggregate
metrics, case IDs, safe retrieved identities, and per-case scores. The pure comparison helper
reports metric deltas plus improved, regressed, newly missed, and newly recovered case IDs.
Optional `max_recall_drop` and `max_mrr_drop` thresholds produce a side-effect-free PASS/FAIL
result for CI; they do not publish, restore, or block a generation automatically. Operational
`RetrievalMetrics` and offline `RetrievalQualityReport` logging remain separate, and neither
adapter logs raw queries, retrieved context, matched PII values, principals, credentials, or group
names.

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
uv run python scripts/ingest.py --mode evaluation
```

Acquisition is separate from normalization. It is intended to create hundreds of real,
overlapping documents when CFPB network access is available; it never manufactures duplicates and
is never called by unit tests. The manifest currently bounds acquisition at 400 pages. Ingestion
combines distinct downloaded records with the original six fixture documents (fixture URLs take
precedence), prints document/chunk/category counts, and rebuilds the dedicated
`consumer_finance_evaluation` hybrid collection. Fixture mode remains isolated in
`consumer_finance_phase1`. Downloaded pages and parsed corpus JSON remain ignored by Git.

## Validation

Run the complete local validation path with:

```bash
uv sync
uv run ruff check .
uv run mypy .
uv run pytest
docker compose config --quiet
```

Run the focused Phase 3 two-generation lifecycle regression with:

```bash
uv run pytest tests/test_phase3_end_to_end.py -v
```

The default test suite is deterministic and offline. It uses SQLite, in-memory Qdrant, and model
doubles; it requires no `.env`, PostgreSQL/Qdrant/MLflow service, Hugging Face download, AWS
credential, Bedrock call, MLflow judge, RAGAS judge, internet access, or paid operation.

`.github/workflows/ci.yml` runs on every pull request and every push to `main`. It uses Python 3.12
and `uv sync --frozen`, then runs Ruff, mypy, the complete pytest suite (including Phase 1/2
compatibility, the Phase 3 lifecycle regression, and the SQLite Alembic upgrade/downgrade smoke
test), Docker Compose configuration validation, and a clean-tree check. Repository branch
protection should require the `CI / Offline validation` status check before merging to `main`.
Migration execution on SQLite validates the current portable schema, but production PostgreSQL
behavior still requires an operator-run migration against the target environment.

## Offline and online evaluation

Offline evaluation runs a fixed corpus and reviewable cases before release, making experiments
repeatable. Production/online evaluation observes real traffic and feedback under privacy and
operational controls. Phase 2 implements offline experiments and production-path traces but does
not claim production monitoring, quality gains, or release thresholds.

Phase 3 currently provides the application PostgreSQL schema, deterministic planning,
transactional lineage/version persistence, an explicit immutable managed-index publisher, and an
explicit generation-pinned managed runtime profile with public/group Qdrant pre-filtering, plus
operator status, bounded/deep validation, retention planning, and confirmed physical cleanup.
It also supports validated restoration of a present RETIRED physical generation into a new
immutable ACTIVE generation. Authentication, rebuild from cleaned history, automatic runtime
reload, concurrent apply locking, in-place reactivation, runtime leases, semantic/LLM-based PII
classification, production
deployment, and formal release thresholds remain future work. Kafka, Airflow, Kubernetes,
Supabase, OpenSearch, and additional orchestration frameworks remain out of scope.

See [architecture details](docs/architecture.md) and the
[Phase 2 interview notes](docs/phase2-interview-notes.md). The first expanded-corpus run and its
non-significant sanity analysis are recorded in
[Phase 2 deterministic validation](docs/phase2-validation.md).
