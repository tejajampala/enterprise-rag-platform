# Phase 2 architecture

## Runtime boundaries

The Phase 1 application path remains plain Python:

```text
RAGPipeline.answer
  HybridRetrievalService.retrieve
    embed_query
    QdrantHybridStore.search_dense       # named HNSW/ANN vector
    QdrantHybridStore.search_sparse      # named BM25 sparse vector
    reciprocal_rank_fusion
    RerankerProvider.rerank
  ContextBuilder.build
  LLMProvider.generate                   # Bedrock Converse implementation
```

`factory.py` is the composition root. Model IDs, candidate depths, RRF K, collection names, AWS
region, and MLflow destinations come from typed `config.py` settings. Provider protocols continue
to isolate Hugging Face embeddings/reranking and Bedrock generation.

Qdrant remains the only retrieval database. PostgreSQL also has a separate, optional Phase 3
application database; it never shares MLflow's database or metadata tables. The initial application
schema records managed corpora, ingestion attempts, stable logical documents, immutable document
versions and chunks, per-run document classifications, and immutable Qdrant index generations with
their chunk membership. Application UUIDs connect versions across the existing content-derived
compatibility IDs. This milestone advances lineage persistence only: change classification
planning and the first transactional lineage write path are implemented. Vector reuse, index
activation, Qdrant mutation, and BM25 refresh are not implemented.

The managed-ingestion write boundary is deliberately small:

```text
LineageApplyService / business logic
        |
        v
LineageRepository
        |
        v
caller-owned SQLAlchemy Session
        |
        v
application PostgreSQL
```

`LineageRepository` persists caller-supplied lineage facts and flushes them without committing.
It does not calculate classifications or commit transactions. `LineageApplyService` consumes the
existing pure `ChangePlan`; it does not reclassify documents. Qdrant publishing and BM25 lifecycle
management remain unimplemented.

Source change planning is a separate pure business-logic boundary:

```text
authoritative current snapshot + previous persisted state
                         |
                         v
versioned fingerprints + plan_document_changes
                         |
                         v
          NEW / CHANGED / UNCHANGED / DELETED
                         |
                         v
       optional lineage persistence service
```

The planner compares normalized content hashes and material metadata fingerprints only. It does
not mutate PostgreSQL, touch Qdrant, rebuild BM25, or treat embedding/chunking configuration as a
source-document change. Missing documents are classified as `DELETED` only when the current
snapshot is explicitly authoritative; otherwise deletion detection is reported as skipped.

The Phase 3 CLI always constructs the same read-side plan:

```text
CorpusSnapshot --------------------> CurrentDocumentState ---+
                                                             |
LineageRepository (read-only) -----> PersistedDocumentState -+-> ChangePlan
```

`LineagePlanningService` is the small ORM/domain adapter. `scripts/ingest_incremental.py` selects
fixture or evaluation corpus assembly and prints the plan. By default it exits without committing;
an absent managed-corpus row is reported as uninitialized and remains absent. Explicit `--apply`
passes that same `ChangePlan` to `LineageApplyService`, which persists corpus/run/document/version/
chunk/classification state only.

The apply transaction model deliberately preserves a failed-attempt audit record:

```text
transaction 1: get/create corpus + commit RUNNING ingestion run
transaction 2: all lineage mutations + SUCCEEDED/NOOP transition
on failure:   rollback transaction 2, then mark the run FAILED separately
```

`NEW` creates a logical document, version 1, and its existing compatibility chunks. `CHANGED`
appends one immutable version and chunks. `UNCHANGED` records observation and advances
`last_seen_run_id` without redundant history; an all-unchanged plan is `NOOP`. Authoritative
`DELETED` soft-deletes the logical document without removing history. A deleted source that
reappears is explicitly unsupported rather than violating the corpus/source uniqueness rule.
Snapshot hashes use versioned canonical JSON (`managed-snapshot-v1`) over sorted source keys and
their content/metadata/version hashes. Advisory locking and restoration semantics are deferred.

The configured Qdrant prefix is registration metadata only. Apply never creates `index_states` or
`index_state_chunks`, calls Qdrant, calculates embeddings, or writes BM25 state. Retrieval therefore
continues to use the unchanged Phase 1/Phase 2 indexes.

The managed-index persistence foundation defines immutable generation metadata without publishing
an index yet. PostgreSQL is the future activation authority and permits only these first-version
transitions:

```text
BUILDING -> READY -> ACTIVE -> RETIRED
    |          |
    +----------+-> FAILED
```

`BUILDING` may temporarily lack an embedding dimension, BM25 state, or canonical index
configuration. A generation cannot become `READY` until those values and their deterministic
hashes are present. The complete BM25 query-encoder state is stored as generation-specific JSONB
(portable JSON in SQLite tests), so future runtime resolution can obtain the physical collection
and matching lexical encoder from one ACTIVE PostgreSQL row. One ingestion snapshot may own
multiple generation attempts, while a partial unique index permits only one ACTIVE generation per
corpus. Generation-number allocation still requires a future explicit locked transaction.

This milestone does not create, retrieve, validate, or mutate Qdrant collections; does not run
embedding inference; and does not activate a managed generation. The existing Phase 1 fixture and
Phase 2 evaluation indexing/retrieval paths remain independent.

The next managed-index input boundary resolves the exact active textual corpus before any index
work begins:

```text
ACTIVE logical documents
        |
        v
latest immutable DocumentVersion
        |
        v
requested ParagraphChunker configuration/fingerprint
        |
        v
validate existing derivation or transactionally materialize it
        |
        v
ActiveChunkManifest
        |
        v
future BM25 + dense embedding + Qdrant generation
```

`ChunkMaterializationService` reconstructs the existing domain `Document`, verifies normalized
text against its persisted content hash, and uses the existing `ParagraphChunker`. A complete
matching derivation is reused only after exact ordinal/ID/hash/text validation. An absent
derivation is added inside the caller's transaction; a partial or drifted derivation fails the
whole pass without rewriting immutable history. Changing chunk size or overlap therefore creates
another derivation of the same `DocumentVersion`, not a new source-document version. Deleted
logical documents remain historical and are excluded.

The manifest orders documents by `source_key` and chunks by `source_key`, ordinal, then
compatibility chunk ID. Its `active-chunk-manifest-v1` SHA-256 covers logical source/version and
chunk identities, text, the requested chunking fingerprint, and the active source snapshot hash;
database UUIDs and timestamps are excluded. The latest completed `SUCCEEDED`/`NOOP` ingestion run
anchors successful lineage state while later failed attempts are ignored. The source hash is
derived from the actual active latest versions (and must equal a latest authoritative run's stored
snapshot), which also handles cumulative non-authoritative observations safely. This boundary
still creates no index state, BM25 state, embedding, Qdrant collection, or generation publication.

BM25 preparation is the next pure planning boundary:

```text
ActiveChunkManifest
        |
        v
plan_bm25_generation (BM25 planner)
    |          |
    v          v
  REUSE      REFIT over every active chunk text
    |          |
    +----+-----+
         v
 verified complete BM25State
         |
         v
 future managed Qdrant publisher
```

BM25 statistics are corpus-wide: vocabulary, document frequencies, indexed-document count, and
average length depend on the complete ordered chunk-text sequence. The planner therefore reuses an
ACTIVE generation's state only after validating its canonical state hash, corpus hash, persisted
configuration fingerprint, chunk count, k1/b, implementation/tokenizer identities, and chunking
fingerprint. Missing or internally contradictory ACTIVE state is treated as lineage corruption;
it is not hidden by a refit. A first generation, text/membership/order change, chunking change, or
BM25 compatibility change refits exactly once across all active chunks.

Citation-only metadata changes can alter `ActiveChunkManifest.manifest_hash` while leaving the
ordered text corpus unchanged, so title/category changes may reuse BM25. Embedding and HNSW-only
configuration changes likewise do not affect BM25 state. `BM25State.document_count` counts chunks
(the BM25 documents), not logical source documents. Its existing length-framed
`bm25_corpus_hash` identifies the exact training text sequence and is deliberately distinct from
the broader lineage/payload `ActiveChunkManifest.manifest_hash`. Planning produces no sparse
vectors, index-state writes, Qdrant collection, or publication.

Dense-vector generation is a separate read-only plan:

```text
ActiveChunkManifest + previous ACTIVE generation
                         |
                         v
              DenseGenerationPlanner
                         |
                         v
               exact reuse validation
                         |
                         v
                  REUSED / COMPUTED
                         |
                         v
        complete prepared dense-vector plan
                         |
                         v
                 future publisher
```

Dense reuse is per chunk because an embedding is dependent on exact text and the complete
embedding fingerprint, not on the rest of the corpus. The planner considers only memberships in
the supplied previous ACTIVE generation, chooses duplicate exact-text candidates deterministically,
and reads their stored Qdrant point IDs in bounded batches. It reuses a vector only after checking
the returned point/payload lineage, hash, exact text, dimension, and finite values. A missing or
unusable individual old point is recomputed; a Qdrant read failure aborts planning. New embeddings
are requested once for only the non-reused texts. The plan preserves manifest order and records
the prior chunk and point provenance for every reuse.

This differs deliberately from BM25: dense vectors are text/model dependent and can be reused for
surviving exact chunks after metadata or chunking changes, while BM25 statistics are corpus
dependent and are reused or refitted as one whole-corpus decision. Dense planning performs no
Qdrant mutation, database mutation, BM25 fitting, index-state transition, or publication.

Point preparation combines the already completed plans without repeating their expensive work:

```text
ActiveChunkManifest + DenseGenerationPlan + BM25GenerationPlan
                              |
                              v
                ManagedPointPreparation
                              |
                              v
             complete immutable point inputs
                              |
                              v
                future ManagedIndexService
```

The preparation layer validates that manifest, dense plan, BM25 plan, and generation identity all
describe the same ordered active corpus. It constructs a `BM25SparseEncoder` from the selected
state and encodes each current chunk exactly once; it never fits BM25 again and never copies old
sparse vectors. Dense vectors are validated and passed through unchanged, without embedding calls
or normalization. Every payload is built from the current `ActiveChunk`, so a metadata-only new
version can reuse its old dense vector while receiving new citation metadata, lineage IDs, and a
point ID derived from its new lineage chunk. Separate prepared membership records retain
`REUSED`/`COMPUTED` provenance and the prior source chunk when applicable.

Prepared point IDs legitimately incorporate database lineage UUIDs. They are deterministic for
the same current lineage chunk and independent of index generation, but separately initialized
databases may produce different point identities. This layer performs no repository access,
Qdrant operation, embedding inference, BM25 fitting, index-state transition, or publication.

Managed publication now orchestrates these completed components without changing their internal
algorithms:

```text
ActiveChunkManifest
        |
        +--> BM25 generation plan
        +--> dense generation plan
                    |
                    v
          committed BUILDING index state
                    |
                    v
              point preparation
                    |
                    v
       immutable Qdrant generation collection
                    |
                    v
             physical validation
                    |
                    v
     atomic metadata + memberships + READY
                    |
                    v
       atomic previous ACTIVE -> RETIRED
              candidate READY -> ACTIVE
```

`ManagedIndexService` commits the BUILDING identity in a short corpus-row-locking transaction,
then releases the lock before BM25 work, embedding, collection creation, upsert, and physical
validation. The previous ACTIVE generation remains unchanged and usable throughout that work.
After physical validation, all build metadata, point memberships, dense provenance, and the READY
transition commit atomically. A second short locked transaction verifies the original ACTIVE
baseline and target ingestion run before retiring the old generation and activating the candidate
in one commit.

Pre-READY failures mark the committed candidate FAILED in a separate transaction while preserving
the original error. Deterministic activation conflicts mark a READY candidate FAILED; unexpected
activation infrastructure/database errors leave it READY for a future retry. Failed, BUILDING,
READY, and RETIRED physical collections are retained for diagnosis—automatic cleanup and
retention are not implemented yet. A no-index-change health check verifies persisted BM25 and
configuration identity plus collection shape and exact point count. Detecting externally replaced
same-count points is intentionally deferred to a later repair/audit command.

The publisher reads source lineage but never creates ingestion runs, document versions, or chunks,
and never changes document observation/deletion state. Managed retrieval is not implemented yet.

The managed Qdrant boundary stores only fully prepared generation data:

```text
ActiveChunkManifest + prepared dense vectors + prepared sparse vectors
                              |
                              v
                     ManagedQdrantStore
                              |
                              v
               immutable generation collection
                              |
                              v
                    physical validation
```

Generation collection names are derived from the registered prefix and generation, such as
`consumer_finance_managed_g000001`. Creation is non-destructive: an existing collection is a
collision error and is never deleted or recreated automatically. The Phase 1 fixture and Phase 2
evaluation collection names are explicitly protected. Managed point IDs use versioned UUID5
identity (`lineage-chunk-uuid5-v1`) derived from the lineage chunk primary key, so the same chunk
has the same point ID in separate physical generations.

The adapter validates already-prepared dense and sparse values, writes deterministic batches with
Qdrant acknowledgement, and exposes bounded dense retrieval, paginated inspection, exact count,
and physical validation. Validation checks collection vector configuration, point identity and
payload, dense presence/dimension/finiteness, and sparse identity/weights using explicit
float32-appropriate tolerances. The adapter does not fit BM25, call embedding providers, decide
dense reuse, create or transition PostgreSQL index states, or activate a generation. Full managed
publishing orchestration remains unimplemented.

Fixture and evaluation indexes are deliberately distinct. `scripts/ingest.py --mode fixture`
writes `consumer_finance_phase1` plus `data/index/bm25_state.json`.
`scripts/ingest.py --mode evaluation` requires downloaded records and writes
`consumer_finance_evaluation` plus `data/index/evaluation_bm25_state.json`. The BM25 state includes
an ordered corpus hash. Evaluation startup compares that hash and document count with the assembled
chunks, then compares every live Qdrant `chunk_id`/`chunk_hash` payload with the same snapshot.

## Tracing design

`observability.py` is a small adapter around current manual MLflow spans. When disabled it returns
a no-op implementation with the same protocol, so instrumentation does not branch or duplicate
the application pipeline.

```text
rag_request (CHAIN)
├── embed_query (EMBEDDING)
├── dense_hnsw_search (RETRIEVER)
├── sparse_bm25_search (RETRIEVER)
├── rrf_fusion (CHAIN)
├── rerank (RERANKER)
├── context_build (RETRIEVER)
└── bedrock_generation (CHAT_MODEL)
```

The root records model/config identity and actual candidate counts. Retrieval spans use chunk IDs,
titles, ranks, and scores rather than dense vectors or unrestricted payloads. `context_build`
emits MLflow's standard retriever document shape for final evidence only. The Bedrock span records
the model/provider, stop reason, request ID, and `mlflow.chat.tokenUsage` when Converse returns all
usage fields. Span duration is the latency source. Cost remains unset unless MLflow can calculate
it from real model/usage metadata.

A **trace** describes one request and its causal span tree. An evaluation **run** groups experiment
parameters, aggregate metrics, artifacts, and links to traces. They answer different questions:
traces explain execution; runs compare experiments.

## Evaluation model and reports

`evaluation/models.py` defines immutable seed cases and report contracts. Each case includes
question, category, difficulty, expected facts, reference answer, relevant document/chunk IDs,
optional grades, notes, and review status. `evaluation/dataset.py` rejects duplicate IDs, invalid
Pydantic categories/grades, and qrels that do not resolve in the selected corpus.

The repository JSON is authoritative and reviewable. `sync_evaluation_dataset.py` merges its
inputs, expectations, and tags into MLflow's Evaluation Dataset API. This requires the SQL backend
provided by Docker Compose; unit tests do not contact it.

All 36 cases remain `review_required` seed cases, not a production golden set. Their qrels are
intentionally anchored to a few stable fixture chunks while the downloaded CFPB documents act as
real retrieval distractors. Broader qrels across the expanded corpus require human review; neither
acquisition nor evaluation invents relevance labels.

`evaluation/metrics.py` implements the formulas directly:

- Recall@K: relevant retrieved / all relevant.
- Precision@K: relevant retrieved / K.
- Hit Rate@K: whether at least one relevant result appears.
- MRR@K: reciprocal rank of the first relevant result.
- DCG/NDCG@K: logarithmically discounted graded gain divided by the ideal ordering.
- MAP@K: mean per-query average precision, with one query-level AP computed by the module.

`evaluation/retrieval_runner.py` calls the existing `retrieve_debug()` and evaluates dense,
sparse, fused, and reranked lists for the same cases. It aggregates globally and by category and
difficulty. `evaluation/reporting.py` writes JSON summaries and per-case CSV rows that Streamlit
can read and future CI can compare. No threshold or improvement claim is embedded.

## MLflow GenAI and RAGAS

`evaluation/mlflow_genai.py` is the only MLflow judge/dataset adapter. Judge construction calls
`require_judge_model` first, preventing MLflow's environment-level default from silently selecting
a billable OpenAI model. It invokes `mlflow.genai.evaluate()`, never the classic model evaluation
API. Correctness uses expected facts/reference response; retrieval groundedness is included only
when every evaluated row has its production trace.

`evaluation/ragas_adapter.py` is the only RAGAS integration. It uses RAGAS 0.4 collections,
`@experiment`, an in-memory experiment backend, `llm_factory`, and retains `MetricResult.value`
and `.reason`. LiteLLM translates the explicitly configured `bedrock:/...` judge URI for Bedrock.
The adapter exposes a narrow protocol/no-op implementation, so disabling or removing RAGAS does
not affect `RAGPipeline`.

RAGAS 0.4.3 transitively installs LangChain, LangGraph, and LangSmith. They are not architectural
components and are not imported in runtime, retrieval, generation, tracing, or UI modules. A test
parses those modules' ASTs to enforce the boundary. The adapter contains an evaluation-local shim
for RAGAS 0.4.3's stale `langchain_community.chat_models.vertexai` import; the sentinel is used
only for legacy type detection and never routes a request.

## Infrastructure

Docker Compose owns three services:

- Qdrant 1.19 for dense and sparse retrieval.
- PostgreSQL 17 on the internal Compose network for MLflow metadata, traces, runs, and Evaluation
  Dataset state.
- MLflow 3.16 on port 5000, with PostgreSQL backend and a local Docker artifact volume.

PostgreSQL is intentionally not exposed to the host. Credentials enter Compose through `.env` and
are not committed. Qdrant and MLflow have health checks; MLflow waits for PostgreSQL health.

## Corpus strategy

The six-document fixture is stable, fast, and offline. `corpus_expansion.py` separately interprets
the versioned CFPB sitemap manifest, filters official Ask CFPB URLs using topic rules, enforces a
hard count, caches raw pages, extracts main content, and emits `SourceRecord` values. Normalization
and indexing remain downstream. `corpus.py` assembles the fixture first, removes acquired records
with duplicate source URLs in deterministic acquisition order, and then normalizes/chunks the
combined corpus. Downloaded data is
ignored by Git and network acquisition never runs in unit tests. The retrieval evaluator validates
qrels against this same assembled snapshot, verifies its dedicated persisted index, and records its
mode, document, chunk, and category counts.

Hugging Face providers retain their model object after lazy initialization. One evaluation process
constructs one retrieval service, explicitly warms its embedding and reranker once through the
same provider interfaces, and reuses both across all cases. `--max-cases` changes only the selected
case list; it does not create an evaluator-specific retrieval implementation.

## Security and deferrals

Retrieved content remains untrusted prompt data. Traces avoid secrets and vectors. Boto3 uses
standard credential resolution. Paid generation and judging require explicit commands/config.
Authentication, ACLs, PII pipelines, managed Qdrant/BM25 indexing, production
deployment, online evaluation, and formal regression gates remain future work.
