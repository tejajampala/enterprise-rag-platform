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
planning and the transactional lineage write path are implemented, followed by explicit managed
generation publication, activation, vector reuse, BM25 lifecycle management, and retrieval.

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
management are separate responsibilities owned by `ManagedIndexService`.

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
activation infrastructure/database errors leave it READY for a future retry. A no-index-change
health check verifies persisted BM25 and
configuration identity plus collection shape and exact point count. Detecting externally replaced
same-count points is intentionally deferred to a later repair/audit command.

The publisher reads source lineage but never creates ingestion runs, document versions, or chunks,
and never changes document observation/deletion state.

The operator boundary is intentionally a separate command from source ingestion:

```text
scripts/ingest_incremental.py --apply
        |
        v
persisted latest SUCCEEDED/NOOP lineage
        |
        v
scripts/index_managed.py --publish
        |
        +--> transaction: materialize/validate active chunks -> COMMIT
        |
        +--> ManagedIndexService.publish(manifest, target_run_id)
        v
PUBLISHED or NO_INDEX_CHANGE
```

`scripts/index_managed.py` without `--publish` reads only the configured managed corpus, latest
successful/no-op lineage run, active logical-document count, and optional ACTIVE index identity.
It does not materialize alternate chunks, construct the embedding provider, fit BM25, or access
Qdrant. Publish mode uses the existing repository run-resolution contract, so a later `FAILED`
ingestion attempt cannot become the index target. The materialization transaction is committed
before embedding inference or any Qdrant operation; the publisher then owns its established short
BUILDING, READY, and activation transactions.

The CLI constructs the existing lazy `HuggingFaceEmbeddingProvider` from model settings and a
dimension-aware `ManagedQdrantStore` factory from the configured Qdrant URL. It contains no BM25,
dense-reuse, point-preparation, upsert, or activation algorithm. Repeating the command against
unchanged lineage/configuration returns `NO_INDEX_CHANGE`; older PostgreSQL generation history is
retained even when a separately confirmed operation later removes eligible physical storage.

Managed generation operations remain separate from publication:

```text
PostgreSQL immutable generation history
                    |
                    v
       ManagedIndexOperationsService
          |        |        |       |
        status  validate   deep   cleanup plan
                                    |
                        explicit --cleanup --confirm
                                    |
                  locked authoritative re-check
                                    |
                   guarded physical Qdrant delete
```

`scripts/manage_index_generations.py` is read-only by default. Status shows every persisted
generation and whether its physical collection exists. Bounded validation checks the single ACTIVE
generation, canonical naming, persisted BM25/configuration hashes, membership and point counts,
vector names, dense dimension, distance, payload compatibility, and collection readability. Deep
validation additionally scrolls points to check point/lineage identity, membership, chunking, ACL,
PII, payload schema, and vector presence. RETIRED collections missing after cleanup and FAILED rows
without collections remain valid historical states; missing ACTIVE storage is an operational error.

The pure cleanup planner protects ACTIVE, READY, and BUILDING states, the newest configured number
of RETIRED generations, and RETIRED generations inside the configured minimum-age grace period.
Older canonical RETIRED collections and present canonical FAILED collections can be eligible.
Managed-looking orphan collections are report-only, and Phase 1/evaluation collections cannot
match the guarded managed name. Before each confirmed delete, the service locks the corpus row,
re-reads PostgreSQL state, re-lists Qdrant, and recomputes retention/age eligibility. This shares
the publication locking convention and prevents stale-plan deletion after a concurrent state
change.

Cleanup never deletes or updates PostgreSQL lineage/index rows. A cleaned RETIRED generation and
all `index_state_chunks`, document lineage, ACL history, and PII history remain queryable. The
minimum age protects common generation-pinned process overlap only as a grace period; it is not a
distributed lease. Runtime leases, rollback/reactivation, repair, and telemetry are deliberately
deferred. No schema migration is required for this operations layer.

Managed retrieval resolution is a separate, explicit factory boundary:

```text
PostgreSQL activation authority
        |
        v
exactly one ACTIVE index_state
        |
        +--> qdrant_collection + dense/sparse vector names
        +--> persisted BM25 state from the same row
        +--> immutable build configuration/fingerprints
        v
ManagedRetrievalRuntime
        |
        v
existing HybridRetrievalService
```

`build_managed_retrieval_runtime()` validates the canonical BM25/configuration hashes, embedding
compatibility, registered generation collection name, vector names, dense dimension, distance,
and exact point count before returning a runtime. It reconstructs `BM25SparseEncoder` directly
from the ACTIVE row; it never reads Phase 1/2 BM25 files, fits BM25, generates document vectors,
or mutates PostgreSQL/Qdrant. The configured reranker, candidate depths, final-context depth, RRF
K, and optional query-time HNSW `ef` remain query-time choices and do not alter the immutable index
fingerprint.

Resolution is process-lifetime pinned: a runtime built for `g000003` keeps both that collection and
its matching BM25 encoder even if `g000004` is activated later. A process restart or future
explicit reload must rebuild the complete runtime; collection names are never hot-swapped alone.
The Phase 1/2 retrieval implementations remain unchanged.

The offline end-to-end regression in `tests/test_phase3_end_to_end.py` proves this complete
two-generation sequence using the production orchestration boundaries:

```text
Source snapshot 1
        |
        v
lineage apply -> publish g1 -> g1 ACTIVE -> runtime_g1 query
        |
        v
Source snapshot 2 (one changed source)
        |
        v
lineage apply -> g2 BUILDING -> g2 READY -> g2 ACTIVE
                                      |
                                      +-> g1 RETIRED
                                      +-> new runtime_g2 resolves g2
                                      +-> existing runtime_g1 remains pinned to g1
```

The regression uses SQLite and in-memory Qdrant with deterministic embedding/reranking doubles,
but retains the real paragraph chunker, change planner, lineage services, BM25 implementation,
managed publisher/store, resolver, and `HybridRetrievalService`. It also verifies per-generation
lineage membership equals the physical point count, dense reuse provenance references exact prior
text/hash, BM25 state is independently valid for each ordered corpus, and unchanged republishing
returns `NO_INDEX_CHANGE`. It does not poll for or hot-reload a newly ACTIVE generation.

Application integration uses a separate explicit profile dispatcher rather than changing the
legacy/evaluation factory entry points:

```text
runtime.profile = fixture    -> Phase 1 collection + Phase 1 BM25 file
runtime.profile = evaluation -> evaluation collection + evaluation BM25 file
runtime.profile = managed    -> ManagedRetrievalRuntime.retrieval_service
                                      |
                                      v
                               backend-agnostic RAGPipeline
```

`fixture` is the default, so existing application users do not require PostgreSQL. The fixture and
evaluation profiles remain static and database-independent. Managed mode requires the application
database and propagates missing/corrupt/incompatible ACTIVE-generation errors without falling back.
Streamlit caches the selected `RetrievalRuntime`, displays the selected profile, and displays a
generation/collection only for managed mode. On the first submitted question it reuses that pinned
retrieval runtime while lazily composing the Bedrock-backed `ApplicationRuntime` and request
tracer. This avoids AWS/MLflow startup work while keeping the collection and BM25 encoder pinned as
one unit for the process/session lifecycle. A restart is required to select a newly activated
generation; there is no polling or automatic hot reload.

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
publishing orchestration is implemented by `ManagedIndexService`, not this storage adapter.

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

## Managed retrieval authorization

Authentication and identity-provider integration are outside this milestone. A caller resolves
identity facts into the immutable request-scoped `AccessContext`; the generation-scoped runtime
never stores a process-global user identity.

```text
Identity provider / application
        |
        v
resolved AccessContext
        |
        v
managed HybridRetrievalService
        |
        v
one Qdrant ACL filter
        +--> dense HNSW/ANN search
        +--> sparse BM25 search
        |
        v
RRF -> reranker -> authorized contexts
```

`DocumentAccessPolicy` supports `PUBLIC` and group-based `RESTRICTED` content. Group identifiers
are trimmed, case-folded, deduplicated, and sorted for persistence and hashing; empty identifiers,
restricted policies without groups, and public policies with groups are rejected. PostgreSQL stores
`access_visibility` and `allowed_groups` on each immutable `document_versions` row. The v0003
migration assigns existing rows the backward-compatible `PUBLIC` policy.

Chunks inherit their owning document version's policy. Managed payload schema v3 serializes the
explicit `access_visibility` and sorted `allowed_groups`; no principal identity or secret is
persisted in Qdrant. At query time, a point is eligible when it is public, or when it is restricted
and its allowed-group array contains any caller group. The identical typed Qdrant filter object is
passed to both candidate searches before fusion. Managed results are also validated against the
request context as defense in depth, but Python-side validation is not the primary enforcement.
Fixture and evaluation services do not enable ACL filtering, so their legacy points remain
backward-compatible.

The security invariants are:

1. Authorization is applied before dense retrieval.
2. Authorization is applied before sparse retrieval.
3. Unauthorized candidates never enter RRF or reranking.
4. Missing access context is public-only, never unrestricted.
5. Policy changes create immutable document-version history.
6. Old index generations retain their historical policy payload.
7. Runtime generation pinning also pins ACL payload state.

Access policy is part of the version metadata fingerprint. Consequently, changing `PUBLIC` to
`RESTRICTED`, or changing the canonical group set, creates a metadata-only document version while
leaving `content_hash` unchanged. The active-manifest hash includes policy, so a new immutable
managed generation is published. If ordered chunk text is unchanged, whole-corpus BM25 state and
exact-text dense vectors are reused; new points still carry fresh lineage IDs and ACL payload.

## Managed PII governance and safe contexts

The first PII milestone is deterministic and entirely local:

```text
resolved AccessContext
        |
        v
Qdrant ACL-filtered dense + sparse retrieval
        |
        v
RRF -> reranker (authorized original text)
        |
        v
PIIHandlingPolicy (managed default: REDACT)
        |
        v
local deterministic redaction + verification
        |
        v
safe final contexts -> downstream LLM
```

`DeterministicPIIDetector` recognizes reasonable emails, practical AU/international phone
formats, checksum-valid nine-digit Australian TFNs, and Luhn-valid 13–19 digit payment-card
candidates. Detector behavior is explicitly versioned as `deterministic-pii-v1`. It stores only
safe match structure in memory—type and offsets—and never copies the matched value into metadata.
Names, physical addresses, medical or demographic facts, semantic/LLM classifiers, and cloud DLP
services remain outside this milestone.

PII classification is derived governance metadata, not source identity. The
`document_version_pii` table is keyed by `(document_version_id, detector_version)` and stores the
classification hash, sorted PII types, match count, and classification timestamp. The hash covers
the detector version, source-text hash, and safe match structure. A changed detector implementation
must use a new detector version; conflicting results under an existing version fail materialization.
Neither a new derived classification nor detector upgrade changes `content_hash`, metadata hash,
source `version_hash`, or the persisted original text.

Chunk classification is deterministically recalculated from exact persisted chunk text during
managed materialization. Payload schema v3 includes only `contains_pii`, sorted `pii_types`, and
`pii_detector_version`; raw matches are not duplicated into Qdrant fields. Detector version is
also part of the managed index configuration fingerprint, so detector upgrades publish a new
immutable generation even though the source snapshot is unchanged. BM25 and dense reuse retain
their existing text-based rules.

ACL and PII have separate responsibilities and ordering. ACL decides whether a candidate may be
retrieved and is applied before both Qdrant searches. Only authorized candidates reach RRF and the
reranker. The PII policy then determines the representation of selected final contexts. Managed
requests default to `REDACT`; `ALLOW` is explicit and is never inferred from groups. Redaction uses
match offsets, preserves surrounding text and lineage identifiers, and performs a second local
scan. If supported PII remains, retrieval raises `PIIRedactionError` instead of returning the
context. Production `RAGPipeline` receives only the final redacted chunks under `REDACT`.

The security invariants are:

1. ACL authorization precedes all candidate retrieval.
2. Unauthorized text never reaches PII handling.
3. Authorized retrieval and reranking may use original text internally.
4. `REDACT` transforms final contexts before downstream generation.
5. Managed missing/default PII policy means `REDACT`, never raw exposure.
6. Supported residual PII causes a fail-closed error.
7. Raw detected values are not duplicated into classification metadata or logs.
8. Original source lineage remains immutable and unredacted.
9. Detector implementation identity is explicit and versioned.
10. Derived PII governance metadata is separate from source-document identity.

Hugging Face providers retain their model object after lazy initialization. One evaluation process
constructs one retrieval service, explicitly warms its embedding and reranker once through the
same provider interfaces, and reuses both across all cases. `--max-cases` changes only the selected
case list; it does not create an evaluator-specific retrieval implementation.

## Security and deferrals

Retrieved content remains untrusted prompt data. Traces avoid secrets and vectors. Boto3 uses
standard credential resolution. Paid generation and judging require explicit commands/config.
Authentication, identity-provider integration, semantic/LLM PII pipelines, production
deployment, online evaluation, and formal regression gates remain future work.
