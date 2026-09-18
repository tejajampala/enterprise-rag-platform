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
compatibility IDs. This milestone defines persistence only: incremental ingestion, change
classification, vector reuse, index activation, and automatic registration of the Phase 1 fixture
or Phase 2 evaluation corpus are not implemented.

The repository boundary for future managed-ingestion work is deliberately small:

```text
future service / business logic
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
Change detection, incremental-ingestion decisions, and Qdrant publishing remain unimplemented.

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
Authentication, ACLs, PII pipelines, lineage service logic, incremental indexing, production
deployment, online evaluation, and formal regression gates remain future work.
