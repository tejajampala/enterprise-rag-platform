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

Qdrant is the only retrieval database. PostgreSQL is introduced only as MLflow's backend store;
there is no Phase 3 application-lineage schema.

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
and indexing remain downstream. Downloaded data is ignored by Git and network acquisition never
runs in unit tests.

## Security and deferrals

Retrieved content remains untrusted prompt data. Traces avoid secrets and vectors. Boto3 uses
standard credential resolution. Paid generation and judging require explicit commands/config.
Authentication, ACLs, PII pipelines, document version history, incremental indexing, production
deployment, online evaluation, and formal regression gates remain Phase 3 work.
