# Phase 2 deterministic retrieval validation

This is a measured local validation run, not a benchmark claim or statistical-significance
claim. It used the production `HybridRetrievalService.retrieve_debug()` path and made no Bedrock,
MLflow-judge, or RAGAS-judge calls.

## Corpus and experiment

- Run date: 2026-09-13
- MLflow run ID: `2f4486ba0fb04bcc8d72e464ad0907b3`
- Seed set: `consumer-finance-seed-evaluation` version `1.0`
- Review status: all 36 cases remain `review_required`
- Acquired: 336 official CFPB Ask CFPB pages from the committed manifest
- Normalized/indexed: 340 documents and 1,556 chunks
- Composition: 6 fixture documents plus 334 distinct downloaded documents; two downloaded
  source URLs overlapped fixture URLs and were excluded with fixture precedence
- Categories: mortgages 136, credit cards 56, credit reporting 47, payments/fraud 42, debt
  collection 30, bank accounts 29
- Models: `Qwen/Qwen3-Embedding-0.6B` and `Qwen/Qwen3-Reranker-0.6B`, CPU
- Qdrant collection: `consumer_finance_evaluation` (the six-chunk fixture remains isolated in
  `consumer_finance_phase1`)
- Candidate depths: dense 20, sparse 20, fused 12, final contexts 5
- RRF K: 60; BM25 k1/b: 1.2/0.75; HNSW M/ef_construct: 16/100

The machine-readable reports are generated at `reports/evaluation/retrieval_summary.json` and
`reports/evaluation/retrieval_cases.csv` and are also artifacts on the MLflow run. Downloaded raw
pages and generated reports remain intentionally ignored by Git.

## Aggregate results at K=5

| Stage | Recall | Precision | Hit rate | MRR | NDCG | MAP |
|---|---:|---:|---:|---:|---:|---:|
| Dense HNSW | 0.833333 | 0.166667 | 0.833333 | 0.597222 | 0.657518 | 0.597222 |
| Sparse BM25 | 0.611111 | 0.122222 | 0.611111 | 0.426389 | 0.471778 | 0.426389 |
| RRF fused | 0.805556 | 0.161111 | 0.805556 | 0.531944 | 0.599707 | 0.531944 |
| Reranked | 0.916667 | 0.183333 | 0.916667 | 0.815741 | 0.841101 | 0.815741 |

Each seed case currently has one relevant chunk, so query-level AP equals reciprocal rank in this
run and aggregate MAP equals aggregate MRR. Precision uses K as its denominator. The reranked list
contains five results, so K=5 is its meaningful final-context comparison.

## Sanity analysis

- Dense was stronger than BM25 for every aggregate quality metric at K=5 and K=10.
- RRF was stronger than BM25 but did not improve over dense alone under this configuration.
- Reranking raised MRR@5 from 0.531944 (fused) to 0.815741 and NDCG@5 from 0.599707 to
  0.841101. It also promoted relevant candidates from fused ranks 6–12, raising Recall@5 from
  0.805556 to 0.916667.
- Mortgages were weakest after reranking: Recall@5 0.666667, MRR@5 0.583333, NDCG@5 0.605155.
  Credit reporting was next weakest at Recall@5 0.833333.
- No case was completely absent from the union of the dense-20 and sparse-20 candidate sets.
- Candidate-union coverage was dense-only 6, sparse-only 1, both 29, neither 0. BM25 uniquely
  recovered `reports-005` even though its standalone aggregate scores were lower.
- Relevant fixture chunks were absent from both the fused top-12 and final reranked top-5 for
  `reports-005`, `mortgages-001`, and `mortgages-006`.
- Equal-weight RRF with K=60 compressed rank differences: candidates occurring in both lists
  outranked these one-list hits. Their full-union RRF ranks were respectively 19, 13, and 30, so
  the fused top-12 truncation removed them before reranking.

These 36 cases remain seed cases whose relevance is concentrated on known fixture chunks. Future
human review should add judgments across the larger real corpus; no qrels were fabricated for this
run.

## Model initialization finding

Providers cache their model objects, and one retrieval service is reused for the whole process.
The earlier apparent stall had two concrete causes: blocked Hugging Face metadata/download retries
on first use, followed by slow CPU inference for the expanded index. Once cached, model warm-up
succeeded. The first 1,556-chunk CPU indexing pass took approximately 83 minutes in this local
environment; this is indexing setup time, not query latency. `scripts/warm_models.py` isolates
model readiness, ingestion reports completed embedding batches, and retrieval evaluation reports
each case while preserving the production retrieval path.

Measured CPU wall times were approximately 325.6 seconds for a five-case smoke evaluation and
2,442.3 seconds (40 minutes 42.3 seconds) for the corrected 36-case run. Model loading and warm-up
occurred once per process; provider instances were reused for all cases.
