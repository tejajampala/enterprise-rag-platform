# Phase 2 interview/reference notes

These definitions map directly to this repository rather than describing abstract benchmark
claims.

## Retrieval metrics

**Recall@K** asks what fraction of all known relevant chunks appeared in the first K results. It is
important when missing one required passage can make an answer incomplete. See `recall_at_k()` in
`evaluation/metrics.py`.

**Precision@K** asks what fraction of K result positions are relevant. It reflects context noise
and uses K as the denominator. See `precision_at_k()`.

**Hit Rate@K** is one when any relevant chunk appears and zero otherwise. It is useful for
single-answer lookup but does not reward retrieving multiple relevant passages. See
`hit_rate_at_k()`.

**MRR@K** uses the reciprocal rank of the first relevant chunk: rank one scores 1, rank two scores
0.5. It emphasizes how quickly a user or reranker sees the first useful result. See
`reciprocal_rank_at_k()`.

**NDCG@K** gives larger gains to higher relevance grades and discounts lower ranks
logarithmically, then divides by the ideal ordering. It measures graded ordering quality. See
`dcg_at_k()` and `ndcg_at_k()`.

**MAP@K** averages query-level average precision. Each relevant hit contributes precision at its
rank, rewarding systems that put all relevant chunks early. See `average_precision_at_k()` and
the stage aggregation in `evaluation/retrieval_runner.py`.

The retrieval CLI evaluates all six metrics independently for dense HNSW/ANN, sparse BM25, RRF,
and reranked outputs from `HybridRetrievalService.retrieve_debug()`. This isolates where a change
helped or hurt instead of attributing everything to “hybrid search.”

## Generation quality

**LLM-as-a-Judge** uses a configured model to grade an answer against inputs, expected facts, or
retrieval traces. It scales nuanced review but is nondeterministic, model-dependent, and billable;
it supplements rather than replaces deterministic metrics and human review. The explicit model
guard and MLflow scorers are in `evaluation/mlflow_genai.py`.

**Groundedness** asks whether answer claims follow from supplied contexts. **Correctness** asks
whether they match the expected facts/reference. A faithful summary of the wrong retrieved source
can be grounded but incorrect. MLflow `Correctness` and trace-based `RetrievalGroundedness` plus
RAGAS `ResponseGroundedness` make that distinction visible.

**Faithfulness** checks whether the response's claims are supported by retrieved evidence. RAGAS
0.4.3 `Faithfulness` is called through `evaluation/ragas_adapter.py`; both its numeric value and
reason are preserved.

## MLflow concepts

An MLflow **trace** is a complete RAG request. A **span** is one timed operation within that trace,
such as `dense_hnsw_search` or `bedrock_generation`. An MLflow **run** groups a repeatable
evaluation configuration, metrics, and artifacts. See `observability.py` for spans and
`evaluation/tracking.py` for runs.

Bedrock Converse returns input/output/total tokens. `bedrock_token_usage()` maps these onto the LLM
span's supported token attribute. MLflow can attribute cost only if it knows the actual provider
and model pricing; otherwise reports retain `null`/“unavailable” instead of a fabricated value.

## Offline versus production evaluation

**Offline evaluation** uses a fixed corpus and source-controlled cases to compare retrieval or
generation changes before release. It is repeatable and supports diagnosis, but its cases may not
represent real traffic. The 36 cases here are explicitly a seed set requiring review.

**Production evaluation** observes real requests, outcomes, feedback, latency, and drift under
privacy controls. Phase 2 adds production-path trace instrumentation but not a production sampling
or feedback system. That distinction prevents offline scores from being presented as live service
quality.
