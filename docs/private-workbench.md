# Private RAG engineering workbench

The private workbench is a separate Streamlit entrypoint for genuine managed requests and
operator-only inspection. Bind it to loopback or appropriate private authentication; never add
its port to Cloudflare.

## Start locally

Configure the managed profile, lineage database, Qdrant, MLflow, AWS credentials, and an enabled
Bedrock model. The public showcase switches may remain false.

```dotenv
RAG__RUNTIME__PROFILE=managed
RAG__PUBLIC_SHOWCASE__ENABLED=false
BEDROCK_GENERATION_MODEL=google.gemma-3-4b-it
AWS_REGION=ap-southeast-2
```

```powershell
docker compose up -d
uv run python scripts/check_environment.py --managed-smoke
uv run streamlit run app/private_workbench.py `
  --server.address 127.0.0.1 `
  --server.port 8502 `
  --client.showErrorDetails none
```

The runtime is cached across Streamlit reruns. Operations shows startup identity, model/device
selection, retrieval depths, context limits, Bedrock configuration, MLflow health, and an explicit
embedding/reranker warm-up. It has no mutation controls and displays no credentials.

## Full request-inspection walkthrough

1. Open **Chat**, confirm the managed ACTIVE generation, and submit a question.
2. The old answer is cleared before submission. A failure reports its request ID and safe failure
   stage, so a prior success cannot look like the failed request's result.
3. On success, use **Inspect this request**. The workbench carries that exact request ID and trace
   ID into Request Inspector. Only sources explicitly cited as `[C<n>]` appear beside the answer.
4. Request Inspector resolves only that exact request ID or trace ID and shows the measured
   end-to-end sequence and bottleneck.
5. Inspect dense, sparse, RRF, and reranker tables. They contain safe chunk identity/title, stage
   rank, preserved stage-specific scores, final-context selection, and an inclusion reason. They
   do not contain passage text.
6. Select **Refresh trace**. The workbench calls `MlflowClient.get_trace()` with only the returned
   trace ID and verifies its `client_request_id`. It never substitutes a latest or other trace.

The verified observation in `data/fixtures/live_request_observation.json` documents generation
`g000004`: six dense, sparse and fused candidates; 31.72 seconds root duration; 1.05 seconds
embedding; 2.10 seconds dense HNSW; 2.06 seconds sparse BM25; 0.52 milliseconds RRF; 25.84 seconds
Qwen reranking; and 661 milliseconds Bedrock generation. It used `google.gemma-3-4b-it`, 928 input
tokens, 75 output tokens, PUBLIC-only ACL, REDACT, and citation `C1`. Reranking was the measured
bottleneck. The fixture intentionally contains no answer or private span content.

## What each score means

- **Dense score** is Qdrant's embedding/HNSW similarity. Larger values rank more similar for the
  configured cosine index.
- **BM25 score** is sparse lexical relevance returned by Qdrant. Its scale is not dense similarity.
- **RRF score** sums reciprocal rank contributions, `1 / (k + rank)`, from the dense and sparse
  lists. It uses ranks, not reconstructed original scores.
- **Reranker score** is the Qwen cross-encoder output for each query/passage pair and determines
  final ordering before PII handling and context construction.

Never compare raw values across methods. The inspector preserves original dense and BM25 scores
rather than inferring them from RRF.

## Cost estimation

Bedrock Converse `inputTokens`, `outputTokens`, and `totalTokens` are the usage source. Rules in
`config/bedrock_pricing.yaml` require an exact provider, model, region, pricing mode, currency,
effective date, and official source. The `us-east-1` Gemma 3 4B standard on-demand rule comes from
the official Amazon Bedrock pricing page.

The value is an estimated generation API cost, not billed AWS cost. A missing exact rule displays
**Cost unavailable**, never `$0.00`. Answers are deduplicated by request ID; judge/evaluation calls
remain separate. Local embedding/reranker compute is not labelled free and receives no invented
per-request cost.

## Missing or delayed spans

- **PENDING / Trace syncing**: the exact ID is not retrievable yet or is still in progress. Wait
  briefly and use **Refresh trace**.
- **UNAVAILABLE**: tracing was disabled or the request returned no trace ID.
- **FAILED**: the read failed, the trace has an error state, or client request ID does not match.
  No fallback trace is shown.
- **COMPLETED**: the exact trace is retrievable. An absent expected span is still **not recorded**;
  absence does not prove the underlying stage did not execute.

A successful answer does not prove all spans persisted. In particular, do not treat a temporarily
absent `bedrock_generation` span as evidence that Bedrock was skipped.

## Reranker latency checklist

Before changing the Qwen reranker or retrieval quality, check:

1. cold-start versus warmed latency using the explicit warm-up action;
2. whether all fused candidates are scored as one batch;
3. configured CPU/GPU device and actual runtime availability;
4. fused candidate count and final context count; and
5. candidate passage lengths and context character limit.

Model replacement is explicit configuration, never an automatic latency fallback.

## Approved private evaluation

The Evaluations page reuses the existing managed evaluator, MLflow GenAI judge adapter, and RAGAS
adapter. It does not implement another judge framework. Both judge backends receive the same
configured identity from `settings.mlflow.judge_model`; generation keeps its separate configured
Bedrock identity. The private selector defaults to GPT-OSS 20B in Sydney and retains the former
Llama 3 70B Virginia route only for explicit historical comparison:

```text
generation: google.gemma-3-4b-it
judge:      bedrock:/openai.gpt-oss-20b-1:0
LiteLLM:    bedrock/converse/openai.gpt-oss-20b-1:0
AWS model: openai.gpt-oss-20b-1:0
region:     ap-southeast-2
```

The installed provider mapping remains explicit: GPT-OSS uses Bedrock Converse while the Llama
baseline uses `bedrock/meta.llama3-70b-instruct-v1:0`. Offline preflight checks model metadata,
Region, the installed Converse registration, and structured `response_format` support. Metadata
success does not prove runtime invocation permission. The installed MLflow Bedrock provider still
lacks `get_endpoint_url`, so built-in MLflow scorers remain **UNAVAILABLE** and are not invoked.

The exact runtime values are displayed before execution. The three controls **MLflow GenAI
judges**, **RAGAS**, and **Save evaluation to MLflow** default on and can be turned off
independently. Disable default-on Chat evaluation globally with:

```dotenv
RAG__PRIVATE_WORKBENCH__AUTOMATIC_EVALUATION_ENABLED=false
```

The Chat option is eligible only when the submitted question exactly matches one unique case in
`managed_public_cases.json`. The case must retain reviewed PUBLIC authorization and empty caller
identity/groups. Arbitrary private questions are never submitted to a judge. A normal Chat trace is
metadata-only and is not treated as content for semantic groundedness.

Default-on means selected for a newly submitted eligible answer. Evaluation happens only as part
of that explicit **Generate** submission. Streamlit reruns, page navigation, displaying an old
answer, trace refresh, and report display do not invoke generation or judges. A session registry
keys results by exact request/dataset, generation, case IDs, model IDs, and switches; that exact
configuration runs at most once unless **Re-evaluate selected configuration** is explicitly used.

Before a dataset evaluation, the page shows model IDs, cases, generation calls, selected metrics,
logical metric count, and available token limits. Logical metrics are not promised as provider-call
counts because a metric may need multiple model invocations. Reference-dependent metrics are
reported as `REFERENCE_REQUIRED` when an approved expected answer is absent; values are never
invented. Each backend reports `COMPLETED`, `FAILED`, `UNAVAILABLE`, `SKIPPED`, or
`REFERENCE_REQUIRED` as applicable. A missing/failing judge does not invalidate an already
successful answer.

Gemma generation token usage and exact-match price estimates are reported separately from MLflow
and RAGAS judge execution. The RAGAS async LiteLLM callback records provider calls by case/metric,
input/output/total tokens, reasoning tokens when exposed, safe failures, latency, and observable
retries without retaining prompts, answers, contexts, rationales, or credentials. Judge usage stays
unavailable unless the provider returns it. A known subtotal is never labeled complete while a
potentially billable component is unaccounted for, and missing cost is never displayed as zero.
All amounts are labeled **Estimated USD**, not AWS invoice charges. Pricing is exact-match by model,
Region, Standard on-demand mode, and effective date. Report schema v3 adds optional execution
trace references while continuing to load v1/v2 reports; the Llama `paid-smoke.json` remains unchanged with judge cost
unavailable because it did not capture provider usage. Reused MLflow judge run IDs are passed to
safe report logging to avoid creating and counting a duplicate run.

### RAGAS execution traces

Each explicitly requested RAGAS evaluation records a separate metadata-only MLflow trace when
MLflow tracing is enabled. Its hierarchy is `ragas_evaluation` -> `ragas_metric:<metric>` ->
`bedrock_provider_call`. Each provider span opens immediately before the actual supplied LiteLLM
async completion callable is awaited and closes after its returned usage/status metadata is
attached. Its native MLflow duration therefore covers the real request/response wait, rather than
callback processing time. Spans are created from actual invocations, not metric counts. Async
task-local context and explicit parents keep concurrent provider calls under the metric that made
them.

Each execution receives a distinct 36-character UUID as its bounded MLflow evaluation client
request ID. The full configuration hash is never truncated or reused for that database-constrained
field. The trace carries only that UUID, the complete original request ID, reviewed case ID,
original RAG trace ID, complete evaluation configuration ID, MLflow evaluation run ID when already
available, GPT-OSS model ID, Region,
provider request ID when exposed, duration, token usage, retry count, safe status/category, and
an exact-match **Estimated USD** value when token usage and pricing are available. It never carries
questions, prompts, passages, answers, rationales, credentials, or exception text. The workbench
does not enable `mlflow.litellm.autolog()`.

The workbench displays the identifiers separately: **RAG generation trace ID**, **RAGAS evaluation
trace ID**, **MLflow evaluation run ID**, and **evaluation client request ID**. Use **View evaluation trace** from Chat,
Request Inspector, Evaluations, or Evaluation Input Inspector; it opens the exact-ID read-only
trace panel. The exporter flushes and reads that exact trace, verifies the bounded client request
ID, and confirms the root and expected child-span count. A local trace ID remains **PENDING / Trace
syncing** until this read succeeds; a correlation mismatch or non-transient export/read error is
**EXPORT_FAILED** and does not change an otherwise successful RAGAS status.
The original RAG trace remains available through Request Inspector. Historical reports without an
execution trace say **Evaluation execution trace not captured.** They are never replayed to create
missing spans.

To exercise only the local MLflow exporter and exact read-back—without retrieval, Bedrock, RAGAS,
or paid inference—run:

```powershell
uv run python scripts/smoke_mlflow_trace_export.py
```

Exit code `0` confirms the root and child, `2` means the exact trace is still syncing, and `3`
means export or verification is unavailable or failed.

### Evaluation input inspection and retention

For a newly generated Chat answer that exactly matches a reviewed PUBLIC case, the workbench keeps
up to eight versioned evaluation-input snapshots in the current Streamlit process for one hour.
The snapshot records exact request/case/configuration correlation, the generated answer, original
selected passages, ContextBuilder-formatted prompt context, and the existing RAGAS input row. It is
never written to safe evaluation JSON or routine MLflow metadata.

The inspector shows truncation explicitly. RAGAS retains its
`ragas-selected-original-chunks-v1` semantics (full selected chunk texts), while Gemma receives the
bounded ContextBuilder text; those values may differ for the final passage. Opening the inspector,
changing pages, and refreshing are read-only. When the process restarts, the TTL expires, or exact
request/case/configuration correlation fails, the UI reports **Input snapshot unavailable** and
does not regenerate or substitute another request. Clearing the Streamlit session or restarting the
private workbench deletes all snapshots. Persistent content snapshots are not implemented.

### Individual judge-call inspection

The **Individual Judge Calls** view is a separate, explicitly enabled diagnostic path for reviewed,
group-free PUBLIC evaluation cases. It is off by default. Enabling **Capture exact RAGAS judge I/O**
before an explicit evaluation records the actual redacted and bounded LiteLLM request and response
at the provider boundary; it never reconstructs content from a score, call order, or a later trace.
Routine RAG and evaluation traces and the normal JSON report remain metadata-only.

Choose an exact evaluation, then a metric, then a provider call. The inspector shows the stable call
ID, evaluation/metric/provider span IDs, request/run/configuration correlation, actual request
messages, structured-output schema, tools and effective parameters, followed by the returned content,
tool calls, finish reason, token usage, cost estimate, and Instructor parse/validation outcome. For
`context_precision`, retrieved chunk IDs are displayed only when their approved text matches the
actual outgoing request. No chunk association is inferred from call sequence.

Content-bearing artifacts are written beneath
`reports/evaluation/private_judge_calls/`, which is Git-ignored. The writer applies bounded text,
collection, and file-size limits, deterministic PII redaction, automatic expiry, and owner-only file
permissions where supported. Operators can delete one exact artifact or purge expired artifacts from
the private workbench. Opening, refreshing, deleting, or purging an artifact never invokes a model.
Historical evaluations without an exact matching artifact show **Individual judge-call content was
not captured for this run.**

One explicit one-case paid capture can be run with:

```powershell
uv run python scripts/evaluate_managed_rag.py --current --generate --limit 1 `
  --with-ragas --log-mlflow --capture-judge-io `
  --judge-io-retention-hours 24
```

This command uses the configured generation and judge models and can incur Bedrock charges. To
perform retention cleanup without evaluating, use `--purge-expired-judge-io`. Do not copy these
artifacts into tickets, logs, or the repository: even after redaction they are intentionally
content-bearing operator diagnostics.

## Page guide

- **Chat** runs one genuine managed request, shows cited sources, and links to exact inspection.
- **Request Inspector** shows candidate identity, rank movement, distinct score provenance,
  measured stages/bottleneck, exact-trace sync state, and cost detail.
- **Evaluations** runs bounded reviewed-PUBLIC evaluation only on explicit actions and reads the
  latest safe report.
- **Lineage & Index** displays the pinned ACTIVE generation and index identity without mutation.
- **Drift & Regression** reads a genuine uploaded managed drift report.
- **Operations** shows readiness/model/device diagnostics and offers explicit model warm-up and
  MLflow health checks.

See the [repository cleanup audit](repository-cleanup-audit.md) for retained compatibility paths
and safe cleanup decisions.

## Real local acceptance test

These commands use the configured live services and can incur Bedrock generation/judge charges.
They are not part of the offline test suite.

```powershell
docker compose up -d
uv run alembic current
uv run python scripts/check_environment.py --managed-smoke
uv run streamlit run app/private_workbench.py `
  --server.address 127.0.0.1 `
  --server.port 8502 `
  --client.showErrorDetails none
```

In the local workbench:

1. In **Chat**, enter exactly `What should a consumer include when disputing an error on a credit
   report?`. Confirm the displayed generation and judge IDs, selected metrics, one case, logical
   metric count, and usage limits before selecting **Generate grounded answer**.
2. Confirm one answer appears with cited links, generation usage/cost separated from judge
   execution, and MLflow/RAGAS statuses. A judge failure or `UNAVAILABLE` must leave the answer.
3. Select **Inspect this request**. Confirm the request/trace IDs match Chat, candidate scores and
   rank changes are genuine, the measured bottleneck is shown, and delayed traces say **Trace
   syncing** until **Refresh trace** retrieves that exact ID.
4. Navigate away and back, refresh the trace, and expand reports. Confirm no additional generation
   or evaluation occurs. Select **Re-evaluate this answer** only if a deliberate second judge run
   is desired.
5. In **Evaluations**, turn each switch off independently. With all desired paid switches selected,
   review the bounded case/call preview before **Run approved evaluation**. An identical ordinary
   rerun must reuse its result; only **Re-evaluate selected configuration** repeats it.

For a one-case paid acceptance using the same adapter and current GPT-OSS identity:

```powershell
# Free of Bedrock generation/judge calls.
uv run python scripts/evaluate_managed_rag.py --current --limit 2

# Explicit paid path: one Gemma answer and the four existing RAGAS metrics.
uv run python scripts/evaluate_managed_rag.py --current --generate --limit 1 `
  --with-ragas --log-mlflow --output reports/evaluation/gpt-oss-paid-smoke.json
```

Before accepting the result, confirm the preview/preflight shows the exact GPT-OSS URI, Sydney
Region, Converse route, and raw AWS model ID. Afterward confirm all four RAGAS metrics are valid,
the report contains actual invocation/token usage and per-case cost when returned, and the older
Llama paid smoke report is still unchanged. Do not add `--with-mlflow-judges` until the installed
MLflow Bedrock provider limitation is resolved upstream.
