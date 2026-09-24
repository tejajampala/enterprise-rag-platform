# End-to-end managed RAG demonstration

For the public portfolio path, including the Cloudflare Quick Tunnel and its fail-closed controls,
see [Local public demo](local-public-demo.md). For a code-level walkthrough from ingestion through
answer evaluation, see [Project learning guide](project-learning-guide.md).

This runbook demonstrates the repository as implemented. It uses the committed fixture for a
small, reproducible managed corpus and keeps every destructive lifecycle action out of the demo.
Run commands from the repository root.

## Safety legend

- **READ ONLY** reads configuration or existing PostgreSQL/Qdrant/MLflow state. A live retrieval
  may load Hugging Face models, but does not change the managed corpus or generation state.
- **MUTATES** intentionally writes application lineage, a Qdrant generation, or a schema revision.
- **SERVICE STATE** starts local containers but does not deliberately recreate named volumes.

Never use `docker compose down -v` when the local PostgreSQL, Qdrant, or MLflow data must be
preserved. This runbook never runs confirmed cleanup or confirmed restoration.

## 1. Configure and start the services

Prerequisites are Python 3.12, `uv`, Docker Compose, and enough memory and disk for the configured
Hugging Face embedding and reranker models.

```powershell
# READ ONLY with respect to application data; installs the locked project environment.
uv sync

# Creates a local, ignored configuration file. Replace every placeholder password.
Copy-Item .env.example .env
```

Select managed retrieval in `.env` and keep the lineage database URL consistent with the Compose
credentials:

```dotenv
RAG__RUNTIME__PROFILE=managed
RAG__DATABASE__URL=postgresql+psycopg://lineage_app:<local-password>@127.0.0.1:5433/consumer_finance_lineage
```

The Bedrock model is unnecessary for ingestion, publication, health checks, retrieval-quality
evaluation, or the operations dashboard. It is required only for generated answers.

```powershell
# READ ONLY: renders and validates interpolation without starting containers.
docker compose config --quiet

# SERVICE STATE: starts existing persistent services; it does not delete volumes.
docker compose up -d
docker compose ps
```

Compose provides Qdrant, a host-visible lineage PostgreSQL instance, an internal PostgreSQL
instance for MLflow, and MLflow. Sensitive host ports bind to `127.0.0.1`. If `localhost` behaves
differently on Windows because of IPv6 or proxy resolution, use `127.0.0.1` consistently.

## 2. Check configuration and initialize the lineage schema

```powershell
# READ ONLY: compare the configured database revision with the repository head.
uv run alembic current
uv run alembic heads

# MUTATES PostgreSQL schema only. This is always an explicit operator action.
uv run alembic upgrade head

# READ ONLY and model-free. Before an ACTIVE generation exists, NOT_READY is expected.
uv run python scripts/check_environment.py
```

The current migration head is `0005_restore_provenance`. Startup, preflight, dashboards, and
retrieval never apply migrations automatically.

## 3. Plan and apply source ingestion

First inspect the exact change classification:

```powershell
# READ ONLY: NEW / CHANGED / UNCHANGED / DELETED planning only.
uv run python scripts/ingest_incremental.py --mode fixture --details
```

Then deliberately persist the displayed plan:

```powershell
# MUTATES PostgreSQL lineage only; it does not touch Qdrant or run embeddings/BM25.
uv run python scripts/ingest_incremental.py --mode fixture --apply --details
```

The first apply registers the managed corpus, records an ingestion run, creates stable logical
documents, appends immutable document versions, and persists chunk derivations and per-run change
classifications. Repeating the same fixture produces an unchanged/no-op lineage observation rather
than duplicate versions. Missing documents are considered deleted only when
`--authoritative-snapshot` is explicitly supplied.

To inspect the persisted publication target without loading models or contacting Qdrant:

```powershell
# READ ONLY: PostgreSQL status and identity only.
uv run python scripts/index_managed.py --details
```

## 4. Publish the initial immutable generation

```powershell
# MUTATES PostgreSQL and Qdrant: explicit build, validation, and ACTIVE switch.
uv run python scripts/index_managed.py --publish --details
```

On the first publication, BM25 is fitted over the complete ordered chunk corpus and every dense
vector is computed. The publisher creates a new collection such as
`consumer_finance_managed_g000001`; it never replaces a collection in place. PostgreSQL records
the immutable generation, build configuration and fingerprints, BM25 state, and one membership
row per Qdrant point. Repeating the command without a lineage or configuration change returns
`NO_INDEX_CHANGE`.

## 5. Verify PostgreSQL and Qdrant together

Use the supported operations boundary instead of ad hoc writes to either store:

```powershell
# READ ONLY and model-free.
uv run python scripts/manage_index_generations.py --status
uv run python scripts/manage_index_generations.py --validate

# READ ONLY, model-free, and exhaustive: scrolls every point and validates payload v3.
uv run python scripts/manage_index_generations.py --validate --deep

# READ ONLY: service, migration, ACTIVE, membership, point-count, and payload smoke.
uv run python scripts/check_environment.py --managed-smoke
```

Expected evidence includes exactly one ACTIVE generation, its canonical collection, equal
PostgreSQL membership and physical Qdrant point counts, valid BM25/configuration fingerprints,
and compatible ACL/PII payload fields. `PASS_WITH_WARNINGS` is serving-ready when `ACTIVE
validation: PASS`; historical RETIRED or FAILED findings remain visible but do not falsely mark the
serving generation unhealthy. An ACTIVE failure is always `NOT_READY`/`FAIL`.

## 6. Run managed hybrid retrieval, ACL, PII, and telemetry

The following uses the real managed runtime API, initializes the configured Hugging Face models,
performs dense and sparse search, RRF fusion, reranking, ACL filtering, and default post-rerank PII
redaction, and emits one safe structured telemetry record:

```powershell
# READ ONLY, but may download/load models on first use.
@'
import logging

from consumer_finance_rag.access import AccessContext
from consumer_finance_rag.config import load_settings
from consumer_finance_rag.factory import build_runtime_retrieval
from consumer_finance_rag.observability import StructuredLoggingRetrievalTelemetrySink
from consumer_finance_rag.pii import PIIHandlingPolicy

logging.basicConfig(level=logging.INFO, format="%(message)s")
settings = load_settings("config/config.yaml")
runtime = build_runtime_retrieval(
    settings,
    telemetry_sink=StructuredLoggingRetrievalTelemetrySink(),
)
print(f"profile={runtime.profile} generation={runtime.generation} collection={runtime.qdrant_collection}")

result = runtime.retrieval_service.retrieve(
    "How can I dispute an unfamiliar card charge?",
    access_context=AccessContext(),  # missing groups means public-only
    pii_policy=PIIHandlingPolicy.REDACT,
)
for item in result.chunks:
    print(item.chunk.metadata.get("source_id"), item.chunk.text)
'@ | uv run python -
```

For a corpus containing restricted sources, pass an already authenticated caller's normalized
groups through `AccessContext(groups=frozenset({...}))`. The identical Qdrant filter is applied to
dense and sparse searches before fusion, followed by defense-in-depth validation. The managed
default is public-only when no context is supplied. `REDACT` operates only on authorized final
contexts after reranking. Do not use `PIIHandlingPolicy.ALLOW` in a demonstration that may expose
real personal data.

The JSON telemetry allowlist includes the runtime-pinned generation and collection, stage
latencies, candidate/final counts, ACL mode, caller group count, rejection count, PII policy, and
aggregate redaction/type counts. It excludes query text, chunk/document text, matched PII values,
principal IDs, group names, credentials, and exception messages. The default sink is a no-op; the
logging sink requires no external monitoring service.

The production boundaries are also exercised deterministically without local services or model
downloads:

```powershell
# READ ONLY offline regression demonstrations.
uv run pytest tests/test_phase3_end_to_end.py -v -k "acl_filtered_retrieval"
uv run pytest tests/test_phase3_end_to_end.py -v -k "managed_pii_redaction"
uv run pytest tests/test_retrieval_telemetry.py -v
```

## 7. Evaluate deterministic retrieval quality

The managed evaluator pins the current ACTIVE runtime once and reuses the production retrieval
path. It calculates identity-based Recall@K, hit rate, and MRR without Bedrock or an LLM judge:

```powershell
# READ ONLY with respect to PostgreSQL/Qdrant; writes the requested local report artifact.
uv run python scripts/evaluate_managed_retrieval.py --current `
  --output artifacts/retrieval_quality_current.json
```

The committed cases are a review-required seed set, not claimed production ground truth. The
report contains safe generation/configuration identity, aggregate metrics, case IDs, retrieved
identities, and per-case scores; it does not serialize queries, contexts, payloads, principals, or
groups.

MLflow tracking is optional and reporting-only:

```powershell
# READ ONLY for the managed index; writes a run/artifact to configured MLflow.
uv run python scripts/evaluate_managed_retrieval.py --current --log-mlflow
```

This does not invoke a paid judge. To exercise the same pinned managed runtime with the reviewed
PUBLIC answer-evaluation dataset, start with its retrieval-only mode:

```powershell
# READ ONLY for PostgreSQL/Qdrant; writes a local, content-free report.
uv run python scripts/evaluate_managed_rag.py --current --limit 2
```

The following commands are explicit live AWS operations and may incur charges. They preserve ACL
filtering and use REDACT-mode contexts; the generation and judge models remain independently
configured:

```powershell
# LIVE MODEL CALL: Bedrock answer generation only.
uv run python scripts/evaluate_managed_rag.py --current --generate --limit 2

# LIVE MODEL/JUDGE CALLS: content-bearing PUBLIC evaluation boundary.
uv run python scripts/evaluate_managed_rag.py --current --generate --limit 2 `
  --with-mlflow-judges --with-ragas --log-mlflow
```

The generated report contains identities, metrics, statuses, latency, and token totals, but omits
question, answer, context, PII values, principals, groups, and judge rationale text. MLflow tracing
and judges are disabled unless explicitly requested; the combined command uses one evaluation run.

## 8. Assess drift and regression against an approved baseline

Select a known, reviewed RETIRED generation whose physical collection is still present. Start with
the model-free assessment:

```powershell
# READ ONLY for PostgreSQL/Qdrant; writes one metadata-only local JSON report.
uv run python scripts/assess_managed_drift.py --baseline-generation 3 `
  --skip-retrieval-evaluation
```

The candidate defaults to ACTIVE and is pinned once. The report shows source NEW/CHANGED/UNCHANGED/
DELETED counts, content versus metadata-only changes, category/count deltas, compatibility,
physical validation, and unavailable evaluation dimensions. A non-authoritative candidate never
turns an absent source into a deletion.

If both generations use the current compatible schema and both physical collections exist, run the
same approved cases through the existing deterministic retrieval evaluator:

```powershell
# READ ONLY for PostgreSQL/Qdrant; loads embedding/reranker models, but calls no Bedrock model.
uv run python scripts/assess_managed_drift.py --baseline-generation 3
```

Only an explicit `--generate-answers` invokes Bedrock. Only explicit `--with-mlflow-judges` or
`--with-ragas` invokes a judge; `--log-mlflow` adds one metadata-only assessment run. Do not use
those flags during an offline or no-cost demonstration. An incompatible schema/model/dataset,
missing historical collection, failed judge, absent telemetry window, or insufficient sample size
is reported as incomplete evidence, never converted into a score. Findings recommend an operator
investigation and execute no lifecycle action.

## 9. Open the read-only operations dashboard

```powershell
# READ ONLY dashboard; model-free and contains no lifecycle mutation controls.
uv run streamlit run app/operations_dashboard.py --server.address 127.0.0.1
```

Use **Refresh** to view service availability, ACTIVE health, historical audit findings, generation
status/provenance, cleanup classifications, safe quality reports, and optional MLflow experiment
availability. Deep validation occurs only when explicitly requested. The key readiness distinction
is `ACTIVE validation: PASS` versus `Historical audit: WARNINGS`.

Upload the generated `managed-drift-assessment-v1` report under **Drift & regression** to show its
baseline/candidate identity, source/index changes, metric states, operational aggregates, and typed
findings. Uploading a report does not rerun evaluation. This section exists only in the private
operator dashboard, not the public showcase.

## 10. Demonstrate lifecycle planning without executing it

```powershell
# READ ONLY: reports protected, retained, recent, eligible, and orphan collections.
uv run python scripts/manage_index_generations.py --cleanup-plan

# READ ONLY: replace 1 with a present, compatible RETIRED generation.
uv run python scripts/manage_index_generations.py --restore-generation 1
```

Do not add `--confirm` during this demonstration. A cleanup plan never deletes PostgreSQL history.
A restoration plan validates the historical generation and proposes a new monotonic generation;
it never reactivates the old row or rolls back source lineage.

## 11. What to show an interviewer

1. The ingestion dry run and apply output: stable source identity plus immutable versions.
2. Publication output: BM25 action, dense computed/reused counts, generation, and collection.
3. Bounded/deep validation: PostgreSQL memberships agree with Qdrant points and payload v3.
4. One managed query: generation-pinned identity, ACL-before-ranking, redacted final context, and
   safe request telemetry.
5. A deterministic quality report and the dashboard's separate ACTIVE and historical status.
6. A model-free drift report, then a compatible retrieval comparison when a retained collection is
   available; point out explicit insufficient-data states.
7. Cleanup and restoration plans only, emphasizing the explicit confirmation boundary.

The five change scenarios and their exact persistence/reuse consequences are detailed in
[System design walkthrough](system-design-walkthrough.md#interview-ready-change-scenarios).

## Live-environment limitations

- The offline suite proves deterministic application behavior with SQLite, in-memory Qdrant, and
  model doubles; it does not prove that a developer's PostgreSQL, Qdrant, MLflow, model cache, AWS
  credentials, or Bedrock entitlement is reachable.
- The fixture is intentionally small and mostly public. Use the ACL regression or an approved
  restricted test corpus to demonstrate group-dependent results; do not manufacture production
  authorization data in the demo.
- The dashboard is a local read-only operator surface, not an authenticated administration plane.
- Runtime generation selection is process-pinned. Restart the application to resolve a newly
  ACTIVE generation.
