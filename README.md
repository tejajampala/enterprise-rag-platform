# Consumer Finance Intelligence

Consumer Finance Intelligence is a production-shaped retrieval-augmented generation platform over
public U.S. consumer-finance material. It combines immutable PostgreSQL lineage, hybrid Qdrant
retrieval, Hugging Face embeddings and reranking, AWS Bedrock generation, ACL and PII governance,
MLflow observability, deterministic quality evaluation, and a Streamlit portfolio interface.

The full platform runs locally on Windows. A temporary Cloudflare Quick Tunnel may expose only the
public Streamlit showcase; PostgreSQL, Qdrant, MLflow, and the operations dashboard stay on
loopback. The project provisions no AWS hosting infrastructure.

> Educational demonstration only. Generated answers are not legal or financial advice.

## What this demonstrates

- Stable source identity with immutable document versions and chunk derivations.
- Incremental generation builds with dense-vector reuse and corpus-wide BM25 reuse or refitting.
- Transactional publication of immutable Qdrant generations and generation-pinned retrieval.
- Dense HNSW + sparse BM25 retrieval, reciprocal rank fusion, cross-encoder reranking, and citations.
- Qdrant ACL pre-filtering plus defense-in-depth payload validation.
- Deterministic PII classification and post-rerank redaction.
- Request-level safe telemetry and request-bound MLflow trace correlation.
- Separate deterministic retrieval, generated-answer, MLflow GenAI, and RAGAS evaluation results.
- Read-only baseline/candidate drift assessment with explicit compatibility and evidence states.
- Read-only validation, cleanup planning, restoration planning, and operations views.

## Architecture

### Platform overview

```mermaid
flowchart LR
    S[Versioned sources] --> P[(PostgreSQL control plane)]
    P --> I[Immutable index build]
    I --> Q[(Qdrant data plane)]
    U[Streamlit visitor] --> R[Managed RAG runtime]
    R --> Q
    R --> B[AWS Bedrock]
    R -. safe metadata .-> M[(MLflow)]
    O[Private operations dashboard] --> P
    O --> Q
```

### Indexing lifecycle

```mermaid
flowchart LR
    A[Observe source] --> V[Immutable source version]
    V --> C[Chunk + fingerprints]
    C --> G[Build generation]
    G --> D{Reuse decision}
    D -->|unchanged text/model| E[Reuse dense vectors]
    D -->|same corpus statistics| B[Reuse BM25 state]
    D -->|corpus changed| F[Refit BM25]
    E --> Q[New Qdrant collection]
    B --> Q
    F --> Q
    Q --> T[Transactional ACTIVE switch]
```

### Public request and governance

```mermaid
flowchart LR
    U[Anonymous visitor] --> L[Length + process-wide usage gate]
    L --> X[Redact visitor PII]
    X --> A[Empty AccessContext]
    A --> Q[Qdrant PUBLIC-only filter]
    Q --> V[Defense-in-depth ACL validation]
    V --> R[Rerank]
    R --> P[REDACT context]
    P --> B[Bedrock answer]
    B --> Z[Redact public output]
    Z --> C[Answer + clickable citations]
```

### Observability and evaluation

```mermaid
flowchart TB
    R[One public request] --> T[Request-bound metadata-only trace]
    T --> O[Stage latency, counts, ACL/PII status]
    E[Reviewed PUBLIC evaluation cases] --> D[Deterministic retrieval metrics]
    E --> G[Explicit Bedrock generation]
    G -. explicit opt-in .-> M[MLflow GenAI judges]
    G -. explicit opt-in .-> A[RAGAS metrics]
    D --> J[Versioned safe report]
    M --> J
    A --> J
```

Phase 4 compares an operator-approved baseline generation with a candidate (ACTIVE by default).
It combines immutable lineage, existing generation validation, existing retrieval-quality reports,
optional managed answer evaluation, and optional safe aggregate telemetry. Findings never publish,
restore, delete, rebuild, or refit an index.

### Local public/private deployment

```mermaid
flowchart LR
    Internet --> CF[Cloudflare Quick Tunnel]
    CF -->|only localhost:8501| UI[Public Streamlit]
    UI --> Q[(Qdrant 127.0.0.1:6333)]
    UI --> P[(Lineage PostgreSQL 127.0.0.1:5433)]
    UI --> M[(MLflow 127.0.0.1:5000)]
    UI --> B[AWS Bedrock API]
    OP[Local operator] --> OD[Private operations dashboard]
    OD --> Q
    OD --> P
```

## Quick start

Prerequisites: Python 3.12, `uv`, Docker Desktop, an approved Bedrock model, and standard AWS
credentials. Copy `.env.example` to `.env`, replace both local database passwords, and configure
the Bedrock model.

```powershell
uv sync
docker compose up -d
uv run alembic upgrade head
uv run python scripts/check_environment.py --managed-smoke
```

The application never migrates, indexes, publishes, restores, cleans up, evaluates, or invokes a
judge during normal startup. On a fresh database, ingest and publish explicitly:

```powershell
uv run python scripts/ingest_incremental.py --mode fixture --authoritative-snapshot --apply
uv run python scripts/index_managed.py --publish
```

Review the dry-run output before either mutation. For an existing environment, start services and
run the preflight; do not republish automatically.

### Start the public showcase locally

The public app fails closed unless all of these settings are explicit:

```dotenv
RAG__RUNTIME__PROFILE=managed
RAG__PUBLIC_SHOWCASE__ENABLED=true
RAG__PUBLIC_SHOWCASE__INFERENCE_ENABLED=true
RAG__PUBLIC_SHOWCASE__APPROVED_CORPUS_KEY=consumer-finance-managed
BEDROCK_GENERATION_MODEL=your-enabled-bedrock-model-id
```

```powershell
uv run streamlit run app/streamlit_app.py --server.address 127.0.0.1 --server.port 8501 `
  --client.showErrorDetails none
```

To expose only that port through a temporary tunnel:

```powershell
cloudflared tunnel --url http://127.0.0.1:8501
```

See [Local public demo](docs/local-public-demo.md) for health checks, cost controls, cross-network
testing, shutdown, and private operator access.

### Private operator tools

```powershell
uv run streamlit run app/private_workbench.py --server.address 127.0.0.1 --server.port 8502
uv run streamlit run app/operations_dashboard.py --server.address 127.0.0.1 --server.port 8503
uv run python scripts/manage_index_generations.py --status
uv run python scripts/manage_index_generations.py --validate
uv run python scripts/manage_index_generations.py --cleanup-plan
uv run python scripts/manage_index_generations.py --restore-generation 3
```

The private workbench, operations dashboard, and MLflow UI must remain local. Never route ports
`8502`, `8503`, `5000`, `6333`, `6334`, or `5433` through the public tunnel. The workbench requires
the managed profile but does not require `RAG__PUBLIC_SHOWCASE__ENABLED=true`.

## Evaluation

Deterministic managed retrieval evaluation makes no Bedrock or judge calls:

```powershell
uv run python scripts/evaluate_managed_retrieval.py --current
uv run python scripts/evaluate_managed_rag.py --current
```

Generated answers and judges are separately opt-in and may incur AWS charges:

```powershell
uv run python scripts/evaluate_managed_rag.py --current --generate --limit 2
uv run python scripts/evaluate_managed_rag.py --current --generate --limit 2 --with-mlflow-judges
uv run python scripts/evaluate_managed_rag.py --current --generate --limit 2 --with-ragas
uv run python scripts/evaluate_managed_rag.py --current --generate --limit 2 `
  --with-mlflow-judges --with-ragas --log-mlflow
```

The managed evaluator pins ACTIVE once, accepts only reviewed synthetic PUBLIC cases, uses the
production ACL/PII pipeline, and writes a versioned metadata-only report. The public Evaluation tab
reads existing reports only; it never starts generation or judges. Phase 2 reports remain visible
separately and preserve the fixture/evaluation compatibility paths.

## Drift and regression assessment

The basic assessment is read-only and does not initialize embedding, reranker, Bedrock, or judge
models:

```powershell
uv run python scripts/assess_managed_drift.py --baseline-generation 3 `
  --skip-retrieval-evaluation
```

Omit `--skip-retrieval-evaluation` to run the existing deterministic managed retrieval evaluator
against the same reviewed cases for both physically available, compatible generations. Answer
generation and each judge backend remain separate, explicit, potentially billable choices:

```powershell
uv run python scripts/assess_managed_drift.py --baseline-generation 3
uv run python scripts/assess_managed_drift.py --baseline-generation 3 --generate-answers
uv run python scripts/assess_managed_drift.py --baseline-generation 3 --generate-answers `
  --with-mlflow-judges --with-ragas --log-mlflow
```

The versioned `managed-drift-assessment-v1` JSON is metadata-only. The private operations
dashboard can display it under **Drift & regression**; the public app has no assessment controls or
private findings. Current default thresholds allow no Recall@K, MRR, or answer-metric drop, require
20 samples per equivalent operational window, warn above a 0.05 failure-rate increase, and warn
above a 25% mean-latency increase. Configure them under `drift_monitoring`; thresholds classify
evidence and never trigger lifecycle actions.

Comparisons are rejected when the generation schemas, embedding/chunking identities, BM25
configuration, PII detector, payload schema, point-ID format, reviewed dataset, or evaluation-model
identity are not equivalent. Missing collections, skipped judges, unavailable MLflow, and too-small
telemetry windows remain explicit `UNAVAILABLE`, `NOT_REQUESTED`, `FAILED`, `INCOMPATIBLE`, or
`INSUFFICIENT_DATA` states rather than synthetic scores.

## Runtime profiles

| Profile | Purpose | PostgreSQL lineage | Generation identity |
|---|---|---:|---:|
| `fixture` | Small Phase 1 local corpus | No | None |
| `evaluation` | Isolated Phase 2 corpus | No | None |
| `managed` | Phase 3 immutable generations | Required | Pinned once per runtime |

Only `managed` is accepted by the public showcase. Anonymous callers cannot submit a principal,
groups, or an `ALLOW` PII policy.

The [private workbench runbook](docs/private-workbench.md) covers request inspection, score
provenance, approved private evaluation, cost separation, and trace-sync troubleshooting. The
[repository cleanup audit](docs/repository-cleanup-audit.md) records why compatibility paths,
migrations, datasets, and historical documents remain.

## Documentation

- [Project learning guide](docs/project-learning-guide.md) — implementation walkthrough and
  interview explanations.
- [Local public demo](docs/local-public-demo.md) — Windows + Cloudflare Quick Tunnel runbook.
- [End-to-end demo](docs/end-to-end-demo.md) — ingestion, publication, retrieval, evaluation,
  validation, cleanup planning, and restoration planning.
- [System design walkthrough](docs/system-design-walkthrough.md) — engineering trade-offs and five
  change scenarios.
- [Detailed architecture](docs/architecture.md) — schemas, invariants, telemetry, and operations.
- [Private workbench](docs/private-workbench.md) — page guide, exact trace inspection, evaluation
  switches and safeguards, cost handling, and a verified request walkthrough.
- [Repository cleanup audit](docs/repository-cleanup-audit.md) — tracked-file/dependency findings
  and safe cleanup classifications.
- [Phase 2 interview notes](docs/phase2-interview-notes.md) and
  [validation record](docs/phase2-validation.md).

No screenshot is committed because the repository contains no verified, portable live-service
capture. The UI renders genuine runtime and report data and intentionally shows unavailable states
instead of fabricated scores or screenshots.

## Validation

```powershell
uv sync
uv run ruff check .
uv run mypy .
uv run pytest
docker compose config --quiet
git diff --check
uv run alembic heads
```

Offline tests use fakes and do not call Bedrock, MLflow judges, or RAGAS judges. A passing offline
suite does not prove live PostgreSQL, Qdrant, MLflow, model-download, AWS credential, Bedrock, or
Cloudflare connectivity.

Never use `docker compose down -v` when preserving PostgreSQL, Qdrant, or MLflow data. A normal
`docker compose down` keeps the named volumes.
