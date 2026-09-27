# Repository cleanup audit

Audit date: 2026-09-25. Scope: all 162 files tracked at the start of the audit, the current
working tree, declared and locked dependencies, imports, entrypoints, Docker/CI configuration,
scripts, tests, migrations, datasets, reports, and documentation links. This audit did not run a
migration, publish or delete an index, delete local data, or remove any tracked file.

## Outcome

No file was demonstrably safe to delete. Several paths overlap in purpose, but each still has an
active caller, test, documented compatibility role, or useful historical explanation. The safest
cleanup is therefore documentation consolidation over time, not removal. Generated caches,
downloaded corpora, local index state, and evaluation reports are already ignored.

The classifications below use:

- **KEEP**: active runtime, compatibility, test, data-contract, migration, or operational value.
- **CONSOLIDATE**: active today, but responsibilities or documentation overlap and may later move.
- **ARCHIVE**: useful history that may move only after links are updated and a deliberate archive
  location is chosen.
- **DELETE**: proven to have no consumer. No candidate met this standard.

## Entry points and active implementation

| Paths | Purpose and consumers | Evidence | Action, risk, and affected tests |
|---|---|---|---|
| `app/streamlit_app.py` | PUBLIC-only showcase | README/runbook entrypoint; public boundary and source-filter tests | **KEEP**. Removing breaks the product surface and public/privacy tests. |
| `app/private_workbench.py` | Six-page private engineering workbench | Private runbook and navigation/inspection/evaluation tests | **KEEP**. Removing loses the current operator interface. |
| `app/operations_dashboard.py`, `src/consumer_finance_rag/operations_dashboard.py` | Earlier private lifecycle/status dashboard and its view models | README/local demo still documents port 8503; `tests/test_operations_dashboard.py` exercises it | **CONSOLIDATE**, not delete. Some read-only views now overlap the workbench, but lifecycle detail and compatibility remain. Update callers/tests before any retirement. |
| `scripts/*.py` | Explicit ingestion, index lifecycle, diagnostics, retrieval/RAG evaluation, drift, and model warm-up commands | Every script is referenced by documentation, a CLI test, or both | **KEEP**. The Phase 1/2 scripts are compatibility profiles, not dead code. |
| `src/consumer_finance_rag/**` | Runtime, governance, evaluation, lineage, observability, and UI services | Import scan and strict mypy include the whole package; tests directly cover every module family | **KEEP**. No orphaned implementation module was found. |

## Compatibility and apparent duplication

| Exact paths | Original purpose and current consumer | Unused evidence | Recommendation |
|---|---|---|---|
| `scripts/evaluate_retrieval.py`, `scripts/evaluate_rag.py`; `evaluation/dataset.py`, `metrics.py`, `retrieval_runner.py`, `reporting.py` | Phase 2 fixture/evaluation-profile evaluation, still used by evaluation tests, public read-only legacy report display, and docs | None; imports and tests remain | **KEEP**, later **CONSOLIDATE** adapters behind the managed report contract only if compatibility tests remain. Risk: breaking Phase 2 profiles and saved-report readers. |
| `scripts/evaluate_managed_retrieval.py`, `scripts/evaluate_managed_rag.py`; `evaluation/managed_rag.py` | Generation-pinned reviewed-PUBLIC managed evaluation | Active CLI/tests/private workbench | **KEEP**. This is the canonical managed path reused by the workbench. |
| `scripts/ingest.py` and `scripts/ingest_incremental.py` | Fixture/Phase 1 ingestion versus managed lineage ingestion | Both are documented and tested through their service paths | **KEEP**. Similar names do not imply identical lifecycle guarantees. |
| `docs/architecture.md`, `system-design-walkthrough.md`, `project-learning-guide.md`, architecture sections in `README.md` | Formal architecture, scenario-based design, learning narrative, and landing-page summary | All are linked from README and cover different reader needs | **CONSOLIDATE** repeated diagrams gradually; do not delete knowledge. Risk is broken links and lost rationale. |
| `docs/end-to-end-demo.md`, `docs/local-public-demo.md`, `docs/private-workbench.md` | Full lifecycle, public tunnel runbook, and private operator runbook | Active commands differ by audience | **KEEP**; cross-link shared prerequisites rather than merge into one oversized guide. |
| `artifacts/retrieval_quality_current.json` | Versioned example/current deterministic retrieval-quality artifact | Referenced by `docs/end-to-end-demo.md` | **KEEP** until the docs and any baseline workflow no longer consume it. Do not confuse it with ignored local reports. |

## Data, schema, and historical records

| Exact paths | Purpose and references | Action, risk, and affected tests |
|---|---|---|
| `migrations/versions/0001_application_lineage.py` through `0005_restore_provenance.py`, migration support files, and `.gitkeep` | Ordered application schema history used by Alembic and fresh/upgrade deployments | **KEEP** without exception. Deletion would make existing databases and clean installs irreconcilable; lineage/database tests and deployment fail. |
| `data/evaluation/managed_public_cases.json` | Reviewed `PUBLIC_SYNTHETIC_REVIEWED` contract used by managed evaluation and the private workbench | **KEEP**. It is the authorization boundary for judge inputs; managed evaluation tests depend on it. |
| `data/evaluation/seed_cases.json` | Phase 2 reviewed evaluation dataset | **KEEP** for the evaluation profile and dataset validation tests. |
| `data/fixtures/cfpb_consumer_finance.json`, `data/fixtures/README.md` | Small deterministic offline corpus and its provenance | **KEEP**. Offline tests, fixture runtime, and quick-start ingestion depend on it. |
| `data/corpus_manifest.json` | Corpus acquisition identity/configuration | **KEEP**. Acquisition and corpus-expansion code/tests consume it. |
| `data/fixtures/live_request_observation.json` | Content-free verified observation of request/trace latency and token metadata | **KEEP**. Private runbook and workbench tests use it; it contains no answer or raw spans. |
| `docs/phase2-interview-notes.md`, `docs/phase2-validation.md` | Historical design/validation record | **ARCHIVE candidate**, not removed. Both remain linked by README and explain compatibility behavior. Moving them requires link updates and preservation of history. |

## Generated and local-only files

The following are correctly ignored and must not be deleted by repository cleanup automation:

- `.env`, `.venv/`, `.uv-cache/`, `.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/`, `.test-tmp/`;
- `data/index/` generated lexical/index state;
- `data/evaluation_corpus/` downloaded/generated evaluation corpus;
- `reports/evaluation/` local safe reports and `.ragas/` state;
- Python bytecode, build output, coverage output, and Streamlit secrets.

`uv.lock` is intentionally tracked for reproducible Python 3.12 environments. Database volumes,
Qdrant collections, MLflow artifacts, model caches, and user reports are external/local state and
were not modified.

## Dependency audit

No declared dependency is safe to remove from `pyproject.toml` or `uv.lock`:

| Dependency group | Active use |
|---|---|
| `alembic`, `sqlalchemy`, `psycopg` | Migration and PostgreSQL lineage/control plane |
| `boto3`, `botocore[crt]` | Bedrock Converse and AWS client configuration |
| `qdrant-client` | Dense/sparse serving and managed collections |
| `sentence-transformers`, `torchvision` | Hugging Face embedding/reranker runtime and compatible model stack |
| `mlflow[genai]`, `ragas`, `litellm` | Existing MLflow GenAI and RAGAS judge adapters; LiteLLM supplies Bedrock judge access |
| `pydantic`, `pydantic-settings`, `pyyaml` | Typed contracts, environment configuration, and versioned YAML pricing/config |
| `pandas` | Evaluation/MLflow tabular adapter inputs |
| `streamlit` | Public and private UIs |
| development dependencies | Ruff, strict mypy, pytest, Bedrock stubs, and YAML typing in CI/local validation |

The configuration scan likewise found no safe field removal. Fixture/evaluation/managed profiles,
Phase 2 report paths, managed dataset paths, public gates, private workbench controls, and drift
thresholds all have active code, documentation, or tests.

## Safe-removal record and follow-up gates

Exact removals performed: **none**.

Before changing any **CONSOLIDATE** or **ARCHIVE** candidate, require all of the following:

1. replace every import, CLI/documentation link, saved-report reader, and test consumer;
2. preserve Phase 1/2 runtime and report compatibility or publish an explicit migration;
3. retain all Alembic history and reviewed evaluation/fixture contracts;
4. run Ruff, strict mypy, the full pytest suite, Compose validation, and a link/CLI smoke check;
5. never remove local databases, reports, index generations, model caches, or user data as part of
   a source cleanup.
