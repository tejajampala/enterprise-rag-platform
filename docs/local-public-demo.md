# Local public demo with a Cloudflare Quick Tunnel

This runbook exposes only the public Streamlit showcase from a Windows development machine. The
lineage PostgreSQL database, Qdrant, MLflow, and operations dashboard stay local. AWS Bedrock remains
the generation provider; no AWS hosting resources are created.

Cloudflare documents Quick Tunnels as temporary development/testing endpoints. They receive a
random `trycloudflare.com` hostname, have a documented 200 concurrent-request limit, and are not a
production deployment mechanism. See the official
[Quick Tunnels documentation](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/do-more-with-tunnels/trycloudflare/)
and [Windows downloads](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/downloads/).

## 1. Configure the fail-closed public boundary

Copy `.env.example` to `.env` and configure local database credentials, the lineage URL, AWS
region/credentials, and an enabled Bedrock generation model. The public settings must be explicit:

```dotenv
RAG__RUNTIME__PROFILE=managed
RAG__PUBLIC_SHOWCASE__ENABLED=true
RAG__PUBLIC_SHOWCASE__INFERENCE_ENABLED=true
RAG__PUBLIC_SHOWCASE__APPROVED_CORPUS_KEY=consumer-finance-managed
RAG__PUBLIC_SHOWCASE__MAX_QUESTION_CHARACTERS=500
RAG__PUBLIC_SHOWCASE__MAX_REQUESTS_PER_WINDOW=20
RAG__PUBLIC_SHOWCASE__RATE_WINDOW_SECONDS=3600
RAG__PUBLIC_SHOWCASE__MAX_CONCURRENT_REQUESTS=2
BEDROCK_GENERATION_MODEL=your-enabled-model-id
```

The first switch controls whether the public application can construct a serving runtime. The
second is the inference kill switch. Leave either `false` until the local health checks pass.

The approved corpus key must equal `RAG__MANAGED_INGESTION__CORPUS_KEY`. The server also requires a
generation-pinned managed runtime with ACL and PII enforcement. Visitor code has no API for passing
a principal, groups, or an `ALLOW` policy.

## 2. Start and verify local services

```powershell
uv sync
docker compose up -d
uv run alembic current
uv run alembic heads
uv run python scripts/check_environment.py --managed-smoke
```

`PASS_WITH_WARNINGS` is serving-ready when ACTIVE validation passes; warnings may describe legacy
RETIRED or incomplete FAILED history. An ACTIVE failure blocks readiness.

Useful local checks:

```powershell
docker compose ps
Invoke-WebRequest http://127.0.0.1:6333/healthz -UseBasicParsing
Invoke-WebRequest http://127.0.0.1:5000/health -UseBasicParsing
uv run python scripts/manage_index_generations.py --validate
```

If `localhost` resolves unexpectedly on Windows, use `127.0.0.1` consistently in PostgreSQL,
Qdrant, MLflow, Streamlit, and tunnel URLs. Do not print or paste `.env` into diagnostics.

## 3. Produce optional genuine evaluation data

The Evaluation tab only reads existing artifacts. It never starts a paid job.

```powershell
# No Bedrock or judge call.
uv run python scripts/evaluate_managed_rag.py --current

# Optional paid generation smoke; no judge.
uv run python scripts/evaluate_managed_rag.py --current --generate --limit 2
```

MLflow and RAGAS judges remain separate explicit options documented in the README. Do not run them
for a public visitor request. Generated reports are local ignored artifacts under
`reports/evaluation/`; the UI shows only values that actually exist and labels non-run stages.

## 4. Start Streamlit on loopback

```powershell
uv run streamlit run app/streamlit_app.py `
  --server.address 127.0.0.1 `
  --server.port 8501 `
  --client.showErrorDetails none `
  --browser.gatherUsageStats false
```

Open <http://127.0.0.1:8501> locally first. Verify:

- the page reports `managed` and a concrete generation;
- a guided question returns a real Bedrock answer and clickable source citations;
- the retrieval timeline contains measured, non-negative values;
- request ID and MLflow trace ID refer to that same request, or MLflow is explicitly unavailable;
- the trace summary says ACL filtering was applied, group count is zero, and PII policy is REDACT;
- the Evaluation tab does not initiate generation or judges.

If startup fails, the visitor sees a bounded message. Use the private terminal, preflight, MLflow,
and operations dashboard for detailed diagnosis.

## 5. Start the Quick Tunnel

Install the current Windows `cloudflared` MSI or executable from Cloudflare's official download
page. In a second PowerShell window:

```powershell
cloudflared version
cloudflared tunnel --url http://127.0.0.1:8501
```

`cloudflared` prints a random HTTPS `trycloudflare.com` URL. Share only that URL. Do not add Qdrant,
PostgreSQL, MLflow, or the operations dashboard as tunnel routes. Quick Tunnels can conflict with a
pre-existing `.cloudflared/config.yaml`; the official documentation recommends removing or
temporarily renaming that configuration for this mode.

## 6. Test from another network

Use a phone with Wi-Fi disabled or another independent network:

1. Open the printed HTTPS URL.
2. Confirm no MLflow or operations links are exposed.
3. Ask one guided question.
4. Confirm citations open the public source URL.
5. Submit an over-length question and verify the bounded validation message.
6. Temporarily set `RAG__PUBLIC_SHOWCASE__INFERENCE_ENABLED=false`, restart Streamlit, and confirm
   that generation is unavailable without a Bedrock call.

The Quick Tunnel URL changes when restarted. Browser success proves only tunnel-to-Streamlit
connectivity; it does not prove that every AWS region/model or future corpus is safe.

## 7. Cost and abuse limits

The app uses process-wide, lock-protected controls—not Streamlit session counters:

- bounded question length;
- a maximum accepted request count per configured time window;
- a non-blocking maximum number of concurrent Bedrock calls;
- an inference kill switch checked for every request.

These are appropriate safeguards for a one-process portfolio demo. They reset on process restart
and are not a distributed quota or identity-aware abuse-prevention service. Keep the request limit
small, use a low Bedrock `max_tokens`, watch the private AWS billing tools, and turn the inference
switch off whenever the demo is unattended.

## 8. Keep operator access private

Run private tools on separate loopback ports:

```powershell
uv run streamlit run app/private_workbench.py `
  --server.address 127.0.0.1 `
  --server.port 8502

uv run streamlit run app/operations_dashboard.py `
  --server.address 127.0.0.1 `
  --server.port 8503
```

- Private workbench: <http://127.0.0.1:8502>
- Legacy operations dashboard: <http://127.0.0.1:8503>
- MLflow: <http://127.0.0.1:5000>
- Qdrant API: <http://127.0.0.1:6333>

Do not expose these endpoints through Quick Tunnel. The public page displays an allowlisted summary,
not a link to the operator MLflow UI.

## 9. Stop safely

1. In the `cloudflared` window, press `Ctrl+C`. The random public URL stops routing.
2. Stop Streamlit with `Ctrl+C`.
3. Set `RAG__PUBLIC_SHOWCASE__INFERENCE_ENABLED=false` if the machine will remain running.
4. Optionally stop local services:

```powershell
docker compose down
```

Never use `docker compose down -v` when preserving Qdrant, lineage PostgreSQL, or MLflow data.
