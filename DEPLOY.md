# Deploying Signal Engine

## The short version

```bash
./run.sh --host 0.0.0.0
```

That builds the UI if needed, installs what is missing, fetches the dataset if
it is absent, and serves the site and the API together on port 8000.

---

## Read this before choosing Vercel

**Vercel can host the front end. It cannot host the API.**

This is not a configuration problem, it is a shape mismatch:

| What the engine does | What serverless allows |
|---|---|
| An analysis runs **60–180 seconds** | Function timeout is 10s (Hobby) / 60s (Pro) |
| Streams SSE for the whole run | Streaming through the serverless proxy is buffered and capped |
| Holds run state in memory across requests | Each invocation is a cold, separate process |
| Reads a 61 MB Parquet file | Bundle and `/tmp` limits make this impractical |
| Renders matplotlib PNGs to disk | No durable filesystem |

So there are two supported topologies.

### Topology A — everything on one box (recommended)

One process, one port, no CORS. This is what `./run.sh` gives you.

```bash
# on the EC2 instance
git clone <repo> && cd HackMIT_new
cp .env.example .env && vim .env          # add credentials
./run.sh --host 0.0.0.0 --port 8000
```

Put it behind nginx or a load balancer for TLS. Open port 8000 (or whatever you
front it with) in the security group.

### Topology B — front end on Vercel, API on a long-lived host

Vercel serves the UI from its CDN; the Python API runs somewhere that allows a
three-minute streaming request. Six steps.

**1. Deploy the API** as in Topology A, and note its public URL. It must be
reachable over **HTTPS** — a Vercel page is HTTPS, and a browser blocks mixed
content, so an `http://` API fails silently mid-stream.

```bash
# on the box (EC2, Fly, Render, Railway — anything long-lived)
git clone <repo> && cd HackMIT_new
cp .env.example .env && vim .env
export SIGNAL_CORS_ORIGINS="https://<your-app>.vercel.app"
./run.sh --host 0.0.0.0 --port 8000 --no-ui
```

`--no-ui` skips the bundle: Vercel is serving it. `SIGNAL_CORS_ORIGINS` is a
comma-separated allowlist and must include the exact Vercel origin, scheme and
all, with no trailing slash.

**2. Point Vercel at the `web/` directory.** From the repo root:

```bash
npm i -g vercel
vercel link            # answer: Root Directory -> web
```

The root directory matters. `web/vercel.json` and `web/package.json` are what
Vercel builds, and `vite.config.ts` switches its output to `web/dist` when it
sees Vercel's `VERCEL=1`, because a build cannot write outside its own root.

**3. Set the API origin** — build-time, so it must exist before the build:

```bash
vercel env add VITE_API_BASE production
# paste: https://api.your-domain.com     (no trailing slash)
```

**4. Ship it.**

```bash
vercel --prod
```

**5. Add the real origin to CORS.** Vercel prints the final URL. If it differs
from the one guessed in step 1, update `SIGNAL_CORS_ORIGINS` on the API and
restart it.

**6. Check the stream**, which is the part that actually breaks:

```bash
curl -sI https://<your-app>.vercel.app | head -1          # 200
curl -s  https://api.your-domain.com/health | head -c 80  # {"status":"ok"...
```

Then open the site, run an analysis, and confirm events arrive. If the page
loads but nothing streams, it is CORS or mixed content — both show up in the
browser console.

> **What Vercel cannot do here:** host the API. An analysis runs 60–180 seconds
> and streams SSE throughout (10s timeout on Hobby, 60s on Pro), holds run state
> in memory between requests, reads a 61 MB Parquet file, and writes matplotlib
> PNGs to disk. None of that survives a serverless function. Deploying the repo
> root to Vercel gives a UI with no backend.

---

## Docker

```bash
docker build -t signal-engine .
docker run -p 8000:8000 --env-file .env \
  -v "$PWD/data:/app/data" -v "$PWD/artifacts:/app/artifacts" \
  signal-engine
```

Two stages: node builds the bundle, and only the Python runtime ships. The
volumes keep uploaded datasets and generated plots across restarts.

### On EC2 with Docker

```bash
sudo yum install -y docker git && sudo systemctl start docker
git clone <repo> && cd HackMIT_new
docker build -t signal-engine .
docker run -d --restart unless-stopped -p 80:8000 \
  --env-file .env -v /opt/signal-data:/app/data signal-engine
```

A `t3.large` (2 vCPU, 8 GB) handles the 3.7M-row TLC file comfortably. The
profiler is single-pass and the screening stage is vectorised, so RAM matters
more than cores — 8 GB is the number to hit.

---

## Environment variables

| Variable | Purpose |
|---|---|
| `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` | OpenAI-compatible endpoint. Without it the engine runs its deterministic path. |
| `ELASTIC_URL` or `ELASTIC_CLOUD_ID`, `ELASTIC_API_KEY` | Evidence memory. Falls back to a local JSONL store. |
| `SIGNAL_HOST`, `SIGNAL_PORT` | Bind address (the Docker image defaults to `0.0.0.0:8000`). |
| `SIGNAL_CORS_ORIGINS` | Comma-separated origins allowed to call the API cross-origin. |
| `VITE_API_BASE` | **Build-time**, front end only. Where the API lives. |

Everything is optional. With nothing set, the engine still runs end to end and
reports exactly what it degraded.

---

## Local development

```bash
# terminal 1 — API with reload
./run.sh --no-ui

# terminal 2 — Vite dev server with HMR, proxying to the API
cd web && npm run dev      # http://localhost:5173
```

The Vite proxy forwards `/analyses`, `/datasets`, `/artifacts` and the rest to
`127.0.0.1:8000`, so there is no CORS setup in development.

---

## Checks

```bash
curl localhost:8000/health        # config (redacted), missing credentials
curl localhost:8000/datasets      # what is available to analyse
pytest -q                         # 400+ tests
cd web && npm run typecheck       # strict TypeScript
```

`/health` returns availability booleans only. No key material reaches a log,
an error message, or an HTTP response.
