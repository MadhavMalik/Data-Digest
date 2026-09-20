# Signal Engine

**An automated researcher for tabular data.** An LLM proposes hypotheses; a deterministic
statistical engine decides what is actually true; Elasticsearch remembers every finding —
including the ones nothing could explain yet — so later evidence can resolve them.

Built for the **Voloridge HackMIT 2026 "Signal in the Noise"** challenge, on the NYC TLC
Trip Record Data listed in that challenge.

---

## The idea in one loop

```
HYPOTHESIZE → COMPUTE → OBSERVE → REFINE → TRANSFORM → COMPUTE
    → VISUALIZE → INTERPRET → REMEMBER → RETRIEVE → REINTERPRET
```

The model never sees the dataset. It sees ~4 KB of column cards and compact statistical
feedback, and it decides *what to test next*. Deterministic code decides *what is true*.

The last step is the one that makes this more than a search loop: a robust relationship the
system cannot explain is **stored as an open question**, not discarded. When related
evidence arrives later, retrieval surfaces the old finding and both are reconsidered
together — which is how a researcher actually works.

---

## Architecture

```
                        RAW PARQUET  (3.7M rows, 61 MiB, never fully loaded)
                              │
                              │  Polars lazy scan · projection + predicate pushdown
                              ▼
   ┌──────────────────────────────────────────────────────────────┐
   │ STAGE A  CHEAP PROFILER                                      │
   │   one pass, 276 aggregations, 0.8 s                          │
   │   → dataset card + one column card each                      │
   │   → semantic types & UNITS from the official data dictionary │
   └──────────────────────────────────────────────────────────────┘
                              │           ~4 KB of cards
                              ▼
   ┌──────────────────────────────────────────────────────────────┐
   │ STAGE B  HYPOTHESIS GENERATION        Llama 4 Scout          │
   │   sees: question + cards + last round's numbers              │
   │         + retrieved prior evidence + remaining budget        │
   │   emits: structured Hypothesis objects (Pydantic-validated)  │
   │   ── falls back to a deterministic planner with no creds ──  │
   └──────────────────────────────────────────────────────────────┘
                              │
                              ▼
   ┌──────────────────────────────────────────────────────────────┐
   │ STAGE C/D/E  DIMENSIONAL PRUNING → TRANSFORM SEARCH          │
   │   dollars + miles        → REJECTED, before reading a row    │
   │   PULocationID * fare    → REJECTED (label has no magnitude) │
   │   fare / distance        → KEPT (USD per mile)               │
   │   ~72% of candidates deleted at zero I/O cost                │
   └──────────────────────────────────────────────────────────────┘
                              │
                              ▼
   ┌──────────────────────────────────────────────────────────────┐
   │ PROGRESSIVE SCREENING LADDER (each rung cheaper than the one │
   │ above it, so the expensive rungs see only a handful)         │
   │   batched materialization → linear/monotonic → mutual info   │
   │   → cross-fold stability → BH-FDR                            │
   └──────────────────────────────────────────────────────────────┘
                              │   survivors only
                              ▼
   ┌──────────────────────────────────────────────────────────────┐
   │ VISUALIZE  chart form follows semantic type AND scale        │
   │ INTERPRET  Llama 4 Scout (multimodal) reads the graph        │
   │ CRITIC     deterministic validation of the CLAIM vs NUMBERS  │
   └──────────────────────────────────────────────────────────────┘
                              │
                              ▼
   ┌──────────────────────────────────────────────────────────────┐
   │ EVIDENCE MEMORY            Elasticsearch                      │
   │   BM25 + dense-vector kNN, fused with RRF                    │
   │   unresolved findings persist as open questions              │
   │   new evidence retrieves them → joint reinterpretation       │
   └──────────────────────────────────────────────────────────────┘
```

Full detail: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) ·
[`docs/OPTIMIZATIONS.md`](docs/OPTIMIZATIONS.md) · [`docs/decisions/`](docs/decisions/)

---

## Why the LLM never sees the whole dataset

Three independent reasons, in order of importance:

1. **It would be wrong.** A language model asked to eyeball 3.7 million rows does not
   compute a correlation — it pattern-matches a plausible-sounding answer. Correlations are
   arithmetic. Arithmetic belongs in NumPy.
2. **It would not fit.** 3.7M rows × 20 columns is roughly 400 million tokens.
3. **It would cost a fortune** for an answer that is worse than `np.corrcoef`.

So the model gets a **dataset card** and **column cards**: name, semantic type, unit,
distribution summary, official description, and known caveats — about **4 KB** standing in
for **61 MiB** on disk. It proposes; the engine verifies.

## Role of Elasticsearch

Elasticsearch is the **persistent semantic evidence memory**. It is deliberately *not* the
statistics engine — a correlation over a local Parquet file is faster in Polars and does not
make correctness depend on a network service.

What it does:

- stores every analyzed relationship as an `EvidenceObject` (statistics, plot URI,
  interpretation, critic report, status, provenance)
- **hybrid retrieval**: BM25 over the finding's text + dense-vector kNN over its embedding,
  merged with **Reciprocal Rank Fusion** (server-side `rrf` retriever where available, with
  an identical client-side fallback)
- structured filters by dataset, status, tags, effect size
- `int8_hnsw` vector quantization: ~4× less memory than float32 with rescoring for recall

## Role of Llama 4

`meta-llama/Llama-4-Scout-17B-16E-Instruct`, natively multimodal, serves **both** stages:

- **text** — hypothesis generation and research planning
- **vision** — reading the generated graph alongside the exact statistics

One model family, one deployment. The provider is abstracted behind an OpenAI-compatible
interface, so switching hosts is an env-var change.

## Evidence memory — the distinguishing idea

```
Evidence 1   airport pickup → unusually high fare per mile     UNRESOLVED
             (statistically robust; no mechanism available)

… later in the search …

Evidence 2   airport pickup → fixed airport surcharge
Evidence 3   airport routes → toll charges

→ retrieval surfaces Evidence 1 alongside 2 and 3
→ "The apparent airport premium may be partly explained by fixed fees and
   tolls rather than by distance itself."
→ Evidence 1 is promoted to EXPLAINED, with resolved_by pointing at 2 and 3
```

That is a *hypothesis about mechanism*, not proof of causation — a distinction the critic
enforces rather than leaving to a prompt.

---

## Why this is not merely a GPT wrapper

| | A GPT wrapper | Signal Engine |
|---|---|---|
| Who computes the statistics | the model, badly | NumPy/SciPy/Polars, exactly |
| What the model sees | raw rows | column cards + compact numeric feedback |
| Candidate space | whatever the model thinks of | generated, then **pruned on unit algebra** |
| Search control | one prompt | budgeted beam search with marginal-improvement stopping |
| Wrong claims | ship | caught by a deterministic critic |
| Memory | the context window | Elasticsearch, across runs and datasets |
| Unexplained findings | invented explanation | stored as open questions, revisited later |
| Code execution | `eval` on model output | whitelisted AST parser; `eval` appears nowhere |

The model is a **search policy over a combinatorial hypothesis space**, steered by
compressed statistical feedback. Remove it and the engine still runs — the deterministic
planner produces real findings, which is also the baseline the LLM path is measured against.

---

## Results on real data

NYC TLC Yellow Taxi, January 2026 — 3,724,889 rows, 20 columns, 61 MiB.
Every figure below is read from instrumentation, not estimated.

| | |
|---|---|
| Profiling (276 aggregations, one pass) | **0.79 s** |
| Rows after documented validity filters | 3,509,466 (5.78% excluded, **every exclusion attributed**) |
| Candidates pruned by dimensional analysis | **~72%**, before reading a single row |
| Statistical tests run | 103 |
| Full pipeline wall-clock | **~30 s** |
| Covariance shortcut vs direct computation | agrees to **2.8 × 10⁻¹⁷** |

Findings the engine recovers, correctly labelled:

- `trip_distance ~ total_amount` — r = +0.867, stability 1.00 · **empirical**
- `trip_duration_minutes ~ total_amount` — r = +0.699, ρ = +0.889 · **empirical**
- `fare_amount ~ total_amount` — r = +0.966 · **mechanical** (accounting identity, not a discovery)
- `RatecodeID = 2 (JFK)` — a flat $70 fare, visible as a zero-variance box in the plot

---

## Installation

```bash
git clone <repo> && cd HackMIT_new
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[elastic,dev]"
cp .env.example .env          # then fill in credentials
```

Python ≥ 3.10. Core deps: Polars, PyArrow, NumPy, SciPy, Matplotlib, Pydantic, FastAPI, httpx.

## Environment variables

| Variable | Required? | Purpose |
|---|---|---|
| `LLM_BASE_URL`, `LLM_API_KEY` | optional | OpenAI-compatible endpoint hosting Llama 4 Scout. Verified working shapes: Together, Groq, Fireworks, DeepInfra, local vLLM, and Meta's own `https://api.meta.ai/v1` |
| `LLM_MODEL` | — | defaults to `meta-llama/Llama-4-Scout-17B-16E-Instruct` |
| `VLM_MODEL` | — | defaults to `LLM_MODEL` (Scout handles both) |
| `ELASTIC_URL` **or** `ELASTIC_CLOUD_ID` | optional | evidence memory endpoint |
| `ELASTIC_API_KEY` | optional | **API-key auth only** — never the `elastic` superuser |
| `ELASTIC_INDEX_PREFIX` | — | defaults to `hackmit_signal` |
| `BRAVE_SEARCH_API_KEY` | optional | external grounding for unexplained findings |
| `CONTEXT_COMPRESSOR` | — | `noop` (default) or `token_company` |

**Everything is optional.** With nothing configured the engine still runs the full loop and
declares what it degraded. See `.env.example` for the complete list including budgets.

## Downloading the dataset

```bash
python scripts/fetch_tlc.py --vehicle yellow --year 2026 --month 1 --output data/raw
```

Resolves the official TLC monthly Parquet URL, streams it to disk, writes atomically
(`.part` → rename), retries with backoff, validates the Parquet footer, skips if already
present (`--force` overrides), and also fetches the taxi-zone lookup and writes the official
data dictionary to `data/metadata/`. Works for any month and vehicle:

```bash
python scripts/fetch_tlc.py --vehicle green --year 2025 --month 6
```

## Running the demo

```bash
python scripts/run_demo.py                          # default question, 3 rounds
python scripts/run_demo.py --rounds 2 --max-visualizations 6
python scripts/run_demo.py --question "What drives tipping behaviour?"
```

Prints the findings, the evidence, the full metrics panel, and any degradations. Writes
`artifacts/<analysis_id>/run_report.md` plus every generated plot.

## API and UI

```bash
uvicorn signal_engine.api.app:app --reload
# → http://127.0.0.1:8000        demo UI
# → http://127.0.0.1:8000/docs   OpenAPI
```

| Endpoint | Purpose |
|---|---|
| `GET  /health` | service status + **redacted** config (availability booleans only) |
| `POST /datasets/profile` | profile a dataset, return all column cards |
| `POST /analyses` | start an analysis (background job) → `analysis_id` |
| `GET  /analyses/{id}` | status, live progress, findings, metrics |
| `GET  /analyses/{id}/evidence` | every evidence object from the run |
| `GET  /evidence?q=…` | hybrid search across evidence memory |
| `GET  /evidence/{id}` | one evidence object |
| `GET  /metrics` | cross-run telemetry |
| `GET  /artifacts/{id}/{file}` | generated plots (path-traversal guarded) |

The UI shows the question, profiler summary, hypotheses, discovered relationships, selected
plots, interpretations, critic findings, retrieved prior evidence, branch stop reasons, and
the full token/cost/cache telemetry panel.

## Running the tests

```bash
pytest tests/unit                       # 270 tests, no network, no credentials
pytest tests/integration                # real TLC data + live Elastic (auto-skips)
pytest tests/e2e                        # full pipeline
pytest                                  # everything
```

Unit tests are deterministic and hermetic. Integration tests skip cleanly when the data is
not downloaded or the service is not configured.

## Running with and without each service

| Missing | What happens |
|---|---|
| **No LLM** | Deterministic planner generates hypotheses from semantics; the interpreter reports the measured association and marks the conclusion `UNRESOLVED` rather than inventing a mechanism. Full statistics, full plots. |
| **LLM misconfigured** | Configuration failures are classified rather than reported as a generic error: HTTP 401/402/403/404 each produce an actionable degradation line naming what to fix (rejected key, billing not configured, unauthorized model, wrong URL). |
| **No Elasticsearch** | `MemoryEvidenceStore` takes over, with the *same* BM25 + vector + RRF retrieval, persisted to JSONL. |
| **No Brave** | External grounding is skipped. It is gated anyway — only strong-but-unexplained findings qualify. |
| **Elastic fails mid-run** | Writes queue for retry, the analysis continues, the degradation is reported. |

Degradations are always listed in the result, never silent.

## Security notes

- **No secrets in the repo.** `.env` is gitignored; `.env.example` holds placeholders only.
- **Credentials never leave the process.** `Settings.redacted_dump()` is the only path config
  takes to a log or an HTTP response, and it emits availability booleans, never key material.
  Elasticsearch transport errors pass through a scrubber that redacts anything key-shaped.
- **TLS verification stays on.** An intercepted API key is worse than a setup error.
- **API-key auth**, least privilege over this project's indices. Never the `elastic` superuser.
- **Model output is untrusted.** No `eval`, no `exec`, no shelling out, no generated SQL.
  Expressions go through a whitelisted recursive-descent parser (`features/parser.py`) that
  only ever produces typed AST nodes; hallucinated columns fail there, not at the data layer.
- **Index deletion is guarded**: `delete_index()` refuses any index whose name lacks `test`.
- **Path traversal guarded** on the artifact endpoint.

---

## Repository layout

```
src/signal_engine/
  config.py              settings + redaction
  jsonutil.py            NumPy-safe JSON sanitization
  reporting.py           run report + evaluation report
  ingestion/             dataset resolution, atomic download, TLC data dictionary
  profiling/             units algebra, semantic typing, profiler, column cards
  features/              expression AST, whitelisted parser, canonical hashing,
                         transformation search, DAG, two-level cache, derived columns
  statistics/            correlation, covariance shortcut, MI, FDR, stability
  search/                budgets, scorer, state, screening ladder, planner, orchestrator
  visualization/         graph selector, matplotlib renderer
  interpretation/        VLM interpretation, deterministic critic
  evidence/              EvidenceObject, memory store, Elasticsearch store, retrieval
  external/              Brave client, context compression
  telemetry/             metrics registry
  api/                   FastAPI app + routes
scripts/                 fetch_tlc.py, run_demo.py
tests/                   unit · integration · e2e
ui/                      single-file demo UI
docs/                    ARCHITECTURE.md · OPTIMIZATIONS.md · decisions/
```

## Data sources

- [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) —
  official monthly Parquet releases
- [AWS Open Data registry entry](https://registry.opendata.aws/nyc-tlc-trip-records-pds/) —
  the Voloridge-listed dataset
- The official Yellow Taxi data dictionary is encoded machine-readably in
  `ingestion/tlc.py` and written to `data/metadata/`.
