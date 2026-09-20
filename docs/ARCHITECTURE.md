# Architecture

## The governing constraint

> **The LLM is the last expensive operation, not the thing doing the data processing.**

By the time a model sees anything, deterministic code has already profiled the data, pruned
the candidate space on units, screened thousands of candidates through a cost-ordered ladder,
and selected a handful worth explaining.

A second constraint follows from the first: **every external dependency is optional.** No
LLM, no Elasticsearch, no Brave — the run still completes with real statistics, real plots
and honest evidence records, and it says out loud what it degraded.

---

## Data flow

```
data/raw/yellow_tripdata_2026-01.parquet        3,724,889 rows · 61 MiB
        │
        │  pl.scan_parquet  — lazy, projection + predicate pushdown
        ▼
┌───────────────────────────────────────────────────────────────────────┐
│ ingestion/         DatasetHandle: path, fingerprint, row count,       │
│                    byte size, provenance, official data dictionary    │
└───────────────────────────────────────────────────────────────────────┘
        ▼
┌───────────────────────────────────────────────────────────────────────┐
│ profiling/         ONE lazy query, 276 aggregations, one collect      │
│   units.py           dimensional algebra (the pruning substrate)      │
│   semantic_types.py  metadata > dtype > name > distribution           │
│   profiler.py        exact where cheap, approximate where flagged     │
│   column_cards.py    ~4 KB total — the ONLY thing the model sees      │
└───────────────────────────────────────────────────────────────────────┘
        ▼
┌───────────────────────────────────────────────────────────────────────┐
│ features/derive.py  derived columns + the ANALYSIS VIEW               │
│                     every filter named, justified, and its exclusions │
│                     counted. Nothing is silently dropped.             │
│                     Provenance recorded → target-leakage detection.   │
└───────────────────────────────────────────────────────────────────────┘
        ▼
╔═══════════════════════════ ROUND LOOP ════════════════════════════════╗
║                                                                       ║
║  search/planner.py    HYPOTHESIZE                                     ║
║    Llama 4 Scout, or the deterministic planner when unavailable.      ║
║    Sees: question + cards + last round's numbers + retrieved          ║
║          evidence + remaining budget.                                 ║
║    Emits: Pydantic-validated Hypothesis objects. Hallucinated         ║
║           column names are stripped here.                             ║
║                              │                                        ║
║  features/transforms.py   PRUNE                                       ║
║    Unit algebra rejects ~72% of candidates at zero I/O cost.          ║
║    features/parser.py — model-proposed expressions pass through a     ║
║    whitelisted recursive-descent parser. No eval, ever.               ║
║                              │                                        ║
║  search/beam.py           SCREEN (progressive ladder)                 ║
║    batched materialize → linear/monotonic → MI → stability → FDR      ║
║                              │                                        ║
║  search/scorer.py         RANK                                        ║
║    weighted geometric mean of strength · stability · relevance ·      ║
║    novelty, minus a bounded cost penalty                              ║
║                              │                                        ║
║  visualization/           VISUALIZE                                   ║
║    form follows semantic type AND scale; deterministic selector       ║
║    wins over any model suggestion it cannot defend                    ║
║                              │                                        ║
║  interpretation/vlm.py    INTERPRET                                   ║
║    image + exact statistics + column definitions + units + filters    ║
║    + caveats + related prior evidence                                 ║
║                              │                                        ║
║  interpretation/critic.py VALIDATE  (deterministic, no model call)    ║
║    direction · strength · identifier misuse · causal language ·       ║
║    dictionary-blocked claims · accounting identities · disclosure     ║
║                              │                                        ║
║  evidence/                REMEMBER → RETRIEVE → REINTERPRET           ║
║    EvidenceObject → Elasticsearch (BM25 + kNN, RRF)                   ║
║    open questions resurface when related evidence arrives             ║
║                              │                                        ║
║  search/budgets.py        STOP?                                       ║
║    hard budgets + marginal-improvement rule; reason recorded          ║
╚═══════════════════════════════════════════════════════════════════════╝
        ▼
   FinalAnswer + artifacts/<id>/run_report.md + every plot
```

---

## Module responsibilities

| Module | Responsibility | Never does |
|---|---|---|
| `ingestion/` | resolve, download atomically, fingerprint, carry the data dictionary | mutate raw files |
| `profiling/` | units, semantic types, one-pass profile, compact cards | send rows anywhere |
| `features/` | safe expression language, pruning, DAG, caching, derived columns | execute model-authored code |
| `statistics/` | decide what is actually true | call an LLM |
| `search/` | plan, budget, screen, rank, orchestrate | compute statistics itself |
| `visualization/` | pick the chart form, render aggregates | draw every row |
| `interpretation/` | propose mechanisms; validate claims against numbers | let prose outrank arithmetic |
| `evidence/` | persist and retrieve findings | compute correlations |
| `external/` | optional grounding and compression | be required |
| `telemetry/` | measure | estimate silently |
| `api/` | expose it | leak credentials |

---

## Key design decisions

### The LLM proposes; deterministic code disposes

Three separable jobs, assigned to whichever is actually good at them:

| Job | Owner | Why |
|---|---|---|
| What is worth testing? | LLM | open-ended, benefits from world knowledge |
| Is it true? | NumPy/SciPy | arithmetic; a model would pattern-match |
| Does the stated claim match the numbers? | deterministic critic | a model that got it wrong cannot reliably catch itself |

### Semantic type ≠ physical dtype

`PULocationID` is an `int32` and a taxi-zone label. Averaging it is meaningless. Inference
runs in strict precedence: **official metadata** (1.00) → **dtype hard constraints** → **name
heuristics** (0.55–0.80) → **distribution heuristics**. Every inference records which rule
fired, so a warning can say *why*.

Physical castability is checked separately: `store_and_fwd_flag` is semantically boolean but
physically a string, so it is excluded from the numeric candidate pool.

### Units are a correctness mechanism, not just a speed one

Because `dollars + miles` cannot be constructed, the engine *cannot* report it. The pruning
speedup is a side effect of making a class of nonsense unrepresentable.

### The critic is deterministic

A model that produced a wrong claim cannot be trusted to catch it. So the critic is code:
it compares the claim's stated direction against the sign of the coefficient, checks strength
against the measured effect, catches correlation-to-causation slippage, enforces the data
dictionary's explicit claim blocks, and flags accounting identities.

**One rule governs it: world knowledge never overrides the data.** A surprising result is
flagged `semantically_surprising` for investigation — a NOTE, never an error, never deleted.
Finding unexpected structure is the point; the failure mode guarded against is a model
*narrating* the numbers wrongly.

### Evidence memory is a product feature

An unexplained-but-robust finding is not a failure to discard; it is a standing question. It
is stored with status `UNRESOLVED`, and `semantic_search_text` — a dense natural-language
rendering including column names, units, direction, strength and interpretation — makes it
findable by both BM25 and vector search when related evidence arrives.

### Elasticsearch is memory, not a calculator

It stores and retrieves. It never computes a correlation. This keeps correctness independent
of a network service and keeps the numerical work where it is fastest.

### Nothing is silently dropped

The analysis view excludes ~5.8% of rows. Every filter is named, carries a written rationale,
and reports exactly how many rows it removes. Excluding a 269,000-mile taxi trip is correct;
hiding that you did is how an analysis becomes untrustworthy.

### Failure is a branch, not an exception

| Failure | Behaviour |
|---|---|
| LLM unavailable | deterministic planner; interpretation reports the association only |
| LLM returns bad JSON | one retry with the validation error fed back, then fall back |
| LLM hallucinates a column | stripped at the schema boundary |
| Elasticsearch down | writes queue for retry; analysis continues; degradation reported |
| Brave down / unkeyed | grounding skipped (it is gated anyway) |
| Plot fails | statistics retained, interpretation marked pending |
| Budget exhausted | search stops, reason recorded and surfaced |

---

## Concurrency model

| Workload | Mechanism | Why |
|---|---|---|
| LLM / Elastic / Brave calls | `asyncio` + bounded semaphores | I/O-bound; prevents provider stampede |
| Numerical work | vectorized Polars / NumPy | released GIL, SIMD |
| CPU stages inside the orchestrator | `asyncio.to_thread` | keeps the event loop free so the API reports progress |
| Multi-column aggregation | Polars internal parallelism | already parallel |

Ordinary Python threads are not used as the primary mechanism for CPU-bound numeric loops.

---

## Extending to another dataset

The engine is NYC-agnostic; only the metadata is domain-specific.

1. Add a source in `ingestion/` (or use `register_local_dataset` on any Parquet/CSV).
2. Supply a data dictionary: `{column: {description, semantic_type, unit, categories,
   caveats, blocks_claims}}`.
3. Declare accounting identities, if the schema has any.
4. Supply domain filters as `FilterRule(name, expression, rationale)`.

Everything downstream — pruning, screening, plotting, interpretation, critique, memory — is
driven by semantic types and units, not by column names.

`blocks_claims` is the mechanism worth reusing: it lets a dataset declare *"this field cannot
support that comparison"* and have it enforced deterministically, which is how the cash-tip
trap is caught.
