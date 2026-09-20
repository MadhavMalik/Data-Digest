# Signal Engine — what it does, and how

> An LLM proposes hypotheses. A deterministic statistical engine decides what is
> actually true. Elasticsearch remembers every finding — including the ones
> nothing could explain yet — so later evidence can resolve them.

Built for the **Voloridge HackMIT 2026 "Signal in the Noise"** challenge, on the
NYC TLC Trip Record Data listed in that challenge. It runs on any tabular
dataset.

---

## Table of contents

1. [The problem we set out to solve](#1-the-problem-we-set-out-to-solve)
2. [The central design decision](#2-the-central-design-decision)
3. [The loop, end to end](#3-the-loop-end-to-end)
4. [Stage A — profiling](#4-stage-a--profiling)
5. [Stage B — hypothesis generation](#5-stage-b--hypothesis-generation)
6. [Stage C — dimensional pruning](#6-stage-c--dimensional-pruning)
7. [Stage D — the screening ladder](#7-stage-d--the-screening-ladder)
8. [Stage E — residual analysis](#8-stage-e--residual-analysis)
9. [Stage F — visualization](#9-stage-f--visualization)
10. [Stage G — multimodal interpretation](#10-stage-g--multimodal-interpretation)
11. [Stage H — the critic](#11-stage-h--the-critic)
12. [Stage I — evidence memory](#12-stage-i--evidence-memory)
13. [The optimizations, measured](#13-the-optimizations-measured)
14. [Working on any dataset](#14-working-on-any-dataset)
15. [The interface](#15-the-interface)
16. [What we got wrong, and how we found out](#16-what-we-got-wrong-and-how-we-found-out)
17. [Honest limitations](#17-honest-limitations)
18. [Repository map](#18-repository-map)

---

## 1. The problem we set out to solve

Hand an LLM a dataset and ask what is interesting in it, and you get fluent
nonsense. Three failure modes, all of which we hit and then engineered against:

**It cannot do the arithmetic.** A language model asked to eyeball 3.7 million
rows does not compute a correlation; it pattern-matches a plausible-sounding
answer. Correlations are arithmetic, and arithmetic belongs in NumPy.

**It does not fit.** 3.7M rows × 20 columns is roughly 400 million tokens.

**It narrates the numbers wrongly.** Even handed correct statistics, a model
will call `r = +0.8` an inverse relationship, or conclude "cash riders tip less"
from a field whose documentation says cash tips are not recorded.

And there is a fourth failure mode that is subtler and, we think, more
interesting:

**Marginal correlation finds only the obvious.** On taxi data,
`corr(distance, fare) = +0.87`. True, dominant, and worthless — it restates that
taxis have meters. Rank by effect size and that relationship plus its algebraic
cousins crowd out everything else.

Signal Engine is our answer to all four.

---

## 2. The central design decision

> **The LLM is the last expensive operation, not the thing doing the data
> processing.**

By the time a model sees anything, deterministic code has already profiled the
data, pruned the candidate space on unit algebra, screened thousands of
candidates through a cost-ordered ladder, and selected a handful worth
explaining.

Three separable jobs, each given to whichever is actually good at it:

| Job | Owner | Why |
|---|---|---|
| What is worth testing? | LLM | Open-ended, benefits from world knowledge |
| Is it true? | NumPy / SciPy / Polars | Arithmetic; a model would pattern-match |
| Does the claim match the numbers? | Deterministic critic | A model that got it wrong cannot reliably catch itself |

A second constraint follows: **every external dependency is optional.** No LLM,
no Elasticsearch, no Brave — the run still completes with real statistics, real
plots and honest evidence, and it says out loud what it degraded.

---

## 3. The loop, end to end

```
HYPOTHESIZE → COMPUTE → OBSERVE → REFINE → TRANSFORM → COMPUTE
    → VISUALIZE → INTERPRET → REMEMBER → RETRIEVE → REINTERPRET
```

```
data/raw/yellow_tripdata_2026-01.parquet      3,724,889 rows · 61 MiB
        │
        │  Polars lazy scan — projection + predicate pushdown
        ▼
┌──────────────────────────────────────────────────────────────────┐
│ A  PROFILE          one query, 276 aggregations, 0.79 s          │
│                     → ~7 KB of column cards. The only thing the  │
│                       model ever sees.                            │
└──────────────────────────────────────────────────────────────────┘
        ▼
┌──────────────────────────────────────────────────────────────────┐
│    ANALYSIS VIEW    derived columns + documented validity filters │
│                     every exclusion named, justified and counted  │
└──────────────────────────────────────────────────────────────────┘
        ▼
╔═══════════════════════ ROUND LOOP ═══════════════════════════════╗
║ B  HYPOTHESIZE   LLM (or deterministic planner), Pydantic-validated
║ C  PRUNE         unit algebra deletes ~72% before reading a row
║ D  SCREEN        linear → mutual information → stability → FDR
║    RANK          weighted geometric mean of measured + heuristic
║ F  VISUALIZE     form follows semantic type AND scale
║ G  INTERPRET     multimodal read of the graph + exact statistics
║ H  CRITIQUE      deterministic validation of claim vs numbers
║ I  REMEMBER      → Elasticsearch → retrieve → reinterpret jointly
║    STOP?         budgets + marginal-improvement rule
╚══════════════════════════════════════════════════════════════════╝
        ▼
┌──────────────────────────────────────────────────────────────────┐
│ E  RESIDUAL      fit a baseline from the obvious drivers,         │
│                  then search what it CANNOT explain               │
└──────────────────────────────────────────────────────────────────┘
        ▼
   Final answer + run report + every plot + full interpretation trace
```

---

## 4. Stage A — profiling

**`src/signal_engine/profiling/`**

The profiler builds **one** Polars lazy query containing every aggregation for
every column and collects it once. A per-column loop would re-read the file N
times.

**Measured: 276 aggregations across 20 columns in 0.79 seconds** on 3.7M rows.

### Semantic typing is the part that matters

A column's *physical* dtype is almost never its *semantic* type. `PULocationID`
is an `int32` and a taxi-zone label; averaging it is meaningless. Getting this
wrong is the single most common way automated analysis produces confident
nonsense.

Inference runs in strict precedence, and every inference records which rule
fired:

1. **Official metadata** (confidence 1.00) — never overridden
2. **Physical dtype constraints** — a timestamp is a datetime, full stop
3. **Name heuristics** (0.55–0.80) — `*_amount` is currency, `*_id` is a label
4. **Distribution heuristics** — few distinct integers over millions of rows is
   an encoded category, not a magnitude

### Exactness policy

Metrics that are cheap in a streaming pass (min, max, mean, std, quantiles, null
counts) are computed over **all** rows. Distinct counts use an approximate
counter and are explicitly flagged in `approximate_metrics`. **The engine never
presents an approximation as exact.**

### What the model receives

A **dataset card** and one **column card** each: name, semantic type, unit,
distribution summary, official description, known caveats.

**~7 KB standing in for 61 MiB.** `01_profiling/EXACT_TEXT_SENT_TO_LLM.txt` in
any export is that exact text, verbatim.

---

## 5. Stage B — hypothesis generation

**`src/signal_engine/search/planner.py`, `src/signal_engine/llm/`**

The model receives the question, the column cards, the previous round's exact
numbers, retrieved prior evidence, and the remaining budget. It returns
**structured** `Hypothesis` objects — never prose parsed by regex.

Schemas are permissive about what they **accept** (models emit `"HIGH"`, `3`,
stray prose) and strict about what they **produce**. Hallucinated column names
are stripped at the schema boundary, before anything reaches the data layer.

### The deterministic planner is not a stub

With no credentials, `deterministic_hypotheses` encodes the same reasoning a
careful analyst applies: prefer columns matching the question, pair quantities
with the target, route categoricals to group comparisons, propose the ratio
features the semantics imply.

It means the engine produces real findings with zero credentials — **and it is
the baseline the LLM path is measured against.** If model hypotheses do not beat
it, the model is not earning its cost.

---

## 6. Stage C — dimensional pruning

**`src/signal_engine/profiling/units.py`, `features/transforms.py`**

This is the headline optimization, and it is a **correctness** mechanism first.

With N columns and 4 binary operations the depth-1 candidate space is
`4·N·(N−1) + 4·N`. Most of it is meaningless:

```
fare_amount + trip_distance        dollars + miles       → REJECTED
PULocationID * fare_amount         label × dollars       → REJECTED
tpep_pickup_datetime * fare        datetime arithmetic   → REJECTED
trip_distance / is_airport_pickup  a filter in disguise  → REJECTED
fare_amount / trip_distance        USD per mile          → KEPT
trip_distance / duration           miles per hour        → KEPT
```

Rejection happens on the unit algebra alone, at **zero I/O cost**.

A small layer tracks integer exponents over base dimensions (currency, length,
time, count, angle, temperature, energy, mass, volume) plus non-quantitative
*kinds* — `DATETIME`, `CATEGORICAL`, `IDENTIFIER`, `BOOLEAN` — whose algebra is
special-cased:

- `datetime − datetime = duration`; `datetime + datetime` is nonsense
- an identifier has **no magnitude**, so every arithmetic operation on it is invalid
- a boolean flag is a **stratification variable**, not an operand — `distance /
  is_airport` is a filter wearing the costume of a feature

**Measured: ~72% of candidates rejected**, every rejection attributed by reason.

Because `dollars + miles` cannot be *constructed*, the engine **cannot report
it**. The speedup is a side effect of making a class of nonsense
unrepresentable.

---

## 7. Stage D — the screening ladder

**`src/signal_engine/search/beam.py`, `statistics/`**

Each rung is strictly cheaper than the one above it, so the expensive rungs only
ever see a handful of candidates:

```
candidates surviving dimensional pruning
    │  batched materialization — one pass over only the needed columns
cheap linear + monotonic screening        O(n), vectorized
    │  top fraction + anything flagged nonlinear
mutual information                        O(n), ~20× the constant
    │  survivors
cross-fold stability                      O(n · folds)
    │  survivors
plot rendering + multimodal interpretation   seconds and money each
```

**Measured on one run: 103 statistical tests → 52 MI tests → 16 stability tests
→ 12 plots → 12 interpretations.**

### Policy decisions baked into the statistics

**Effect size over p-value.** At n = 3.5M, `r = 0.002` has a p-value near zero.
We compute p-values and BH-FDR q-values, but ranking leads with effect
magnitude, and the report says so explicitly.

**Cross-fold stability is the real robustness signal.** Folds are assigned by a
seeded permutation, not by row order — TLC files arrive time-ordered, and
contiguous blocks would measure temporal drift rather than robustness.

**Never Pearson an identifier.** Categoricals route to `grouped_comparison`
(eta-squared + per-group means) or `categorical_association` (Cramér's V).

**Nonlinearity is detected, not assumed.** For X symmetric about zero and
Y = X², Pearson r ≈ 0 despite a deterministic relationship. Mutual information
catches it. That case is a unit test.

### The sufficient-statistics shortcut

For `z = aᵀX` and `w = bᵀX`:

```
Cov(z,w) = aᵀ Σ b        corr(z,w) = aᵀΣb / √(aᵀΣa · bᵀΣb)
```

Compute Σ once and any linear combination costs k×k arithmetic instead of an
N-row scan — with N = 3.5M and k = 19, roughly 10⁴× per candidate.

**The correctness condition is the whole reason this needs care.** The identity
holds only when z and w are evaluated on the rows Σ was estimated from. A
*pairwise-complete* Σ is not the covariance matrix of any single dataset, need
not be positive semi-definite, and returns numbers matching no real computation.

So Σ is estimated on **complete cases only**, the retained fraction is recorded,
and `is_valid()` **refuses** the shortcut below 50% complete — the caller falls
back to direct computation.

**Measured agreement with a direct row scan: 2.8 × 10⁻¹⁷.** The refusal path is
tested under heavy column-specific missingness.

---

## 8. Stage E — residual analysis

**`src/signal_engine/statistics/residual.py`**

This is the stage that finds what is *not* obvious, and it exists because our
first working version produced only obvious findings.

Fit a simple baseline from the strongest **empirical** drivers, subtract it, and
search what explains the remainder:

```
BASELINE   trip_distance + average_speed_mph → total_amount    R² = 0.762
           unexplained spread: 20.86 → 10.19

WHAT EXPLAINS THE REMAINDER

RatecodeID              η = 0.600    spread $58.19
    Negotiated fare          +39.62
    Nassau or Westchester    +33.32
    Newark                   +31.01
    Standard rate             −0.33

is_airport_pickup       η = 0.233    +6.47 vs −1.29     ← the airport fee
VendorID                η = 0.173    Myle Technologies −20.56
```

Those numbers say: *after accounting for how far and how fast you went, these
rate codes cost dramatically more.* Marginal correlation cannot see it, because
`RatecodeID` barely correlates with fare on its own.

Design notes that matter:

- The baseline is deliberately **simple** (OLS on a few strong predictors). A
  flexible model would absorb the very structure we are surfacing.
- Baseline predictors are chosen from **empirical** drivers, never accounting
  components of the target — regressing a total on its own parts leaves a
  residual of rounding error.
- Integer-coded categories are excluded from the baseline and from numeric
  residual tests.
- Everything is reported in the target's own units, because **"+$34.15" is
  actionable and "η = 0.638" is not**.

**Proof it works on an unseen domain:** a unit test builds a synthetic
energy-grid dataset with a hidden per-plant price premium, confirms the premium
is invisible marginally (|r| < 0.25), and asserts the residual stage recovers
every per-plant effect to within $1.50 of the injected truth.

### Shape characterization

**`src/signal_engine/statistics/shape.py`**

A coefficient says "these move together"; the **shape** says how, and that is
usually the finding. Candidate forms — linear, logarithmic, power, quadratic,
inverse — are fitted to the **conditional mean** E[y|x] over quantile bins, not
to the raw cloud. A more complex form must beat the linear baseline by a margin
to be declared better, so quadratic cannot win by absorbing noise.

```
fare_per_mile vs trip_distance
    inverse, R² = 0.549 (linear: 0.034)
    slope falls 475× across the range
    mean goes $391/mi → $3.86/mi
```

---

## 9. Stage F — visualization

**`src/signal_engine/visualization/`**

One relationship gets the chart form that most clearly exposes it. A misleading
chart is worse than none: the model faithfully interprets the misleading
picture, and that claim enters evidence memory.

| x | y | n | Form |
|---|---|---|---|
| continuous | continuous | ≤ 20k | scatter, every row |
| continuous | continuous | 20k–100k | stratified sampled scatter |
| continuous | continuous | ≥ 100k | **density + conditional mean + CI + fitted form** |
| datetime | continuous | any | time-binned line |
| categorical ≤15 levels | continuous | any | box plot |
| categorical ≤30 | continuous | any | bar + 95% CI |
| categorical | categorical | any | contingency heatmap |

Selection is **deterministic and runs first**. The model may propose an
override; `validate_override` accepts it only if defensible. Deterministic code
wins on conflict.

### The framing rule

The default at scale is density **plus** the conditional mean — a bare hexbin of
3.5M rows shows where the mass is but not how y moves with x, and the second
question is the finding.

The view is fitted to the **curve**, not the raw data. On a heavy-tailed column
the raw range runs to $1000/mi while the conditional mean lives under $40;
percentile-clipping still squashes the curve flat. Anything outside the frame is
**reported in the notes**, never silently cropped.

### Telling the model what it is looking at

`describe_for_vlm()` states the rendering explicitly — *"colour is how many
records fall in each cell, NOT the value of a third variable"*. The exact
statistics are **burned into the image** as a caption, so the authoritative
numbers are in the model's visual field.

### Colour

Validated categorical palette. One hue per series, never cycled. Sequential is
one hue light→dark, never a rainbow. Diverging is blue↔red with a **neutral gray
midpoint** so "no correlation" reads as nothing. Never a dual y-axis. Identity is
never colour-alone.

---

## 10. Stage G — multimodal interpretation

**`src/signal_engine/interpretation/vlm.py`**

The model receives the image **and** everything needed to read it honestly:
exact statistics, column definitions and units, the transformation applied, the
row filters, the dataset caveats, the fitted functional form, and the top
related prior findings.

A naked graph invites invention. The same plot with *"tip_amount records
credit-card tips only; cash tips are structurally absent"* attached does not.

Output separates **observation** (what the numbers show) from **interpretation**
(what it might mean) from **plausible_mechanisms** (candidate explanations) from
**confounders** — and the critic enforces the separation.

**An honest `UNRESOLVED` is a first-class answer.** The prompt says so, the
schema supports it, and the evidence store keeps it as an open question rather
than discarding it.

---

## 11. Stage H — the critic

**`src/signal_engine/interpretation/critic.py`**

Deterministic. No model call. A model that produced a wrong claim cannot be
relied on to catch it; comparing a stated direction to `sign(r)` is a boolean
expression and should not be probabilistic.

| Check | Catches |
|---|---|
| `direction_mismatch` | "inverse relationship" attached to `r = +0.8` |
| `strength_overstated` | "very strong" on an effect of 0.15 |
| `identifier_as_quantity` | a correlation against arbitrary zone codes |
| `causal_claim_from_association` | unhedged "causes" / "drives" / "leads to" |
| `claim_blocked_by_data_dictionary` | claims the data provably cannot support |
| `accounting_identity_presented_as_discovery` | a total against its own components |
| `filters_not_disclosed` | omitting that filters shaped the result |

### This fired live, on GPT-4o

```
graph      : payment_type groups
model said : "Payment type is moderately associated with the recorded tip
              amount, with credit card payments showing higher recorded tips"
status     : explained, confidence 0.8

CRITIC [ERROR] claim_blocked_by_data_dictionary
    tip_amount only captures credit-card tips; cash tips are structurally
    absent from the data, so the comparison is undefined.

status AFTER critic: mechanical, confidence 0.25
```

That is the entire thesis of the project, caught on a real model's real output.

### The rule that governs the module

> **World knowledge never overrides the data.**

A surprising result is flagged `semantically_surprising` — severity **NOTE**,
never an error, never deleted. Finding unexpected structure is the point; the
failure mode guarded against is a model *narrating* the numbers wrongly, not the
numbers being inconvenient.

---

## 12. Stage I — evidence memory

**`src/signal_engine/evidence/`**

Every analyzed relationship becomes an `EvidenceObject` — statistics, plot URI,
interpretation, critic report, status, provenance — **including the ones nothing
could explain.**

```
Evidence 1   airport pickup → unusually high fare per mile     UNRESOLVED
             (statistically robust; no mechanism available)

… later in the search …

Evidence 2   airport pickup → fixed airport surcharge
Evidence 3   airport routes → toll charges

→ retrieval surfaces Evidence 1 alongside 2 and 3
→ "The apparent airport premium may be partly explained by fixed fees and
   tolls rather than by distance itself."
→ Evidence 1 promoted to EXPLAINED, resolved_by pointing at 2 and 3
```

**Measured on one live run: 60 retrievals, 5 joint reinterpretations, 8
previously-unresolved findings promoted to explained.**

### Elasticsearch is memory, not a calculator

It stores and retrieves. It never computes a correlation — that would make
correctness depend on a network service, and Polars on a local file is faster
anyway.

**Hybrid retrieval:** BM25 over the finding's text, dense-vector kNN over its
embedding, merged with **Reciprocal Rank Fusion**. The two score scales are
incomparable; RRF uses only the *ranks*, so no normalization or tuning is
needed.

```
score(d) = Σᵢ wᵢ / (k + rankᵢ(d))        k = 60
```

Three-tier strategy: native server-side `rrf` retriever (confirmed in use on
Elasticsearch 9.5.4), an identical client-side fallback, then BM25 only. The
resolved mode is recorded in telemetry.

`dense_vector` uses `int8_hnsw` — ~4× less memory than float32, with rescoring
from full precision to recover recall.

**Indexing lag:** Elasticsearch is near-real-time, so a document written moments
ago is not searchable. Within one analysis the engine writes evidence and
immediately searches for related evidence. Without intervention it saw *zero*.
A lazy refresh — once before a search, only when dirty — took it to **48
retrievals per run**.

---

## 13. The optimizations, measured

Every figure below comes from instrumentation on the real 3.7M-row file.

| Optimization | Effect |
|---|---|
| Single-pass profiling | 276 aggregations in **0.79 s** |
| Dimensional pruning | **~72%** of candidates deleted at zero I/O cost |
| Sufficient statistics (Σ once) | ~10⁴× per linear combination; **agrees to 2.8e-17** |
| Progressive screening ladder | 103 tests → 52 MI → 16 stability → 12 plots |
| Column projection | 2 of 20 columns read for a 2-column expression |
| Profile cache (fingerprint-keyed) | second question pays **zero** profiling cost |
| Expression DAG + canonical hashing | `add(A,B) ≡ add(B,A)`; shared subexpressions computed once |
| Two-level feature cache | byte-budgeted LRU over an on-disk `.npy` store |
| LLM response cache | **disabled above temperature 0.35** — caching a sampled response is a correctness change |
| Context minimization | **~7 KB** of cards stands in for **61 MiB** |
| Protected-span compression | statistics masked out, restored byte-identically, or the compression is refused |
| Result deduplication | five spellings of one finding collapse to the simplest |
| Target-leakage rejection | drops expressions that algebraically rebuild the target |
| `asyncio.to_thread` for CPU stages | 13 sites; the API stays responsive mid-analysis |
| `int8_hnsw` quantization | ~4× less vector memory |

**Full pipeline: ~95–110 seconds** for 3.7M rows, 3 rounds, 12–15 graph
interpretations, ~55k tokens.

### Deliberately NOT done, and why

**Sharding vectors across GPU cores.** Elasticsearch already distributes search
across shards. GPUs belong on batched inference.

**"Smaller inputs with more hidden layers."** Not an available lever. With a
hosted model you cannot change hidden layers; training your own and adding
layers *adds* compute. The effort went into input selection, retrieval,
quantization, caching, batching and representation.

**Statistics in Elasticsearch.** Slower than Polars locally and it would make
correctness contingent on a network service.

**Percentile-based outlier trimming.** Every filter is a documented
plausibility bound with a counted exclusion. Trimming by percentile silently
deletes exactly the extreme records that carry signal.

---

## 14. Working on any dataset

**`src/signal_engine/datasets.py`**

Metadata is a bonus layer, never a requirement.

```
known dataset    → curated dictionary, accounting identities, filters,
                   target hints, domain notes
unknown dataset  → semantic types inferred from dtype, name and distribution;
                   definitional validity filters; target inferred from the question
```

A dataset is matched on filename, then on **column signature** — so a file
renamed on upload still resolves.

### What happens with no metadata at all

Derived features come from **semantics**, not names:

- every datetime pair → `duration_seconds` / `duration_minutes`
- every datetime → `event_hour`, `event_day_of_week`, `event_date`
- every dimensionally coherent ratio → currency/length, currency/time,
  currency/energy, length/time, count/time
- `cost per unit of X` even when X's unit is unidentified — on an unfamiliar
  dataset that is usually the most informative feature available, and the unit
  label is marked approximate so nothing downstream over-claims

Generic filters are **definitional only**: a duration cannot be negative, a
count cannot be negative. Everything beyond that is domain knowledge and belongs
in a `DatasetSpec`. Guessing would mean percentile trimming.

Target inference scores columns by curated hint → name match → description match
→ semantic kind ("what do people pay" wants a currency column) → variance.

### Verified on a different domain

An energy-grid dataset — `dispatch_start`, `energy_mwh`, `settlement_price_usd`,
`plant_id` — with zero curated metadata:

```
recognised as   : Generic tabular dataset
typing          : plant_id → categorical_identifier (excluded from correlation)
                  settlement_price_usd → currency
                  energy_mwh → continuous, MWh
derived         : duration_minutes, event_hour, event_day_of_week, cost-per-unit ratios
target inferred : settlement_price_usd
baseline        : energy_mwh + duration_minutes    R² = 0.972
residual driver : plant_id   η = 0.864   spread $45.02
```

The per-plant premiums it recovered matched the injected ground truth to within
$1.10. **Adding another challenge dataset is one `DatasetSpec` — no engine
changes.**

---

## 15. The interface

**`web/` (React + TypeScript + Vite), `src/signal_engine/api/`**

Two lanes, streaming live:

```
┌── 01 FINDING AGENT ──────────┐ ┌── 02 ANALYSIS AGENT ─────────┐
│ profile · 20 columns, 0.79s  │ │ handoff · density_trend       │
│ hypothesis: distance → total │ │ [chart appears]               │
│ pruning: 283/432 on units    │ │ interpreting…                 │
│ result: r=+0.867 very strong │ │ interpretation [mechanical]   │
│ residual baseline R²=0.762   │ │ critic: 1 violation           │
│ driver: RatecodeID η=0.600   │ │ evidence memory: 4 retrieved  │
└──────────────────────────────┘ └───────────────────────────────┘
```

Cards enter sharp at the top and **age out** — a `--age` custom property drives
opacity, blur, saturation and scale, so attention lands on what is happening
now while history stays scannable. Hovering restores a card fully.
`prefers-reduced-motion` disables it.

### Streaming

`src/signal_engine/events.py` is an append-only, thread-safe, **replayable**
event log. The engine emits from worker threads; the SSE endpoint consumes from
the event loop. Readers track their own cursor, so:

- `cursor=0` replays the whole run — a late or reconnecting browser lands in
  exactly the same state
- `emit` never blocks on a consumer, so the analysis cannot deadlock on a slow
  client

The TypeScript event contract is a **discriminated union**, so the compiler
catches a handler reading a field the event does not have.

### Deployment

```bash
./run.sh --host 0.0.0.0        # builds UI, installs deps, fetches data, serves
docker build -t signal-engine . && docker run -p 8000:8000 --env-file .env signal-engine
```

One process serves the API and the site. See [DEPLOY.md](DEPLOY.md) for EC2 and
for the Vercel split (front end on Vercel, API on a long-lived host — a 100-second
SSE analysis exceeds serverless limits, and the doc explains exactly why).

---

## 16. What we got wrong, and how we found out

Every one of these came from **running the thing and reading the output**, not
from reasoning in advance. They are the most informative part of the project.

**The engine "discovered" that a quantity equals itself.** First full run's top
finding: `total_per_mile × trip_distance ~ total_amount`, r = **1.000**, n =
3.5M. It is `(total/d) × d`. Fixed with derivation-provenance tracking: a
candidate whose closure contains the target is dropped before materialization.

**Ten of fifteen findings were arithmetic.** `total_amount` is the sum of its
components, so every component correlates ~0.97 with it and the family
monopolised the interpretation budget. Now recorded once as definitional,
excluded from plots and model calls.

**The graphs were clouds.** A hexbin of 3.5M rows *is* a cloud. We had built a
binned-trend plot and never selected it. Now the default is density + the
conditional mean, framed on the curve.

**Boolean flags were arithmetic operands.** The search produced `trip_distance /
is_airport_pickup` — a filter in disguise that cut n from 3.5M to 221k. The
engine's own missing-data warning caught it and the model dutifully carried the
finding forward, which is exactly the failure this project exists to prevent.

**The disk feature cache never persisted anything.** `np.save` appends `.npy`,
so the atomic rename always failed, silently.

**Every LLM cache read would have crashed.** `to_dict()` emitted a derived field
the constructor rejects.

**Elasticsearch rejected every write.** An empty-string `updated_at` against a
`date` field.

**Excluded IDs leaked back through the kNN stream.** The exclusion was only on
the lexical query.

**A payload field named `kind` collided with the event envelope** and silently
killed the entire residual stage at the emit call. Fixed structurally: the
envelope is written *after* the payload spreads, so it always wins.

---

## 17. Honest limitations

**The critic catches inconsistency, not falsehood.** It verifies that a claim
agrees with the evidence, the units and the documented semantics. It cannot
catch a claim that is consistent with the evidence and still wrong about the
world. That is what the human/domain evaluation report is for.

**Residual analysis assumes an additive baseline.** An effect that is purely
multiplicative and absorbed by the OLS fit will not appear in the residual.

**Generic filters are minimal by design.** On an unknown dataset the engine
excludes almost nothing, so genuinely corrupt records reach the statistics. It
reports this rather than guessing at domain bounds.

**Shape fitting is over the conditional mean, not the joint distribution.** It
characterizes E[y|x] well; it says nothing about heteroscedasticity.

**The local embedder captures lexical overlap, not deep semantics.** It finds
"airport fee" from "airport surcharge"; it will not connect "taxi" to "cab".
With a managed inference endpoint, Elasticsearch does the embedding instead.

**Findings are associational.** The engine says "may be partly explained by",
never "causes", and the critic enforces it. There is no identification strategy
here and the reports say so.

---

## 18. Repository map

```
src/signal_engine/
  config.py            settings + credential redaction
  datasets.py          dataset registry + generic target inference
  events.py            replayable event log for the live UI
  export.py            human-readable analysis export
  jsonutil.py          NumPy-safe JSON
  reporting.py         run report + evaluation report
  ingestion/           resolution, atomic download, TLC data dictionary
  profiling/           unit algebra, semantic typing, one-pass profiler
  features/            expression AST, whitelisted parser, canonical hashing,
                       transformation search, DAG, caches, derived columns
  statistics/          correlation, covariance shortcut, MI, FDR, stability,
                       shape characterization, residual analysis
  search/              budgets, scorer, state, screening ladder, planner,
                       orchestrator
  visualization/       graph selector, matplotlib renderer
  interpretation/      multimodal interpretation, deterministic critic
  evidence/            EvidenceObject, memory store, Elasticsearch, retrieval
  external/            Brave grounding, context compression
  telemetry/           metrics registry
  api/                 FastAPI app, routes, SSE stream, upload

web/                   React + TypeScript + Vite front end
scripts/               fetch_tlc · run_demo · run_full_analysis · evaluate · make_visuals
tests/                 400+ tests: unit · integration (live Elastic) · e2e
docs/                  ARCHITECTURE · OPTIMIZATIONS · 8 decision records
```

### Security posture

- **No secrets in the repo.** `.env` is gitignored; `.env.example` holds
  placeholders only.
- **Credentials never leave the process.** `redacted_dump()` emits availability
  booleans, never key material. Elasticsearch transport errors pass through a
  scrubber.
- **TLS verification always on.** An intercepted key is worse than a setup error.
- **Model output is untrusted.** No `eval`, no `exec`, no shell, no generated
  SQL. Expressions go through a whitelisted recursive-descent parser whose only
  possible outputs are four AST node types.
- **Index deletion is guarded** — refuses any index whose name lacks `test`.
- **Path traversal guarded** on the artifact endpoint; uploads are extension-
  and size-capped.

---

## Running it

```bash
./run.sh                                   # → http://127.0.0.1:8000
python scripts/run_full_analysis.py        # full export to nyc_taxi_analysis/
python scripts/make_visuals.py             # 7 presentation figures
python scripts/evaluate.py                 # domain validation report
pytest -q                                  # 400+ tests
```
