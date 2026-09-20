# Optimization log

Every optimization here is **implemented and measured**. Where a number appears it came from
instrumentation on the real NYC TLC January 2026 file (3,724,889 rows, 61 MiB), not an
estimate. Where an optimization has a correctness precondition, the precondition and its
guard are stated — a fast wrong answer is worse than a slow right one.

The organizing principle, in priority order:

> **Can we avoid the work entirely?** → **Can we reuse an earlier result?** →
> **Can we compute a sufficient statistic once?** → **Can we prune before touching N rows?**
> → only then, make the remaining work fast.

---

## 1. Dimensional pruning — delete nonsense before reading a row

**The headline optimization.** With N base columns and 4 binary operations the depth-1
candidate space is `4·N·(N−1) + 4·N`. Most of it is semantically meaningless:

```
fare_amount + trip_distance     dollars + miles          → REJECTED
PULocationID * fare_amount      label × dollars          → REJECTED
tpep_pickup_datetime * fare     datetime arithmetic      → REJECTED
fare_amount / trip_distance     USD per mile             → KEPT
trip_distance / duration        miles per hour           → KEPT
```

Rejection happens on the **unit algebra alone**, at zero I/O cost. A small dimensional layer
(`profiling/units.py`) tracks integer exponents over base dimensions plus non-quantitative
*kinds* (datetime, categorical, identifier) whose algebra is special-cased.

**Measured on the real schema:** 432 candidates considered → 120 emitted, **~72% pruned**.
Rejection reasons, counted:

| Reason | Share |
|---|---|
| categorical has no magnitude | largest |
| incompatible units for +/− | |
| datetime arithmetic not meaningful | |
| identifier has no magnitude | |
| same-unit ratio (dimensionless) | policy |

This is also a **correctness** mechanism: it is structurally impossible for the engine to
report a correlation against an arbitrary zone ID, because that candidate never exists.

## 2. Sufficient statistics — Σ once, then k×k instead of N

For `z = aᵀX` and `w = bᵀX`:

```
Cov(z,w) = aᵀ Σ b        Var(z) = aᵀ Σ a
corr(z,w) = aᵀΣb / √(aᵀΣa · bᵀΣb)
```

Compute Σ **once** over the k base numeric variables, then any linear combination costs k×k
arithmetic instead of an N-row scan. With N = 3.5 M and k = 19 that is a ~10⁴× reduction
*per candidate*, amortized against one Σ build.

**Correctness precondition — this is the whole reason it needs care.** The identity holds
only when z and w are evaluated on the rows Σ was estimated from. A *pairwise-complete* Σ is
not the covariance matrix of any single dataset, need not be positive semi-definite, and
would return numbers matching no real computation.

So `build_covariance_model` uses **complete cases only**, records the retained fraction, and
`is_valid()` **refuses the shortcut** below 50% complete — the caller then falls back to
direct computation. On the real file: 19 variables, 71.7% complete, shortcut **valid**.

Proven in `tests/unit/test_statistics.py::TestCovarianceShortcut`:
**shortcut vs direct agree to 2.8 × 10⁻¹⁷**, and the refusal path is tested under heavy
column-specific missingness.

## 3. Progressive computation — each rung cheaper than the one above

```
candidates surviving dimensional pruning
    │  batched materialization, one pass over only the needed columns
cheap linear + monotonic screening        O(n), vectorized
    │  top fraction + anything flagged nonlinear
mutual information                        O(n), ~20× the constant
    │  survivors
cross-fold stability                      O(n · folds)
    │  survivors
plot rendering + VLM interpretation       seconds and money per item
```

**Measured:** 103 statistical tests → 52 MI tests → 16 stability tests → 10 plots → 10
interpretations. A flat pipeline that rendered a plot per candidate would spend hours and
dollars on relationships that do not exist.

## 4. Single-pass profiling

A per-column loop over 3.7 M rows re-reads the file N times. The profiler builds **one**
Polars lazy query containing every aggregation for every column and collects it once.

**Measured: 276 aggregations across 20 columns in 0.79 s.**

Exactness policy: metrics cheap in a streaming pass (min/max/mean/std/quantiles/null counts)
are computed over **all** rows. Distinct counts use Polars' approximate counter and are
explicitly flagged in `approximate_metrics` — the engine never presents an approximation as
exact. Low-cardinality categoricals get exact value counts in one parallel `collect_all`.

## 5. Profile once, cache on the fingerprint

Datasets are fingerprinted by size + head/mid/tail content samples (full SHA-256 of a
multi-GB Parquet costs minutes and buys nothing). Schema, column cards, semantic typing and
dictionary enrichment are cached against that fingerprint. A second question about the same
dataset pays **zero** profiling cost.

## 6. Expression DAG + canonical hashing

Expressions form a DAG; shared subexpressions are computed once. Canonicalization makes
equivalent expressions collide:

```
add(A,B) ≡ add(B,A)        commutative → operands sorted by their own canonical key
mul(A,B) ≡ mul(B,A)
sub(A,B) ≢ sub(B,A)        order preserved
div(A,B) ≢ div(B,A)
```

Plus genuine algebraic identities: `−(−x) → x`, `abs(abs(x)) → abs(x)`, `x/1 → x`, `x+0 → x`.

Batched evaluation matters as much: requesting 200 features one at a time means 200 passes
over the file. `materialize_many` compiles them into **one** Polars `select`.

**Measured:** materializing `fare_amount / trip_distance` over 3.5 M rows reads exactly **2
of 20 columns** and takes 0.086 s.

## 7. Two-level feature cache

- **L1** bounded in-memory LRU, sized in **bytes** — a cap on entry *count* is the wrong
  control when one 3.5 M-row float64 column is 28 MB.
- **L2** `.npy` files on disk, surviving process restarts.

Cache keys combine dataset fingerprint + analysis-view hash + canonical expression hash, so
a vector can never be served for a different dataset version or a different row filter.

## 8. LLM response cache

Keyed on provider + model + prompt-template version + normalized messages + temperature +
max_tokens + json_mode. Whitespace-normalized, so cosmetically different prompts share a key.

**Caching is disabled above temperature 0.35** — caching a sampled response silently turns a
stochastic call deterministic, which is a correctness change disguised as an optimization.

## 9. Context minimization

The model receives the **question**, **column cards**, **last round's exact numbers**, **the
few retrieved evidence items that matter**, and **the remaining budget**. Never rows, never
the full evidence history.

**Measured: ~4 KB of column cards stands in for 61 MiB on disk.**

Numbers are passed as compact structured lines (`pearson=+0.834 n=3,509,466`) rather than
narrative — fewer tokens and less ambiguity than a sentence saying the same thing.

## 10. Protected-span context compression

`ContextCompressor` is an interface with `NoOpContextCompressor` (default) and a Token
Company adapter. Compression applies to **narrative prose only**. Exact statistics, column
names and units are masked out before compression and restored byte-identically afterwards;
if any protected span fails to return, the **original text is kept** — a compression that
loses a protected span is a failed compression, not a smaller prompt.

Embedding vectors are never compressed: a text-token compressor has no meaning applied to a
float array, and our vectors never enter an LLM context anyway.

## 11. Result-level deduplication

`trip_duration_minutes`, `log(trip_duration_minutes)` and `abs(trip_duration_minutes)`
against the same target are **one finding reported four ways**. Grouping is on *what was
measured* (same base-column set, effect within tolerance), and the **simplest** expression
survives — a human would rather see `trip_duration_minutes` than `log1p(...)` when they
carry the same signal.

Exact unit-rescalings (`trip_duration_seconds` vs `_minutes`) are excluded from the
candidate pool outright: a rescale carries no information the original does not.

## 12. Target-leakage rejection

Derived columns carry **provenance** (which base columns they came from). A candidate whose
derivation closure contains the target is dropped before materialization.

Without it the engine "discovers" that `(total_amount / trip_distance) × trip_distance`
correlates **+1.000** with `total_amount` — an algebraic identity dressed up as a finding.
This was a real bug caught by running the pipeline and reading the output.

## 13. Rendering is aggregation, not plotting every row

Nothing draws 3.5 M marks. Continuous-vs-continuous at that scale becomes a **hexbin density
plot** (log colour scale, clipped to the 0.5–99.5 percentile range so outliers cannot
compress all the structure into one cell). Grouped comparisons aggregate to box plots or
means with CIs. The chosen strategy is recorded in the `PlotSpec` and **told to the VLM**, so
it cannot describe binned means as if they were individual trips.

## 14. Budgets and marginal-improvement stopping

The candidate space is combinatorial; pretending to exhaust it would be dishonest. Instead:

- hard budgets: depth, beam width, candidates, tests, LLM calls, VLM calls, wall clock
- **marginal improvement**: a branch that has not improved its best score by more than
  `min_improvement` for `no_improvement_rounds` consecutive rounds is closed

Every stop is recorded with its reason and surfaced in the API, the report and the UI.

## 15. Adaptive escalation — deterministic code before model calls

A cheap deterministic path answers whatever it can, and only genuinely ambiguous cases reach
a model:

| Question | Answered by |
|---|---|
| Does a relationship exist, and how strong? | NumPy/SciPy — **no model call** |
| Which chart form exposes it? | deterministic `GraphSelector` |
| Is the stated claim consistent with the numbers? | deterministic critic — **no model call** |
| Is this an accounting identity? | dictionary metadata — **no model call** |
| What mechanism might explain it? | VLM (the expensive path) |
| Does new evidence resolve an old open question? | LLM, only when retrieval finds one |

## 16. Correct concurrency for each workload

- **network I/O** (LLM, Elasticsearch, Brave) — `asyncio`, with bounded semaphores so a wide
  fan-out cannot stampede a provider
- **numerical work** — vectorized Polars/NumPy first
- **CPU-bound stages inside the async orchestrator** — dispatched with `asyncio.to_thread`
  (13 call sites). Polars and NumPy release the GIL for their kernels, so this genuinely
  overlaps with request handling. Without it the event loop froze for tens of seconds and the
  API stopped answering mid-analysis — another bug found by actually running the thing.

Ordinary Python threads are **not** used as the primary mechanism for CPU-bound numeric
loops.

## 17. Vector quantization in Elasticsearch

`dense_vector` uses `int8_hnsw`: roughly 4× less memory than float32, with Elasticsearch
rescoring from full-precision values to recover recall. At larger corpus sizes BBQ would be
the next step; at this scale int8 is the right point on the curve.

## 18. Hybrid retrieval with RRF

Lexical and vector scores live on incomparable scales. **Reciprocal Rank Fusion** uses only
the *ranks*, so no normalization or tuning is needed:

```
score(d) = Σᵢ wᵢ / (k + rankᵢ(d))
```

Server-side `rrf` retriever where the cluster supports it (confirmed `native_rrf` on
Elasticsearch 9.5.4), with an identical client-side implementation as fallback so ranking
behaviour does not change with the cluster version.

## 19. Lazy index refresh

Elasticsearch indexing is near-real-time: a document written moments ago is not yet
searchable. Within one analysis the engine writes evidence and then immediately searches for
related evidence, so without a refresh it would never see its own findings.

Refresh is **lazy** — once, before a search, only when dirty — rather than on every write.
**Measured: 0 → 48 prior-evidence retrievals per run** after this change.

---

## Optimizations deliberately NOT done, and why

**Sharding vectors across GPU cores.** Elasticsearch already distributes search across
shards. GPUs belong on batched embedding/vision inference, not on hand-rolled vector sharding.

**"Smaller inputs with more hidden layers."** Not an available lever. With a hosted model you
cannot change its hidden layers; if you train your own, adding layers *adds* compute. The
effort belongs in input selection, retrieval, quantization, caching, batching, routing and
representation — which is where it went.

**Pushing statistics into Elasticsearch.** Slower than Polars on a local file and it would
make correctness depend on a network service. Elasticsearch is memory, not a calculator.

**Image embeddings for evidence retrieval.** The plan is `graph image → multimodal
interpretation`, not `graph → embedding → LLM`. Evidence is retrieved on the graph's *textual*
metadata and interpretation. Image embeddings are a later option, not an MVP requirement.

**Percentile-based outlier trimming.** Every filter is a documented physical-plausibility
bound with a counted exclusion. Trimming by percentile would silently delete real extreme
trips, which is exactly the "signal" the challenge is about.
