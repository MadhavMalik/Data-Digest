# 0007 — Target-leakage rejection and result-level deduplication

Both of these were found by **running the pipeline and reading the output**, not by
reasoning about it in advance. They are recorded here because the failure modes are general,
not NYC-specific.

---

## Part A — Target leakage

### The bug

The first full run reported, as its single strongest finding:

```
total_per_mile_times_trip_distance ~ total_amount
  r = +1.000   ρ = +1.000   MI = 0.998   stability = 1.00   n = 3,509,466
```

Perfect correlation, perfect stability, three and a half million rows. It is also
meaningless: `total_per_mile` is *defined* as `total_amount / trip_distance`, so the
expression is `(total_amount / d) × d = total_amount`. The engine had discovered that a
quantity equals itself.

### Why the existing guards missed it

The critic had a `derived_feature_shares_a_term` check, but it compared **display names**:
`"total_amount"` is not a substring of `"total_per_mile_times_trip_distance"`. The dependency
was real but invisible at the name level.

### Candidate fixes

1. **Better name matching** — regex over generated names.
2. **Post-hoc filter** — drop findings with `|r| > 0.9999`.
3. **Derivation provenance** — record which base columns each derived column came from, and
   reject candidates whose derivation closure contains the target.

### Chosen: (3)

Each derived column registers its sources (`total_per_mile ← {total_amount, trip_distance}`).
`derivation_closure()` expands a candidate's columns transitively to a fixed point;
`reconstructs_target()` returns True when the target is inside that closure. Such candidates
are dropped **before materialization**.

### Why not the others

- **(1) Name matching** is fragile in both directions: it misses renamed derivations and
  would falsely reject legitimate pairs that happen to share a word.
- **(2) An `|r| > 0.9999` filter** treats the symptom. It would also delete genuine
  near-deterministic relationships, which are exactly the kind of finding worth surfacing —
  and it would leave *partial* leakage (r = 0.97) in place.

Provenance also draws the distinction that matters: `fare_per_mile × trip_distance` vs
`total_amount` is **not** exact leakage (fare_amount ≠ total_amount) — it is an accounting
identity, and it is correctly labelled `[mechanical]` rather than dropped.

### Implications

Runtime cost is a set operation per candidate. The finding quality improvement is large: the
top result went from a tautology to `trip_distance ~ total_amount`, r = +0.867.

### Tests

`tests/integration/test_nyc_domain.py::TestTargetLeakage`, and the e2e assertion that no
reported finding has `|r| > 0.9999`.

---

## Part B — Result-level deduplication

### The bug

The evidence set was dominated by restatements of one finding:

```
trip_duration_minutes ~ total_amount        r = +0.699
trip_duration_seconds ~ total_amount        r = +0.699
log(trip_duration_minutes) ~ total_amount   r = +0.685
log1p(trip_duration_minutes) ~ total_amount r = +0.698
abs(trip_duration_minutes) ~ total_amount   r = +0.699
```

Five evidence objects, five plots, five interpretation calls — for one relationship.

### Chosen fix: three complementary rules

1. **Exclude exact unit-rescalings from the candidate pool.** `trip_duration_seconds` and
   `trip_duration_minutes` are one variable; a rescale carries no information the original
   does not. The redundant one stays available for filters and display, but never enters the
   search.
2. **Do not propose `abs()` on a strictly-positive column.** It is the identity function.
   (Similarly `log`/`sqrt` are only proposed for columns known positive, so they do not
   produce mostly-null features.)
3. **Group findings by what was measured** — same base-column set, effect within tolerance —
   and keep the **simplest** expression (fewest AST nodes), recording the collapsed forms as
   `equivalent_forms` on the survivor.

Simplicity is the right tiebreak: a human would rather be shown `trip_duration_minutes` than
`log1p(trip_duration_minutes)` when the two carry the same signal.

### Why not deduplicate on the correlation value alone

Two genuinely different variables can coincidentally share an effect size. Requiring the
**same base-column set** means only true restatements collapse.

### Implications

Evidence now covers distinct findings (distance, duration, distance×duration, the mechanical
identities) instead of five spellings of one. Expensive VLM calls go to different
relationships rather than synonyms.

### Tests

Covered by `tests/unit/test_units_and_profiling.py::TestDimensionalPruning`
(`test_abs_of_a_positive_column_is_not_proposed`,
`test_log_is_only_proposed_for_positive_columns`) and observed in the run report.
