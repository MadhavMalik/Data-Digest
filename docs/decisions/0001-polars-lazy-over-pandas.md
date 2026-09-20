# 0001 — Polars lazy scanning as the data substrate

## Problem

The engine must repeatedly evaluate derived features over a 3.7 M-row Parquet file, using a
different 2–3 column subset each time, inside a search loop that runs hundreds of such
evaluations. Loading the file into memory once and slicing it is the obvious approach and the
wrong one.

## Candidate designs

1. **pandas, load once** — `pd.read_parquet()` then operate in memory.
2. **Polars eager** — same shape, faster kernels.
3. **Polars lazy + projection pushdown** — `scan_parquet`, materialize only what each
   expression needs.
4. **DuckDB** — SQL over Parquet with its own optimizer.

## Chosen

**(3) Polars lazy.**

## Why

- **Projection pushdown is the whole game.** Evaluating `fare_amount / trip_distance` reads
  2 of 20 columns. Measured: 0.086 s over 3.5 M rows. pandas would read all 20 and hold
  ~600 MB resident.
- **Predicate pushdown** applies the analysis-view filters during the scan, not after.
- **Lazy composition** lets the analysis view be built once as an unmaterialized plan and
  reused by every expression, so filters are never re-applied by hand.
- **`collect_all`** runs independent plans in one scheduled batch (used for categorical
  value counts).
- Polars releases the GIL in its kernels, which is what makes `asyncio.to_thread` genuinely
  overlap with request handling rather than just deferring work.

## Rejected

- **pandas** — memory-resident by construction; a 3.7 M × 20 frame is ~600 MB before any
  derived feature exists, and the search loop creates hundreds. Still used where a library
  demands it (nowhere in the hot path).
- **Polars eager** — gives up pushdown, which is the main win.
- **DuckDB** — excellent, but the engine's expression layer is a typed Python AST that
  compiles to Polars expressions. Targeting SQL would mean generating SQL strings from model
  input, which reintroduces exactly the injection surface the whitelisted parser exists to
  remove. Not worth it for no measured gain here.

## Implications

- **Runtime:** column projection makes per-candidate cost scale with columns touched, not
  table width.
- **Memory:** bounded by the materialized feature cache (byte-budgeted LRU), not by dataset
  size.
- **Accuracy:** none — same arithmetic.
- **Cost:** none.

## Tests

- `tests/integration/test_nyc_domain.py::test_only_needed_columns_are_projected` asserts
  exactly 2 columns are read for a 2-column expression.
- `tests/integration/test_nyc_domain.py::test_parsed_expression_matches_manual_computation`
  proves the lazy path agrees with direct computation to 1e-9.
