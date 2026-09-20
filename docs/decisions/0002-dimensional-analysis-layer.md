# 0002 — A dimensional-analysis layer for candidate pruning

## Problem

The transformation search generates `O(N²)` candidates per generation and explodes with
depth. Most are semantically meaningless — `dollars + miles`, `zone_id × dollars`. Testing
them costs I/O and, worse, *produces results*: a Pearson correlation against an arbitrary
zone ID is a real number that a downstream model will happily narrate.

## Candidate designs

1. **No units** — generate everything, let effect size sort it out.
2. **Type tags only** — a flat enum (currency, distance, …), reject mismatched pairs.
3. **Integer-exponent dimension vectors** + non-quantitative kinds.
4. **A full symbolic units library** (Pint, astropy.units).

## Chosen

**(3)** — `Dimension` as a sorted tuple of `(base, exponent)` pairs, plus a `Kind` enum
(QUANTITY / DATETIME / CATEGORICAL / IDENTIFIER / BOOLEAN / TEXT / UNKNOWN) whose algebra is
special-cased.

## Why

- **Composite units are the useful ones.** A flat enum can reject `dollars + miles` but
  cannot *derive* `USD/mile` from `USD ÷ mile`, so it cannot type the very features the
  engine most wants to find. Exponent vectors give closure under × and ÷ for free.
- **Non-quantities need different rules, not a dimension.** A zone ID is not "dimensionless"
  — it has no magnitude at all, so *every* arithmetic operation on it is invalid. Modelling
  that as a `Kind` rather than a dimension is what makes `identifier × anything` rejectable.
- **Datetime algebra is genuinely special**: `datetime − datetime = duration`,
  `datetime + duration = datetime`, `datetime + datetime` is nonsense. Three rules, worth
  hand-writing.
- **Confidence propagates.** Metadata-derived units are 1.0, name-heuristic units lower; the
  minimum propagates through every operation, so a derived feature can never look more
  certain than its weakest input.

## Rejected

- **(1) No units** — the failure isn't speed, it's that it *generates confident nonsense*.
  This layer is a correctness mechanism first.
- **(2) Flat type tags** — cannot express `USD/mile`, so it prunes the good candidates along
  with the bad.
- **(4) Pint/astropy** — heavyweight, opinionated about unit *conversion* (which we do not
  want: converting miles to km silently would change a feature's meaning), and neither models
  "this integer is a label with no magnitude", which is the case that matters most here.

## Implications

- **Runtime:** ~72% of candidates rejected at zero I/O cost on the real schema.
- **Memory:** negligible — frozen dataclasses, hashable, shared.
- **Accuracy:** strictly improved. A correlation against an identifier cannot be produced
  because the candidate cannot be constructed.
- **Token cost:** fewer junk findings reaching the interpretation stage.

## Known deliberate imprecision

`log()` of a dimensional quantity is strictly undefined. It is also standard practice in data
analysis. We permit it, return a DIMENSIONLESS result, and keep the provenance in the label
(`log(USD)`) so no interpretation can claim the result is still dollars.

## Tests

`tests/unit/test_units_and_profiling.py` — `TestRejectedOperations` (what must be deleted),
`TestAllowedOperations` (what must survive), `TestDimensionalPruning` (measured prune rate,
attributed rejection reasons).
