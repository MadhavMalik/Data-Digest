# 0008 — Chart form follows semantic type and scale; deterministic selection wins

## Problem

Every discovered relationship gets a graph, and the graph is then read by a vision model.
A misleading chart is worse than no chart: the VLM will faithfully interpret the misleading
picture and the resulting claim enters evidence memory.

Two specific failure modes:

1. **One chart type for everything.** A bar chart of a continuous-vs-continuous relationship
   hides the shape entirely.
2. **Ignoring scale.** A scatter plot of 3.5 M points is a solid black rectangle. It takes a
   minute to render and shows nothing.

## Candidate designs

1. **Always scatter** — simple, wrong most of the time.
2. **Let the LLM choose** — flexible, unvalidated.
3. **Deterministic selector from semantic types + row count.**
4. **Deterministic selector, with a validated LLM override.**

## Chosen

**(4).** The deterministic selector runs first; the model may propose an alternative, and
`validate_override` accepts it only if it is defensible for these semantic types and this
scale. Rejections are recorded with a reason.

## The selection rule

| x | y | n | Form |
|---|---|---|---|
| continuous | continuous | ≤ 20 k | scatter, every row |
| continuous | continuous | 20 k–100 k | stratified sampled scatter |
| continuous | continuous | ≥ 100 k | **hexbin**, log colour, 0.5–99.5 pct clip |
| datetime | continuous | any | time-binned line |
| categorical (≤15 levels) | continuous | any | box plot |
| categorical (≤30) | continuous | any | bar + 95% CI |
| categorical | categorical | any | contingency heatmap |
| continuous | — | any | histogram |

Hard rejections in `validate_override`: scatter above the over-plot threshold; a line plot on
a non-time axis (it would imply an ordering that does not exist); a continuous plot against a
categorical code; a grouped plot where neither variable is categorical.

## Why deterministic wins on conflict

The selector's inputs are exactly the facts that determine the right answer: the two semantic
types and the row count. There is no judgement left for a model to add — only the opportunity
to be wrong. The override path exists because occasionally a violin genuinely reads better
than a box, and that is the kind of choice a model can contribute.

## Telling the VLM what it is looking at

`PlotSpec.describe_for_vlm()` states the rendering strategy explicitly — *"colour encodes how
many records fall in each hexagonal cell (log scale), **not the value of a third variable**"*,
*"points are conditional means within quantile bins; individual records are not shown"*.

Without this a VLM will describe binned means as if they were individual trips.

The exact statistics are additionally **burned into the image** as a caption, so the
authoritative numbers are in the model's visual field and it cannot read a direction off the
picture that contradicts the arithmetic.

## Colour policy

Colours come from a validated categorical palette (checked with the accompanying validator in
both light and dark mode):

- **one hue** for a single series; the categorical order is fixed, never cycled
- **sequential** encoding (density, counts) is one hue light→dark — never a rainbow, which
  implies an ordering its luminance does not carry and is unreadable under colour-vision
  deficiency
- **diverging** encoding (correlation) is blue↔red with a **neutral gray midpoint**, so "no
  correlation" reads as nothing
- **never a dual y-axis**
- identity is never colour-alone: axes are labelled, bars are direct-labelled, group sizes
  annotated
- recessive grid and axes — the data is the darkest thing on the page

## Implications

- **Runtime:** aggregation before rendering keeps every plot sub-second at 3.5 M rows.
- **Accuracy:** the VLM is told the sampling strategy, so it cannot mistake aggregates for
  records.
- **Cost:** plots are only rendered for candidates that survived the full screening ladder.

## Tests

`tests/unit/test_visualization.py` — 31 tests: selection by type, scale thresholds, both
axis orientations, all five override rejection rules, every plot type rendering to a real
PNG, downsampling disclosure, the statistics caption, empty-data handling, and a figure-leak
check (a leaked matplotlib figure per plot would exhaust memory over a long run).
