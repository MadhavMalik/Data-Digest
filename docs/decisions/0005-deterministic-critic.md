# 0005 — A deterministic critic, and never letting priors override data

## Problem

A multimodal model reading a graph produces fluent prose. Fluent prose can contradict the
arithmetic it is describing: calling `r = +0.8` an "inverse relationship", asserting causation
from an observational association, or announcing that "cash passengers tip less" from a field
the data dictionary says does not record cash tips.

## Candidate designs

1. **Prompt harder** — instruct the model not to do these things.
2. **LLM-as-judge** — a second model call checks the first.
3. **Deterministic rule-based critic** over (claim, statistics, metadata).
4. **Critic + human review** for the final report.

## Chosen

**(3)**, with **(4)** as the evaluation report.

## Why

- **A model that produced a wrong claim cannot be relied on to catch it.** The error modes
  are correlated: the same misreading that generated "inverse" will likely survive a
  self-check. Comparing a stated direction to `sign(r)` is a boolean expression; it should
  not be probabilistic.
- **It is cheap and always runs.** No token cost, no latency, no budget to exhaust — so it
  runs on *every* interpretation, not just the ones we can afford to double-check.
- **The dataset can declare its own constraints.** `blocks_claims` lets the TLC dictionary
  state "tip_amount cannot support a cash-vs-card tipping comparison, because cash tips are
  structurally absent". That is enforced as data, not remembered as a prompt sentence.
- **Violations are stored with the evidence.** The failure is visible in the record rather
  than quietly patched, which is what makes the evaluation report honest.

## Rejected

- **(1) Prompt harder** — necessary but not sufficient. The prompts *do* carry these rules;
  the critic is what makes them enforceable.
- **(2) LLM-as-judge** — doubles cost and latency, adds a second sampling process to the
  failure surface, and is still probabilistic about facts that are deterministic. Reasonable
  for subjective quality; wrong for "does this sentence agree with this number".

## The rule that governs the whole module

> **World knowledge never overrides the data.**

A surprising empirical result is flagged `semantically_surprising` — severity **NOTE**, never
an error, never deleted, never downgraded. Finding unexpected structure is the entire point
of the exercise; "this contradicts my prior" is not evidence against a measurement. The
failure mode guarded against is a model *narrating* the numbers wrongly, not the numbers
being inconvenient.

## Severity model

| Severity | Meaning | Effect |
|---|---|---|
| ERROR | the claim contradicts the evidence | status forced to INVALID/MECHANICAL, confidence −0.4 |
| WARNING | the claim overreaches or omits a limit | confidence −0.15 |
| NOTE | worth recording, not a defect | no penalty |

Penalties accumulate and are capped at 0.9, so evidence is never assigned negative confidence.

## Implications

- **Runtime:** microseconds. Pure regex and comparison.
- **Cost:** zero tokens.
- **Accuracy:** the specific failure classes the challenge brief names are caught
  structurally rather than probabilistically.
- **Limitation, stated plainly:** the critic catches *inconsistency between a claim and the
  evidence*. It cannot catch a claim that is consistent with the evidence and still wrong
  about the world. That is what the human/domain evaluation report is for.

## Tests

`tests/unit/test_critic.py` — 25 tests. Direction inversion in both directions (structured
field and prose), identifier misuse, hedged vs unhedged causal language, the **cash-tip
block**, accounting identities detected from base columns rather than display names, strength
over/understatement, filter disclosure, and the surprising-result path asserting severity is
NOTE and status is *not* downgraded.
