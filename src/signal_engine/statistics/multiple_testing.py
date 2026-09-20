"""Multiple-testing correction.

The search tests thousands of candidate relationships.  At alpha = 0.05 and
4,000 tests, ~200 pure-noise relationships clear significance by chance.
Benjamini-Hochberg controls the false discovery rate instead of the
family-wise error rate, which is the right choice for a screening pipeline:
we are selecting candidates to look at more closely, not making final
accept/reject decisions.

A caveat this engine states out loud rather than hiding: with n in the
millions, essentially everything is "significant", so FDR-corrected q-values
mostly confirm that the sample is large. Ranking is therefore driven by effect
size and stability, with q-values reported alongside rather than used as the
gate.
"""

from __future__ import annotations

import numpy as np


def benjamini_hochberg(p_values: list[float] | np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Return BH-adjusted q-values, aligned to the input order.

    NaN p-values pass through as NaN so a skipped test never silently becomes
    a discovery.
    """
    p = np.asarray(list(p_values), dtype=np.float64)
    q = np.full(p.shape, np.nan, dtype=np.float64)

    valid = np.isfinite(p)
    m = int(valid.sum())
    if m == 0:
        return q

    pv = p[valid]
    order = np.argsort(pv)
    ranked = pv[order]

    adjusted = ranked * m / np.arange(1, m + 1)
    # Enforce monotonicity from the largest p downward.
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)

    out = np.empty(m, dtype=np.float64)
    out[order] = adjusted
    q[valid] = out
    return q


def apply_fdr(results: list, alpha: float = 0.05) -> list:
    """Attach `q_value` to a list of RelationshipResult objects in place."""
    p_values = [
        (r.pearson_p if r.pearson_p is not None else float("nan")) for r in results
    ]
    q = benjamini_hochberg(p_values, alpha=alpha)
    for result, qv in zip(results, q):
        result.q_value = None if np.isnan(qv) else float(qv)
    return results


def significance_note(n: int, tests: int) -> str:
    """A one-line honesty statement to carry into reports."""
    return (
        f"{tests:,} relationships tested on up to {n:,} rows. At this sample size almost any "
        f"non-zero association is statistically significant, so findings are ranked by effect "
        f"size and cross-subsample stability, with BH-FDR q-values reported for reference only."
    )
