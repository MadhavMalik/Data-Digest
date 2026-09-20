"""Mutual information — the nonlinear-dependence detector.

Why this exists: for X symmetric about zero and Y = X², Pearson r ≈ 0 even
though Y is a deterministic function of X.  A pipeline that screens on linear
correlation alone throws that relationship away.  MI catches it.

Implementation: histogram (binned plug-in) estimator with the
Miller-Madow bias correction.  It is O(n) with a tiny constant, which matters
because MI runs as the SECOND screening stage over the candidates that survived
the cheap linear pass — the progressive-computation ladder only works if each
rung is genuinely cheaper than the one above it.

Binning uses quantile edges, not equal-width, so heavy-tailed columns (every
currency column in this dataset) do not collapse into one bin.
"""

from __future__ import annotations

import math

import numpy as np

DEFAULT_BINS = 24
MIN_SAMPLE = 100


def mutual_information(
    x: np.ndarray,
    y: np.ndarray,
    *,
    bins: int = DEFAULT_BINS,
    sample: int | None = 200_000,
    rng_seed: int = 4242,
    normalize: bool = True,
) -> float:
    """Estimated mutual information in nats (or normalized to 0-1).

    `normalize=True` divides by sqrt(H(X)·H(Y)), giving a 0-1 quantity
    comparable to |r|.  Returns NaN when there is not enough data.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = x.size
    if n < MIN_SAMPLE:
        return float("nan")

    if sample is not None and n > sample:
        rng = np.random.default_rng(rng_seed)
        idx = rng.choice(n, size=sample, replace=False)
        x, y = x[idx], y[idx]
        n = sample

    xi = _quantile_bin(x, bins)
    yi = _quantile_bin(y, bins)
    if xi is None or yi is None:
        return float("nan")

    nx, ny = xi.max() + 1, yi.max() + 1
    if nx < 2 or ny < 2:
        return 0.0

    joint = np.zeros((nx, ny), dtype=np.float64)
    np.add.at(joint, (xi, yi), 1.0)
    joint /= n

    px = joint.sum(axis=1)
    py = joint.sum(axis=0)

    nz = joint > 0
    mi = float((joint[nz] * np.log(joint[nz] / np.outer(px, py)[nz])).sum())

    # Miller-Madow: the plug-in estimator is biased upward by roughly
    # (#occupied cells - #occupied rows - #occupied cols + 1) / (2n).
    occupied = int(nz.sum())
    correction = (occupied - int((px > 0).sum()) - int((py > 0).sum()) + 1) / (2.0 * n)
    mi = max(0.0, mi - correction)

    if not normalize:
        return mi

    hx = _entropy(px)
    hy = _entropy(py)
    denom = math.sqrt(hx * hy)
    if denom <= 0:
        return 0.0
    return float(min(1.0, mi / denom))


def _quantile_bin(values: np.ndarray, bins: int) -> np.ndarray | None:
    """Assign values to quantile-edged bins; None when the column is constant."""
    if values.size == 0:
        return None
    lo, hi = values.min(), values.max()
    if lo == hi:
        return None

    qs = np.linspace(0, 1, bins + 1)[1:-1]
    edges = np.unique(np.quantile(values, qs))
    if edges.size == 0:
        # Fewer distinct values than bins: fall back to distinct-value coding.
        _, inverse = np.unique(values, return_inverse=True)
        return inverse.astype(np.int64)
    return np.searchsorted(edges, values, side="right").astype(np.int64)


def _entropy(p: np.ndarray) -> float:
    nz = p > 0
    return float(-(p[nz] * np.log(p[nz])).sum())


def conditional_mutual_information(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    bins: int = 12,
    sample: int | None = 200_000,
) -> float:
    """I(X;Y|Z), estimated by averaging MI within bins of Z.

    Used for confounder screening: when I(X;Y) is large but I(X;Y|Z) collapses,
    Z explains the association and the interpretation should say so.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[mask], y[mask], z[mask]
    if x.size < MIN_SAMPLE * 4:
        return float("nan")

    zi = _quantile_bin(z, bins)
    if zi is None:
        return mutual_information(x, y, bins=bins, sample=sample)

    total = 0.0
    n = x.size
    for level in np.unique(zi):
        sel = zi == level
        weight = sel.sum() / n
        if sel.sum() < MIN_SAMPLE:
            continue
        mi = mutual_information(x[sel], y[sel], bins=max(4, bins // 2), sample=sample)
        if not math.isnan(mi):
            total += weight * mi
    return total
