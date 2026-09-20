"""Functional-form characterization.

A correlation coefficient answers "do these move together?".  It cannot answer
"how?", which is where the actual finding usually lives.

On the NYC taxi data, `corr(distance, fare) = +0.87` is true and useless — it
restates the meter's existence.  The interesting fact is the SHAPE: fare per
mile falls from ~$361/mi on sub-half-mile trips to ~$3.89/mi above 12 miles, a
~90x decay that a scatter or hexbin renders as an undifferentiated cloud.

So this module fits candidate functional forms to the CONDITIONAL MEAN curve
E[y|x] rather than to the raw points.  Binning first is what makes the fit
robust: at 3.5M rows the point cloud is dominated by within-bin variance, and
the shape only becomes visible once you average it out.

Candidate forms (all cheap closed-form least squares):

    linear        y = a + b·x
    logarithmic   y = a + b·log(x)          saturating growth
    power         y = A·x^b                 constant elasticity
    quadratic     y = a + b·x + c·x²        turning points
    inverse       y = a + b/x               hyperbolic decay

The winner is chosen by R² ON THE BINNED MEANS, with a margin requirement so a
more complex form must earn its extra parameter rather than win by rounding.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

MIN_BINS = 8
DEFAULT_BINS = 24
MIN_PER_BIN = 20
# A more complex form must beat the linear baseline by this much R² to be
# declared the better description.  Without it, quadratic always "wins" by
# absorbing noise.
COMPLEXITY_MARGIN = 0.02


@dataclass
class BinnedCurve:
    """The conditional mean curve E[y|x], with uncertainty."""

    x: np.ndarray
    y: np.ndarray
    se: np.ndarray
    counts: np.ndarray
    edges: np.ndarray

    @property
    def n_bins(self) -> int:
        return len(self.x)

    def to_rows(self) -> list[dict]:
        return [
            {
                "bin": i + 1,
                "x_low": float(self.edges[i]),
                "x_high": float(self.edges[i + 1]),
                "x_center": float(self.x[i]),
                "mean_y": float(self.y[i]),
                "se": float(self.se[i]),
                "n": int(self.counts[i]),
            }
            for i in range(self.n_bins)
        ]


@dataclass
class ShapeFit:
    """The best-fitting functional form and what it implies."""

    form: str = "unknown"
    r2: float = 0.0
    linear_r2: float = 0.0
    params: dict = field(default_factory=dict)
    equation: str = ""
    monotonic: str = "none"          # increasing | decreasing | non_monotonic | none
    saturating: bool = False
    turning_point: float | None = None
    curvature_ratio: float | None = None
    description: str = ""
    alternatives: list[tuple[str, float]] = field(default_factory=list)
    curve: BinnedCurve | None = field(default=None, repr=False)

    @property
    def is_nonlinear(self) -> bool:
        """Does a nonlinear form describe this materially better than a line?"""
        return self.form != "linear" and (self.r2 - self.linear_r2) > COMPLEXITY_MARGIN

    def to_dict(self) -> dict:
        return {
            "form": self.form,
            "r2_on_conditional_mean": round(self.r2, 5),
            "linear_r2_on_conditional_mean": round(self.linear_r2, 5),
            "is_nonlinear": self.is_nonlinear,
            "equation": self.equation,
            "monotonic": self.monotonic,
            "saturating": self.saturating,
            "turning_point": self.turning_point,
            "curvature_ratio": self.curvature_ratio,
            "description": self.description,
            "alternatives": [(f, round(r, 4)) for f, r in self.alternatives],
            "n_bins": self.curve.n_bins if self.curve else 0,
        }

    def to_prompt_text(self) -> str:
        """Compact rendering for the interpretation prompt."""
        lines = [f"functional form: {self.form} (R^2 on the conditional mean = {self.r2:.3f})"]
        if self.equation:
            lines.append(f"fitted: {self.equation}")
        if self.is_nonlinear:
            lines.append(
                f"a straight line only reaches R^2 = {self.linear_r2:.3f}, so the relationship "
                f"is materially nonlinear"
            )
        lines.append(f"monotonicity: {self.monotonic}")
        if self.saturating:
            lines.append("the curve SATURATES: it rises but with a steadily decreasing slope")
        if self.turning_point is not None:
            lines.append(f"turning point near x = {self.turning_point:.4g}")
        if self.curvature_ratio is not None:
            lines.append(
                f"slope at the low end is {self.curvature_ratio:.1f}x the slope at the high end"
            )
        if self.description:
            lines.append(self.description)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Binning
# ---------------------------------------------------------------------------


def conditional_mean_curve(
    x: np.ndarray,
    y: np.ndarray,
    *,
    bins: int = DEFAULT_BINS,
    min_per_bin: int = MIN_PER_BIN,
    clip_quantiles: tuple[float, float] | None = (0.001, 0.999),
) -> BinnedCurve | None:
    """Compute E[y|x] over quantile bins of x.

    Quantile bins, not equal-width: every currency and distance column here is
    heavy-tailed, and equal-width bins would put 99% of the data in bin 1.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < bins * min_per_bin:
        bins = max(MIN_BINS, x.size // max(min_per_bin, 1))
    if x.size < MIN_BINS * min_per_bin:
        return None

    if clip_quantiles:
        lo, hi = np.quantile(x, clip_quantiles)
        if hi > lo:
            keep = (x >= lo) & (x <= hi)
            x, y = x[keep], y[keep]

    edges = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)))
    if edges.size < MIN_BINS:
        return None

    idx = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, edges.size - 2)

    centers, means, ses, counts, kept_edges = [], [], [], [], []
    for b in range(edges.size - 1):
        sel = idx == b
        n = int(sel.sum())
        if n < min_per_bin:
            continue
        yv = y[sel]
        centers.append(float(np.median(x[sel])))
        means.append(float(yv.mean()))
        ses.append(float(yv.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0)
        counts.append(n)
        kept_edges.append((edges[b], edges[b + 1]))

    if len(centers) < MIN_BINS:
        return None

    flat_edges = np.array([e[0] for e in kept_edges] + [kept_edges[-1][1]])
    return BinnedCurve(
        x=np.array(centers), y=np.array(means), se=np.array(ses),
        counts=np.array(counts), edges=flat_edges,
    )


# ---------------------------------------------------------------------------
# Form fitting
# ---------------------------------------------------------------------------


def _r2(y: np.ndarray, pred: np.ndarray, weights: np.ndarray | None = None) -> float:
    if weights is None:
        weights = np.ones_like(y)
    mean = np.average(y, weights=weights)
    ss_res = float(np.sum(weights * (y - pred) ** 2))
    ss_tot = float(np.sum(weights * (y - mean) ** 2))
    if ss_tot <= 0:
        return 0.0
    return max(0.0, 1.0 - ss_res / ss_tot)


def _wls(design: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray | None:
    """Weighted least squares; bins are weighted by their sample count."""
    try:
        sw = np.sqrt(w)
        beta, *_ = np.linalg.lstsq(design * sw[:, None], y * sw, rcond=None)
        return beta
    except np.linalg.LinAlgError:
        return None


def fit_shape(curve: BinnedCurve) -> ShapeFit:
    """Fit candidate forms to a conditional-mean curve and pick the winner."""
    x, y, w = curve.x, curve.y, curve.counts.astype(np.float64)
    fits: list[tuple[str, float, dict, str]] = []

    # linear
    beta = _wls(np.column_stack([np.ones_like(x), x]), y, w)
    if beta is not None:
        r2 = _r2(y, beta[0] + beta[1] * x, w)
        fits.append(("linear", r2, {"a": beta[0], "b": beta[1]},
                     f"y = {beta[0]:.4g} + {beta[1]:.4g}·x"))
    linear_r2 = fits[0][1] if fits else 0.0

    # quadratic
    beta = _wls(np.column_stack([np.ones_like(x), x, x**2]), y, w)
    if beta is not None:
        r2 = _r2(y, beta[0] + beta[1] * x + beta[2] * x**2, w)
        fits.append(("quadratic", r2, {"a": beta[0], "b": beta[1], "c": beta[2]},
                     f"y = {beta[0]:.4g} + {beta[1]:.4g}·x + {beta[2]:.4g}·x²"))

    # logarithmic (x must be positive)
    if np.all(x > 0):
        lx = np.log(x)
        beta = _wls(np.column_stack([np.ones_like(x), lx]), y, w)
        if beta is not None:
            r2 = _r2(y, beta[0] + beta[1] * lx, w)
            fits.append(("logarithmic", r2, {"a": beta[0], "b": beta[1]},
                         f"y = {beta[0]:.4g} + {beta[1]:.4g}·ln(x)"))

        # inverse
        beta = _wls(np.column_stack([np.ones_like(x), 1.0 / x]), y, w)
        if beta is not None:
            r2 = _r2(y, beta[0] + beta[1] / x, w)
            fits.append(("inverse", r2, {"a": beta[0], "b": beta[1]},
                         f"y = {beta[0]:.4g} + {beta[1]:.4g}/x"))

        # power law (fit in log-log, score in the original space)
        if np.all(y > 0):
            beta = _wls(np.column_stack([np.ones_like(x), lx]), np.log(y), w)
            if beta is not None:
                pred = np.exp(beta[0]) * x ** beta[1]
                r2 = _r2(y, pred, w)
                fits.append(("power", r2, {"A": float(np.exp(beta[0])), "b": beta[1]},
                             f"y = {np.exp(beta[0]):.4g}·x^{beta[1]:.4g}"))

    if not fits:
        return ShapeFit(curve=curve)

    fits.sort(key=lambda f: -f[1])

    # A more complex form must clear the margin over linear to be declared best.
    best = fits[0]
    if best[0] != "linear" and (best[1] - linear_r2) <= COMPLEXITY_MARGIN:
        best = next((f for f in fits if f[0] == "linear"), best)

    fit = ShapeFit(
        form=best[0], r2=best[1], linear_r2=linear_r2, params=best[2],
        equation=best[3], curve=curve,
        alternatives=[(f[0], f[1]) for f in fits if f[0] != best[0]][:4],
    )
    _characterize(fit, curve)
    return fit


def _characterize(fit: ShapeFit, curve: BinnedCurve) -> None:
    """Describe the curve in terms a person (or a model) can act on."""
    x, y = curve.x, curve.y
    diffs = np.diff(y)

    up = float((diffs > 0).sum()) / max(len(diffs), 1)
    if up >= 0.85:
        fit.monotonic = "increasing"
    elif up <= 0.15:
        fit.monotonic = "decreasing"
    else:
        fit.monotonic = "non_monotonic"

    # Compare the slope over the first third against the last third.
    third = max(2, len(x) // 3)
    lo_slope = (y[third] - y[0]) / max(x[third] - x[0], 1e-12)
    hi_slope = (y[-1] - y[-third]) / max(x[-1] - x[-third], 1e-12)
    if abs(hi_slope) > 1e-12:
        ratio = abs(lo_slope) / abs(hi_slope)
        fit.curvature_ratio = float(ratio)
        # Same-sign slopes that shrink materially = saturation.
        if lo_slope * hi_slope > 0 and ratio > 1.8:
            fit.saturating = True

    if fit.form == "quadratic":
        b, c = fit.params.get("b", 0.0), fit.params.get("c", 0.0)
        if abs(c) > 1e-15:
            turn = -b / (2 * c)
            if x.min() < turn < x.max():
                fit.turning_point = float(turn)

    parts = []
    if fit.monotonic == "increasing":
        parts.append("y rises with x")
    elif fit.monotonic == "decreasing":
        parts.append("y falls as x rises")
    else:
        parts.append("y does not move in one consistent direction")

    if fit.saturating and fit.curvature_ratio:
        parts.append(
            f"but the effect flattens out: the slope over the low range is about "
            f"{fit.curvature_ratio:.1f}x the slope over the high range"
        )
    if fit.turning_point is not None:
        parts.append(f"with a turning point near x = {fit.turning_point:.4g}")
    if fit.is_nonlinear:
        parts.append(
            f"a {fit.form} form fits the conditional mean at R^2 = {fit.r2:.3f} versus "
            f"{fit.linear_r2:.3f} for a straight line"
        )

    span = ""
    if len(y) >= 2 and abs(y[0]) > 1e-12:
        span = f" Across the measured range the mean of y goes from {y[0]:.4g} to {y[-1]:.4g}."
    fit.description = ", ".join(parts) + "." + span


def describe_relationship(
    x: np.ndarray, y: np.ndarray, *, bins: int = DEFAULT_BINS
) -> ShapeFit | None:
    """Convenience: bin, fit, and characterize in one call."""
    curve = conditional_mean_curve(x, y, bins=bins)
    if curve is None:
        return None
    return fit_shape(curve)
