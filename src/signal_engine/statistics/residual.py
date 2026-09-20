"""Residual analysis — searching what the obvious drivers do NOT explain.

The core problem with marginal correlation search on a dataset like this:

    corr(trip_distance, fare_amount) = +0.87

is true, dominant, and worthless.  It restates that taxis have meters.  Because
the engine ranks by effect size, that relationship and its many algebraic
cousins crowd out everything else, and the "findings" become a list of
tautologies.

The fix is to ask a different question.  Fit a baseline from the obvious
drivers, subtract it, and search what explains the REMAINDER:

    fare ~ distance + duration           ->  R² = 0.779
    residual std                          ->  $8.15 of unexplained fare
    what explains the residual?           ->  RatecodeID, eta = 0.638

    Newark             +$21.23 above the distance/duration prediction
    Nassau/Westchester +$34.15
    Negotiated fare    +$32.99
    Standard rate       -$1.45

THAT is a finding: after accounting for how far and how long you travelled,
these rate codes cost dramatically more.  It is invisible to marginal
correlation because RatecodeID barely correlates with fare on its own — the
signal only appears once distance is held constant.

Design notes:

* The baseline is deliberately SIMPLE (OLS on a few strong predictors).  A
  flexible model would absorb the very structure we are trying to surface.
* Baseline predictors are chosen from the *empirical* drivers, never from
  accounting components of the target — regressing `total_amount` on
  `fare_amount` would leave a residual of pure rounding.
* Everything is reported in the target's own units, because "+$34.15" is
  actionable and "eta = 0.638" is not.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

MIN_ROWS = 500
MIN_GROUP = 100
MAX_LEVELS = 40


@dataclass
class BaselineModel:
    """An OLS baseline over the obvious drivers."""

    predictors: list[str]
    coefficients: dict[str, float]
    intercept: float
    r_squared: float
    residual_std: float
    target_std: float
    n: int
    mask: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    residuals: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))

    @property
    def variance_explained_pct(self) -> float:
        return 100.0 * self.r_squared

    @property
    def unexplained_pct(self) -> float:
        return 100.0 * (1.0 - self.r_squared)

    def to_dict(self) -> dict:
        return {
            "predictors": self.predictors,
            "coefficients": {k: round(v, 6) for k, v in self.coefficients.items()},
            "intercept": round(self.intercept, 6),
            "r_squared": round(self.r_squared, 5),
            "variance_explained_pct": round(self.variance_explained_pct, 2),
            "residual_std": round(self.residual_std, 4),
            "target_std": round(self.target_std, 4),
            "n": self.n,
        }

    def equation(self, target: str) -> str:
        terms = " + ".join(f"{c:.4g}·{name}" for name, c in self.coefficients.items())
        return f"{target} ≈ {self.intercept:.4g} + {terms}"

    def to_prompt_text(self, target: str) -> str:
        return (
            f"BASELINE: {self.equation(target)}\n"
            f"  explains {self.variance_explained_pct:.1f}% of the variance in {target} "
            f"(R² = {self.r_squared:.4f}, n = {self.n:,})\n"
            f"  unexplained spread: {self.residual_std:.4g} "
            f"(vs {self.target_std:.4g} before the baseline)\n"
            f"  the findings below concern what is LEFT after this baseline is removed"
        )


@dataclass
class ResidualDriver:
    """Something that explains part of the residual."""

    name: str
    kind: str                    # "categorical" | "numeric"
    effect: float                # eta for categorical, |r| for numeric
    n: int
    # Categorical
    level_effects: list[dict] = field(default_factory=list)
    spread: float | None = None  # max - min of level means, in target units
    # Numeric
    pearson_r: float | None = None
    spearman_rho: float | None = None
    slope: float | None = None
    shape: dict | None = None

    @property
    def is_meaningful(self) -> bool:
        return self.effect >= 0.08 and self.n >= MIN_ROWS

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "effect": round(self.effect, 5),
            "n": self.n,
            "spread_in_target_units": round(self.spread, 4) if self.spread is not None else None,
            "level_effects": self.level_effects,
            "pearson_r": round(self.pearson_r, 5) if self.pearson_r is not None else None,
            "spearman_rho": round(self.spearman_rho, 5) if self.spearman_rho is not None else None,
            "slope": round(self.slope, 6) if self.slope is not None else None,
            "shape": self.shape,
        }

    def to_prompt_text(self, target: str, unit: str = "") -> str:
        u = f" {unit}" if unit else ""
        if self.kind == "categorical":
            lines = [
                f"{self.name}: explains residual {target} with eta = {self.effect:.3f} "
                f"(n = {self.n:,})",
                f"  spread across levels: {self.spread:.4g}{u}",
                "  level effects, expressed as deviation from the baseline prediction:",
            ]
            for lv in self.level_effects[:10]:
                lines.append(
                    f"    {lv['label']}: {lv['mean_residual']:+.4g}{u} (n = {lv['n']:,})"
                )
            return "\n".join(lines)
        lines = [
            f"{self.name}: correlates with the residual at r = {self.pearson_r:+.4f}, "
            f"rho = {self.spearman_rho:+.4f} (n = {self.n:,})"
        ]
        if self.slope is not None:
            lines.append(f"  slope: {self.slope:+.4g}{u} per unit of {self.name}")
        if self.shape:
            lines.append(f"  shape: {self.shape.get('description', '')}")
        return "\n".join(lines)


@dataclass
class ResidualAnalysis:
    target: str
    baseline: BaselineModel
    drivers: list[ResidualDriver] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)

    def meaningful(self) -> list[ResidualDriver]:
        return [d for d in self.drivers if d.is_meaningful]

    def to_dict(self) -> dict:
        return {
            "target": self.target,
            "baseline": self.baseline.to_dict(),
            "drivers": [d.to_dict() for d in self.drivers],
            "meaningful_driver_count": len(self.meaningful()),
            "skipped": [{"name": n, "reason": r} for n, r in self.skipped],
        }

    def to_prompt_text(self, unit: str = "") -> str:
        parts = [self.baseline.to_prompt_text(self.target), ""]
        if not self.meaningful():
            parts.append("No variable explained a meaningful share of the residual.")
            return "\n".join(parts)
        parts.append("WHAT EXPLAINS THE REMAINDER (ranked):")
        for d in self.meaningful()[:8]:
            parts.append(d.to_prompt_text(self.target, unit))
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------


def build_baseline(
    predictors: dict[str, np.ndarray],
    target: np.ndarray,
    *,
    target_name: str = "y",
) -> BaselineModel | None:
    """Fit an OLS baseline of `target` on `predictors` over complete cases."""
    if not predictors:
        return None

    names = list(predictors)
    cols = [np.asarray(predictors[n], dtype=np.float64) for n in names]
    y = np.asarray(target, dtype=np.float64)

    stacked = np.column_stack(cols + [y])
    mask = np.all(np.isfinite(stacked), axis=1)
    n = int(mask.sum())
    if n < MIN_ROWS:
        return None

    X = np.column_stack([np.ones(n)] + [c[mask] for c in cols])
    yv = y[mask]

    try:
        beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
    except np.linalg.LinAlgError:
        return None

    fitted = X @ beta
    residuals = yv - fitted
    target_var = float(yv.var())
    r2 = 1.0 - float(residuals.var()) / target_var if target_var > 0 else 0.0

    return BaselineModel(
        predictors=names,
        coefficients={name: float(b) for name, b in zip(names, beta[1:])},
        intercept=float(beta[0]),
        r_squared=max(0.0, r2),
        residual_std=float(residuals.std()),
        target_std=float(yv.std()),
        n=n,
        mask=mask,
        residuals=residuals,
    )


# ---------------------------------------------------------------------------
# Residual drivers
# ---------------------------------------------------------------------------


def categorical_residual_effect(
    residuals: np.ndarray,
    groups: np.ndarray,
    *,
    name: str,
    labels: dict | None = None,
    max_levels: int = MAX_LEVELS,
    min_group: int = MIN_GROUP,
) -> ResidualDriver | None:
    """How much of the residual does group membership explain?"""
    residuals = np.asarray(residuals, dtype=np.float64)
    groups = np.asarray(groups)

    mask = np.isfinite(residuals)
    if groups.dtype.kind == "f":
        mask &= np.isfinite(groups)
    r, g = residuals[mask], groups[mask]
    if r.size < MIN_ROWS:
        return None

    levels, inverse = np.unique(g, return_inverse=True)
    if len(levels) < 2 or len(levels) > max_levels:
        return None

    counts = np.bincount(inverse, minlength=len(levels)).astype(np.float64)
    sums = np.bincount(inverse, weights=r, minlength=len(levels))
    means = sums / np.maximum(counts, 1)

    grand = float(r.mean())
    ss_between = float((counts * (means - grand) ** 2).sum())
    ss_total = float(((r - grand) ** 2).sum())
    eta = math.sqrt(ss_between / ss_total) if ss_total > 0 else 0.0

    keep = counts >= min_group
    if keep.sum() < 2:
        return None

    level_effects = []
    for i in np.argsort(-np.abs(means * keep)):
        if not keep[i]:
            continue
        level_effects.append({
            "level": _py(levels[i]),
            "label": _label(levels[i], labels),
            "n": int(counts[i]),
            "mean_residual": round(float(means[i]), 5),
        })

    shown = [m for m, k in zip(means, keep) if k]
    return ResidualDriver(
        name=name, kind="categorical", effect=eta, n=int(r.size),
        level_effects=level_effects[:max_levels],
        spread=float(max(shown) - min(shown)),
    )


def numeric_residual_effect(
    residuals: np.ndarray,
    values: np.ndarray,
    *,
    name: str,
    with_shape: bool = True,
) -> ResidualDriver | None:
    """How much of the residual does a numeric variable explain?"""
    from signal_engine.statistics.correlation import pairwise_relationship

    result = pairwise_relationship(
        np.asarray(values, dtype=np.float64),
        np.asarray(residuals, dtype=np.float64),
        x_name=name, y_name="residual",
    )
    if result.skipped_reason is not None:
        return None

    shape = None
    if with_shape:
        from signal_engine.statistics.shape import describe_relationship

        fit = describe_relationship(values, residuals)
        if fit is not None:
            shape = fit.to_dict()

    return ResidualDriver(
        name=name, kind="numeric", effect=result.effect, n=result.n,
        pearson_r=result.pearson_r, spearman_rho=result.spearman_rho,
        slope=result.slope, shape=shape,
    )


def analyze_residual(
    *,
    target_name: str,
    target: np.ndarray,
    baseline_predictors: dict[str, np.ndarray],
    categorical_candidates: dict[str, np.ndarray] | None = None,
    numeric_candidates: dict[str, np.ndarray] | None = None,
    category_labels: dict[str, dict] | None = None,
) -> ResidualAnalysis | None:
    """Fit the baseline, then rank what explains what it leaves behind."""
    baseline = build_baseline(baseline_predictors, target, target_name=target_name)
    if baseline is None:
        return None

    analysis = ResidualAnalysis(target=target_name, baseline=baseline)
    mask, residuals = baseline.mask, baseline.residuals
    labels = category_labels or {}

    for name, values in (categorical_candidates or {}).items():
        if name in baseline.predictors:
            analysis.skipped.append((name, "already in the baseline"))
            continue
        arr = np.asarray(values)
        if arr.shape[0] != mask.shape[0]:
            analysis.skipped.append((name, "length mismatch"))
            continue
        driver = categorical_residual_effect(
            residuals, arr[mask], name=name, labels=labels.get(name)
        )
        if driver is None:
            analysis.skipped.append((name, "too few levels, too many levels, or too few rows"))
        else:
            analysis.drivers.append(driver)

    for name, values in (numeric_candidates or {}).items():
        if name in baseline.predictors:
            analysis.skipped.append((name, "already in the baseline"))
            continue
        arr = np.asarray(values, dtype=np.float64)
        if arr.shape[0] != mask.shape[0]:
            analysis.skipped.append((name, "length mismatch"))
            continue
        driver = numeric_residual_effect(residuals, arr[mask], name=name)
        if driver is None:
            analysis.skipped.append((name, "not testable against the residual"))
        else:
            analysis.drivers.append(driver)

    analysis.drivers.sort(key=lambda d: -d.effect)
    return analysis


def _py(value):
    return value.item() if hasattr(value, "item") else value


def _label(value, labels: dict | None) -> str:
    if labels:
        try:
            key = int(value)
        except (TypeError, ValueError):
            key = value
        if key in labels:
            return f"{key} ({labels[key]})"
    return str(_py(value))
