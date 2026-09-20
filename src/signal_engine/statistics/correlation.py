"""Pairwise relationship testing.

Policy decisions baked in here, because getting them wrong is how automated
analysis produces confident nonsense:

* **Effect size over p-value.**  At n = 3.5 million, a Pearson r of 0.002 has a
  p-value near zero.  We compute p-values, but ranking and reporting lead with
  |r|, and `is_meaningful` requires an effect floor, not just significance.

* **Never Pearson an identifier.**  A correlation with `PULocationID` is
  arithmetic on arbitrary labels.  Callers must route categoricals to
  `categorical_association` / `grouped_comparison`.  `pairwise_relationship`
  refuses non-quantities outright.

* **Joint masking.**  Two features are compared only on rows where BOTH are
  present, so a correlation can never be computed across misaligned subsets.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
from scipy import stats as sps

# Below this |r| we treat a relationship as noise regardless of p-value.
MEANINGFUL_EFFECT_FLOOR = 0.10
STRONG_EFFECT = 0.5
MIN_SAMPLE = 30


@dataclass
class RelationshipResult:
    """The outcome of testing one pair of features."""

    x_name: str
    y_name: str
    n: int = 0
    pearson_r: float | None = None
    pearson_p: float | None = None
    spearman_rho: float | None = None
    spearman_p: float | None = None
    covariance: float | None = None
    mutual_information: float | None = None
    r_squared: float | None = None
    slope: float | None = None
    intercept: float | None = None
    method: str = "pearson_spearman"
    stability: float | None = None
    q_value: float | None = None
    # eta (grouped comparison) or Cramer's V (categorical pair).  Both live on
    # the same 0-1 scale as |r|, so they rank directly against it.
    eta: float | None = None
    warnings: list[str] = field(default_factory=list)
    skipped_reason: str | None = None
    extra: dict = field(default_factory=dict)

    # ---- summary views --------------------------------------------------
    @property
    def effect(self) -> float:
        """The headline effect magnitude used for ranking."""
        candidates = [
            abs(v)
            for v in (self.pearson_r, self.spearman_rho, self.eta)
            if v is not None and not math.isnan(v)
        ]
        if self.mutual_information is not None and not math.isnan(self.mutual_information):
            # MI is on a different scale; normalise it into a comparable 0-1 band.
            candidates.append(min(1.0, self.mutual_information / 1.5))
        return max(candidates, default=0.0)

    @property
    def direction(self) -> str:
        """Sign of the relationship as the numbers actually state it.

        The critic compares any natural-language claim against THIS, which is
        how "positively correlated" can never be attached to a negative r.
        """
        r = self.pearson_r if self.pearson_r is not None else self.spearman_rho
        if r is None or math.isnan(r):
            return "undetermined"
        if abs(r) < 0.02:
            return "none"
        return "positive" if r > 0 else "negative"

    @property
    def is_meaningful(self) -> bool:
        return self.skipped_reason is None and self.effect >= MEANINGFUL_EFFECT_FLOOR

    @property
    def is_nonlinear_signal(self) -> bool:
        """Weak linear but strong monotonic/MI signal -> worth a nonlinear look."""
        lin = abs(self.pearson_r) if self.pearson_r is not None else 0.0
        mono = abs(self.spearman_rho) if self.spearman_rho is not None else 0.0
        mi = self.mutual_information or 0.0
        return lin < 0.2 and (mono > 0.35 or mi > 0.2)

    def strength_label(self) -> str:
        e = self.effect
        if e >= 0.8:
            return "very strong"
        if e >= STRONG_EFFECT:
            return "strong"
        if e >= 0.3:
            return "moderate"
        if e >= MEANINGFUL_EFFECT_FLOOR:
            return "weak"
        return "negligible"

    def to_dict(self) -> dict:
        return {
            "x": self.x_name,
            "y": self.y_name,
            "n": self.n,
            "pearson_r": _r(self.pearson_r),
            "pearson_p": _p(self.pearson_p),
            "spearman_rho": _r(self.spearman_rho),
            "spearman_p": _p(self.spearman_p),
            "mutual_information": _r(self.mutual_information),
            "r_squared": _r(self.r_squared),
            "slope": _r(self.slope),
            "covariance": _r(self.covariance),
            "effect": round(self.effect, 4),
            "direction": self.direction,
            "strength": self.strength_label(),
            "eta": _r(self.eta),
            "stability": _r(self.stability),
            "q_value": _p(self.q_value),
            "method": self.method,
            "warnings": self.warnings,
            "skipped_reason": self.skipped_reason,
            **({"extra": self.extra} if self.extra else {}),
        }

    def to_compact_text(self) -> str:
        """Dense one-liner for LLM context — exact numbers, no prose."""
        if self.skipped_reason:
            return f"{self.x_name} ~ {self.y_name}: SKIPPED ({self.skipped_reason})"
        bits = [f"n={self.n:,}"]
        if self.pearson_r is not None:
            bits.append(f"pearson={self.pearson_r:+.3f}")
        if self.spearman_rho is not None:
            bits.append(f"spearman={self.spearman_rho:+.3f}")
        if self.mutual_information is not None:
            bits.append(f"MI={self.mutual_information:.3f}")
        if self.stability is not None:
            bits.append(f"stability={self.stability:.2f}")
        if self.q_value is not None:
            bits.append(f"q={self.q_value:.2e}")
        bits.append(f"dir={self.direction}")
        bits.append(self.strength_label())
        line = f"{self.x_name} ~ {self.y_name}: " + " ".join(bits)
        if self.warnings:
            line += " | " + "; ".join(self.warnings)
        return line


def _r(v: float | None) -> float | None:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return round(float(v), 6)


def _p(v: float | None) -> float | None:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return float(f"{v:.6g}")


# ---------------------------------------------------------------------------
# Core pairwise test
# ---------------------------------------------------------------------------


def joint_mask(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Rows where both arrays are finite.  The ONLY masking rule in the engine."""
    return np.isfinite(x) & np.isfinite(y)


def pairwise_relationship(
    x: np.ndarray,
    y: np.ndarray,
    *,
    x_name: str = "x",
    y_name: str = "y",
    compute_spearman: bool = True,
    spearman_sample: int = 200_000,
    min_sample: int = MIN_SAMPLE,
    rng_seed: int = 12345,
) -> RelationshipResult:
    """Pearson + Spearman + OLS slope for two numeric vectors."""
    res = RelationshipResult(x_name=x_name, y_name=y_name)

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape:
        res.skipped_reason = f"shape mismatch: {x.shape} vs {y.shape}"
        return res

    mask = joint_mask(x, y)
    n = int(mask.sum())
    res.n = n
    if n < min_sample:
        res.skipped_reason = f"only {n} jointly-present rows (minimum {min_sample})"
        return res

    xs, ys = x[mask], y[mask]
    dropped = len(x) - n
    if dropped:
        frac = dropped / len(x)
        if frac > 0.5:
            res.warnings.append(
                f"{frac:.0%} of rows dropped for missing values; the surviving subset may be biased"
            )

    sx, sy = xs.std(), ys.std()
    if sx == 0 or sy == 0:
        constant = x_name if sx == 0 else y_name
        res.skipped_reason = f"{constant} is constant over the analysed rows"
        return res

    # Pearson via the covariance form; identical to scipy but avoids a second pass.
    xc, yc = xs - xs.mean(), ys - ys.mean()
    cov = float(xc @ yc) / (n - 1)
    res.covariance = cov
    r = cov / (sx * sy * n / (n - 1))
    r = float(np.clip(r, -1.0, 1.0))
    res.pearson_r = r
    res.pearson_p = _pearson_pvalue(r, n)
    res.r_squared = r * r
    res.slope = cov / (sx * sx * n / (n - 1))
    res.intercept = float(ys.mean() - res.slope * xs.mean())

    if compute_spearman:
        # Spearman ranks cost O(n log n); on millions of rows we sample, and
        # say so, rather than pretending the full-data value was computed.
        if n > spearman_sample:
            rng = np.random.default_rng(rng_seed)
            idx = rng.choice(n, size=spearman_sample, replace=False)
            rho, p = sps.spearmanr(xs[idx], ys[idx])
            res.extra["spearman_sampled_n"] = spearman_sample
        else:
            rho, p = sps.spearmanr(xs, ys)
        if not (isinstance(rho, float) and math.isnan(rho)):
            res.spearman_rho = float(rho)
            res.spearman_p = float(p)

    if res.spearman_rho is not None and abs(res.spearman_rho) - abs(r) > 0.25:
        res.warnings.append(
            "monotonic association is much stronger than the linear one; "
            "the relationship is probably nonlinear"
        )

    return res


def _pearson_pvalue(r: float, n: int) -> float:
    """Two-sided p-value for Pearson r via the t transform."""
    if n <= 2:
        return float("nan")
    if abs(r) >= 1.0:
        return 0.0
    df = n - 2
    t = r * math.sqrt(df / max(1e-300, (1.0 - r * r)))
    return float(2 * sps.t.sf(abs(t), df))


# ---------------------------------------------------------------------------
# Categorical relationships
# ---------------------------------------------------------------------------


def grouped_comparison(
    values: np.ndarray,
    groups: np.ndarray,
    *,
    value_name: str = "value",
    group_name: str = "group",
    max_groups: int = 40,
    labels: dict | None = None,
) -> RelationshipResult:
    """Compare a numeric value ACROSS category levels.

    This is what replaces a (meaningless) Pearson correlation against an
    integer-coded category.  Reports eta-squared — the fraction of variance
    explained by group membership — plus per-group means, which is what a plot
    and an interpretation actually need.
    """
    res = RelationshipResult(x_name=group_name, y_name=value_name, method="grouped_eta_squared")

    values = np.asarray(values, dtype=np.float64)
    groups = np.asarray(groups)
    mask = np.isfinite(values)
    if groups.dtype.kind == "f":
        mask &= np.isfinite(groups)
    values, groups = values[mask], groups[mask]

    res.n = int(values.size)
    if res.n < MIN_SAMPLE:
        res.skipped_reason = f"only {res.n} usable rows"
        return res

    levels, inverse = np.unique(groups, return_inverse=True)
    if len(levels) < 2:
        res.skipped_reason = f"{group_name} has a single level over the analysed rows"
        return res
    if len(levels) > max_groups:
        res.skipped_reason = f"{group_name} has {len(levels)} levels (> {max_groups}); use a binned view"
        return res

    grand = values.mean()
    counts = np.bincount(inverse, minlength=len(levels)).astype(np.float64)
    sums = np.bincount(inverse, weights=values, minlength=len(levels))
    means = sums / np.maximum(counts, 1)

    ss_between = float((counts * (means - grand) ** 2).sum())
    ss_total = float(((values - grand) ** 2).sum())
    eta_sq = ss_between / ss_total if ss_total > 0 else 0.0

    # One-way ANOVA F, for a significance figure alongside the effect size.
    k = len(levels)
    ss_within = ss_total - ss_between
    df_b, df_w = k - 1, res.n - k
    p_value = float("nan")
    if df_w > 0 and ss_within > 0:
        f_stat = (ss_between / df_b) / (ss_within / df_w)
        p_value = float(sps.f.sf(f_stat, df_b, df_w))
        res.extra["f_statistic"] = round(f_stat, 4)

    res.pearson_p = p_value
    res.extra["eta_squared"] = round(eta_sq, 6)
    res.extra["n_levels"] = k
    res.extra["group_means"] = [
        {
            "level": _label(levels[i], labels),
            "n": int(counts[i]),
            "mean": round(float(means[i]), 6),
        }
        for i in np.argsort(-counts)[:max_groups]
    ]
    # eta is the correlation-scale analogue of eta-squared, so it ranks
    # comparably against |r| elsewhere in the engine.
    res.mutual_information = None
    res.spearman_rho = None
    res.pearson_r = None
    res.extra["eta"] = round(math.sqrt(eta_sq), 6)
    res.extra["effect_basis"] = "eta (sqrt of variance explained by group)"
    res.eta = math.sqrt(eta_sq)
    res.warnings.append(
        f"{group_name} is categorical; compared as group means, not as a correlation"
    )
    return res


def categorical_association(
    a: np.ndarray,
    b: np.ndarray,
    *,
    a_name: str = "a",
    b_name: str = "b",
    max_levels: int = 50,
) -> RelationshipResult:
    """Cramer's V between two categorical variables."""
    res = RelationshipResult(x_name=a_name, y_name=b_name, method="cramers_v")

    a = np.asarray(a)
    b = np.asarray(b)
    mask = np.ones(a.shape, dtype=bool)
    for arr in (a, b):
        if arr.dtype.kind == "f":
            mask &= np.isfinite(arr)
    a, b = a[mask], b[mask]
    res.n = int(a.size)
    if res.n < MIN_SAMPLE:
        res.skipped_reason = f"only {res.n} usable rows"
        return res

    la, ia = np.unique(a, return_inverse=True)
    lb, ib = np.unique(b, return_inverse=True)
    if len(la) < 2 or len(lb) < 2:
        res.skipped_reason = "one variable has a single level"
        return res
    if len(la) > max_levels or len(lb) > max_levels:
        res.skipped_reason = f"too many levels ({len(la)} x {len(lb)}); exceeds {max_levels}"
        return res

    table = np.zeros((len(la), len(lb)), dtype=np.float64)
    np.add.at(table, (ia, ib), 1.0)

    chi2, p, _, _ = sps.chi2_contingency(table, correction=False)
    v = math.sqrt((chi2 / res.n) / max(1, min(len(la) - 1, len(lb) - 1)))

    res.pearson_p = float(p)
    res.extra["chi2"] = round(float(chi2), 4)
    res.extra["cramers_v"] = round(v, 6)
    res.extra["shape"] = [len(la), len(lb)]
    res.eta = v
    res.warnings.append("both variables are categorical; association measured with Cramer's V")
    return res


def _label(value, labels: dict | None):
    if labels:
        try:
            key = int(value)
        except (TypeError, ValueError):
            key = value
        if key in labels:
            return f"{value} ({labels[key]})"
    return value.item() if hasattr(value, "item") else value


def rank_results(results: Iterable[RelationshipResult]) -> list[RelationshipResult]:
    """Rank by effect magnitude, then stability, then sample size."""
    return sorted(
        (r for r in results if r.skipped_reason is None),
        key=lambda r: (r.effect, r.stability or 0.0, r.n),
        reverse=True,
    )
