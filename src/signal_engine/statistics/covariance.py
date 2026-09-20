"""The sufficient-statistics shortcut for linear combinations.

Testing a linear combination normally costs a pass over N rows.  But for
z = aᵀX and w = bᵀX:

    Cov(z, w) = aᵀ Σ b
    Var(z)    = aᵀ Σ a
    corr(z,w) = aᵀΣb / sqrt(aᵀΣa · bᵀΣb)

So once Σ (a k×k covariance matrix over the k base variables) is computed ONCE,
any number of linear combinations can be evaluated with k×k arithmetic instead
of N-row arithmetic.  With N = 3.5M and k = 20, that is a ~10,000x reduction
per candidate.

**The correctness condition, which is the whole reason this needs care.**
The identity above holds only when z and w are computed over the SAME rows that
Σ was estimated on.  If Σ is built pairwise-complete (each entry using whatever
rows that pair happens to share), the resulting matrix is not the covariance of
any single dataset — it need not even be positive semi-definite — and the
algebra silently returns numbers that match no real computation.

So `build_covariance_model` uses COMPLETE-CASE rows only: rows where every
tracked variable is present.  It records the retained fraction, and
`is_valid()` refuses the shortcut when too much data was dropped, forcing the
caller back to direct computation.  That trade is deliberate: a slower correct
answer beats a fast meaningless one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from signal_engine.statistics.correlation import RelationshipResult

# Below this share of complete cases the shortcut is refused: the complete-case
# subset is too small a slice of the data to speak for it.
MIN_COMPLETE_FRACTION = 0.5


@dataclass
class CovarianceModel:
    """Σ over a fixed set of base variables, estimated on complete cases."""

    names: list[str]
    mean: np.ndarray
    cov: np.ndarray
    n_complete: int
    n_total: int
    index: dict[str, int] = field(default_factory=dict)
    build_seconds: float = 0.0

    def __post_init__(self) -> None:
        if not self.index:
            self.index = {n: i for i, n in enumerate(self.names)}

    # ---- validity -------------------------------------------------------
    @property
    def complete_fraction(self) -> float:
        return self.n_complete / self.n_total if self.n_total else 0.0

    def is_valid(self, min_complete_fraction: float = MIN_COMPLETE_FRACTION) -> bool:
        return (
            self.n_complete >= 30
            and self.complete_fraction >= min_complete_fraction
            and np.all(np.isfinite(self.cov))
        )

    def invalid_reason(self, min_complete_fraction: float = MIN_COMPLETE_FRACTION) -> str | None:
        if self.n_complete < 30:
            return f"only {self.n_complete} complete cases"
        if self.complete_fraction < min_complete_fraction:
            return (
                f"complete cases are {self.complete_fraction:.1%} of rows "
                f"(< {min_complete_fraction:.0%}); the shortcut would describe a biased subset"
            )
        if not np.all(np.isfinite(self.cov)):
            return "covariance matrix contains non-finite entries"
        return None

    # ---- vector helpers -------------------------------------------------
    def vector(self, weights: dict[str, float]) -> np.ndarray:
        """Turn {'trip_distance': 2, 'fare_amount': -1} into a weight vector."""
        v = np.zeros(len(self.names), dtype=np.float64)
        for name, w in weights.items():
            if name not in self.index:
                raise KeyError(f"{name!r} is not part of this covariance model")
            v[self.index[name]] = float(w)
        return v

    # ---- the shortcut ---------------------------------------------------
    def covariance_of(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(a @ self.cov @ b)

    def variance_of(self, a: np.ndarray) -> float:
        return float(a @ self.cov @ a)

    def correlation_of(self, a: np.ndarray, b: np.ndarray) -> float:
        va, vb = self.variance_of(a), self.variance_of(b)
        if va <= 0 or vb <= 0:
            return float("nan")
        return float(np.clip(self.covariance_of(a, b) / math.sqrt(va * vb), -1.0, 1.0))

    def mean_of(self, a: np.ndarray) -> float:
        return float(a @ self.mean)

    def pairwise_correlation_matrix(self) -> np.ndarray:
        sd = np.sqrt(np.clip(np.diag(self.cov), 0, None))
        denom = np.outer(sd, sd)
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.where(denom > 0, self.cov / denom, np.nan)
        return np.clip(corr, -1.0, 1.0)

    def to_dict(self) -> dict:
        return {
            "names": self.names,
            "n_complete": self.n_complete,
            "n_total": self.n_total,
            "complete_fraction": round(self.complete_fraction, 6),
            "valid": self.is_valid(),
            "invalid_reason": self.invalid_reason(),
            "build_seconds": round(self.build_seconds, 3),
        }


def build_covariance_model(
    data: dict[str, np.ndarray],
    *,
    n_total: int | None = None,
) -> CovarianceModel:
    """Estimate Σ over `data`'s variables using complete cases only."""
    import time

    started = time.time()
    names = list(data)
    if not names:
        raise ValueError("no variables supplied")

    matrix = np.column_stack([np.asarray(data[n], dtype=np.float64) for n in names])
    total = n_total if n_total is not None else matrix.shape[0]

    complete = np.all(np.isfinite(matrix), axis=1)
    n_complete = int(complete.sum())

    if n_complete < 2:
        k = len(names)
        return CovarianceModel(
            names=names,
            mean=np.full(k, np.nan),
            cov=np.full((k, k), np.nan),
            n_complete=n_complete,
            n_total=total,
            build_seconds=time.time() - started,
        )

    sub = matrix[complete]
    mean = sub.mean(axis=0)
    centered = sub - mean
    cov = (centered.T @ centered) / (n_complete - 1)

    return CovarianceModel(
        names=names,
        mean=mean,
        cov=cov,
        n_complete=n_complete,
        n_total=total,
        build_seconds=time.time() - started,
    )


def linear_combination_correlation(
    model: CovarianceModel,
    weights_a: dict[str, float],
    weights_b: dict[str, float],
    *,
    name_a: str = "z",
    name_b: str = "w",
) -> RelationshipResult:
    """Correlate two linear combinations using Σ alone (no row scan)."""
    res = RelationshipResult(x_name=name_a, y_name=name_b, method="covariance_shortcut")

    reason = model.invalid_reason()
    if reason:
        res.skipped_reason = f"covariance shortcut unavailable: {reason}"
        return res

    a = model.vector(weights_a)
    b = model.vector(weights_b)

    va, vb = model.variance_of(a), model.variance_of(b)
    if va <= 0 or vb <= 0:
        res.skipped_reason = "one linear combination is constant over complete cases"
        return res

    res.n = model.n_complete
    res.covariance = model.covariance_of(a, b)
    r = model.correlation_of(a, b)
    res.pearson_r = r
    res.r_squared = r * r
    res.slope = res.covariance / va
    res.extra["computed_without_row_scan"] = True
    res.extra["complete_fraction"] = round(model.complete_fraction, 6)

    from signal_engine.statistics.correlation import _pearson_pvalue

    res.pearson_p = _pearson_pvalue(r, model.n_complete)

    if model.complete_fraction < 1.0:
        res.warnings.append(
            f"computed on {model.complete_fraction:.1%} complete-case rows "
            f"({model.n_complete:,} of {model.n_total:,})"
        )
    return res


def direct_linear_combination_correlation(
    data: dict[str, np.ndarray],
    weights_a: dict[str, float],
    weights_b: dict[str, float],
    *,
    name_a: str = "z",
    name_b: str = "w",
) -> RelationshipResult:
    """Reference implementation: build both combinations and correlate directly.

    Used as the fallback when the shortcut is invalid, and as the oracle the
    shortcut is unit-tested against.
    """
    from signal_engine.statistics.correlation import pairwise_relationship

    def combine(weights: dict[str, float]) -> np.ndarray:
        arrays = [np.asarray(data[n], dtype=np.float64) * w for n, w in weights.items()]
        return np.sum(arrays, axis=0)

    z = combine(weights_a)
    w = combine(weights_b)
    res = pairwise_relationship(z, w, x_name=name_a, y_name=name_b, compute_spearman=False)
    res.method = "direct_row_scan"
    return res
