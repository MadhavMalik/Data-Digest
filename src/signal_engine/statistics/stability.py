"""Cross-subsample stability.

With millions of rows, p-values stop discriminating.  What still discriminates
is whether a relationship holds up when you cut the data into independent
folds and re-measure it.

A real relationship gives a similar coefficient in every fold.  An artifact
driven by a handful of extreme rows gives wildly different coefficients, and
stability collapses.  This is the engine's primary robustness signal and it
feeds directly into candidate ranking.

Folds are assigned by a seeded permutation rather than by row order, because
TLC files arrive time-ordered and contiguous blocks would each cover a
different part of the month — that would measure temporal drift, not
robustness.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class StabilityReport:
    score: float
    fold_values: list[float]
    mean: float
    std: float
    sign_agreement: float
    n_folds: int
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 4),
            "fold_values": [round(v, 5) for v in self.fold_values],
            "mean": round(self.mean, 5),
            "std": round(self.std, 5),
            "sign_agreement": round(self.sign_agreement, 4),
            "n_folds": self.n_folds,
            "note": self.note,
        }


def stability_score(
    x: np.ndarray,
    y: np.ndarray,
    *,
    folds: int = 5,
    rng_seed: int = 90210,
    min_fold_size: int = 50,
) -> StabilityReport:
    """Measure how consistent the correlation is across independent folds.

    The score combines two things a single number should not hide:
      * sign agreement — do all folds agree on the direction?
      * relative spread — is the magnitude consistent?

    score = sign_agreement * (1 - min(1, std / (|mean| + eps)))

    so a relationship that flips sign between folds, or whose magnitude varies
    as much as its mean, scores near zero.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = x.size

    if n < folds * min_fold_size:
        usable = max(2, min(folds, n // max(min_fold_size, 1)))
        if usable < 2:
            return StabilityReport(
                score=float("nan"),
                fold_values=[],
                mean=float("nan"),
                std=float("nan"),
                sign_agreement=float("nan"),
                n_folds=0,
                note=f"only {n} rows; too few for fold-based stability",
            )
        folds = usable

    rng = np.random.default_rng(rng_seed)
    order = rng.permutation(n)
    fold_ids = np.array_split(order, folds)

    values: list[float] = []
    for idx in fold_ids:
        if idx.size < 3:
            continue
        xi, yi = x[idx], y[idx]
        sx, sy = xi.std(), yi.std()
        if sx == 0 or sy == 0:
            continue
        r = float(np.corrcoef(xi, yi)[0, 1])
        if not math.isnan(r):
            values.append(r)

    if len(values) < 2:
        return StabilityReport(
            score=float("nan"),
            fold_values=values,
            mean=float("nan"),
            std=float("nan"),
            sign_agreement=float("nan"),
            n_folds=len(values),
            note="fewer than 2 usable folds (constant values within folds)",
        )

    arr = np.asarray(values)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1))

    positives = int((arr > 0).sum())
    sign_agreement = max(positives, len(arr) - positives) / len(arr)

    relative_spread = std / (abs(mean) + 1e-9)
    score = float(sign_agreement * max(0.0, 1.0 - min(1.0, relative_spread)))

    note = ""
    if sign_agreement < 1.0:
        note = "folds disagree on the direction of the relationship"
    elif relative_spread > 0.5:
        note = "magnitude varies substantially across folds"

    return StabilityReport(
        score=score,
        fold_values=values,
        mean=mean,
        std=std,
        sign_agreement=sign_agreement,
        n_folds=len(values),
        note=note,
    )


def subgroup_consistency(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    min_group_size: int = 500,
    max_groups: int = 12,
) -> dict:
    """Recompute the correlation within each category level.

    A Simpson's-paradox detector: when the pooled correlation has one sign and
    most subgroups have the other, the pooled number is an aggregation
    artifact and the interpretation must say so.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    groups = np.asarray(groups)

    mask = np.isfinite(x) & np.isfinite(y)
    if groups.dtype.kind == "f":
        mask &= np.isfinite(groups)
    x, y, groups = x[mask], y[mask], groups[mask]

    if x.size < min_group_size:
        return {"available": False, "reason": "not enough rows"}

    overall = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else float("nan")

    levels, counts = np.unique(groups, return_counts=True)
    keep = levels[np.argsort(-counts)][:max_groups]

    per_group: list[dict] = []
    for level in keep:
        sel = groups == level
        if sel.sum() < min_group_size:
            continue
        xi, yi = x[sel], y[sel]
        if xi.std() == 0 or yi.std() == 0:
            continue
        r = float(np.corrcoef(xi, yi)[0, 1])
        per_group.append(
            {"level": _py(level), "n": int(sel.sum()), "r": round(r, 5)}
        )

    if len(per_group) < 2 or math.isnan(overall):
        return {"available": False, "reason": "too few usable subgroups", "overall_r": _round(overall)}

    signs = [g["r"] > 0 for g in per_group]
    overall_sign = overall > 0
    disagreeing = sum(1 for s in signs if s != overall_sign)
    reversal = disagreeing > len(signs) / 2

    return {
        "available": True,
        "overall_r": _round(overall),
        "per_group": per_group,
        "groups_disagreeing_with_pooled_sign": disagreeing,
        "simpson_reversal_suspected": reversal,
        "note": (
            "most subgroups show the opposite sign to the pooled correlation; "
            "the pooled value is an aggregation artifact"
            if reversal
            else ""
        ),
    }


def _py(v):
    return v.item() if hasattr(v, "item") else v


def _round(v: float) -> float | None:
    return None if (v is None or math.isnan(v)) else round(float(v), 5)
