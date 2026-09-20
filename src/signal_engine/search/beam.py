"""Progressive screening: spend compute only as evidence becomes promising.

The ladder, with each rung strictly cheaper than the one above it:

    candidates surviving dimensional pruning
        |  (batched materialization, one pass over the columns needed)
    cheap linear/monotonic screening          O(n) per pair, vectorized
        |  keep the top fraction + anything flagged nonlinear
    mutual information                        O(n) but ~20x the constant
        |  keep survivors
    cross-fold stability                      O(n * folds)
        |  keep survivors
    plot rendering + VLM interpretation       seconds and money per item

The point of the ordering is that the expensive rungs only ever see a handful
of candidates.  A flat pipeline that rendered a plot for every candidate would
spend hours and dollars on relationships that do not exist.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from signal_engine.features.dag import ExpressionDAG
from signal_engine.features.transforms import CandidateFeature
from signal_engine.statistics.correlation import RelationshipResult, pairwise_relationship
from signal_engine.statistics.multiple_testing import apply_fdr
from signal_engine.statistics.mutual_information import mutual_information
from signal_engine.statistics.stability import stability_score

# How many of the linear-screened candidates advance to the MI rung.
NONLINEAR_KEEP_FRACTION = 0.25
NONLINEAR_MIN_KEEP = 8
# Candidates below this effect are dropped unless flagged as nonlinear.
SCREEN_EFFECT_FLOOR = 0.05


@dataclass
class ScreenStats:
    materialized: int = 0
    linear_tested: int = 0
    linear_survived: int = 0
    mi_tested: int = 0
    mi_survived: int = 0
    stability_tested: int = 0
    dropped_constant: int = 0
    dropped_insufficient_rows: int = 0
    batches: int = 0

    def to_dict(self) -> dict:
        return {
            "materialized": self.materialized,
            "linear_tested": self.linear_tested,
            "linear_survived": self.linear_survived,
            "mi_tested": self.mi_tested,
            "mi_survived": self.mi_survived,
            "stability_tested": self.stability_tested,
            "dropped_constant": self.dropped_constant,
            "dropped_insufficient_rows": self.dropped_insufficient_rows,
            "batches": self.batches,
        }


@dataclass
class ScreenedCandidate:
    candidate: CandidateFeature
    result: RelationshipResult
    reached_stage: str = "linear"
    vector: np.ndarray | None = field(default=None, repr=False)

    @property
    def effect(self) -> float:
        return self.result.effect


def screen_candidates(
    dag: ExpressionDAG,
    candidates: list[CandidateFeature],
    target_name: str,
    target_vector: np.ndarray,
    *,
    max_tests: int,
    mi_sample: int = 200_000,
    stability_folds: int = 5,
    min_sample: int = 100,
    metrics=None,
    keep_vectors: bool = False,
    batch_size: int = 64,
) -> tuple[list[ScreenedCandidate], ScreenStats]:
    """Run the screening ladder over `candidates` against one target."""
    stats = ScreenStats()
    if not candidates:
        return [], stats

    candidates = candidates[:max_tests]
    screened: list[ScreenedCandidate] = []

    # ---- rung 1: batched materialization + linear screening -------------
    for start in range(0, len(candidates), batch_size):
        batch = candidates[start : start + batch_size]
        try:
            vectors = dag.materialize_many([c.expr for c in batch])
        except Exception:  # noqa: BLE001 - a bad batch must not kill the round
            vectors = []
            for c in batch:
                try:
                    vectors.append(dag.materialize(c.expr))
                except Exception:  # noqa: BLE001
                    vectors.append(np.full(target_vector.shape, np.nan))
        stats.batches += 1
        stats.materialized += len(batch)

        for candidate, vector in zip(batch, vectors):
            if vector.shape != target_vector.shape:
                stats.dropped_insufficient_rows += 1
                continue

            finite = np.isfinite(vector)
            if finite.sum() < min_sample:
                stats.dropped_insufficient_rows += 1
                continue
            if np.nanstd(vector[finite]) == 0:
                stats.dropped_constant += 1
                continue

            result = pairwise_relationship(
                vector,
                target_vector,
                x_name=candidate.name,
                y_name=target_name,
                compute_spearman=True,
                min_sample=min_sample,
            )
            stats.linear_tested += 1
            if metrics is not None:
                metrics.incr("statistical_tests")

            if result.skipped_reason is not None:
                continue
            if result.effect < SCREEN_EFFECT_FLOOR and not result.is_nonlinear_signal:
                continue

            screened.append(
                ScreenedCandidate(
                    candidate=candidate,
                    result=result,
                    vector=vector if keep_vectors else None,
                )
            )
            # Keep the vector for the next rungs regardless; dropped below.
            screened[-1]._vec = vector  # type: ignore[attr-defined]

    stats.linear_survived = len(screened)
    if not screened:
        return [], stats

    # ---- rung 2: mutual information on the promising fraction -----------
    screened.sort(key=lambda s: -s.effect)
    keep_n = max(NONLINEAR_MIN_KEEP, int(len(screened) * NONLINEAR_KEEP_FRACTION))
    mi_pool = screened[:keep_n]
    # Anything flagged as nonlinear gets in regardless of its linear rank —
    # that is the whole point of having an MI rung.
    for s in screened[keep_n:]:
        if s.result.is_nonlinear_signal:
            mi_pool.append(s)

    for s in mi_pool:
        vector = getattr(s, "_vec", None)
        if vector is None:
            continue
        mi = mutual_information(vector, target_vector, sample=mi_sample)
        if not np.isnan(mi):
            s.result.mutual_information = float(mi)
        s.reached_stage = "mutual_information"
        stats.mi_tested += 1
        if metrics is not None:
            metrics.incr("mutual_information_tests")

    stats.mi_survived = sum(1 for s in mi_pool if s.effect >= SCREEN_EFFECT_FLOOR)

    # ---- rung 3: stability on the survivors -----------------------------
    mi_pool.sort(key=lambda s: -s.effect)
    stability_pool = mi_pool[: max(NONLINEAR_MIN_KEEP, keep_n // 2)]
    for s in stability_pool:
        vector = getattr(s, "_vec", None)
        if vector is None:
            continue
        report = stability_score(vector, target_vector, folds=stability_folds)
        if not np.isnan(report.score):
            s.result.stability = report.score
            if report.note:
                s.result.warnings.append(report.note)
            s.result.extra["stability_detail"] = report.to_dict()
        s.reached_stage = "stability"
        stats.stability_tested += 1
        if metrics is not None:
            metrics.incr("stability_tests")

    # ---- FDR across everything tested this round ------------------------
    apply_fdr([s.result for s in screened])

    # Release the held vectors unless the caller asked to keep them.
    for s in screened:
        if keep_vectors:
            s.vector = getattr(s, "_vec", None)
        if hasattr(s, "_vec"):
            delattr(s, "_vec")

    screened.sort(key=lambda s: -s.effect)
    return screened, stats


def deduplicate_findings(
    screened: list[ScreenedCandidate], *, effect_tolerance: float = 0.01
) -> tuple[list[ScreenedCandidate], int]:
    """Collapse candidates that are restatements of the same finding.

    `trip_duration_seconds`, `trip_duration_minutes`, `log(trip_duration_minutes)`
    and `abs(trip_duration_minutes)` against the same target are ONE result
    reported four ways: a monotone rescaling changes Spearman not at all and
    Pearson barely.  Reporting them separately pads the evidence set and
    crowds genuinely different findings out of the expensive stage.

    Grouping is on what was measured, not on syntax: same base-column set, and
    effect sizes within `effect_tolerance`.  The surviving representative is
    the SIMPLEST expression in the group (fewest nodes), because
    `trip_duration_minutes` is a better thing to show a human than
    `log1p(trip_duration_minutes)` when they carry the same signal.
    """
    groups: dict[tuple, list[ScreenedCandidate]] = {}
    for item in screened:
        key = (
            frozenset(item.candidate.expr.columns()),
            round(item.effect / max(effect_tolerance, 1e-9)),
        )
        groups.setdefault(key, []).append(item)

    kept: list[ScreenedCandidate] = []
    dropped = 0
    for members in groups.values():
        if len(members) == 1:
            kept.append(members[0])
            continue
        members.sort(key=lambda s: (s.candidate.expr.size, -s.effect))
        winner = members[0]
        winner.result.extra["equivalent_forms"] = [
            m.candidate.name for m in members[1:][:6]
        ]
        winner.result.warnings.append(
            f"{len(members) - 1} equivalent transformation(s) of the same columns gave the "
            f"same effect within {effect_tolerance}; reporting the simplest form"
        )
        kept.append(winner)
        dropped += len(members) - 1

    kept.sort(key=lambda s: -s.effect)
    return kept, dropped


def select_for_visualization(
    screened: list[ScreenedCandidate],
    *,
    limit: int,
    min_effect: float = 0.1,
    diversity_by_feature: bool = True,
) -> list[ScreenedCandidate]:
    """Pick which findings earn a plot and a VLM call.

    Diversity matters more than raw ranking here: ten variations on
    `fare_amount` tell the user one thing.  At most two findings per base
    feature reach the expensive stage, so the evidence set covers the question
    rather than one corner of it.
    """
    eligible = [s for s in screened if s.effect >= min_effect]
    if not diversity_by_feature:
        return eligible[:limit]

    chosen: list[ScreenedCandidate] = []
    per_feature: dict[str, int] = {}

    for s in eligible:
        base = sorted(s.candidate.expr.columns())
        key = base[0] if base else s.candidate.name
        if per_feature.get(key, 0) >= 2:
            continue
        per_feature[key] = per_feature.get(key, 0) + 1
        chosen.append(s)
        if len(chosen) >= limit:
            break

    # Backfill if diversity left slots unused.
    if len(chosen) < limit:
        seen = {id(c) for c in chosen}
        for s in eligible:
            if id(s) not in seen:
                chosen.append(s)
                if len(chosen) >= limit:
                    break
    return chosen
