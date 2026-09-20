"""Bounded candidate-feature generation with dimensional pruning.

This module is where the headline optimization lives.  Given N base columns and
4 binary operations, the naive candidate space is O(N^2) per generation and
explodes combinatorially with depth.  Most of it is nonsense:

    fare_amount + trip_distance     dollars + miles      -> rejected
    PULocationID * fare_amount      label * dollars      -> rejected
    fare_amount / trip_distance     dollars per mile     -> KEPT

Rejection happens on the unit algebra alone, before a single row is read.  On
the real NYC yellow-taxi schema this removes the large majority of candidates
at zero I/O cost, and every rejection is counted by reason so the demo panel
can show exactly what the dimensional layer bought.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations

from signal_engine.features.canonicalize import canonicalize, expression_hash, suggest_name
from signal_engine.features.expressions import BinaryOp, Col, Expr, UnaryOp
from signal_engine.profiling.units import Kind, Unit

BINARY_OPS = ("add", "sub", "mul", "div")
UNARY_OPS = ("log", "log1p", "sqrt", "abs")


@dataclass
class CandidateFeature:
    """A derived feature that survived pruning."""

    expr: Expr
    unit: Unit
    name: str
    expr_hash: str
    depth: int
    origin: str = "generated"
    rationale: str = ""

    @property
    def display(self) -> str:
        return self.expr.display()

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "expression": self.display,
            "expr_hash": self.expr_hash,
            "unit": self.unit.to_dict(),
            "depth": self.depth,
            "origin": self.origin,
            "rationale": self.rationale,
        }


@dataclass
class PruneStats:
    """Telemetry for the dimensional-pruning stage."""

    considered: int = 0
    emitted: int = 0
    pruned_by_units: int = 0
    pruned_by_dedup: int = 0
    pruned_by_budget: int = 0
    pruned_by_policy: int = 0
    reasons: Counter = field(default_factory=Counter)

    @property
    def prune_rate(self) -> float:
        return (self.considered - self.emitted) / self.considered if self.considered else 0.0

    def to_dict(self) -> dict:
        return {
            "considered": self.considered,
            "emitted": self.emitted,
            "pruned_by_units": self.pruned_by_units,
            "pruned_by_dedup": self.pruned_by_dedup,
            "pruned_by_budget": self.pruned_by_budget,
            "pruned_by_policy": self.pruned_by_policy,
            "prune_rate": round(self.prune_rate, 4),
            "top_rejection_reasons": dict(self.reasons.most_common(8)),
        }

    def merge(self, other: PruneStats) -> None:
        self.considered += other.considered
        self.emitted += other.emitted
        self.pruned_by_units += other.pruned_by_units
        self.pruned_by_dedup += other.pruned_by_dedup
        self.pruned_by_budget += other.pruned_by_budget
        self.pruned_by_policy += other.pruned_by_policy
        self.reasons.update(other.reasons)


def _short_reason(message: str) -> str:
    """Collapse a unit error into a bucket label for telemetry."""
    m = message.lower()
    if "not defined for identifier" in m:
        return "identifier has no magnitude"
    if "not defined for categorical" in m:
        return "categorical has no magnitude"
    if "not defined for text" in m:
        return "text has no magnitude"
    if "datetime" in m:
        return "datetime arithmetic not meaningful"
    if "cannot add" in m or "cannot subtract" in m:
        return "incompatible units for +/-"
    if "root" in m:
        return "non-integral dimension root"
    return m[:60]


def generate_candidates(
    base_columns: list[str],
    units: dict[str, Unit],
    *,
    binary_ops: tuple[str, ...] = BINARY_OPS,
    unary_ops: tuple[str, ...] = UNARY_OPS,
    max_candidates: int | None = None,
    exclude_hashes: set[str] | None = None,
    positive_only: set[str] | None = None,
    allow_self_ratio: bool = False,
) -> tuple[list[CandidateFeature], PruneStats]:
    """Generate depth-1 derived features over `base_columns`.

    `positive_only` names columns known to be strictly positive; `log` and
    `sqrt` are only proposed for those, since proposing log() of a column with
    negative values generates a feature that is mostly null.
    """
    stats = PruneStats()
    seen: set[str] = set(exclude_hashes or set())
    out: list[CandidateFeature] = []
    positive_only = positive_only or set()

    def emit(expr: Expr, unit: Unit, origin: str, rationale: str) -> bool:
        """Return False when the candidate budget is exhausted."""
        if max_candidates is not None and len(out) >= max_candidates:
            stats.pruned_by_budget += 1
            return False
        canonical = canonicalize(expr)
        h = expression_hash(canonical)
        if h in seen:
            stats.pruned_by_dedup += 1
            return True
        seen.add(h)
        out.append(
            CandidateFeature(
                expr=canonical,
                unit=unit,
                name=suggest_name(canonical),
                expr_hash=h,
                depth=canonical.depth - 1,
                origin=origin,
                rationale=rationale,
            )
        )
        stats.emitted += 1
        return True

    # ---- unary transforms ----------------------------------------------
    for name in base_columns:
        u = units.get(name)
        if u is None or u.kind in {Kind.CATEGORICAL, Kind.IDENTIFIER, Kind.TEXT, Kind.DATETIME}:
            continue
        if u.kind is Kind.BOOLEAN:
            # log/sqrt/abs of a 0/1 flag is either the flag itself or a
            # constant.  Pure noise in the candidate pool.
            stats.considered += len(unary_ops)
            stats.pruned_by_policy += len(unary_ops)
            stats.reasons["unary transform of a boolean flag is degenerate"] += len(unary_ops)
            continue
        for op in unary_ops:
            stats.considered += 1
            if op in {"log", "sqrt"} and name not in positive_only:
                stats.pruned_by_policy += 1
                stats.reasons[f"{op} needs a strictly positive column"] += 1
                continue
            if op == "abs" and name in positive_only:
                # abs() of a strictly positive column IS the column.
                stats.pruned_by_policy += 1
                stats.reasons["abs of an already-positive column is the identity"] += 1
                continue
            expr = UnaryOp(op, Col(name))
            derived, err = expr.unit(units)
            if err or derived is None:
                stats.pruned_by_units += 1
                stats.reasons[_short_reason(err or "unknown")] += 1
                continue
            if not emit(expr, derived, "unary", f"{op} of {name}"):
                return out, stats

    # ---- binary transforms ---------------------------------------------
    for a, b in combinations(base_columns, 2):
        ua, ub = units.get(a), units.get(b)
        if ua is None or ub is None:
            continue
        for op in binary_ops:
            # Non-commutative ops must be tried in both orders.
            orders = [(a, b)] if op in {"add", "mul"} else [(a, b), (b, a)]
            for left, right in orders:
                stats.considered += 1
                expr = BinaryOp(op, Col(left), Col(right))
                derived, err = expr.unit(units)
                if err or derived is None:
                    stats.pruned_by_units += 1
                    stats.reasons[_short_reason(err or "unknown")] += 1
                    continue
                if op == "div" and not allow_self_ratio and derived.dimension.is_dimensionless:
                    # A/B where A and B share units is a unitless ratio.  Useful
                    # sometimes, but it floods the space; gated behind a flag.
                    stats.pruned_by_policy += 1
                    stats.reasons["same-unit ratio (dimensionless)"] += 1
                    continue
                if not emit(expr, derived, "binary", f"{left} {op} {right}"):
                    return out, stats

    return out, stats


def expand_candidate(
    parent: CandidateFeature,
    base_columns: list[str],
    units: dict[str, Unit],
    *,
    binary_ops: tuple[str, ...] = ("mul", "div"),
    max_candidates: int | None = 200,
    exclude_hashes: set[str] | None = None,
) -> tuple[list[CandidateFeature], PruneStats]:
    """Grow one promising candidate by combining it with each base column.

    Only the promising parents from the previous generation are expanded — this
    is the beam-search step that keeps depth-2 and depth-3 tractable.  Additive
    ops are dropped by default here: at depth >= 2 they mostly re-derive linear
    combinations that the covariance shortcut already covers analytically.
    """
    stats = PruneStats()
    seen: set[str] = set(exclude_hashes or set())
    out: list[CandidateFeature] = []
    parent_units = dict(units)
    parent_units[parent.name] = parent.unit

    for name in base_columns:
        u = units.get(name)
        if u is None or u.kind in {Kind.CATEGORICAL, Kind.IDENTIFIER, Kind.TEXT, Kind.DATETIME}:
            continue
        for op in binary_ops:
            orders = [(parent.expr, Col(name))] if op in {"add", "mul"} else [
                (parent.expr, Col(name)),
                (Col(name), parent.expr),
            ]
            for left, right in orders:
                stats.considered += 1
                expr = BinaryOp(op, left, right)
                derived, err = expr.unit(units)
                if err or derived is None:
                    stats.pruned_by_units += 1
                    stats.reasons[_short_reason(err or "unknown")] += 1
                    continue
                canonical = canonicalize(expr)
                h = expression_hash(canonical)
                if h in seen:
                    stats.pruned_by_dedup += 1
                    continue
                if max_candidates is not None and len(out) >= max_candidates:
                    stats.pruned_by_budget += 1
                    return out, stats
                seen.add(h)
                out.append(
                    CandidateFeature(
                        expr=canonical,
                        unit=derived,
                        name=suggest_name(canonical),
                        expr_hash=h,
                        depth=canonical.depth - 1,
                        origin="expansion",
                        rationale=f"expanded {parent.name} with {name} via {op}",
                    )
                )
                stats.emitted += 1

    return out, stats


def theoretical_space_size(n_columns: int, depth: int) -> int:
    """Size of the UNPRUNED candidate space, for the demo metrics panel.

    Depth 1 counts ordered pairs over 4 binary ops plus 4 unary transforms.
    Deeper generations multiply by the per-generation branching factor.  This
    is the number the pruning rate is measured against — it is a count of what
    a brute-force search would have had to consider, not a claim that the
    engine enumerated them.
    """
    if n_columns <= 1 or depth < 1:
        return 0
    per_gen = 4 * n_columns * (n_columns - 1) + 4 * n_columns
    total = per_gen
    frontier = per_gen
    for _ in range(depth - 1):
        frontier = frontier * 2 * n_columns
        total += frontier
    return total
