"""Derived columns and the configurable "analysis view".

Two jobs:

1. Materialize features the arithmetic grammar cannot express, because a
   timestamp is not a number: trip duration, pickup hour, day of week.  Once
   materialized they are ordinary columns with ordinary units, so there is
   exactly one evaluation path for everything downstream.

2. Build the analysis view — a documented, configurable row filter.

On (2) the rule is: NOTHING IS SILENTLY DELETED.  Every filter is named,
its rationale is recorded, and the number of rows it excludes is counted and
reported.  A raw TLC file contains a 269,000-mile taxi trip and negative
fares; excluding them is correct, but hiding that you did is how an analysis
becomes untrustworthy.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import polars as pl

from signal_engine.profiling import units as U
from signal_engine.profiling.profiler import DatasetProfile
from signal_engine.profiling.semantic_types import SemanticType
from signal_engine.profiling.units import Unit

SECONDS_PER_HOUR = 3600.0


@dataclass(frozen=True)
class FilterRule:
    """One named, justified row filter."""

    name: str
    expression: str
    rationale: str

    def to_polars(self, available: set[str]) -> pl.Expr | None:
        """Compile to a Polars predicate, or None when its columns are absent."""
        needed = _referenced_columns(self.expression)
        if not needed.issubset(available):
            return None
        return _compile_predicate(self.expression)


@dataclass
class AnalysisView:
    frame: pl.LazyFrame
    applied: list[FilterRule]
    skipped: list[FilterRule]
    derived_units: dict[str, Unit]
    derived_descriptions: dict[str, str]
    # derived column -> the base columns it was computed from.  This is what
    # lets the engine refuse to "discover" that (total_amount/d) * d equals
    # total_amount.  See `derivation_closure`.
    derived_provenance: dict[str, set[str]] = field(default_factory=dict)
    # Derived columns that are exact linear rescalings of another derived
    # column (seconds vs minutes).  They stay available for filters and for
    # display, but they are kept OUT of the candidate pool: a rescale carries
    # no information the original does not, and including it doubles the
    # search space for nothing.
    redundant_derived: set[str] = field(default_factory=set)
    report: dict = field(default_factory=dict)

    @property
    def view_hash(self) -> str:
        payload = json.dumps(
            {"filters": [r.expression for r in self.applied]}, sort_keys=True
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Derived column definitions
# ---------------------------------------------------------------------------


def derive_tlc_columns(
    lf: pl.LazyFrame, profile: DatasetProfile
) -> tuple[pl.LazyFrame, dict[str, Unit], dict[str, str], dict[str, set[str]], set[str]]:
    """Add NYC-TLC derived columns where the source columns exist.

    Also returns a provenance map recording which base columns each derived
    column was computed from.  Without it the transformation search happily
    "discovers" that `(total_amount / trip_distance) * trip_distance`
    correlates 1.000 with `total_amount` -- an algebraic identity dressed up
    as a finding.
    """
    names = set(lf.collect_schema().names())
    units: dict[str, Unit] = {}
    descriptions: dict[str, str] = {}
    provenance: dict[str, set[str]] = {}
    redundant: set[str] = set()
    additions: list[pl.Expr] = []

    pickup, dropoff = _timestamp_pair(profile, names)

    if pickup and dropoff:
        duration_s = (pl.col(dropoff) - pl.col(pickup)).dt.total_seconds().cast(pl.Float64)
        additions.append(duration_s.alias("trip_duration_seconds"))
        additions.append((duration_s / 60.0).alias("trip_duration_minutes"))
        units["trip_duration_seconds"] = U.SECONDS
        units["trip_duration_minutes"] = U.MINUTES
        provenance["trip_duration_seconds"] = {pickup, dropoff}
        provenance["trip_duration_minutes"] = {pickup, dropoff}
        # minutes == seconds / 60 exactly, so the two are one variable.
        redundant.add("trip_duration_seconds")
        descriptions["trip_duration_seconds"] = f"{dropoff} minus {pickup}, in seconds."
        descriptions["trip_duration_minutes"] = f"{dropoff} minus {pickup}, in minutes."

    if pickup:
        additions += [
            pl.col(pickup).dt.hour().cast(pl.Int32).alias("pickup_hour"),
            pl.col(pickup).dt.weekday().cast(pl.Int32).alias("pickup_day_of_week"),
            pl.col(pickup).dt.date().alias("pickup_date"),
            pl.col(pickup).dt.hour().is_between(7, 9).or_(
                pl.col(pickup).dt.hour().is_between(16, 18)
            ).cast(pl.Int32).alias("is_rush_hour"),
        ]
        for derived_name in ("pickup_hour", "pickup_day_of_week", "pickup_date", "is_rush_hour"):
            provenance[derived_name] = {pickup}
        units["pickup_hour"] = Unit("hour_of_day", U.Dimension.dimensionless(), U.Kind.CATEGORICAL, 1.0)
        units["pickup_day_of_week"] = Unit(
            "day_of_week", U.Dimension.dimensionless(), U.Kind.CATEGORICAL, 1.0
        )
        units["pickup_date"] = U.TIMESTAMP
        units["is_rush_hour"] = U.BOOLEAN
        descriptions["pickup_hour"] = "Hour of day (0-23) the meter was engaged."
        descriptions["pickup_day_of_week"] = "ISO day of week of pickup (1=Monday .. 7=Sunday)."
        descriptions["pickup_date"] = "Calendar date of pickup."
        descriptions["is_rush_hour"] = "1 when pickup falls in 07:00-09:59 or 16:00-18:59."

    # Guarded rate features.  `trip_distance > 0` is required, not clamped:
    # a zero-distance trip has no defined cost per mile.
    if {"fare_amount", "trip_distance"} <= names:
        additions.append(_safe_ratio("fare_amount", "trip_distance").alias("fare_per_mile"))
        units["fare_per_mile"] = U.USD_PER_MILE
        provenance["fare_per_mile"] = {"fare_amount", "trip_distance"}
        descriptions["fare_per_mile"] = "fare_amount / trip_distance; null when distance is 0."

    if {"total_amount", "trip_distance"} <= names:
        additions.append(_safe_ratio("total_amount", "trip_distance").alias("total_per_mile"))
        units["total_per_mile"] = U.USD_PER_MILE
        provenance["total_per_mile"] = {"total_amount", "trip_distance"}
        descriptions["total_per_mile"] = "total_amount / trip_distance; null when distance is 0."

    if pickup and dropoff and "trip_distance" in names:
        duration_h = (pl.col(dropoff) - pl.col(pickup)).dt.total_seconds().cast(pl.Float64) / SECONDS_PER_HOUR
        additions.append(
            pl.when(duration_h > 0)
            .then(pl.col("trip_distance") / duration_h)
            .otherwise(None)
            .alias("average_speed_mph")
        )
        units["average_speed_mph"] = U.MPH
        provenance["average_speed_mph"] = {"trip_distance", pickup, dropoff}
        descriptions["average_speed_mph"] = (
            "trip_distance / trip duration in hours; null for non-positive durations."
        )

    if {"tip_amount", "fare_amount"} <= names:
        additions.append(_safe_ratio("tip_amount", "fare_amount").alias("tip_fraction_of_fare"))
        units["tip_fraction_of_fare"] = U.UNITLESS
        provenance["tip_fraction_of_fare"] = {"tip_amount", "fare_amount"}
        descriptions["tip_fraction_of_fare"] = (
            "tip_amount / fare_amount. Inherits the credit-card-only caveat of tip_amount."
        )

    airport = _first_present(names, ["airport_fee", "Airport_fee"])
    if airport:
        additions.append((pl.col(airport) > 0).cast(pl.Int32).alias("is_airport_pickup"))
        units["is_airport_pickup"] = U.BOOLEAN
        provenance["is_airport_pickup"] = {airport}
        descriptions["is_airport_pickup"] = (
            f"1 when {airport} is positive, i.e. a LaGuardia or JFK pickup."
        )

    if not additions:
        return lf, units, descriptions, provenance, redundant
    return lf.with_columns(additions), units, descriptions, provenance, redundant


def _safe_ratio(numerator: str, denominator: str) -> pl.Expr:
    return (
        pl.when(pl.col(denominator).abs() > 1e-9)
        .then(pl.col(numerator) / pl.col(denominator))
        .otherwise(None)
    )


def _timestamp_pair(profile: DatasetProfile, names: set[str]) -> tuple[str | None, str | None]:
    """Find the pickup/dropoff timestamp pair (tpep_* for yellow, lpep_* for green)."""
    dt_cols = [
        n
        for n, c in profile.columns.items()
        if c.semantic_type is SemanticType.DATETIME and n in names
    ]
    pickup = next((n for n in dt_cols if "pickup" in n.lower()), None)
    dropoff = next((n for n in dt_cols if "dropoff" in n.lower()), None)
    if pickup is None and len(dt_cols) >= 2:
        pickup, dropoff = dt_cols[0], dt_cols[1]
    return pickup, dropoff


def _first_present(names: set[str], candidates: list[str]) -> str | None:
    return next((c for c in candidates if c in names), None)


# ---------------------------------------------------------------------------
# Analysis view
# ---------------------------------------------------------------------------

# Thresholds are deliberately loose.  The goal is to exclude records that are
# physically impossible or are refund/void artifacts, NOT to make the data look
# tidy.  Each bound is justified; none is a percentile-based outlier trim,
# because trimming by percentile would silently delete real extreme trips.
TLC_ANALYSIS_FILTERS: list[FilterRule] = [
    FilterRule(
        "positive_distance",
        "trip_distance > 0",
        "A zero-distance trip has no defined cost-per-mile and is usually a meter error.",
    ),
    FilterRule(
        "plausible_distance",
        "trip_distance < 200",
        "The longest plausible NYC yellow-taxi trip is well under 200 miles. The raw file "
        "contains trips of 269,000 miles, which are meter faults.",
    ),
    FilterRule(
        "non_negative_fare",
        "fare_amount >= 0",
        "Negative fares are refunds/voided trips, not priced journeys.",
    ),
    FilterRule(
        "non_negative_total",
        "total_amount >= 0",
        "Negative totals are refunds/voided trips.",
    ),
    FilterRule(
        "plausible_duration",
        "trip_duration_seconds >= 30",
        "Trips under 30 seconds are meter engage/disengage errors.",
    ),
    FilterRule(
        "bounded_duration",
        "trip_duration_seconds <= 21600",
        "A metered trip longer than 6 hours indicates a meter left running.",
    ),
    FilterRule(
        "plausible_speed",
        "average_speed_mph <= 100",
        "Average speeds above 100 mph are not physically achievable in NYC traffic.",
    ),
]


def build_analysis_view(
    lf: pl.LazyFrame,
    profile: DatasetProfile,
    *,
    filters: list[FilterRule] | None = None,
    count_exclusions: bool = True,
) -> AnalysisView:
    """Derive columns, then apply the documented filters with full accounting."""
    (
        frame,
        derived_units,
        derived_desc,
        derived_provenance,
        redundant_derived,
    ) = derive_tlc_columns(lf, profile)
    available = set(frame.collect_schema().names())

    rules = filters if filters is not None else TLC_ANALYSIS_FILTERS
    applied: list[FilterRule] = []
    skipped: list[FilterRule] = []
    predicates: list[pl.Expr] = []

    for rule in rules:
        pred = rule.to_polars(available)
        if pred is None:
            skipped.append(rule)
            continue
        applied.append(rule)
        predicates.append(pred)

    report: dict = {
        "filters_applied": [r.name for r in applied],
        "filters_skipped": [{"name": r.name, "reason": "column not present"} for r in skipped],
        "rationale": {r.name: r.rationale for r in applied},
    }

    if count_exclusions and predicates:
        # One pass computes total rows plus, per rule, how many rows it alone
        # excludes -- including nulls, which a bare predicate would drop
        # silently.
        exprs = [pl.len().alias("__total")]
        for rule, pred in zip(applied, predicates):
            exprs.append((~pred.fill_null(False)).sum().alias(f"__x_{rule.name}"))
        exprs.append(
            pl.all_horizontal([p.fill_null(False) for p in predicates]).sum().alias("__kept")
        )
        counts = frame.select(exprs).collect().to_dicts()[0]

        total = int(counts["__total"])
        kept = int(counts["__kept"])
        report["rows_before"] = total
        report["rows_after"] = kept
        report["rows_excluded"] = total - kept
        report["exclusion_fraction"] = round((total - kept) / total, 6) if total else 0.0
        report["excluded_by_rule"] = {
            r.name: int(counts[f"__x_{r.name}"]) for r in applied
        }

    if predicates:
        frame = frame.filter(pl.all_horizontal([p.fill_null(False) for p in predicates]))

    return AnalysisView(
        frame=frame,
        applied=applied,
        skipped=skipped,
        derived_units=derived_units,
        derived_descriptions=derived_desc,
        derived_provenance=derived_provenance,
        redundant_derived=redundant_derived,
        report=report,
    )


def derivation_closure(columns: set[str], provenance: dict[str, set[str]]) -> set[str]:
    """Expand a set of columns to include every base column they derive from.

    `{total_per_mile}` expands to `{total_per_mile, total_amount, trip_distance}`,
    which is what makes target leakage detectable.  Iterates to a fixed point so
    a derived column built from another derived column still resolves, and is
    bounded so a cyclic map cannot hang the search.
    """
    closure = set(columns)
    for _ in range(8):
        expanded = set(closure)
        for name in closure:
            expanded |= provenance.get(name, set())
        if expanded == closure:
            break
        closure = expanded
    return closure


def reconstructs_target(
    columns: set[str], target: str, provenance: dict[str, set[str]]
) -> bool:
    """True when an expression over `columns` contains the target's own value.

    Correlating such an expression against the target measures algebra, not
    the world.  The candidate is dropped before it is ever materialized.
    """
    if not target or columns == {target}:
        return False
    return target in derivation_closure(columns, provenance)


def materializable_numeric_columns(frame: pl.LazyFrame, units: dict[str, Unit]) -> list[str]:
    """Columns that are BOTH semantically known and physically castable to float.

    Semantic type and physical dtype can disagree: TLC's `store_and_fwd_flag`
    is semantically a boolean but physically a string, so `log1p()` of it would
    blow up at cast time.  Candidate generation must intersect the two.
    """
    schema = frame.collect_schema()
    out: list[str] = []
    for name in schema.names():
        if name not in units:
            continue
        dtype = schema[name]
        if dtype.is_numeric() or dtype == pl.Boolean:
            out.append(name)
    return out


def positive_columns(frame: pl.LazyFrame, candidates: list[str]) -> set[str]:
    """Which of `candidates` are strictly positive, in one pass.

    Used to gate `log`/`sqrt` proposals: proposing log() of a column that holds
    zeros or negatives generates a feature that is mostly null, which wastes a
    slot in the beam.
    """
    if not candidates:
        return set()
    row = frame.select([pl.col(c).min().alias(c) for c in candidates]).collect().to_dicts()[0]
    out: set[str] = set()
    for name, value in row.items():
        try:
            if value is not None and float(value) > 0:
                out.add(name)
        except (TypeError, ValueError):
            continue
    return out


# ---------------------------------------------------------------------------
# Predicate compilation (whitelisted, no eval)
# ---------------------------------------------------------------------------

_COMPARISONS = ("<=", ">=", "==", "!=", "<", ">")


def _referenced_columns(expression: str) -> set[str]:
    return {_split_predicate(expression)[0]}


def _split_predicate(expression: str) -> tuple[str, str, float]:
    for op in _COMPARISONS:
        if op in expression:
            left, right = expression.split(op, 1)
            return left.strip(), op, float(right.strip())
    raise ValueError(f"unsupported filter expression: {expression!r}")


def _compile_predicate(expression: str) -> pl.Expr:
    """Compile `column <op> number` to a Polars predicate.

    Filters come from this module's own constants or from an operator-supplied
    config, never from a model, but they still go through a restricted parser
    rather than `eval` so the boundary holds no matter who supplies them.
    """
    column, op, value = _split_predicate(expression)
    c = pl.col(column)
    return {
        "<=": c <= value,
        ">=": c >= value,
        "==": c == value,
        "!=": c != value,
        "<": c < value,
        ">": c > value,
    }[op]
