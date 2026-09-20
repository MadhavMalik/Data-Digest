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
import re
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


def derive_columns(
    lf: pl.LazyFrame, profile: DatasetProfile
) -> tuple[pl.LazyFrame, dict[str, Unit], dict[str, str], dict[str, set[str]], set[str]]:
    """Derive features generically, from SEMANTICS rather than column names.

    Nothing here knows about taxis. The rules are:

      * every datetime pair (earliest, latest) yields a duration
      * every datetime yields its calendar parts (hour, weekday, date)
      * every dimensionally meaningful ratio is materialised: currency/length,
        currency/count, length/time, currency/time -- i.e. anything whose unit
        algebra produces a coherent composite
      * a positive currency amount divided by another yields a share

    That reproduces `fare_per_mile`, `average_speed_mph` and `trip_duration`
    on the TLC data without naming any of them, and produces the equivalent
    features on a dataset the engine has never seen.

    Also returns provenance (which base columns each derived column came from,
    for target-leakage detection) and the set of redundant unit-rescalings that
    should stay out of the candidate pool.
    """
    names = set(lf.collect_schema().names())
    units: dict[str, Unit] = {}
    descriptions: dict[str, str] = {}
    provenance: dict[str, set[str]] = {}
    redundant: set[str] = set()
    additions: list[pl.Expr] = []

    # ---- 1. datetime pairs -> duration ---------------------------------
    start, end = _timestamp_pair(profile, names)
    if start and end:
        duration_s = (pl.col(end) - pl.col(start)).dt.total_seconds().cast(pl.Float64)
        additions.append(duration_s.alias("duration_seconds"))
        additions.append((duration_s / 60.0).alias("duration_minutes"))
        units["duration_seconds"] = U.SECONDS
        units["duration_minutes"] = U.MINUTES
        provenance["duration_seconds"] = {start, end}
        provenance["duration_minutes"] = {start, end}
        descriptions["duration_seconds"] = f"{end} minus {start}, in seconds."
        descriptions["duration_minutes"] = f"{end} minus {start}, in minutes."
        # minutes == seconds / 60 exactly: one variable, two scales.
        redundant.add("duration_seconds")

    # ---- 2. datetimes -> calendar parts --------------------------------
    if start:
        additions += [
            pl.col(start).dt.hour().cast(pl.Int32).alias("event_hour"),
            pl.col(start).dt.weekday().cast(pl.Int32).alias("event_day_of_week"),
            pl.col(start).dt.date().alias("event_date"),
        ]
        for name in ("event_hour", "event_day_of_week", "event_date"):
            provenance[name] = {start}
        units["event_hour"] = Unit("hour_of_day", U.Dimension.dimensionless(),
                                   U.Kind.CATEGORICAL, 1.0)
        units["event_day_of_week"] = Unit("day_of_week", U.Dimension.dimensionless(),
                                          U.Kind.CATEGORICAL, 1.0)
        units["event_date"] = U.TIMESTAMP
        descriptions["event_hour"] = f"Hour of day (0-23) of {start}."
        descriptions["event_day_of_week"] = f"ISO weekday of {start} (1=Monday .. 7=Sunday)."
        descriptions["event_date"] = f"Calendar date of {start}."

    # ---- 3. dimensionally coherent ratios ------------------------------
    base_units = {n: u for n, u in profile.units().items() if n in names}
    base_units.update({n: u for n, u in units.items() if n in {"duration_seconds", "duration_minutes"}})

    for name, unit in units.items():
        if name in ("duration_seconds", "duration_minutes"):
            continue

    ratios = _meaningful_ratios(profile, base_units, names, units)
    for numerator, denominator, alias, unit, why in ratios:
        if alias in units:
            continue
        expr = _safe_ratio_expr(numerator, denominator, lf, units)
        if expr is None:
            continue
        additions.append(expr.alias(alias))
        units[alias] = unit
        provenance[alias] = {numerator, denominator}
        descriptions[alias] = why

    if not additions:
        return lf, units, descriptions, provenance, redundant
    return lf.with_columns(additions), units, descriptions, provenance, redundant


# Ratio pairs worth materialising, keyed by (numerator dimension, denominator
# dimension). These are the composites that name a real quantity rather than an
# arbitrary quotient.
_RATIO_RULES: list[tuple[dict, dict, str, str]] = [
    ({"currency": 1}, {"length": 1}, "per_{d}", "cost per unit of {d}"),
    ({"currency": 1}, {"time": 1}, "per_{d}", "cost per unit of {d}"),
    ({"currency": 1}, {"count": 1}, "per_{d}", "cost per unit of {d}"),
    ({"length": 1}, {"time": 1}, "{n}_per_{d}", "speed: {n} per {d}"),
    ({"count": 1}, {"time": 1}, "{n}_per_{d}", "rate: {n} per {d}"),
    ({"currency": 1}, {"energy": 1}, "per_{d}", "cost per unit of {d}"),
    ({"currency": 1}, {"mass": 1}, "per_{d}", "cost per unit of {d}"),
    ({"currency": 1}, {"volume": 1}, "per_{d}", "cost per unit of {d}"),
]

# "Cost per unit of X" is meaningful for ANY quantity X, including one whose
# unit could not be identified. On an unfamiliar dataset that single derived
# feature is usually the most informative one available, so it is generated
# even when the denominator's unit is unknown -- with the unit label marked
# unknown so nothing downstream over-claims.
COST_PER_UNKNOWN = True

MAX_DERIVED_RATIOS = 12


def _meaningful_ratios(profile, base_units, names, already):
    """Enumerate ratios whose units name a real quantity.

    Capped, and ordered so the most interpretable come first: without a cap a
    wide dataset would materialise hundreds of columns before any analysis
    began.
    """
    from signal_engine.profiling.units import Dimension, Kind, Unit

    out: list[tuple] = []
    candidates = [
        (n, u) for n, u in base_units.items()
        if u.kind is Kind.QUANTITY and not u.dimension.is_dimensionless
    ]
    unknown_quantities = [
        (n, u) for n, u in base_units.items()
        if u.kind is Kind.UNKNOWN and n not in {c[0] for c in candidates}
    ]
    currencies = [(n, u) for n, u in candidates
                  if u.dimension == Dimension.of(currency=1)]

    for num_name, num_unit in candidates:
        for den_name, den_unit in candidates:
            if num_name == den_name:
                continue
            for num_dim, den_dim, pattern, why in _RATIO_RULES:
                if num_unit.dimension != Dimension.of(**num_dim):
                    continue
                if den_unit.dimension != Dimension.of(**den_dim):
                    continue
                try:
                    unit = num_unit.divide(den_unit)
                except Exception:  # noqa: BLE001
                    continue
                alias = pattern.format(n=_short(num_name), d=_short(den_name))
                alias = f"{_short(num_name)}_{alias}" if alias.startswith("per_") else alias
                alias = re.sub(r"[^0-9a-zA-Z_]", "_", alias)[:60]
                if alias in already:
                    continue
                out.append((
                    num_name, den_name, alias, unit,
                    why.format(n=num_name, d=den_name) + f" ({num_name} / {den_name}).",
                ))
                break

    if COST_PER_UNKNOWN:
        for cur_name, cur_unit in currencies:
            for den_name, den_unit in unknown_quantities:
                alias = re.sub(r"[^0-9a-zA-Z_]", "_",
                               f"{_short(cur_name)}_per_{_short(den_name)}")[:60]
                if alias in already:
                    continue
                unit = Unit(f"{cur_unit.label}/{den_unit.label}",
                            Dimension.of(currency=1), Kind.QUANTITY,
                            min(cur_unit.confidence, 0.4))
                out.append((
                    cur_name, den_name, alias, unit,
                    f"cost per unit of {den_name} ({cur_name} / {den_name}). "
                    f"The denominator's unit could not be identified, so the "
                    f"composite unit is approximate.",
                ))

    # Prefer short, readable names -- they are the ones a person can interpret.
    out.sort(key=lambda r: (len(r[2]), r[2]))
    return out[:MAX_DERIVED_RATIOS]


def _short(name: str) -> str:
    """Trim a column name to its distinctive part for use in a derived alias."""
    trimmed = re.sub(r"_(amount|value|total|count|id)$", "", name.lower())
    return trimmed or name.lower()


def _safe_ratio_expr(numerator: str, denominator: str, lf, units) -> pl.Expr | None:
    schema = lf.collect_schema()
    if numerator not in schema.names() or denominator not in schema.names():
        return None
    if not (schema[numerator].is_numeric() and schema[denominator].is_numeric()):
        return None
    return _safe_ratio(numerator, denominator)


# Back-compat alias: the TLC-specific entry point is now just the generic one.
derive_tlc_columns = derive_columns


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
        "duration_seconds >= 30",
        "Trips under 30 seconds are meter engage/disengage errors.",
    ),
    FilterRule(
        "bounded_duration",
        "duration_seconds <= 21600",
        "A metered trip longer than 6 hours indicates a meter left running.",
    ),
]


def generic_filters(profile: DatasetProfile, available: set[str]) -> list[FilterRule]:
    """Validity filters that hold for ANY dataset.

    Deliberately minimal. Only rules that are true BY DEFINITION go here:

      * a duration cannot be negative — time does not run backwards
      * a count cannot be negative — you cannot have minus three passengers

    Everything beyond that is domain knowledge (is a 200-mile taxi trip
    plausible? only someone who knows the domain can say), and belongs in a
    DatasetSpec rather than being guessed from the distribution. Guessing would
    mean trimming by percentile, which silently deletes exactly the extreme
    records that carry the most signal.
    """
    from signal_engine.profiling.semantic_types import SemanticType

    rules: list[FilterRule] = []
    for name, card in profile.columns.items():
        if name not in available:
            continue
        if card.semantic_type is SemanticType.DURATION:
            rules.append(FilterRule(
                f"non_negative_{name}", f"{name} >= 0",
                f"{name} is a duration; a negative elapsed time is impossible.",
            ))
        elif card.semantic_type is SemanticType.COUNT:
            rules.append(FilterRule(
                f"non_negative_{name}", f"{name} >= 0",
                f"{name} is a count; a negative count is impossible.",
            ))

    # Derived durations are not in the profile (they did not exist when it ran),
    # so they are added explicitly.
    for derived in ("duration_seconds", "duration_minutes"):
        if derived in available:
            rules.append(FilterRule(
                f"non_negative_{derived}", f"{derived} >= 0",
                f"{derived} is derived from a timestamp difference; a negative value "
                f"means the end precedes the start, which is a data error.",
            ))
            break

    return rules


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
    ) = derive_columns(lf, profile)
    available = set(frame.collect_schema().names())

    # Explicit filters win; otherwise a known dataset contributes its curated
    # domain rules and an unknown one falls back to the definitional set.
    if filters is not None:
        rules = filters
    else:
        from signal_engine.datasets import resolve_spec

        spec = resolve_spec(getattr(profile, "path", None) or profile.dataset.dataset_id,
                            set(profile.columns))
        rules = spec.filters if spec.filters is not None else generic_filters(profile, available)
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
