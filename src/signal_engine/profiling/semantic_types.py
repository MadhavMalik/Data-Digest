"""Semantic type inference.

A column's *physical* dtype (int64) is almost never its *semantic* type.
`PULocationID` is an int64 and a taxi zone label; averaging it is meaningless.
Getting this wrong is the single most common way an automated analysis produces
confident nonsense, so inference runs in strict precedence order:

    1. official dataset metadata      (confidence 1.00 — never overridden)
    2. physical dtype hard constraints(a timestamp is a datetime, period)
    3. column-name heuristics         (confidence 0.55-0.80)
    4. value-distribution heuristics  (integer-coded-categorical detection)

Every inference records which rule fired, so a downstream warning can say
*why* a column was typed the way it was.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from signal_engine.profiling import units as U
from signal_engine.profiling.units import Kind, Unit


class SemanticType(str, Enum):
    CONTINUOUS_MEASUREMENT = "continuous_measurement"
    CURRENCY = "currency"
    COUNT = "count"
    RATE = "rate"
    DURATION = "duration"
    DATETIME = "datetime"
    CATEGORICAL = "categorical"
    CATEGORICAL_IDENTIFIER = "categorical_identifier"
    BOOLEAN = "boolean"
    TEXT = "text"
    GEO_COORDINATE = "geo_coordinate"
    UNKNOWN = "unknown"

    @property
    def is_numeric_quantity(self) -> bool:
        """True when arithmetic and correlation are meaningful on raw values."""
        return self in {
            SemanticType.CONTINUOUS_MEASUREMENT,
            SemanticType.CURRENCY,
            SemanticType.COUNT,
            SemanticType.RATE,
            SemanticType.DURATION,
            SemanticType.GEO_COORDINATE,
        }

    @property
    def is_categorical(self) -> bool:
        return self in {
            SemanticType.CATEGORICAL,
            SemanticType.CATEGORICAL_IDENTIFIER,
            SemanticType.BOOLEAN,
        }


@dataclass(frozen=True)
class TypeInference:
    semantic_type: SemanticType
    unit: Unit
    confidence: float
    rule: str
    notes: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Name heuristics
# ---------------------------------------------------------------------------

_CURRENCY_PAT = re.compile(
    r"(?:^|_)(amount|fare|price|cost|revenue|fee|surcharge|tax|tip|toll|charge|payment|usd|dollar)s?(?:_|$)",
    re.I,
)
_DISTANCE_PAT = re.compile(r"(?:^|_)(distance|miles|mileage|km|kilometers|meters|length)(?:_|$)", re.I)
_DURATION_PAT = re.compile(r"(?:^|_)(duration|elapsed|seconds|minutes|hours|secs|mins)(?:_|$)", re.I)
_DATETIME_PAT = re.compile(r"(?:^|_)(datetime|timestamp|date|time)(?:_|$)|(?:_|^)(at|on)$", re.I)
_COUNT_PAT = re.compile(r"(?:^|_)(count|num|number|qty|quantity|passengers?|trips?)(?:_|$)", re.I)
_ID_PAT = re.compile(r"(?:^|_)(id|ids|uuid|guid|key|code|zone|locationid)(?:_|$)|id$", re.I)
_BOOL_PAT = re.compile(r"(?:^|_)(flag|is|has|was|should)(?:_|$)|_flag$", re.I)
_LAT_PAT = re.compile(r"(?:^|_)(lat|latitude)(?:_|$)", re.I)
_LON_PAT = re.compile(r"(?:^|_)(lon|lng|long|longitude)(?:_|$)", re.I)
_RATE_PAT = re.compile(r"_per_|(?:^|_)(rate|ratio|pct|percent|speed|mph|kph)(?:_|$)", re.I)
_ENERGY_PAT = re.compile(r"(?:^|_)(mwh|kwh|gwh|wh|energy|generation|consumption|load)(?:_|$)", re.I)
_MASS_PAT = re.compile(r"(?:^|_)(tons?|tonnes?|kg|kilograms?|mass|weight|co2|emissions?)(?:_|$)", re.I)
_VOLUME_PAT = re.compile(r"(?:^|_)(litres?|liters?|gallons?|m3|volume|barrels?)(?:_|$)", re.I)
_TEMP_PAT = re.compile(r"(?:^|_)(temp|temperature|celsius|fahrenheit|degc|degf)(?:_|$)", re.I)

# Column-name -> unit, only consulted when metadata is absent.
_NAME_UNITS: list[tuple[re.Pattern, Unit, float]] = [
    (_CURRENCY_PAT, U.USD, 0.75),
    (_DISTANCE_PAT, U.MILES, 0.65),
    (_DURATION_PAT, U.SECONDS, 0.65),
    (_LAT_PAT, U.DEGREES, 0.8),
    (_LON_PAT, U.DEGREES, 0.8),
    (_COUNT_PAT, U.COUNT, 0.6),
    (_ENERGY_PAT, U.MWH, 0.65),
    (_MASS_PAT, U.TONNES, 0.6),
    (_VOLUME_PAT, U.LITRES, 0.6),
    (_TEMP_PAT, U.CELSIUS, 0.6),
]


_UNIT_LOOKUP: dict[str, Unit] = {
    "mwh": U.MWH, "kwh": U.KWH, "tonnes": U.TONNES, "kg": U.KG,
    "litres": U.LITRES, "celsius": U.CELSIUS,
    "usd": U.USD,
    "dollars": U.USD,
    "miles": U.MILES,
    "mile": U.MILES,
    "km": U.KM,
    "seconds": U.SECONDS,
    "second": U.SECONDS,
    "minutes": U.MINUTES,
    "hours": U.HOURS,
    "count": U.COUNT,
    "timestamp": U.TIMESTAMP,
    "category": U.CATEGORICAL,
    "id": U.IDENTIFIER,
    "bool": U.BOOLEAN,
    "text": U.TEXT,
    "degrees": U.DEGREES,
    "unitless": U.UNITLESS,
    "usd/miles": U.USD_PER_MILE,
    "miles/hours": U.MPH,
}


def unit_from_label(label: str | None, *, confidence: float = 1.0) -> Unit:
    """Map a metadata unit string onto a Unit object."""
    if not label:
        return U.UNKNOWN_UNIT
    base = _UNIT_LOOKUP.get(label.strip().lower())
    if base is None:
        return Unit(label, U.Dimension.dimensionless(), Kind.UNKNOWN, min(confidence, 0.4))
    if confidence >= 1.0:
        return base
    return Unit(base.label, base.dimension, base.kind, confidence)


_SEMANTIC_DEFAULT_UNIT: dict[SemanticType, Unit] = {
    SemanticType.CURRENCY: U.USD,
    SemanticType.CONTINUOUS_MEASUREMENT: U.UNKNOWN_UNIT,
    SemanticType.COUNT: U.COUNT,
    SemanticType.DURATION: U.SECONDS,
    SemanticType.DATETIME: U.TIMESTAMP,
    SemanticType.CATEGORICAL: U.CATEGORICAL,
    SemanticType.CATEGORICAL_IDENTIFIER: U.IDENTIFIER,
    SemanticType.BOOLEAN: U.BOOLEAN,
    SemanticType.TEXT: U.TEXT,
    SemanticType.GEO_COORDINATE: U.DEGREES,
    SemanticType.RATE: U.UNITLESS,
    SemanticType.UNKNOWN: U.UNKNOWN_UNIT,
}


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


@dataclass
class ValueEvidence:
    """The cheap distributional facts inference is allowed to consult."""

    dtype: str = ""
    is_integral: bool = False
    is_float: bool = False
    is_temporal: bool = False
    is_string: bool = False
    is_boolean: bool = False
    distinct_count: int | None = None
    row_count: int = 0
    non_null_count: int = 0
    min_value: float | None = None
    max_value: float | None = None
    has_negative: bool = False
    has_fractional: bool = False
    distinct_ratio: float | None = None


# An integer column is treated as an integer-coded CATEGORY when it has few
# distinct values relative to the data.  Thresholds are deliberately
# conservative: mis-typing a real measurement as categorical loses signal,
# while mis-typing a code as continuous produces confident nonsense.  We accept
# the former risk only where the evidence is strong.
MAX_CATEGORICAL_DISTINCT = 60
MAX_CATEGORICAL_RATIO = 0.001


def infer_semantic_type(
    name: str,
    evidence: ValueEvidence,
    metadata: dict | None = None,
) -> TypeInference:
    """Infer a column's semantic type and unit."""
    notes: list[str] = []

    # ---- 1. official metadata wins outright -----------------------------
    if metadata:
        declared = metadata.get("semantic_type")
        if declared:
            st = _coerce_semantic_type(declared)
            unit = unit_from_label(metadata.get("unit")) if metadata.get("unit") else _SEMANTIC_DEFAULT_UNIT[st]
            if metadata.get("is_identifier_like") and st is SemanticType.CATEGORICAL:
                notes.append("metadata marks this as an integer-coded label, not a magnitude")
            return TypeInference(st, unit, 1.0, "official_metadata", tuple(notes))

    # ---- 2. physical dtype hard constraints -----------------------------
    if evidence.is_temporal:
        return TypeInference(SemanticType.DATETIME, U.TIMESTAMP, 0.98, "dtype_temporal")
    if evidence.is_boolean:
        return TypeInference(SemanticType.BOOLEAN, U.BOOLEAN, 0.98, "dtype_boolean")
    if evidence.is_string:
        # A low-cardinality string is a category; a high-cardinality one is text.
        if evidence.distinct_count is not None and evidence.distinct_count <= MAX_CATEGORICAL_DISTINCT:
            if evidence.distinct_count <= 2:
                return TypeInference(SemanticType.BOOLEAN, U.BOOLEAN, 0.85, "string_binary")
            return TypeInference(SemanticType.CATEGORICAL, U.CATEGORICAL, 0.9, "string_low_cardinality")
        return TypeInference(SemanticType.TEXT, U.TEXT, 0.9, "string_high_cardinality")

    # ---- 3/4. numeric: name heuristics + distribution --------------------
    if evidence.is_integral or evidence.is_float:
        # 3a. Geo coordinates are recognisable by name AND range.
        if _LAT_PAT.search(name) and _in_range(evidence, -90, 90):
            return TypeInference(SemanticType.GEO_COORDINATE, U.DEGREES, 0.9, "name_and_range_latitude")
        if _LON_PAT.search(name) and _in_range(evidence, -180, 180):
            return TypeInference(SemanticType.GEO_COORDINATE, U.DEGREES, 0.9, "name_and_range_longitude")

        # 3b. Identifier-looking name + integral + no fractional part.
        looks_like_id = bool(_ID_PAT.search(name)) and evidence.is_integral
        low_cardinality = _is_low_cardinality(evidence)

        if looks_like_id:
            notes.append("column name matches an identifier pattern; values are integral")
            return TypeInference(
                SemanticType.CATEGORICAL_IDENTIFIER, U.IDENTIFIER, 0.8, "name_identifier", tuple(notes)
            )

        if evidence.is_integral and low_cardinality and not evidence.has_fractional:
            if evidence.distinct_count is not None and evidence.distinct_count <= 2:
                return TypeInference(SemanticType.BOOLEAN, U.BOOLEAN, 0.7, "integer_binary")
            notes.append(
                f"only {evidence.distinct_count} distinct integer values across "
                f"{evidence.non_null_count:,} rows; treated as an encoded category, not a magnitude"
            )
            return TypeInference(
                SemanticType.CATEGORICAL, U.CATEGORICAL, 0.7, "integer_low_cardinality", tuple(notes)
            )

        # 3c. Unit-bearing names.
        if _RATE_PAT.search(name):
            return TypeInference(SemanticType.RATE, U.UNITLESS, 0.6, "name_rate")
        for pattern, unit, conf in _NAME_UNITS:
            if pattern in (_LAT_PAT, _LON_PAT):
                # Geo columns were already decided above, where the NAME had to
                # agree with the VALUE RANGE. Re-matching on the name alone here
                # would silently undo that check and type a -500..900 column as
                # a latitude.
                continue
            if pattern.search(name):
                st = _semantic_for_unit(unit)
                return TypeInference(st, Unit(unit.label, unit.dimension, unit.kind, conf), conf, "name_unit")

        if _COUNT_PAT.search(name) and evidence.is_integral:
            return TypeInference(SemanticType.COUNT, U.COUNT, 0.6, "name_count")

        # 3d. Fall back to an untyped continuous measurement.
        notes.append("no metadata or name signal; unit unknown, so unit-based pruning is relaxed")
        return TypeInference(
            SemanticType.CONTINUOUS_MEASUREMENT, U.UNKNOWN_UNIT, 0.35, "numeric_default", tuple(notes)
        )

    return TypeInference(SemanticType.UNKNOWN, U.UNKNOWN_UNIT, 0.2, "unrecognized_dtype")


def _semantic_for_unit(unit: Unit) -> SemanticType:
    if unit.dimension == U.Dimension.of(currency=1):
        return SemanticType.CURRENCY
    if unit.dimension == U.Dimension.of(length=1):
        return SemanticType.CONTINUOUS_MEASUREMENT
    if unit.dimension == U.Dimension.of(time=1):
        return SemanticType.DURATION
    if unit.dimension == U.Dimension.of(count=1):
        return SemanticType.COUNT
    if unit.dimension == U.Dimension.of(angle=1):
        return SemanticType.GEO_COORDINATE
    if unit.dimension in (
        U.Dimension.of(energy=1), U.Dimension.of(mass=1),
        U.Dimension.of(volume=1), U.Dimension.of(temperature=1),
    ):
        return SemanticType.CONTINUOUS_MEASUREMENT
    return SemanticType.CONTINUOUS_MEASUREMENT


def _is_low_cardinality(ev: ValueEvidence) -> bool:
    if ev.distinct_count is None or ev.non_null_count == 0:
        return False
    if ev.distinct_count > MAX_CATEGORICAL_DISTINCT:
        return False
    ratio = ev.distinct_count / max(ev.non_null_count, 1)
    # Small datasets legitimately have a high distinct ratio, so the ratio test
    # only applies once there are enough rows for it to mean anything.
    if ev.non_null_count >= 10_000:
        return ratio <= MAX_CATEGORICAL_RATIO or ev.distinct_count <= 20
    return ev.distinct_count <= 20


def _in_range(ev: ValueEvidence, lo: float, hi: float) -> bool:
    if ev.min_value is None or ev.max_value is None:
        return False
    return ev.min_value >= lo and ev.max_value <= hi


def _coerce_semantic_type(raw: str) -> SemanticType:
    try:
        return SemanticType(raw)
    except ValueError:
        return {
            "continuous": SemanticType.CONTINUOUS_MEASUREMENT,
            "numeric": SemanticType.CONTINUOUS_MEASUREMENT,
            "measurement": SemanticType.CONTINUOUS_MEASUREMENT,
            "money": SemanticType.CURRENCY,
            "id": SemanticType.CATEGORICAL_IDENTIFIER,
            "identifier": SemanticType.CATEGORICAL_IDENTIFIER,
            "category": SemanticType.CATEGORICAL,
            "bool": SemanticType.BOOLEAN,
            "time": SemanticType.DATETIME,
        }.get(raw.lower(), SemanticType.UNKNOWN)
