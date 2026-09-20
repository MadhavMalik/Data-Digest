"""A small dimensional-analysis layer.

This is the single highest-leverage optimization in the engine: it deletes
semantically nonsensical candidate features *before* a single row is touched.

    fare_amount / trip_distance   -> USD/mile      KEEP
    fare_amount + trip_distance   -> incoherent    REJECT
    PULocationID * fare_amount    -> incoherent    REJECT (ID has no magnitude)

It is deliberately NOT a full symbolic physics package.  It tracks a small set
of base dimensions as integer exponents, plus a few non-quantitative kinds
(datetime, categorical, identifier) whose algebra is special-cased.

Design note on logarithms: strictly, log() of a dimensional quantity is
undefined.  In data analysis log(fare_amount) is both common and useful, so we
permit it and mark the result DIMENSIONLESS with a `log(USD)` display label.
The provenance stays in the label so an interpretation can never silently
claim the result is still dollars.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum


class UnitAlgebraError(ValueError):
    """Raised when an operation is dimensionally incoherent."""


class BaseDimension(str, Enum):
    """Base quantities we track with integer exponents."""

    CURRENCY = "currency"
    LENGTH = "length"
    TIME = "time"
    COUNT = "count"
    ANGLE = "angle"
    TEMPERATURE = "temperature"
    INFORMATION = "information"


class Kind(str, Enum):
    """How a value behaves under arithmetic.

    QUANTITY      ordinary dimensional number (dollars, miles, seconds, counts)
    DATETIME      an instant.  datetime - datetime = duration; datetime + datetime is nonsense.
    CATEGORICAL   a label.  No arithmetic at all.
    IDENTIFIER    an integer-coded label (PULocationID).  Looks numeric, has no magnitude.
    BOOLEAN       0/1 flag.  Averaging is meaningful, so it behaves as a dimensionless quantity.
    TEXT          free text.  No arithmetic.
    UNKNOWN       could not be determined; arithmetic allowed but flagged low-confidence.
    """

    QUANTITY = "quantity"
    DATETIME = "datetime"
    CATEGORICAL = "categorical"
    IDENTIFIER = "identifier"
    BOOLEAN = "boolean"
    TEXT = "text"
    UNKNOWN = "unknown"


_NON_ARITHMETIC = {Kind.CATEGORICAL, Kind.IDENTIFIER, Kind.TEXT}


@dataclass(frozen=True, order=True)
class Dimension:
    """Integer exponents over the base dimensions.  Immutable and hashable."""

    exponents: tuple[tuple[str, int], ...] = ()

    # ---- construction ---------------------------------------------------
    @staticmethod
    def of(**kwargs: int) -> Dimension:
        """Dimension.of(currency=1, length=-1) -> USD/mile."""
        items = tuple(sorted((k, v) for k, v in kwargs.items() if v != 0))
        for name, _ in items:
            if name not in {d.value for d in BaseDimension}:
                raise UnitAlgebraError(f"unknown base dimension: {name}")
        return Dimension(items)

    @staticmethod
    def dimensionless() -> Dimension:
        return Dimension(())

    # ---- algebra --------------------------------------------------------
    @property
    def is_dimensionless(self) -> bool:
        return not self.exponents

    def as_dict(self) -> dict[str, int]:
        return dict(self.exponents)

    def __mul__(self, other: Dimension) -> Dimension:
        merged = self.as_dict()
        for k, v in other.exponents:
            merged[k] = merged.get(k, 0) + v
        return Dimension(tuple(sorted((k, v) for k, v in merged.items() if v != 0)))

    def __truediv__(self, other: Dimension) -> Dimension:
        return self * other.inverse()

    def inverse(self) -> Dimension:
        return Dimension(tuple(sorted((k, -v) for k, v in self.exponents)))

    def power(self, n: int) -> Dimension:
        if n == 0:
            return Dimension.dimensionless()
        return Dimension(tuple(sorted((k, v * n) for k, v in self.exponents)))

    def root(self, n: int) -> Dimension:
        """Integer root; raises when the exponents do not divide evenly."""
        out: list[tuple[str, int]] = []
        for k, v in self.exponents:
            if v % n != 0:
                raise UnitAlgebraError(f"cannot take root {n} of dimension {self}")
            out.append((k, v // n))
        return Dimension(tuple(sorted(out)))

    def __str__(self) -> str:  # pragma: no cover - display only
        if not self.exponents:
            return "1"
        num = [k if v == 1 else f"{k}^{v}" for k, v in self.exponents if v > 0]
        den = [k if v == -1 else f"{k}^{-v}" for k, v in self.exponents if v < 0]
        left = "*".join(num) if num else "1"
        return left if not den else f"{left}/{'*'.join(den)}"


@dataclass(frozen=True)
class Unit:
    """A dimension plus a human label, a kind, and a confidence.

    `label` is what a human reads ("USD/mile").  `dimension` is what the algebra
    uses.  `confidence` records how sure we are — metadata-derived units are
    1.0, name-heuristic units are lower, and the confidence propagates through
    every operation so a derived feature can never look more certain than its
    weakest input.
    """

    label: str = "unitless"
    dimension: Dimension = Dimension.dimensionless()
    kind: Kind = Kind.QUANTITY
    confidence: float = 1.0

    # ---- predicates -----------------------------------------------------
    @property
    def arithmetic_allowed(self) -> bool:
        return self.kind not in _NON_ARITHMETIC

    @property
    def is_additive(self) -> bool:
        """Can this participate in + / - at all?"""
        return self.kind in {Kind.QUANTITY, Kind.BOOLEAN, Kind.DATETIME, Kind.UNKNOWN}

    # ---- algebra --------------------------------------------------------
    def _guard(self, other: Unit, op: str) -> None:
        for u in (self, other):
            if not u.arithmetic_allowed:
                raise UnitAlgebraError(
                    f"{op} is not defined for {u.kind.value} value '{u.label}' "
                    f"(no meaningful magnitude)"
                )

    def add(self, other: Unit) -> Unit:
        self._guard(other, "addition")
        if self.kind is Kind.DATETIME and other.kind is Kind.DATETIME:
            raise UnitAlgebraError("datetime + datetime is not meaningful")
        if self.kind is Kind.DATETIME or other.kind is Kind.DATETIME:
            dt, delta = (self, other) if self.kind is Kind.DATETIME else (other, self)
            if delta.dimension != Dimension.of(time=1):
                raise UnitAlgebraError("only a duration may be added to a datetime")
            return Unit(dt.label, dt.dimension, Kind.DATETIME, _min_conf(self, other))
        if self.dimension != other.dimension:
            raise UnitAlgebraError(
                f"cannot add incompatible units: {self.label} + {other.label} "
                f"({self.dimension} vs {other.dimension})"
            )
        return Unit(self.label, self.dimension, _merge_kind(self, other), _min_conf(self, other))

    def subtract(self, other: Unit) -> Unit:
        self._guard(other, "subtraction")
        if self.kind is Kind.DATETIME and other.kind is Kind.DATETIME:
            # The one genuinely useful datetime operation.
            return Unit("seconds", Dimension.of(time=1), Kind.QUANTITY, _min_conf(self, other))
        if self.kind is Kind.DATETIME:
            if other.dimension != Dimension.of(time=1):
                raise UnitAlgebraError("only a duration may be subtracted from a datetime")
            return Unit(self.label, self.dimension, Kind.DATETIME, _min_conf(self, other))
        if other.kind is Kind.DATETIME:
            raise UnitAlgebraError("cannot subtract a datetime from a non-datetime")
        if self.dimension != other.dimension:
            raise UnitAlgebraError(
                f"cannot subtract incompatible units: {self.label} - {other.label} "
                f"({self.dimension} vs {other.dimension})"
            )
        return Unit(self.label, self.dimension, _merge_kind(self, other), _min_conf(self, other))

    def multiply(self, other: Unit) -> Unit:
        self._guard(other, "multiplication")
        if Kind.DATETIME in (self.kind, other.kind):
            raise UnitAlgebraError("multiplication involving a datetime is not meaningful")
        dim = self.dimension * other.dimension
        return Unit(_compose_label(self, other, "*"), dim, Kind.QUANTITY, _min_conf(self, other))

    def divide(self, other: Unit) -> Unit:
        self._guard(other, "division")
        if Kind.DATETIME in (self.kind, other.kind):
            raise UnitAlgebraError("division involving a datetime is not meaningful")
        dim = self.dimension / other.dimension
        return Unit(_compose_label(self, other, "/"), dim, Kind.QUANTITY, _min_conf(self, other))

    def log(self) -> Unit:
        if not self.arithmetic_allowed:
            raise UnitAlgebraError(f"log is not defined for {self.kind.value} values")
        if self.kind is Kind.DATETIME:
            raise UnitAlgebraError("log of a datetime is not meaningful")
        return Unit(f"log({self.label})", Dimension.dimensionless(), Kind.QUANTITY, self.confidence)

    def sqrt(self) -> Unit:
        if not self.arithmetic_allowed or self.kind is Kind.DATETIME:
            raise UnitAlgebraError(f"sqrt is not defined for {self.kind.value} values")
        return Unit(f"sqrt({self.label})", self.dimension.root(2), Kind.QUANTITY, self.confidence)

    def abs(self) -> Unit:
        if not self.arithmetic_allowed:
            raise UnitAlgebraError(f"abs is not defined for {self.kind.value} values")
        return Unit(self.label, self.dimension, self.kind, self.confidence)

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "dimension": str(self.dimension),
            "kind": self.kind.value,
            "confidence": round(self.confidence, 3),
        }


def _min_conf(a: Unit, b: Unit) -> float:
    return min(a.confidence, b.confidence)


def _merge_kind(a: Unit, b: Unit) -> Kind:
    if a.kind is b.kind:
        return a.kind
    if Kind.UNKNOWN in (a.kind, b.kind):
        return Kind.UNKNOWN
    return Kind.QUANTITY


def _compose_label(a: Unit, b: Unit, op: str) -> str:
    left = a.label if a.label != "unitless" else "1"
    right = b.label if b.label != "unitless" else "1"
    if op == "/" and right == "1":
        return left
    if op == "*" and left == "1":
        return right
    if op == "*" and right == "1":
        return left
    return f"{left}{op}{right}"


# ---------------------------------------------------------------------------
# Common units
# ---------------------------------------------------------------------------

UNITLESS = Unit("unitless", Dimension.dimensionless(), Kind.QUANTITY, 1.0)
UNKNOWN_UNIT = Unit("unknown", Dimension.dimensionless(), Kind.UNKNOWN, 0.2)
USD = Unit("USD", Dimension.of(currency=1), Kind.QUANTITY, 1.0)
MILES = Unit("miles", Dimension.of(length=1), Kind.QUANTITY, 1.0)
KM = Unit("km", Dimension.of(length=1), Kind.QUANTITY, 1.0)
SECONDS = Unit("seconds", Dimension.of(time=1), Kind.QUANTITY, 1.0)
MINUTES = Unit("minutes", Dimension.of(time=1), Kind.QUANTITY, 1.0)
HOURS = Unit("hours", Dimension.of(time=1), Kind.QUANTITY, 1.0)
COUNT = Unit("count", Dimension.of(count=1), Kind.QUANTITY, 1.0)
TIMESTAMP = Unit("timestamp", Dimension.of(time=1), Kind.DATETIME, 1.0)
CATEGORICAL = Unit("category", Dimension.dimensionless(), Kind.CATEGORICAL, 1.0)
IDENTIFIER = Unit("id", Dimension.dimensionless(), Kind.IDENTIFIER, 1.0)
BOOLEAN = Unit("bool", Dimension.dimensionless(), Kind.BOOLEAN, 1.0)
TEXT = Unit("text", Dimension.dimensionless(), Kind.TEXT, 1.0)
DEGREES = Unit("degrees", Dimension.of(angle=1), Kind.QUANTITY, 1.0)

# Derived / composite units that show up constantly in this domain.
USD_PER_MILE = Unit("USD/miles", Dimension.of(currency=1, length=-1), Kind.QUANTITY, 1.0)
MPH = Unit("miles/hours", Dimension.of(length=1, time=-1), Kind.QUANTITY, 1.0)


def is_compatible_for_addition(a: Unit, b: Unit) -> bool:
    """Non-raising predicate used by the candidate generator's fast path."""
    try:
        a.add(b)
        return True
    except UnitAlgebraError:
        return False


def try_op(op: str, units: Iterable[Unit]) -> tuple[Unit | None, str | None]:
    """Apply `op` to `units`, returning (unit, None) or (None, reason).

    The candidate generator calls this millions of times, so it returns a
    reason string instead of raising — exceptions in a hot loop are slow and
    we want the rejection reason for telemetry anyway.
    """
    us = list(units)
    try:
        if op in {"add", "sub", "mul", "div"}:
            if len(us) != 2:
                return None, f"{op} needs exactly 2 operands"
            a, b = us
            return {
                "add": a.add,
                "sub": a.subtract,
                "mul": a.multiply,
                "div": a.divide,
            }[op](b), None
        if op in {"log", "log1p"}:
            return us[0].log(), None
        if op == "sqrt":
            return us[0].sqrt(), None
        if op == "abs":
            return us[0].abs(), None
        if op == "neg":
            return us[0], None
        return None, f"unknown operation: {op}"
    except UnitAlgebraError as exc:
        return None, str(exc)
    except IndexError:
        return None, f"{op} requires an operand"
