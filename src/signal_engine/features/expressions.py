"""A tiny, closed expression language.

The LLM may *propose* features, but it never gets to run code.  It emits a
string in this grammar, a whitelisted parser turns it into this AST, and only
then does the AST compile to a Polars expression.  `eval`/`exec` appear nowhere
in the engine.

Every operation carries three things: how it evaluates (Polars), how its unit
derives (dimensional analysis), and what guard protects it (division by
near-zero, log of a non-positive, sqrt of a negative).  Keeping them in one
place is what stops the three from drifting apart.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

import polars as pl

from signal_engine.profiling.units import Unit, try_op

# Denominators smaller than this are treated as missing rather than producing
# an infinity.  Absolute (not relative) because the alternative -- silently
# rescaling -- would change the feature's meaning.
DIVISION_EPSILON = 1e-9

BinaryOpName = Literal["add", "sub", "mul", "div"]
UnaryOpName = Literal["log", "log1p", "sqrt", "abs", "neg"]

COMMUTATIVE: frozenset[str] = frozenset({"add", "mul"})

BINARY_SYMBOLS: dict[str, str] = {"add": "+", "sub": "-", "mul": "*", "div": "/"}


class Expr:
    """Base class for expression nodes.  Nodes are frozen and hashable."""

    # ---- traversal ------------------------------------------------------
    def children(self) -> tuple[Expr, ...]:
        return ()

    def walk(self) -> Iterator[Expr]:
        yield self
        for child in self.children():
            yield from child.walk()

    def columns(self) -> set[str]:
        return {n.name for n in self.walk() if isinstance(n, Col)}

    @property
    def depth(self) -> int:
        kids = self.children()
        return 1 + max((c.depth for c in kids), default=0)

    @property
    def size(self) -> int:
        return 1 + sum(c.size for c in self.children())

    # ---- compilation / typing (implemented by subclasses) ---------------
    def to_polars(self) -> pl.Expr:  # pragma: no cover - abstract
        raise NotImplementedError

    def unit(self, units: dict[str, Unit]) -> tuple[Unit | None, str | None]:  # pragma: no cover
        raise NotImplementedError

    def display(self) -> str:  # pragma: no cover - abstract
        raise NotImplementedError

    def __str__(self) -> str:
        return self.display()

    # ---- sugar ----------------------------------------------------------
    def __add__(self, other: Expr) -> Expr:
        return BinaryOp("add", self, other)

    def __sub__(self, other: Expr) -> Expr:
        return BinaryOp("sub", self, other)

    def __mul__(self, other: Expr) -> Expr:
        return BinaryOp("mul", self, other)

    def __truediv__(self, other: Expr) -> Expr:
        return BinaryOp("div", self, other)


@dataclass(frozen=True)
class Col(Expr):
    """A reference to a dataset column."""

    name: str

    def to_polars(self) -> pl.Expr:
        return pl.col(self.name)

    def unit(self, units: dict[str, Unit]) -> tuple[Unit | None, str | None]:
        u = units.get(self.name)
        if u is None:
            return None, f"unknown column: {self.name}"
        return u, None

    def display(self) -> str:
        return self.name


@dataclass(frozen=True)
class Const(Expr):
    """A numeric literal.  Dimensionless by construction."""

    value: float

    def to_polars(self) -> pl.Expr:
        return pl.lit(self.value)

    def unit(self, units: dict[str, Unit]) -> tuple[Unit | None, str | None]:
        from signal_engine.profiling.units import UNITLESS

        return UNITLESS, None

    def display(self) -> str:
        v = self.value
        return str(int(v)) if float(v).is_integer() else repr(v)


@dataclass(frozen=True)
class UnaryOp(Expr):
    op: str
    operand: Expr

    def children(self) -> tuple[Expr, ...]:
        return (self.operand,)

    def to_polars(self) -> pl.Expr:
        inner = self.operand.to_polars()
        if self.op == "neg":
            return -inner
        if self.op == "abs":
            return inner.abs()
        if self.op == "sqrt":
            # sqrt of a negative would be NaN; make it explicitly missing.
            return pl.when(inner >= 0).then(inner.sqrt()).otherwise(None)
        if self.op == "log":
            return pl.when(inner > 0).then(inner.log()).otherwise(None)
        if self.op == "log1p":
            return pl.when(inner > -1).then(inner.log1p()).otherwise(None)
        raise ValueError(f"unsupported unary op: {self.op}")

    def unit(self, units: dict[str, Unit]) -> tuple[Unit | None, str | None]:
        inner, err = self.operand.unit(units)
        if err or inner is None:
            return None, err
        return try_op(self.op, [inner])

    def display(self) -> str:
        if self.op == "neg":
            return f"-({self.operand.display()})"
        return f"{self.op}({self.operand.display()})"


@dataclass(frozen=True)
class BinaryOp(Expr):
    op: str
    left: Expr
    right: Expr

    def children(self) -> tuple[Expr, ...]:
        return (self.left, self.right)

    def to_polars(self) -> pl.Expr:
        a, b = self.left.to_polars(), self.right.to_polars()
        if self.op == "add":
            return a + b
        if self.op == "sub":
            return a - b
        if self.op == "mul":
            return a * b
        if self.op == "div":
            # Guard, not clamp: a near-zero denominator makes the result
            # undefined for that row, and the statistics engine excludes it.
            return pl.when(b.abs() > DIVISION_EPSILON).then(a / b).otherwise(None)
        raise ValueError(f"unsupported binary op: {self.op}")

    def unit(self, units: dict[str, Unit]) -> tuple[Unit | None, str | None]:
        lu, err = self.left.unit(units)
        if err or lu is None:
            return None, err
        ru, err = self.right.unit(units)
        if err or ru is None:
            return None, err
        return try_op(self.op, [lu, ru])

    def display(self) -> str:
        sym = BINARY_SYMBOLS[self.op]
        return f"({self.left.display()} {sym} {self.right.display()})"


# ---------------------------------------------------------------------------
# Constructors
# ---------------------------------------------------------------------------


def col(name: str) -> Col:
    return Col(name)


def const(value: float) -> Const:
    return Const(float(value))


def binary(op: str, left: Expr, right: Expr) -> BinaryOp:
    if op not in BINARY_SYMBOLS:
        raise ValueError(f"unsupported binary op: {op}")
    return BinaryOp(op, left, right)


def unary(op: str, operand: Expr) -> UnaryOp:
    if op not in {"log", "log1p", "sqrt", "abs", "neg"}:
        raise ValueError(f"unsupported unary op: {op}")
    return UnaryOp(op, operand)


# ---------------------------------------------------------------------------
# Datetime-derived features
# ---------------------------------------------------------------------------
#
# Duration and calendar parts are not expressible in the arithmetic grammar
# above (a timestamp is not a number), so they are materialised as real columns
# by `derive.build_analysis_view` and then referenced as ordinary Col nodes.
# That keeps ONE evaluation path instead of two.


def is_safe_expression(expr: Expr, allowed_columns: set[str]) -> tuple[bool, str | None]:
    """Reject anything referencing a column that does not exist.

    This is the gate that stops an LLM-hallucinated column name from ever
    reaching the dataframe layer.
    """
    referenced = expr.columns()
    unknown = referenced - allowed_columns
    if unknown:
        return False, f"unknown column(s): {', '.join(sorted(unknown))}"
    if not referenced:
        return False, "expression references no dataset columns"
    return True, None
