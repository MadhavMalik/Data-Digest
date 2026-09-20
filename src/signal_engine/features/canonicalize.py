"""Canonical forms and stable hashes for expressions.

Two expressions that compute the same thing must hash the same, or the cache
misses and we pay to recompute work we already did.  The rules:

    add(A,B) == add(B,A)        commutative -> operands sorted
    mul(A,B) == mul(B,A)        commutative -> operands sorted
    sub(A,B) != sub(B,A)        NOT commutative -> order preserved
    div(A,B) != div(B,A)        NOT commutative -> order preserved

Sorting uses each operand's own canonical key, so canonicalization is
bottom-up and stable regardless of how the tree was built.

Beyond commutativity we apply a few algebraic identities that show up
constantly in generated candidates (double negation, `x/1`, `abs(abs(x))`).
Each is a genuine identity over the reals with the guards applied, so
collapsing them is safe, not merely convenient.
"""

from __future__ import annotations

import hashlib

from signal_engine.features.expressions import (
    COMMUTATIVE,
    BinaryOp,
    Col,
    Const,
    Expr,
    UnaryOp,
)

HASH_LENGTH = 16


def canonicalize(expr: Expr) -> Expr:
    """Rewrite `expr` into its canonical form (bottom-up)."""
    if isinstance(expr, (Col, Const)):
        return expr

    if isinstance(expr, UnaryOp):
        inner = canonicalize(expr.operand)
        # -(-x) == x  and  abs(abs(x)) == abs(x)
        if expr.op == "neg" and isinstance(inner, UnaryOp) and inner.op == "neg":
            return inner.operand
        if expr.op == "abs" and isinstance(inner, UnaryOp) and inner.op in {"abs", "neg"}:
            return UnaryOp("abs", inner.operand)
        return UnaryOp(expr.op, inner)

    if isinstance(expr, BinaryOp):
        left = canonicalize(expr.left)
        right = canonicalize(expr.right)

        # x / 1 == x ;  x * 1 == x ;  x + 0 == x ;  x - 0 == x
        if isinstance(right, Const):
            if expr.op in {"div", "mul"} and right.value == 1.0:
                return left
            if expr.op in {"add", "sub"} and right.value == 0.0:
                return left
        if isinstance(left, Const) and expr.op == "mul" and left.value == 1.0:
            return right

        if expr.op in COMMUTATIVE:
            a, b = sorted((left, right), key=canonical_key)
            return BinaryOp(expr.op, a, b)
        return BinaryOp(expr.op, left, right)

    return expr


def canonical_key(expr: Expr) -> str:
    """A deterministic prefix-notation string for a canonicalized expression.

    Prefix notation means no parentheses and no operator-precedence ambiguity,
    so string equality is exactly structural equality.
    """
    if isinstance(expr, Col):
        return f"c:{expr.name}"
    if isinstance(expr, Const):
        # Normalize -0.0 and integral floats so 2 and 2.0 agree.
        v = expr.value + 0.0
        return f"k:{int(v)}" if float(v).is_integer() else f"k:{v!r}"
    if isinstance(expr, UnaryOp):
        return f"{expr.op}({canonical_key(expr.operand)})"
    if isinstance(expr, BinaryOp):
        if expr.op in COMMUTATIVE:
            keys = sorted((canonical_key(expr.left), canonical_key(expr.right)))
        else:
            keys = [canonical_key(expr.left), canonical_key(expr.right)]
        return f"{expr.op}({keys[0]},{keys[1]})"
    raise TypeError(f"not an expression node: {type(expr)!r}")


def expression_hash(expr: Expr) -> str:
    """Stable short hash of an expression's canonical form."""
    key = canonical_key(canonicalize(expr))
    return hashlib.sha256(key.encode()).hexdigest()[:HASH_LENGTH]


def expressions_equivalent(a: Expr, b: Expr) -> bool:
    return canonical_key(canonicalize(a)) == canonical_key(canonicalize(b))


def suggest_name(expr: Expr) -> str:
    """A readable, filesystem-safe column name for a derived feature."""

    def render(node: Expr) -> str:
        if isinstance(node, Col):
            return node.name
        if isinstance(node, Const):
            return node.display().replace(".", "p").replace("-", "neg")
        if isinstance(node, UnaryOp):
            return f"{node.op}_{render(node.operand)}"
        if isinstance(node, BinaryOp):
            word = {"add": "plus", "sub": "minus", "mul": "times", "div": "per"}[node.op]
            return f"{render(node.left)}_{word}_{render(node.right)}"
        raise TypeError(type(node))

    name = render(canonicalize(expr))
    name = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in name)
    if len(name) > 90:
        name = f"{name[:80]}_{expression_hash(expr)[:8]}"
    return name
