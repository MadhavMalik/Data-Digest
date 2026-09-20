"""A whitelisted expression parser.

This is a security boundary, not a convenience.  Model output reaches this
function as an untrusted string and leaves as a typed AST containing only
nodes from `expressions.py`.  There is no `eval`, no `exec`, no `compile`, and
no attribute access anywhere in the path.

Grammar (recursive descent, standard precedence):

    expression := term (("+" | "-") term)*
    term       := factor (("*" | "/") factor)*
    factor     := ("-")? primary
    primary    := NUMBER | FUNC "(" expression ")" | IDENT | "(" expression ")"

Identifiers must appear in the allowed-column set, so a hallucinated column
name fails here rather than at the dataframe layer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from signal_engine.features.expressions import (
    BinaryOp,
    Col,
    Const,
    Expr,
    UnaryOp,
)

ALLOWED_FUNCTIONS = frozenset({"log", "log1p", "sqrt", "abs"})
MAX_EXPRESSION_LENGTH = 400
MAX_DEPTH = 12


class ExpressionParseError(ValueError):
    """Raised when an expression is malformed, too complex, or references an
    unknown column or function."""


_TOKEN_RE = re.compile(
    r"""
    (?P<ws>\s+)
  | (?P<number>\d+\.\d*(?:[eE][-+]?\d+)?|\.\d+(?:[eE][-+]?\d+)?|\d+(?:[eE][-+]?\d+)?)
  | (?P<ident>[A-Za-z_][A-Za-z0-9_]*)
  | (?P<op>[-+*/()])
    """,
    re.VERBOSE,
)


@dataclass(frozen=True)
class Token:
    kind: str
    text: str
    pos: int


def tokenize(source: str) -> list[Token]:
    tokens: list[Token] = []
    pos = 0
    while pos < len(source):
        m = _TOKEN_RE.match(source, pos)
        if not m:
            raise ExpressionParseError(f"unexpected character {source[pos]!r} at position {pos}")
        kind = m.lastgroup or ""
        if kind != "ws":
            tokens.append(Token(kind, m.group(), pos))
        pos = m.end()
    return tokens


class _Parser:
    def __init__(self, tokens: list[Token], allowed_columns: set[str]) -> None:
        self.tokens = tokens
        self.i = 0
        self.allowed = allowed_columns
        # Case-insensitive resolution, because a model will confidently write
        # `airport_fee` when the file says `Airport_fee`.
        self.lower_map = {c.lower(): c for c in allowed_columns}

    # ---- token helpers --------------------------------------------------
    def peek(self) -> Token | None:
        return self.tokens[self.i] if self.i < len(self.tokens) else None

    def next(self) -> Token:
        tok = self.peek()
        if tok is None:
            raise ExpressionParseError("unexpected end of expression")
        self.i += 1
        return tok

    def accept(self, text: str) -> bool:
        tok = self.peek()
        if tok is not None and tok.text == text:
            self.i += 1
            return True
        return False

    def expect(self, text: str) -> None:
        if not self.accept(text):
            tok = self.peek()
            got = tok.text if tok else "end of expression"
            raise ExpressionParseError(f"expected {text!r} but found {got!r}")

    # ---- grammar --------------------------------------------------------
    def parse(self) -> Expr:
        expr = self.expression()
        if self.peek() is not None:
            raise ExpressionParseError(f"unexpected trailing input at position {self.peek().pos}")
        return expr

    def expression(self) -> Expr:
        node = self.term()
        while True:
            if self.accept("+"):
                node = BinaryOp("add", node, self.term())
            elif self.accept("-"):
                node = BinaryOp("sub", node, self.term())
            else:
                return node

    def term(self) -> Expr:
        node = self.factor()
        while True:
            if self.accept("*"):
                node = BinaryOp("mul", node, self.factor())
            elif self.accept("/"):
                node = BinaryOp("div", node, self.factor())
            else:
                return node

    def factor(self) -> Expr:
        if self.accept("-"):
            return UnaryOp("neg", self.factor())
        if self.accept("+"):
            return self.factor()
        return self.primary()

    def primary(self) -> Expr:
        tok = self.next()

        if tok.kind == "number":
            return Const(float(tok.text))

        if tok.kind == "ident":
            nxt = self.peek()
            if nxt is not None and nxt.text == "(":
                fname = tok.text.lower()
                if fname not in ALLOWED_FUNCTIONS:
                    raise ExpressionParseError(
                        f"function {tok.text!r} is not allowed "
                        f"(allowed: {', '.join(sorted(ALLOWED_FUNCTIONS))})"
                    )
                self.expect("(")
                inner = self.expression()
                self.expect(")")
                return UnaryOp(fname, inner)

            resolved = self.lower_map.get(tok.text.lower())
            if resolved is None:
                raise ExpressionParseError(
                    f"unknown column {tok.text!r}; it is not present in this dataset"
                )
            return Col(resolved)

        if tok.text == "(":
            inner = self.expression()
            self.expect(")")
            return inner

        raise ExpressionParseError(f"unexpected token {tok.text!r} at position {tok.pos}")


def parse_expression(source: str, allowed_columns: set[str] | list[str]) -> Expr:
    """Parse `source` into an AST, or raise ExpressionParseError.

    Never raises anything else, so callers can treat a bad model response as a
    normal, recoverable outcome.
    """
    if not isinstance(source, str):
        raise ExpressionParseError(f"expression must be a string, got {type(source).__name__}")
    source = source.strip()
    if not source:
        raise ExpressionParseError("empty expression")
    if len(source) > MAX_EXPRESSION_LENGTH:
        raise ExpressionParseError(
            f"expression too long ({len(source)} > {MAX_EXPRESSION_LENGTH} characters)"
        )

    allowed = set(allowed_columns)
    tokens = tokenize(source)
    if not tokens:
        raise ExpressionParseError("expression contains no tokens")

    expr = _Parser(tokens, allowed).parse()

    if expr.depth > MAX_DEPTH:
        raise ExpressionParseError(f"expression nested too deeply ({expr.depth} > {MAX_DEPTH})")
    if not expr.columns():
        raise ExpressionParseError("expression references no dataset columns")
    return expr
