"""Expression language, canonicalization, caching, and parser security.

Spec sections 15.2 and 22.  The parser is a security boundary: model output
arrives as an untrusted string, so these tests assert on what it REFUSES as
much as on what it accepts.
"""

from __future__ import annotations

import numpy as np
import pytest

from signal_engine.features.cache import ExpressionCache
from signal_engine.features.canonicalize import (
    canonical_key,
    canonicalize,
    expression_hash,
    expressions_equivalent,
    suggest_name,
)
from signal_engine.features.expressions import (
    BinaryOp,
    Col,
    Const,
    UnaryOp,
    is_safe_expression,
)
from signal_engine.features.parser import (
    ExpressionParseError,
    parse_expression,
    tokenize,
)

COLUMNS = {"fare_amount", "trip_distance", "tip_amount", "extra", "passenger_count"}


class TestCanonicalHashing:
    """15.2 — commutative operations must collide; non-commutative must not."""

    def test_addition_is_commutative(self):
        a = parse_expression("fare_amount + extra", COLUMNS)
        b = parse_expression("extra + fare_amount", COLUMNS)
        assert expression_hash(a) == expression_hash(b)
        assert expressions_equivalent(a, b)

    def test_multiplication_is_commutative(self):
        a = parse_expression("fare_amount * trip_distance", COLUMNS)
        b = parse_expression("trip_distance * fare_amount", COLUMNS)
        assert expression_hash(a) == expression_hash(b)

    def test_subtraction_is_not_commutative(self):
        a = parse_expression("fare_amount - extra", COLUMNS)
        b = parse_expression("extra - fare_amount", COLUMNS)
        assert expression_hash(a) != expression_hash(b), "A-B and B-A are different quantities"

    def test_division_is_not_commutative(self):
        a = parse_expression("fare_amount / trip_distance", COLUMNS)
        b = parse_expression("trip_distance / fare_amount", COLUMNS)
        assert expression_hash(a) != expression_hash(b)

    def test_hash_is_stable_across_calls(self):
        e = parse_expression("fare_amount / trip_distance", COLUMNS)
        assert expression_hash(e) == expression_hash(e)
        assert len(expression_hash(e)) == 16

    def test_nested_commutativity_canonicalizes_bottom_up(self):
        a = parse_expression("(extra + fare_amount) * trip_distance", COLUMNS)
        b = parse_expression("trip_distance * (fare_amount + extra)", COLUMNS)
        assert expression_hash(a) == expression_hash(b)

    def test_canonical_key_is_prefix_notation(self):
        e = parse_expression("fare_amount / trip_distance", COLUMNS)
        assert canonical_key(canonicalize(e)) == "div(c:fare_amount,c:trip_distance)"


class TestAlgebraicSimplification:
    def test_double_negation_collapses(self):
        e = UnaryOp("neg", UnaryOp("neg", Col("fare_amount")))
        assert canonicalize(e) == Col("fare_amount")

    def test_abs_of_abs_collapses(self):
        e = UnaryOp("abs", UnaryOp("abs", Col("fare_amount")))
        assert canonicalize(e) == UnaryOp("abs", Col("fare_amount"))

    def test_divide_by_one_collapses(self):
        e = BinaryOp("div", Col("fare_amount"), Const(1.0))
        assert canonicalize(e) == Col("fare_amount")

    def test_add_zero_collapses(self):
        e = BinaryOp("add", Col("fare_amount"), Const(0.0))
        assert canonicalize(e) == Col("fare_amount")

    def test_integral_and_float_constants_agree(self):
        assert canonical_key(Const(2.0)) == canonical_key(Const(2))


class TestParserSecurity:
    """22 — the parser is the boundary between model output and execution."""

    @pytest.mark.parametrize(
        "malicious",
        [
            "__import__('os').system('ls')",
            "eval('1+1')",
            "exec('x=1')",
            "open('/etc/passwd').read()",
            "fare_amount.__class__.__bases__",
            "lambda: 1",
            "[x for x in range(10)]",
            "fare_amount; import os",
            "globals()",
            "().__class__",
        ],
    )
    def test_code_injection_is_rejected(self, malicious):
        with pytest.raises(ExpressionParseError):
            parse_expression(malicious, COLUMNS)

    def test_hallucinated_column_is_rejected(self):
        with pytest.raises(ExpressionParseError, match="unknown column"):
            parse_expression("nonexistent_column * 2", COLUMNS)

    def test_unlisted_function_is_rejected(self):
        with pytest.raises(ExpressionParseError, match="not allowed"):
            parse_expression("exp(fare_amount)", COLUMNS)

    def test_overlong_expression_is_rejected(self):
        with pytest.raises(ExpressionParseError, match="too long"):
            parse_expression("fare_amount + " * 200 + "extra", COLUMNS)

    def test_deeply_nested_expression_is_rejected(self):
        expression = "log(" * 20 + "fare_amount" + ")" * 20
        with pytest.raises(ExpressionParseError):
            parse_expression(expression, COLUMNS)

    def test_expression_with_no_columns_is_rejected(self):
        with pytest.raises(ExpressionParseError):
            parse_expression("1 + 2", COLUMNS)

    def test_unbalanced_parentheses_are_rejected(self):
        with pytest.raises(ExpressionParseError):
            parse_expression("(fare_amount + extra", COLUMNS)

    def test_empty_expression_is_rejected(self):
        with pytest.raises(ExpressionParseError):
            parse_expression("   ", COLUMNS)

    def test_non_string_input_is_rejected(self):
        with pytest.raises(ExpressionParseError):
            parse_expression(None, COLUMNS)  # type: ignore[arg-type]

    def test_is_safe_expression_catches_unknown_columns(self):
        expr = BinaryOp("add", Col("fare_amount"), Col("ghost_column"))
        ok, reason = is_safe_expression(expr, COLUMNS)
        assert not ok
        assert "ghost_column" in reason


class TestParserCorrectness:
    def test_operator_precedence(self):
        e = parse_expression("fare_amount + trip_distance * 2", COLUMNS)
        assert isinstance(e, BinaryOp) and e.op == "add"
        assert isinstance(e.right, BinaryOp) and e.right.op == "mul"

    def test_parentheses_override_precedence(self):
        e = parse_expression("(fare_amount + trip_distance) * 2", COLUMNS)
        assert isinstance(e, BinaryOp) and e.op == "mul"

    def test_case_insensitive_column_resolution(self):
        """A model writes `airport_fee`; the file says `Airport_fee`."""
        e = parse_expression("AIRPORT_FEE * 2", {"Airport_fee"})
        assert e.columns() == {"Airport_fee"}

    def test_unary_minus(self):
        e = parse_expression("-fare_amount", COLUMNS)
        assert isinstance(e, UnaryOp) and e.op == "neg"

    def test_allowed_functions_parse(self):
        for fn in ("log", "log1p", "sqrt", "abs"):
            e = parse_expression(f"{fn}(fare_amount)", COLUMNS)
            assert isinstance(e, UnaryOp) and e.op == fn

    def test_scientific_notation(self):
        e = parse_expression("fare_amount * 1e3", COLUMNS)
        assert e.columns() == {"fare_amount"}

    def test_columns_are_collected(self):
        e = parse_expression("(fare_amount + extra) / trip_distance", COLUMNS)
        assert e.columns() == {"fare_amount", "extra", "trip_distance"}

    def test_tokenizer_rejects_stray_characters(self):
        with pytest.raises(ExpressionParseError):
            tokenize("fare_amount @ extra")


class TestNaming:
    def test_generated_names_are_identifier_safe(self):
        e = parse_expression("fare_amount / trip_distance", COLUMNS)
        name = suggest_name(e)
        assert name.replace("_", "").isalnum()
        assert "per" in name

    def test_long_names_are_truncated_with_a_hash(self):
        expression = " + ".join(sorted(COLUMNS)) + " + " + " + ".join(sorted(COLUMNS))
        e = parse_expression(expression, COLUMNS)
        assert len(suggest_name(e)) <= 90


class TestExpressionCache:
    def test_store_and_retrieve_from_memory(self, tmp_path):
        cache = ExpressionCache(directory=tmp_path)
        key = cache.key("abc123", dataset_fingerprint="fp0000000000", view_hash="v1")
        array = np.arange(100, dtype=np.float64)

        assert cache.get(key) is None
        assert cache.stats.misses == 1

        cache.put(key, array)
        retrieved = cache.get(key)
        assert retrieved is not None
        np.testing.assert_array_equal(retrieved, array)
        assert cache.stats.memory_hits == 1

    def test_disk_cache_survives_memory_eviction(self, tmp_path):
        cache = ExpressionCache(directory=tmp_path)
        key = cache.key("abc", dataset_fingerprint="fp", view_hash="v")
        array = np.arange(50, dtype=np.float64)
        cache.put(key, array)

        cache.clear_memory()
        retrieved = cache.get(key)
        assert retrieved is not None
        np.testing.assert_array_equal(retrieved, array)
        assert cache.stats.disk_hits == 1

    def test_lru_eviction_respects_a_byte_budget(self, tmp_path):
        """The cap is in BYTES, because one wide column can be 30 MB."""
        cache = ExpressionCache(directory=None, memory_budget_bytes=8 * 100)
        for i in range(20):
            cache.put(f"k{i}", np.zeros(100, dtype=np.float64), persist=False)
        assert cache.stats.evictions > 0
        assert cache.stats.bytes_in_memory <= cache.memory_budget_bytes

    def test_different_datasets_never_share_a_key(self, tmp_path):
        cache = ExpressionCache(directory=tmp_path)
        k1 = cache.key("same_expr", dataset_fingerprint="datasetA", view_hash="v1")
        k2 = cache.key("same_expr", dataset_fingerprint="datasetB", view_hash="v1")
        assert k1 != k2, "a cached vector must never be served across datasets"

    def test_different_views_never_share_a_key(self, tmp_path):
        cache = ExpressionCache(directory=tmp_path)
        k1 = cache.key("expr", dataset_fingerprint="fp", view_hash="filtersA")
        k2 = cache.key("expr", dataset_fingerprint="fp", view_hash="filtersB")
        assert k1 != k2, "different row filters must not share a cached vector"

    def test_corrupt_disk_entry_is_a_miss_not_a_crash(self, tmp_path):
        cache = ExpressionCache(directory=tmp_path)
        key = cache.key("x", dataset_fingerprint="fp")
        path = cache._path(key)
        path.write_bytes(b"this is not a numpy file")
        assert cache.get(key) is None
