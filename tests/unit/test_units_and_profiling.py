"""Dimensional analysis, semantic typing, and the profiler.

Spec section 15.3.  The dimensional layer is the engine's largest optimization
AND a correctness guard, so these tests pin both: what it rejects (nonsense)
and what it must keep (genuinely meaningful composites).
"""

from __future__ import annotations

import polars as pl
import pytest

from signal_engine.features.transforms import (
    generate_candidates,
    theoretical_space_size,
)
from signal_engine.profiling import units as U
from signal_engine.profiling.semantic_types import (
    SemanticType,
    ValueEvidence,
    infer_semantic_type,
)
from signal_engine.profiling.units import Dimension, Kind, Unit, UnitAlgebraError, try_op


class TestDimensionAlgebra:
    def test_multiplication_adds_exponents(self):
        assert U.MILES.multiply(U.MILES).dimension == Dimension.of(length=2)

    def test_division_subtracts_exponents(self):
        speed = U.MILES.divide(U.HOURS)
        assert speed.dimension == Dimension.of(length=1, time=-1)

    def test_dimension_is_hashable_and_order_independent(self):
        assert Dimension.of(currency=1, length=-1) == Dimension.of(length=-1, currency=1)
        assert hash(Dimension.of(currency=1)) == hash(Dimension.of(currency=1))

    def test_inverse_round_trips(self):
        d = Dimension.of(currency=1, length=-1)
        assert d.inverse().inverse() == d

    def test_root_of_odd_exponent_is_refused(self):
        with pytest.raises(UnitAlgebraError):
            Dimension.of(length=1).root(2)

    def test_root_of_even_exponent_works(self):
        assert Dimension.of(length=2).root(2) == Dimension.of(length=1)


class TestRejectedOperations:
    """These are the combinations the pruning stage must delete."""

    def test_dollars_plus_miles_is_rejected(self):
        with pytest.raises(UnitAlgebraError, match="cannot add"):
            U.USD.add(U.MILES)

    def test_identifier_arithmetic_is_rejected(self):
        """A zone ID has no magnitude, so nothing may be computed with it."""
        for op in ("add", "sub", "mul", "div"):
            unit, reason = try_op(op, [U.IDENTIFIER, U.USD])
            assert unit is None
            assert "identifier" in reason.lower()

    def test_categorical_arithmetic_is_rejected(self):
        unit, reason = try_op("div", [U.CATEGORICAL, U.USD])
        assert unit is None
        assert "categorical" in reason.lower()

    def test_datetime_plus_datetime_is_rejected(self):
        with pytest.raises(UnitAlgebraError):
            U.TIMESTAMP.add(U.TIMESTAMP)

    def test_datetime_times_anything_is_rejected(self):
        with pytest.raises(UnitAlgebraError):
            U.TIMESTAMP.multiply(U.USD)

    def test_currency_plus_seconds_is_rejected(self):
        unit, reason = try_op("add", [U.USD, U.SECONDS])
        assert unit is None


class TestAllowedOperations:
    """These must survive: rejecting them would destroy real signal."""

    def test_dollars_per_mile(self):
        unit = U.USD.divide(U.MILES)
        assert unit.dimension == Dimension.of(currency=1, length=-1)
        assert unit.label == "USD/miles"

    def test_miles_per_hour(self):
        assert U.MILES.divide(U.HOURS).dimension == Dimension.of(length=1, time=-1)

    def test_datetime_minus_datetime_is_a_duration(self):
        duration = U.TIMESTAMP.subtract(U.TIMESTAMP)
        assert duration.dimension == Dimension.of(time=1)
        assert duration.kind is Kind.QUANTITY

    def test_duration_added_to_datetime_stays_a_datetime(self):
        assert U.TIMESTAMP.add(U.SECONDS).kind is Kind.DATETIME

    def test_count_per_duration(self):
        assert U.COUNT.divide(U.HOURS).dimension == Dimension.of(count=1, time=-1)

    def test_same_units_add(self):
        assert U.USD.add(U.USD).dimension == Dimension.of(currency=1)


class TestUnitConfidence:
    def test_confidence_propagates_as_the_minimum(self):
        low = Unit("USD", Dimension.of(currency=1), Kind.QUANTITY, 0.4)
        assert U.USD.divide(low).confidence == 0.4

    def test_log_records_provenance_in_the_label(self):
        """log() of dollars is dimensionless, but must not look like dollars."""
        logged = U.USD.log()
        assert logged.dimension.is_dimensionless
        assert logged.label == "log(USD)"


class TestSemanticTypeInference:
    def test_official_metadata_always_wins(self):
        evidence = ValueEvidence(is_integral=True, distinct_count=250, non_null_count=100_000)
        inference = infer_semantic_type(
            "PULocationID",
            evidence,
            {"semantic_type": "categorical_identifier", "unit": "id"},
        )
        assert inference.semantic_type is SemanticType.CATEGORICAL_IDENTIFIER
        assert inference.confidence == 1.0
        assert inference.rule == "official_metadata"

    def test_integer_id_column_is_not_a_quantity(self):
        """The failure mode this whole layer exists to prevent."""
        evidence = ValueEvidence(
            is_integral=True, distinct_count=263, non_null_count=3_000_000, min_value=1, max_value=265
        )
        inference = infer_semantic_type("PULocationID", evidence, None)
        assert inference.semantic_type is SemanticType.CATEGORICAL_IDENTIFIER
        assert not inference.semantic_type.is_numeric_quantity

    def test_low_cardinality_integer_is_treated_as_a_code(self):
        evidence = ValueEvidence(
            is_integral=True, distinct_count=6, non_null_count=3_000_000, min_value=1, max_value=99
        )
        inference = infer_semantic_type("payment_type", evidence, None)
        assert inference.semantic_type is SemanticType.CATEGORICAL
        assert any("distinct" in n for n in inference.notes)

    def test_high_cardinality_float_is_continuous(self):
        evidence = ValueEvidence(
            is_float=True,
            has_fractional=True,
            distinct_count=50_000,
            non_null_count=3_000_000,
            min_value=0.0,
            max_value=200.0,
        )
        inference = infer_semantic_type("trip_distance", evidence, None)
        assert inference.semantic_type.is_numeric_quantity

    def test_currency_name_heuristic(self):
        evidence = ValueEvidence(is_float=True, has_fractional=True, distinct_count=9000, non_null_count=10**6)
        inference = infer_semantic_type("total_amount", evidence, None)
        assert inference.semantic_type is SemanticType.CURRENCY
        assert inference.unit.dimension == Dimension.of(currency=1)

    def test_timestamp_dtype_is_authoritative(self):
        inference = infer_semantic_type("whatever", ValueEvidence(is_temporal=True), None)
        assert inference.semantic_type is SemanticType.DATETIME

    def test_latitude_needs_both_name_and_range(self):
        in_range = ValueEvidence(is_float=True, min_value=40.0, max_value=41.0, distinct_count=10**5, non_null_count=10**6)
        assert infer_semantic_type("pickup_latitude", in_range, None).semantic_type is SemanticType.GEO_COORDINATE

        out_of_range = ValueEvidence(is_float=True, min_value=-500.0, max_value=900.0, distinct_count=10**5, non_null_count=10**6)
        assert infer_semantic_type("pickup_latitude", out_of_range, None).semantic_type is not SemanticType.GEO_COORDINATE

    def test_high_cardinality_string_is_text(self):
        evidence = ValueEvidence(is_string=True, distinct_count=900_000, non_null_count=10**6)
        assert infer_semantic_type("notes", evidence, None).semantic_type is SemanticType.TEXT

    def test_binary_string_is_boolean(self):
        evidence = ValueEvidence(is_string=True, distinct_count=2, non_null_count=10**6)
        assert infer_semantic_type("store_and_fwd_flag", evidence, None).semantic_type is SemanticType.BOOLEAN


class TestDimensionalPruning:
    """The optimization: nonsense deleted before any row is read."""

    def test_incoherent_candidates_are_pruned(self):
        units = {
            "fare_amount": U.USD,
            "trip_distance": U.MILES,
            "PULocationID": U.IDENTIFIER,
            "payment_type": U.CATEGORICAL,
            "pickup_time": U.TIMESTAMP,
        }
        candidates, stats = generate_candidates(list(units), units)

        assert stats.pruned_by_units > 0
        assert stats.prune_rate > 0.5, "most of this candidate space is nonsense"

        emitted = {c.display for c in candidates}
        # Nothing involving an identifier or a category may survive.
        assert not any("PULocationID" in e for e in emitted)
        assert not any("payment_type" in e for e in emitted)
        # Nothing may add dollars to miles.
        assert not any("fare_amount + trip_distance" in e for e in emitted)

    def test_meaningful_ratio_survives(self):
        units = {"fare_amount": U.USD, "trip_distance": U.MILES}
        candidates, _ = generate_candidates(list(units), units)
        labels = {c.unit.label for c in candidates}
        assert "USD/miles" in labels, "cost per mile must survive pruning"

    def test_rejection_reasons_are_recorded(self):
        units = {"fare": U.USD, "dist": U.MILES, "zone": U.IDENTIFIER}
        _, stats = generate_candidates(list(units), units)
        assert stats.reasons, "every rejection must be attributable"
        assert any("identifier" in r for r in stats.reasons)

    def test_log_is_only_proposed_for_positive_columns(self):
        units = {"a": U.USD, "b": U.USD}
        candidates, stats = generate_candidates(list(units), units, positive_only={"a"})
        logs = {c.display for c in candidates if c.display.startswith("log(")}
        assert "log(a)" in logs
        assert "log(b)" not in logs

    def test_abs_of_a_positive_column_is_not_proposed(self):
        units = {"a": U.USD}
        candidates, _ = generate_candidates(list(units), units, positive_only={"a"})
        assert not any(c.display.startswith("abs(") for c in candidates)

    def test_candidate_budget_is_respected(self):
        units = {f"c{i}": U.USD for i in range(30)}
        candidates, stats = generate_candidates(list(units), units, max_candidates=25)
        assert len(candidates) <= 25
        assert stats.pruned_by_budget > 0

    def test_excluded_hashes_are_not_re_emitted(self):
        units = {"a": U.USD, "b": U.MILES}
        first, _ = generate_candidates(list(units), units)
        seen = {c.expr_hash for c in first}
        second, _ = generate_candidates(list(units), units, exclude_hashes=seen)
        assert not second

    def test_theoretical_space_grows_as_documented(self):
        assert theoretical_space_size(10, 1) < theoretical_space_size(10, 2)
        assert theoretical_space_size(1, 1) == 0


class TestProfiler:
    def test_profile_covers_every_column(self, synthetic_profile, synthetic_frame):
        assert len(synthetic_profile.columns) == len(synthetic_frame.columns)

    def test_row_count_is_exact_not_sampled(self, synthetic_profile, synthetic_frame):
        for card in synthetic_profile.columns.values():
            assert card.row_count == synthetic_frame.height

    def test_numeric_summary_matches_the_data(self, synthetic_profile, synthetic_frame):
        card = synthetic_profile.card("fare_amount")
        actual = synthetic_frame["fare_amount"]
        assert card.numeric.min == pytest.approx(actual.min(), rel=1e-6)
        assert card.numeric.max == pytest.approx(actual.max(), rel=1e-6)
        assert card.numeric.mean == pytest.approx(actual.mean(), rel=1e-6)

    def test_zone_id_is_typed_as_an_identifier(self, synthetic_profile):
        assert synthetic_profile.card("zone_id").semantic_type in {
            SemanticType.CATEGORICAL_IDENTIFIER,
            SemanticType.CATEGORICAL,
        }
        assert "zone_id" not in synthetic_profile.numeric_columns()

    def test_approximate_metrics_are_labelled(self, synthetic_profile):
        card = synthetic_profile.card("fare_amount")
        assert "distinct_count" in card.approximate_metrics
        assert card.categories.approximate is True

    def test_prompt_text_is_compact(self, synthetic_profile):
        """The whole point: the model sees kilobytes, not the dataset."""
        text = synthetic_profile.to_prompt_text()
        assert len(text) < 8000
        assert "fare_amount" in text

    def test_profile_cache_round_trips(self, synthetic_handle, tmp_path):
        from signal_engine.profiling.profiler import profile_dataset

        first = profile_dataset(synthetic_handle, cache_dir=tmp_path)
        assert first.stats["cache_hit"] is False
        second = profile_dataset(synthetic_handle, cache_dir=tmp_path)
        assert second.stats["cache_hit"] is True
        assert set(first.columns) == set(second.columns)
        assert first.card("fare_amount").semantic_type == second.card("fare_amount").semantic_type

    def test_single_pass_aggregation_is_used(self, synthetic_handle, tmp_path):
        from signal_engine.profiling.profiler import profile_dataset

        profile = profile_dataset(synthetic_handle, cache_dir=tmp_path, use_cache=False)
        assert profile.stats["aggregations_in_single_pass"] > len(profile.columns)


class TestAccountingIdentityWarnings:
    def test_identity_members_are_annotated(self, tmp_path):
        from signal_engine.ingestion.base import register_local_dataset
        from signal_engine.profiling.profiler import profile_dataset

        path = tmp_path / "fares.parquet"
        pl.DataFrame(
            {"total_amount": [10.0, 20.0, 30.0] * 100, "fare_amount": [8.0, 16.0, 24.0] * 100}
        ).write_parquet(path)

        handle = register_local_dataset(path, dataset_id="fares")
        profile = profile_dataset(
            handle,
            accounting_identities=[
                {"target": "total_amount", "components": ["fare_amount"], "relation": "sum"}
            ],
            use_cache=False,
        )
        assert any("sum" in c for c in profile.card("total_amount").caveats)
        assert any("component" in c for c in profile.card("fare_amount").caveats)

    def test_identity_matching_is_case_insensitive(self, tmp_path):
        """TLC alternates `airport_fee` / `Airport_fee` between months."""
        from signal_engine.ingestion.base import register_local_dataset
        from signal_engine.profiling.profiler import profile_dataset

        path = tmp_path / "ap.parquet"
        pl.DataFrame(
            {"total_amount": [10.0] * 200, "Airport_fee": [1.75] * 200}
        ).write_parquet(path)
        handle = register_local_dataset(path, dataset_id="ap")
        profile = profile_dataset(
            handle,
            accounting_identities=[
                {"target": "total_amount", "components": ["airport_fee"], "relation": "sum"}
            ],
            use_cache=False,
        )
        assert any("component" in c for c in profile.card("Airport_fee").caveats)
