"""NYC TLC domain tests against the REAL downloaded file.

Spec section 15.6.  These assert on facts from the official Yellow Taxi data
dictionary and on physical plausibility — the "does this make sense to a
human" layer that pure numerical tests cannot reach.

Skipped automatically when the data has not been downloaded.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from signal_engine.features.dag import ExpressionDAG
from signal_engine.features.derive import (
    build_analysis_view,
    derivation_closure,
    reconstructs_target,
)
from signal_engine.features.expressions import Col
from signal_engine.features.parser import parse_expression
from signal_engine.ingestion.tlc import (
    YELLOW_ACCOUNTING_IDENTITIES,
    TLCSource,
    TLCVehicle,
)
from signal_engine.profiling.semantic_types import SemanticType
from signal_engine.profiling.units import Dimension
from signal_engine.statistics.correlation import grouped_comparison, pairwise_relationship
from tests.conftest import requires_tlc

pytestmark = [pytest.mark.integration, requires_tlc]


class TestSourceResolution:
    def test_url_follows_the_official_naming_scheme(self):
        source = TLCSource(vehicle=TLCVehicle.YELLOW, year=2026, month=1)
        assert source.local_filename() == "yellow_tripdata_2026-01.parquet"
        assert source.resolve_url().endswith("/trip-data/yellow_tripdata_2026-01.parquet")

    def test_month_is_zero_padded(self):
        assert TLCSource(year=2026, month=9).local_filename() == "yellow_tripdata_2026-09.parquet"

    def test_other_vehicles_resolve(self):
        assert "green_tripdata" in TLCSource(vehicle=TLCVehicle.GREEN).local_filename()
        assert "fhvhv_tripdata" in TLCSource(vehicle=TLCVehicle.FHVHV).local_filename()

    @pytest.mark.parametrize("month", [0, 13, -1])
    def test_invalid_month_is_rejected(self, month):
        with pytest.raises(ValueError):
            TLCSource(year=2026, month=month)


class TestSemanticTyping:
    """The dictionary-driven typing that everything else depends on."""

    def test_trip_distance_is_a_continuous_distance(self, tlc_profile):
        card = tlc_profile.card("trip_distance")
        assert card.semantic_type is SemanticType.CONTINUOUS_MEASUREMENT
        assert card.unit.dimension == Dimension.of(length=1)
        assert card.unit.label == "miles"

    @pytest.mark.parametrize(
        "column",
        ["fare_amount", "tip_amount", "total_amount", "tolls_amount", "congestion_surcharge"],
    )
    def test_money_columns_are_currency(self, tlc_profile, column):
        card = tlc_profile.card(column)
        assert card.semantic_type is SemanticType.CURRENCY
        assert card.unit.dimension == Dimension.of(currency=1)

    @pytest.mark.parametrize("column", ["RatecodeID", "payment_type", "VendorID"])
    def test_code_columns_are_categorical(self, tlc_profile, column):
        card = tlc_profile.card(column)
        assert card.semantic_type.is_categorical
        assert not card.semantic_type.is_numeric_quantity, (
            f"{column} is an integer-coded category; treating it as a quantity "
            "produces meaningless averages"
        )

    @pytest.mark.parametrize("column", ["PULocationID", "DOLocationID"])
    def test_zone_ids_are_identifiers(self, tlc_profile, column):
        card = tlc_profile.card(column)
        assert card.semantic_type is SemanticType.CATEGORICAL_IDENTIFIER
        assert column not in tlc_profile.numeric_columns()
        assert any("arbitrary" in c for c in card.caveats)

    def test_timestamps_are_datetimes(self, tlc_profile):
        for column in ("tpep_pickup_datetime", "tpep_dropoff_datetime"):
            assert tlc_profile.card(column).semantic_type is SemanticType.DATETIME

    def test_airport_fee_capitalization_variant_resolves(self, tlc_profile):
        """The file uses `Airport_fee`; the dictionary key is `airport_fee`."""
        name = next(n for n in tlc_profile.columns if n.lower() == "airport_fee")
        card = tlc_profile.card(name)
        assert card.semantic_type is SemanticType.CURRENCY
        assert card.description, "the dictionary entry must resolve despite the case difference"
        assert any("LaGuardia" in c or "fixed" in c for c in card.caveats)


class TestDictionaryCaveats:
    def test_cash_tip_caveat_is_attached(self, tlc_profile):
        card = tlc_profile.card("tip_amount")
        assert any("cash tips are NOT recorded" in c or "Cash tips are not" in c for c in card.caveats)
        assert card.blocks_claims, "the cash-tip claim block must be present"
        block = card.blocks_claims[0]
        assert "payment_type" in block["involves_columns"]

    def test_total_amount_identity_is_recorded(self, tlc_profile):
        card = tlc_profile.card("total_amount")
        assert any("SUM" in c or "sum of" in c for c in card.caveats)

    def test_fare_components_are_labelled(self, tlc_profile):
        for column in ("fare_amount", "tip_amount", "tolls_amount"):
            assert tlc_profile.card(column).is_fare_component

    def test_ratecode_flat_fare_caveat(self, tlc_profile):
        card = tlc_profile.card("RatecodeID")
        assert any("flat" in c.lower() for c in card.caveats)


class TestDerivedFeatures:
    @pytest.fixture(scope="class")
    def view(self, tlc_handle, tlc_profile):
        return build_analysis_view(pl.scan_parquet(tlc_handle.path), tlc_profile)

    def test_required_derived_features_exist(self, view):
        for name in (
            "trip_duration_seconds",
            "trip_duration_minutes",
            "pickup_hour",
            "pickup_day_of_week",
            "pickup_date",
            "fare_per_mile",
            "total_per_mile",
            "average_speed_mph",
        ):
            assert name in view.derived_units, f"{name} is a required derived feature"

    def test_duration_is_computed_correctly(self, tlc_handle, view):
        sample = (
            view.frame.select(
                ["tpep_pickup_datetime", "tpep_dropoff_datetime", "trip_duration_seconds"]
            )
            .head(500)
            .collect()
        )
        expected = (
            sample["tpep_dropoff_datetime"] - sample["tpep_pickup_datetime"]
        ).dt.total_seconds()
        np.testing.assert_allclose(
            sample["trip_duration_seconds"].to_numpy(), expected.to_numpy(), rtol=1e-9
        )

    def test_minutes_is_exactly_seconds_over_sixty(self, view):
        sample = view.frame.select(["trip_duration_seconds", "trip_duration_minutes"]).head(500).collect()
        np.testing.assert_allclose(
            sample["trip_duration_minutes"].to_numpy(),
            sample["trip_duration_seconds"].to_numpy() / 60.0,
            rtol=1e-9,
        )

    def test_fare_per_mile_has_no_infinities(self, view):
        """The division guard: zero distance must become null, never inf."""
        values = view.frame.select("fare_per_mile").head(500_000).collect()["fare_per_mile"].to_numpy()
        finite = values[~np.isnan(values)]
        assert np.all(np.isfinite(finite)), "guarded division must never produce an infinity"

    def test_units_of_derived_features_are_correct(self, view):
        assert view.derived_units["fare_per_mile"].dimension == Dimension.of(currency=1, length=-1)
        assert view.derived_units["average_speed_mph"].dimension == Dimension.of(length=1, time=-1)
        assert view.derived_units["trip_duration_seconds"].dimension == Dimension.of(time=1)

    def test_pickup_hour_is_in_range(self, view):
        hours = view.frame.select("pickup_hour").head(100_000).collect()["pickup_hour"].to_numpy()
        assert hours.min() >= 0 and hours.max() <= 23

    def test_day_of_week_is_in_range(self, view):
        days = view.frame.select("pickup_day_of_week").head(100_000).collect()["pickup_day_of_week"]
        assert days.min() >= 1 and days.max() <= 7


class TestAnalysisView:
    @pytest.fixture(scope="class")
    def view(self, tlc_handle, tlc_profile):
        return build_analysis_view(pl.scan_parquet(tlc_handle.path), tlc_profile)

    def test_no_record_is_silently_dropped(self, view):
        report = view.report
        assert report["rows_before"] > report["rows_after"]
        assert report["rows_excluded"] == report["rows_before"] - report["rows_after"]
        assert report["excluded_by_rule"], "every filter must report what it removed"
        for name in report["filters_applied"]:
            assert name in report["rationale"], f"{name} must carry a documented rationale"

    def test_exclusions_are_a_small_minority(self, view):
        """A filter set that removes most of the data is a bug, not a filter."""
        assert view.report["exclusion_fraction"] < 0.2

    def test_implausible_records_are_actually_removed(self, view):
        stats = (
            view.frame.select(
                [
                    pl.col("trip_distance").max().alias("max_distance"),
                    pl.col("trip_distance").min().alias("min_distance"),
                    pl.col("fare_amount").min().alias("min_fare"),
                    pl.col("average_speed_mph").max().alias("max_speed"),
                ]
            )
            .collect()
            .to_dicts()[0]
        )
        assert stats["max_distance"] < 200, "the 269,000-mile trip must be gone"
        assert stats["min_distance"] > 0
        assert stats["min_fare"] >= 0, "refund rows must be gone"
        assert stats["max_speed"] <= 100

    def test_view_hash_changes_with_the_filters(self, tlc_handle, tlc_profile):
        from signal_engine.features.derive import TLC_ANALYSIS_FILTERS

        full = build_analysis_view(pl.scan_parquet(tlc_handle.path), tlc_profile)
        partial = build_analysis_view(
            pl.scan_parquet(tlc_handle.path), tlc_profile, filters=TLC_ANALYSIS_FILTERS[:2]
        )
        assert full.view_hash != partial.view_hash


class TestTargetLeakage:
    """An expression that rebuilds the target is algebra, not a finding."""

    def test_total_per_mile_times_distance_is_rejected(self, tlc_handle, tlc_profile):
        view = build_analysis_view(pl.scan_parquet(tlc_handle.path), tlc_profile)
        columns = {"total_per_mile", "trip_distance"}
        assert reconstructs_target(columns, "total_amount", view.derived_provenance)

    def test_an_honest_predictor_is_not_rejected(self, tlc_handle, tlc_profile):
        view = build_analysis_view(pl.scan_parquet(tlc_handle.path), tlc_profile)
        assert not reconstructs_target(
            {"trip_distance", "trip_duration_seconds"}, "total_amount", view.derived_provenance
        )

    def test_derivation_closure_expands_transitively(self, tlc_handle, tlc_profile):
        view = build_analysis_view(pl.scan_parquet(tlc_handle.path), tlc_profile)
        closure = derivation_closure({"fare_per_mile"}, view.derived_provenance)
        assert {"fare_amount", "trip_distance"} <= closure


class TestDomainSemantics:
    """Physical-plausibility checks on real numbers."""

    @pytest.fixture(scope="class")
    def dag_and_view(self, tlc_handle, tlc_profile):
        view = build_analysis_view(pl.scan_parquet(tlc_handle.path), tlc_profile)
        dag = ExpressionDAG.from_path(
            tlc_handle.path,
            dataset_fingerprint=tlc_handle.fingerprint,
            view_hash=view.view_hash,
            frame=view.frame,
        )
        return dag, view

    def test_distance_and_fare_are_positively_related(self, dag_and_view):
        """The sanity anchor: longer trips cost more. If this inverts, the
        engine is broken, not the world."""
        dag, _ = dag_and_view
        data = dag.materialize_columns(["trip_distance", "fare_amount"])
        result = pairwise_relationship(
            data["trip_distance"], data["fare_amount"],
            x_name="trip_distance", y_name="fare_amount",
        )
        assert result.pearson_r > 0.5, (
            f"fare must rise with distance on cleaned standard-rate data; got {result.pearson_r}"
        )
        assert result.direction == "positive"

    def test_duration_and_fare_are_positively_related(self, dag_and_view):
        dag, _ = dag_and_view
        data = dag.materialize_columns(["trip_duration_seconds", "fare_amount"])
        result = pairwise_relationship(data["trip_duration_seconds"], data["fare_amount"])
        assert result.pearson_r > 0.3
        assert result.direction == "positive"

    def test_total_amount_is_mechanically_tied_to_fare_amount(self, dag_and_view):
        dag, _ = dag_and_view
        data = dag.materialize_columns(["fare_amount", "total_amount"])
        result = pairwise_relationship(data["fare_amount"], data["total_amount"])
        assert result.pearson_r > 0.9, "this is an accounting identity, so it must be near-perfect"

        identity = YELLOW_ACCOUNTING_IDENTITIES[0]
        assert identity["target"] == "total_amount"
        assert "fare_amount" in identity["components"]

    def test_cash_trips_record_near_zero_tips_by_construction(self, dag_and_view, tlc_profile):
        """The trap. The NUMBERS say cash trips tip ~0; the DICTIONARY says
        that is a recording artifact. Both must be true in the system."""
        dag, _ = dag_and_view
        data = dag.materialize_columns(["payment_type", "tip_amount"])

        cash = data["tip_amount"][data["payment_type"] == 2]
        card = data["tip_amount"][data["payment_type"] == 1]
        cash = cash[np.isfinite(cash)]
        card = card[np.isfinite(card)]

        if cash.size > 1000 and card.size > 1000:
            assert cash.mean() < card.mean(), "the raw numbers do show this gap"

        # And the metadata must make the naive conclusion unsupportable.
        blocks = tlc_profile.card("tip_amount").blocks_claims
        assert blocks
        assert "cash" in blocks[0]["forbidden_claim"].lower()
        assert "credit-card" in blocks[0]["reason"].lower()

    def test_jfk_ratecode_has_a_flat_fare(self, dag_and_view):
        """RatecodeID 2 is the JFK flat fare, so its variance must be tiny."""
        dag, _ = dag_and_view
        data = dag.materialize_columns(["RatecodeID", "fare_amount"])
        jfk = data["fare_amount"][data["RatecodeID"] == 2]
        jfk = jfk[np.isfinite(jfk)]
        if jfk.size > 500:
            assert jfk.std() < jfk.mean() * 0.35, "a flat fare must have low dispersion"

    def test_ratecode_grouping_uses_eta_not_correlation(self, dag_and_view, tlc_profile):
        dag, _ = dag_and_view
        data = dag.materialize_columns(["RatecodeID", "fare_amount"])
        result = grouped_comparison(
            data["fare_amount"], data["RatecodeID"],
            value_name="fare_amount", group_name="RatecodeID",
            labels=tlc_profile.card("RatecodeID").category_labels,
        )
        assert result.method == "grouped_eta_squared"
        assert result.pearson_r is None, "a category must never get a correlation coefficient"
        assert result.extra["group_means"]

    def test_average_speed_is_physically_plausible(self, dag_and_view):
        dag, _ = dag_and_view
        speed = dag.materialize(Col("average_speed_mph"))
        speed = speed[np.isfinite(speed)]
        median = float(np.median(speed))
        assert 3.0 < median < 30.0, f"NYC median taxi speed should be ~10-15 mph, got {median}"


class TestExpressionsOnRealData:
    def test_parsed_expression_matches_manual_computation(self, tlc_handle, tlc_profile):
        view = build_analysis_view(pl.scan_parquet(tlc_handle.path), tlc_profile)
        dag = ExpressionDAG.from_path(
            tlc_handle.path,
            dataset_fingerprint=tlc_handle.fingerprint,
            view_hash=view.view_hash,
            frame=view.frame,
        )
        expr = parse_expression("fare_amount / trip_distance", set(dag.allowed_columns))
        computed = dag.materialize(expr)
        base = dag.materialize_columns(["fare_amount", "trip_distance"])
        expected = base["fare_amount"] / base["trip_distance"]

        mask = np.isfinite(computed) & np.isfinite(expected)
        assert mask.sum() > 1000
        np.testing.assert_allclose(computed[mask], expected[mask], rtol=1e-9)

    def test_cache_hit_on_repeated_materialization(self, tlc_handle, tlc_profile, tmp_path):
        view = build_analysis_view(pl.scan_parquet(tlc_handle.path), tlc_profile)
        dag = ExpressionDAG.from_path(
            tlc_handle.path,
            dataset_fingerprint=tlc_handle.fingerprint,
            cache_dir=tmp_path,
            view_hash=view.view_hash,
            frame=view.frame,
        )
        expr = parse_expression("fare_amount / trip_distance", set(dag.allowed_columns))
        dag.materialize(expr)
        materializations = dag.stats.materializations
        dag.materialize(expr)
        assert dag.stats.materializations == materializations, "the second call must hit the cache"
        assert dag.stats.cache_hits >= 1

    def test_only_needed_columns_are_projected(self, tlc_handle, tlc_profile):
        """Column projection: reading 2 of 20 columns, not all 20."""
        view = build_analysis_view(pl.scan_parquet(tlc_handle.path), tlc_profile)
        dag = ExpressionDAG.from_path(
            tlc_handle.path,
            dataset_fingerprint=tlc_handle.fingerprint,
            view_hash=view.view_hash,
            frame=view.frame,
        )
        expr = parse_expression("fare_amount / trip_distance", set(dag.allowed_columns))
        dag.materialize(expr)
        assert dag.stats.columns_projected == 2
