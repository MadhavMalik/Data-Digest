"""The engine must work on datasets it has never seen.

The NYC TLC path is well-covered elsewhere. These tests use a completely
different domain — energy-grid dispatch — with no curated dictionary, to prove
that the semantics layer, the derived features, the filters and the target
inference all work from first principles rather than from hardcoded column
names.

The headline test injects a known per-plant price premium and asserts the
residual stage recovers it. That is the whole product claim, on a dataset the
engine has no knowledge of.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from signal_engine.datasets import infer_target_generic, resolve_spec, spec_for_key
from signal_engine.features.derive import build_analysis_view, generic_filters
from signal_engine.ingestion.base import register_local_dataset
from signal_engine.profiling.profiler import profile_dataset
from signal_engine.profiling.semantic_types import SemanticType
from signal_engine.statistics.residual import analyze_residual

# The per-plant premium the generator injects on top of the price model.
PLANT_PREMIUM = np.array([0.0, 18.0, -9.0, 31.0, 4.0, -14.0])


@pytest.fixture(scope="module")
def energy_path(tmp_path_factory):
    """An energy-grid dataset with a hidden per-plant price premium."""
    rng = np.random.default_rng(7)
    n = 120_000

    start = (
        np.datetime64("2026-01-01") + rng.integers(0, 60 * 24 * 31, n).astype("timedelta64[m]")
    ).astype("datetime64[us]")
    duration_min = rng.gamma(3, 40, n) + 5
    plant = rng.integers(0, len(PLANT_PREMIUM), n)
    mwh = np.abs(rng.gamma(2, 30, n)) + 1

    price = (
        12 + 2.4 * mwh + 0.35 * duration_min + PLANT_PREMIUM[plant] + rng.normal(0, 9, n)
    )
    end = start + (duration_min * 60 * 1e6).astype("timedelta64[us]")

    path = tmp_path_factory.mktemp("generic") / "energy_dispatch.parquet"
    pl.DataFrame({
        "dispatch_start": start,
        "dispatch_end": end,
        "energy_mwh": mwh,
        "settlement_price_usd": price,
        "plant_id": plant,
        "region_code": rng.integers(0, 4, n),
        "operator_count": rng.integers(1, 6, n),
    }).write_parquet(path)
    return path


@pytest.fixture(scope="module")
def energy_profile(energy_path):
    handle = register_local_dataset(energy_path, dataset_id="energy_dispatch")
    spec = resolve_spec(handle.path)
    return profile_dataset(handle, dictionary=spec.dictionary, use_cache=False), handle, spec


class TestDatasetRegistry:
    def test_unknown_dataset_falls_back_to_generic(self, energy_profile):
        _, _, spec = energy_profile
        assert spec.key == "generic"
        assert spec.dictionary == {}
        assert spec.filters is None

    def test_tlc_resolves_by_filename(self):
        assert resolve_spec("yellow_tripdata_2026-01.parquet").key == "nyc_tlc_yellow"
        assert resolve_spec("green_tripdata_2025-06.parquet").key == "nyc_tlc_green"

    def test_tlc_resolves_by_columns_when_renamed(self):
        """An uploaded file is often renamed; the column signature still identifies it."""
        spec = resolve_spec(
            "some_users_upload.parquet",
            {"tpep_pickup_datetime", "trip_distance", "fare_amount", "total_amount"},
        )
        assert spec.key == "nyc_tlc_yellow"

    def test_known_spec_carries_its_metadata(self):
        spec = spec_for_key("nyc_tlc_yellow")
        assert spec.dictionary
        assert spec.accounting_identities
        assert spec.filters is not None


class TestGenericSemantics:
    def test_types_are_inferred_without_a_dictionary(self, energy_profile):
        profile, _, _ = energy_profile
        assert profile.card("dispatch_start").semantic_type is SemanticType.DATETIME
        assert profile.card("settlement_price_usd").semantic_type is SemanticType.CURRENCY
        assert profile.card("energy_mwh").semantic_type.is_numeric_quantity

    def test_id_columns_are_not_treated_as_quantities(self, energy_profile):
        profile, _, _ = energy_profile
        for column in ("plant_id", "region_code"):
            card = profile.card(column)
            assert card.semantic_type.is_categorical
            assert column not in profile.numeric_columns()

    def test_energy_units_are_recognised(self, energy_profile):
        profile, _, _ = energy_profile
        assert profile.card("energy_mwh").unit.label in {"MWh", "kWh"}


class TestGenericDerivation:
    def test_duration_is_derived_from_any_datetime_pair(self, energy_profile):
        profile, handle, _ = energy_profile
        view = build_analysis_view(pl.scan_parquet(handle.path), profile)
        assert "duration_minutes" in view.derived_units
        assert "duration_seconds" in view.derived_units
        # Nothing named "trip" exists in this dataset.
        assert not any("trip" in name for name in view.derived_units)

    def test_calendar_parts_are_derived(self, energy_profile):
        profile, handle, _ = energy_profile
        view = build_analysis_view(pl.scan_parquet(handle.path), profile)
        for name in ("event_hour", "event_day_of_week", "event_date"):
            assert name in view.derived_units

    def test_cost_per_unit_ratio_is_derived(self, energy_profile):
        """The single most informative derived feature on an unfamiliar dataset."""
        profile, handle, _ = energy_profile
        view = build_analysis_view(pl.scan_parquet(handle.path), profile)
        ratios = [
            name for name, unit in view.derived_units.items()
            if "/" in unit.label and "USD" in unit.label
        ]
        assert ratios, f"expected a cost-per-unit ratio, got {list(view.derived_units)}"

    def test_generic_filters_are_definitional_only(self, energy_profile):
        """Generic filters must not invent domain knowledge."""
        profile, handle, _ = energy_profile
        view = build_analysis_view(pl.scan_parquet(handle.path), profile)
        available = set(view.frame.collect_schema().names())
        rules = generic_filters(profile, available)
        for rule in rules:
            assert ">= 0" in rule.expression, (
                "a generic filter may only assert definitional non-negativity; "
                f"got {rule.expression!r}"
            )

    def test_generic_filters_drop_almost_nothing(self, energy_profile):
        profile, handle, _ = energy_profile
        view = build_analysis_view(pl.scan_parquet(handle.path), profile)
        report = view.report
        assert report["rows_after"] >= report["rows_before"] * 0.98


class TestGenericTargetInference:
    def test_price_question_finds_the_price_column(self, energy_profile):
        profile, _, spec = energy_profile
        target = infer_target_generic("What drives the settlement price?", profile.columns, spec)
        assert target == "settlement_price_usd"

    def test_question_about_volume_finds_the_quantity(self, energy_profile):
        profile, _, spec = energy_profile
        target = infer_target_generic(
            "What affects how much energy is dispatched?", profile.columns, spec
        )
        assert target in {"energy_mwh", "settlement_price_usd"}

    def test_returns_none_when_nothing_is_analysable(self):
        assert infer_target_generic("anything at all", {}) is None


class TestHiddenStructureRecovery:
    """The product claim, on a dataset the engine has never seen."""

    def test_residual_stage_recovers_an_injected_premium(self, energy_path):
        frame = pl.read_parquet(energy_path)
        mwh = frame["energy_mwh"].to_numpy()
        price = frame["settlement_price_usd"].to_numpy()
        plant = frame["plant_id"].to_numpy()
        duration = (
            (frame["dispatch_end"] - frame["dispatch_start"]).dt.total_seconds().to_numpy() / 60.0
        )

        # Marginally, plant_id tells you almost nothing about price.
        marginal = abs(np.corrcoef(plant.astype(float), price)[0, 1])
        assert marginal < 0.25, f"setup broken: marginal corr {marginal}"

        analysis = analyze_residual(
            target_name="settlement_price_usd",
            target=price,
            baseline_predictors={"energy_mwh": mwh, "duration_minutes": duration},
            categorical_candidates={"plant_id": plant},
        )
        assert analysis is not None
        assert analysis.baseline.r_squared > 0.9

        drivers = analysis.meaningful()
        assert drivers, "the injected premium must be found"
        top = drivers[0]
        assert top.name == "plant_id"
        assert top.effect > 0.7

        # And the recovered per-level effects must match the injected premium,
        # mean-centred (a residual has zero mean by construction).
        expected = PLANT_PREMIUM - PLANT_PREMIUM.mean()
        recovered = {int(lv["level"]): lv["mean_residual"] for lv in top.level_effects}
        for plant_id, want in enumerate(expected):
            assert recovered[plant_id] == pytest.approx(want, abs=1.5), (
                f"plant {plant_id}: expected {want:+.2f}, recovered {recovered[plant_id]:+.2f}"
            )
