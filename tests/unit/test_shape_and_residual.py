"""Shape characterization and residual analysis.

These two stages exist because marginal correlation was producing true but
useless findings. The tests pin the behaviours that make them useful:

  * shape must recover a KNOWN functional form from noisy data
  * shape must prefer a straight line unless a curve materially beats it
  * residual analysis must find a driver that is INVISIBLE to marginal
    correlation — the whole point of the stage
"""

from __future__ import annotations

import numpy as np
import pytest

from signal_engine.statistics.residual import (
    analyze_residual,
    build_baseline,
    categorical_residual_effect,
)
from signal_engine.statistics.shape import (
    conditional_mean_curve,
    describe_relationship,
)


class TestConditionalMeanCurve:
    def test_recovers_a_known_curve(self):
        rng = np.random.default_rng(0)
        x = rng.uniform(1, 50, 200_000)
        y = 10 + 3 * x + rng.normal(0, 20, 200_000)

        curve = conditional_mean_curve(x, y, bins=20)
        assert curve is not None
        assert curve.n_bins >= 15
        # The binned means must track the true line despite heavy noise.
        expected = 10 + 3 * curve.x
        assert np.allclose(curve.y, expected, rtol=0.12)

    def test_uses_quantile_bins_not_equal_width(self):
        """Heavy-tailed x must not collapse into one bin."""
        rng = np.random.default_rng(1)
        x = rng.exponential(2.0, 100_000)
        y = x + rng.normal(0, 0.5, 100_000)
        curve = conditional_mean_curve(x, y, bins=20)
        assert curve is not None
        counts = curve.counts
        # Quantile bins are roughly balanced; equal-width bins would be wildly not.
        assert counts.max() / counts.min() < 8

    def test_too_little_data_returns_none(self):
        assert conditional_mean_curve(np.arange(5.0), np.arange(5.0)) is None


class TestShapeRecovery:
    @pytest.mark.parametrize(
        "generator,expected_forms",
        [
            (lambda x: 5 + 2 * x, {"linear"}),
            (lambda x: 100 / x, {"inverse", "power"}),
            (lambda x: 20 * np.log(x), {"logarithmic", "power"}),
            (lambda x: 3 * x**2, {"quadratic", "power"}),
        ],
    )
    def test_recovers_the_generating_form(self, generator, expected_forms):
        rng = np.random.default_rng(7)
        x = rng.uniform(1, 30, 200_000)
        clean = generator(x)
        y = clean + rng.normal(0, np.std(clean) * 0.10, x.size)

        fit = describe_relationship(x, y)
        assert fit is not None
        assert fit.form in expected_forms, (
            f"expected one of {expected_forms}, got {fit.form} (R^2={fit.r2:.3f})"
        )
        assert fit.r2 > 0.9

    def test_linear_data_is_not_overfitted_to_a_curve(self):
        """A more complex form must EARN its extra parameter."""
        rng = np.random.default_rng(3)
        x = rng.uniform(1, 40, 200_000)
        y = 4 + 1.5 * x + rng.normal(0, 3, x.size)

        fit = describe_relationship(x, y)
        assert fit.form == "linear"
        assert not fit.is_nonlinear

    def test_detects_saturation(self):
        rng = np.random.default_rng(5)
        x = rng.uniform(1, 60, 200_000)
        y = 50 * np.log(x) + rng.normal(0, 5, x.size)

        fit = describe_relationship(x, y)
        assert fit.monotonic == "increasing"
        assert fit.saturating, "a log curve rises with a steadily falling slope"
        assert fit.curvature_ratio and fit.curvature_ratio > 1.8

    def test_detects_a_turning_point(self):
        rng = np.random.default_rng(11)
        x = rng.uniform(0, 20, 200_000)
        y = -((x - 10) ** 2) + rng.normal(0, 5, x.size)

        fit = describe_relationship(x, y)
        assert fit.form == "quadratic"
        assert fit.monotonic == "non_monotonic"
        assert fit.turning_point is not None
        assert 8 < fit.turning_point < 12

    def test_nonlinear_flag_requires_a_material_margin(self):
        rng = np.random.default_rng(13)
        x = rng.uniform(1, 30, 100_000)
        y = 2 * x + rng.normal(0, 1, x.size)
        fit = describe_relationship(x, y)
        assert (fit.r2 - fit.linear_r2) <= 0.02 or fit.form == "linear"

    def test_prompt_text_states_the_comparison(self):
        rng = np.random.default_rng(17)
        x = rng.uniform(1, 40, 150_000)
        y = 200 / x + rng.normal(0, 1, x.size)
        fit = describe_relationship(x, y)
        text = fit.to_prompt_text()
        assert "functional form" in text
        assert "R^2" in text
        assert fit.form in text


class TestBaseline:
    def test_recovers_known_coefficients(self):
        rng = np.random.default_rng(2)
        n = 100_000
        a = rng.uniform(0, 10, n)
        b = rng.uniform(0, 5, n)
        y = 3.0 + 2.0 * a + 4.0 * b + rng.normal(0, 0.5, n)

        model = build_baseline({"a": a, "b": b}, y)
        assert model is not None
        assert model.intercept == pytest.approx(3.0, abs=0.05)
        assert model.coefficients["a"] == pytest.approx(2.0, abs=0.02)
        assert model.coefficients["b"] == pytest.approx(4.0, abs=0.02)
        assert model.r_squared > 0.99

    def test_residual_std_is_smaller_than_target_std(self):
        rng = np.random.default_rng(4)
        x = rng.uniform(0, 10, 50_000)
        y = 5 * x + rng.normal(0, 2, 50_000)
        model = build_baseline({"x": x}, y)
        assert model.residual_std < model.target_std

    def test_too_few_rows_returns_none(self):
        assert build_baseline({"x": np.arange(10.0)}, np.arange(10.0)) is None

    def test_complete_cases_only(self):
        rng = np.random.default_rng(6)
        n = 20_000
        x = rng.uniform(0, 10, n)
        y = 2 * x + rng.normal(0, 1, n)
        x[:5_000] = np.nan
        model = build_baseline({"x": x}, y)
        assert model.n == 15_000


class TestResidualDiscovery:
    """The stage's reason to exist: find what marginal correlation cannot."""

    @pytest.fixture
    def hidden_effect_data(self):
        """A driver that is invisible marginally but obvious in the residual.

        `group` is assigned independently of `y`, so corr(group, y) is ~0. But
        each group shifts y by a fixed amount on top of the x effect, so once x
        is removed the group effect is unmistakable.
        """
        rng = np.random.default_rng(42)
        n = 120_000
        x = rng.uniform(0, 100, n)
        group = rng.integers(0, 4, n)
        offsets = np.array([0.0, 25.0, -25.0, 50.0])
        y = 2.0 * x + offsets[group] + rng.normal(0, 5, n)
        return x, group, y, offsets

    def test_marginal_correlation_misses_it(self, hidden_effect_data):
        x, group, y, _ = hidden_effect_data
        # The group barely correlates with y on its own ...
        marginal = abs(np.corrcoef(group.astype(float), y)[0, 1])
        assert marginal < 0.25, f"setup broken: marginal corr is {marginal}"

    def test_residual_analysis_finds_it(self, hidden_effect_data):
        x, group, y, offsets = hidden_effect_data

        analysis = analyze_residual(
            target_name="y", target=y,
            baseline_predictors={"x": x},
            categorical_candidates={"group": group},
        )
        assert analysis is not None
        assert analysis.baseline.r_squared > 0.5

        drivers = analysis.meaningful()
        assert drivers, "the hidden group effect must be found"
        top = drivers[0]
        assert top.name == "group"
        assert top.effect > 0.8, f"eta should be high, got {top.effect}"

        # And the recovered level effects must match the injected offsets.
        recovered = {int(lv["level"]): lv["mean_residual"] for lv in top.level_effects}
        centered = offsets - offsets.mean()
        for level, expected in enumerate(centered):
            assert recovered[level] == pytest.approx(expected, abs=1.0)

    def test_spread_is_reported_in_target_units(self, hidden_effect_data):
        x, group, y, offsets = hidden_effect_data
        analysis = analyze_residual(
            target_name="y", target=y,
            baseline_predictors={"x": x},
            categorical_candidates={"group": group},
        )
        top = analysis.meaningful()[0]
        assert top.spread == pytest.approx(offsets.max() - offsets.min(), abs=2.0)

    def test_baseline_predictors_are_not_retested(self, hidden_effect_data):
        x, group, y, _ = hidden_effect_data
        analysis = analyze_residual(
            target_name="y", target=y,
            baseline_predictors={"x": x},
            numeric_candidates={"x": x},
        )
        assert any(name == "x" and "baseline" in reason for name, reason in analysis.skipped)

    def test_prompt_text_leads_with_the_baseline(self, hidden_effect_data):
        x, group, y, _ = hidden_effect_data
        analysis = analyze_residual(
            target_name="y", target=y,
            baseline_predictors={"x": x},
            categorical_candidates={"group": group},
        )
        text = analysis.to_prompt_text(unit="USD")
        assert "BASELINE" in text
        assert "explains" in text
        assert "WHAT EXPLAINS THE REMAINDER" in text
        assert "USD" in text

    def test_category_labels_are_used(self):
        rng = np.random.default_rng(8)
        n = 30_000
        x = rng.uniform(0, 10, n)
        g = rng.integers(1, 3, n)
        y = 2 * x + np.where(g == 1, 0.0, 20.0) + rng.normal(0, 1, n)

        model = build_baseline({"x": x}, y)
        driver = categorical_residual_effect(
            model.residuals, g[model.mask], name="g", labels={1: "Standard", 2: "Premium"}
        )
        labels = [lv["label"] for lv in driver.level_effects]
        assert any("Standard" in x for x in labels)
        assert any("Premium" in x for x in labels)
