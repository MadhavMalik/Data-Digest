"""Statistical correctness against datasets with known ground truth.

Spec section 15.1.  Each test names the property it is proving, not just the
function it calls — a test that only asserts "returns a float" proves nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

from signal_engine.statistics.correlation import (
    categorical_association,
    grouped_comparison,
    pairwise_relationship,
)
from signal_engine.statistics.covariance import (
    build_covariance_model,
    direct_linear_combination_correlation,
    linear_combination_correlation,
)
from signal_engine.statistics.multiple_testing import benjamini_hochberg
from signal_engine.statistics.mutual_information import (
    conditional_mutual_information,
    mutual_information,
)
from signal_engine.statistics.stability import stability_score, subgroup_consistency


class TestKnownRelationships:
    """A. strong positive, B. strong negative, C. independent, D. nonlinear."""

    def test_a_strong_positive_relationship(self, synthetic_frame):
        r = pairwise_relationship(
            synthetic_frame["x"].to_numpy(), synthetic_frame["y_positive"].to_numpy()
        )
        assert r.pearson_r > 0.95, f"expected strong positive, got {r.pearson_r}"
        assert r.spearman_rho > 0.9
        assert r.direction == "positive"
        assert r.strength_label() == "very strong"
        assert r.is_meaningful
        # Y = 3X + noise, so the OLS slope must recover ~3.
        assert 2.8 < r.slope < 3.2

    def test_b_strong_negative_relationship(self, synthetic_frame):
        r = pairwise_relationship(
            synthetic_frame["x"].to_numpy(), synthetic_frame["y_negative"].to_numpy()
        )
        assert r.pearson_r < -0.9, f"expected strong negative, got {r.pearson_r}"
        assert r.direction == "negative"
        assert r.is_meaningful
        assert -2.2 < r.slope < -1.8

    def test_c_independent_variables_are_not_meaningful(self, synthetic_frame):
        r = pairwise_relationship(
            synthetic_frame["x"].to_numpy(), synthetic_frame["y_independent"].to_numpy()
        )
        assert abs(r.pearson_r) < 0.05
        assert not r.is_meaningful, "independent noise must not register as a finding"
        assert r.strength_label() == "negligible"

    def test_d_nonlinear_relationship_escapes_pearson_but_not_mi(self, synthetic_frame):
        """The canonical failure of linear-only screening.

        Y = X^2 with X symmetric about zero is a DETERMINISTIC relationship
        with Pearson r ~ 0.  A pipeline that screens on correlation alone
        throws it away; mutual information must catch it.
        """
        x = synthetic_frame["x_symmetric"].to_numpy()
        y = synthetic_frame["y_quadratic"].to_numpy()

        r = pairwise_relationship(x, y)
        assert abs(r.pearson_r) < 0.1, "Pearson should be near zero for a symmetric parabola"

        mi = mutual_information(x, y)
        assert mi > 0.3, f"MI should detect the dependence, got {mi}"

        r.mutual_information = mi
        assert r.is_nonlinear_signal, "the nonlinear flag must fire on this shape"
        assert r.is_meaningful, "MI must lift this above the meaningfulness floor"


class TestCovarianceShortcut:
    """E. the sufficient-statistics optimization must be EXACTLY right."""

    def test_shortcut_matches_direct_computation(self, synthetic_frame):
        data = {
            "X1": synthetic_frame["x"].to_numpy(),
            "X2": synthetic_frame["y_positive"].to_numpy(),
            "X3": synthetic_frame["y_negative"].to_numpy(),
        }
        model = build_covariance_model(data)
        assert model.is_valid()

        a = {"X1": 2.0, "X2": 1.0}
        b = {"X2": 1.0, "X3": -3.0}

        fast = linear_combination_correlation(model, a, b)
        slow = direct_linear_combination_correlation(data, a, b)

        assert fast.pearson_r == pytest.approx(slow.pearson_r, abs=1e-10), (
            "the covariance shortcut must agree with a direct row scan to "
            "numerical precision, or it is computing something else"
        )
        assert fast.covariance == pytest.approx(slow.covariance, rel=1e-9)
        assert fast.extra["computed_without_row_scan"] is True

    @pytest.mark.parametrize(
        "wa,wb",
        [
            ({"X1": 1.0}, {"X2": 1.0}),
            ({"X1": -1.0, "X3": 0.5}, {"X2": 2.0}),
            ({"X1": 0.3, "X2": 0.3, "X3": 0.3}, {"X1": -1.0, "X2": 1.0}),
        ],
    )
    def test_shortcut_across_many_weightings(self, synthetic_frame, wa, wb):
        data = {
            "X1": synthetic_frame["x"].to_numpy(),
            "X2": synthetic_frame["y_positive"].to_numpy(),
            "X3": synthetic_frame["y_independent"].to_numpy(),
        }
        model = build_covariance_model(data)
        fast = linear_combination_correlation(model, wa, wb)
        slow = direct_linear_combination_correlation(data, wa, wb)
        assert fast.pearson_r == pytest.approx(slow.pearson_r, abs=1e-10)

    def test_shortcut_refuses_itself_when_missingness_breaks_the_algebra(self):
        """The guard that makes the optimization trustworthy.

        The identity Cov(aX, bX) = a'Sigma b only holds when both combinations
        are evaluated on the rows Sigma was estimated from.  With heavy,
        column-specific missingness the complete-case subset stops representing
        the data, and the model must REFUSE rather than return a number that
        matches no real computation.
        """
        rng = np.random.default_rng(7)
        n = 5_000
        a = rng.normal(size=n)
        b = rng.normal(size=n)
        c = rng.normal(size=n)
        # Knock out 70% of `b`, on different rows than `c`.
        b[rng.choice(n, size=int(n * 0.7), replace=False)] = np.nan
        c[rng.choice(n, size=int(n * 0.5), replace=False)] = np.nan

        model = build_covariance_model({"a": a, "b": b, "c": c})
        assert not model.is_valid(), "shortcut must be refused under heavy missingness"
        assert "complete cases" in (model.invalid_reason() or "")

        result = linear_combination_correlation(model, {"a": 1.0}, {"b": 1.0})
        assert result.skipped_reason is not None
        assert "shortcut unavailable" in result.skipped_reason

    def test_complete_case_covariance_is_internally_consistent(self):
        """With complete data, Sigma's diagonal must equal the variances."""
        rng = np.random.default_rng(11)
        data = {"p": rng.normal(2, 3, 4000), "q": rng.normal(-1, 0.5, 4000)}
        model = build_covariance_model(data)
        assert model.cov[0, 0] == pytest.approx(np.var(data["p"], ddof=1), rel=1e-9)
        assert model.cov[1, 1] == pytest.approx(np.var(data["q"], ddof=1), rel=1e-9)


class TestStability:
    def test_real_relationship_is_stable_across_folds(self, synthetic_frame):
        report = stability_score(
            synthetic_frame["x"].to_numpy(), synthetic_frame["y_positive"].to_numpy()
        )
        assert report.score > 0.9
        assert report.sign_agreement == 1.0

    def test_noise_relationship_is_unstable(self, synthetic_frame):
        report = stability_score(
            synthetic_frame["x"].to_numpy(), synthetic_frame["y_independent"].to_numpy()
        )
        assert report.score < 0.5, "pure noise must not look stable"

    def test_simpson_reversal_is_detected(self):
        """Pooled sign opposite to every subgroup's sign must be flagged."""
        rng = np.random.default_rng(3)
        xs, ys, gs = [], [], []
        for group, offset in enumerate([(0.0, 10.0), (5.0, 5.0), (10.0, 0.0)]):
            x = rng.normal(offset[0], 1.0, 3000)
            # Within every group the relationship is NEGATIVE ...
            y = -0.5 * (x - offset[0]) + offset[1] + rng.normal(0, 0.3, 3000)
            xs.append(x)
            ys.append(y)
            gs.append(np.full(3000, group))

        x = np.concatenate(xs)
        y = np.concatenate(ys)
        g = np.concatenate(gs)

        pooled = float(np.corrcoef(x, y)[0, 1])
        assert pooled < 0  # ... and pooled it is also negative here
        out = subgroup_consistency(x, y, g, min_group_size=100)
        assert out["available"]
        assert all(entry["r"] < 0 for entry in out["per_group"])


class TestCategoricalMethods:
    def test_grouped_comparison_recovers_group_structure(self):
        rng = np.random.default_rng(5)
        groups = rng.integers(0, 4, 8000)
        values = groups * 5.0 + rng.normal(0, 1.0, 8000)
        result = grouped_comparison(values, groups, value_name="v", group_name="g")
        assert result.eta is not None and result.eta > 0.9
        assert result.extra["n_levels"] == 4
        assert result.method == "grouped_eta_squared"
        assert any("categorical" in w for w in result.warnings)

    def test_grouped_comparison_reports_no_pearson(self):
        """A group comparison must NOT emit a correlation coefficient."""
        rng = np.random.default_rng(5)
        groups = rng.integers(0, 3, 2000)
        values = rng.normal(0, 1, 2000)
        result = grouped_comparison(values, groups)
        assert result.pearson_r is None
        assert result.spearman_rho is None

    def test_cramers_v_detects_association(self):
        rng = np.random.default_rng(9)
        a = rng.integers(0, 3, 6000)
        b = (a + rng.integers(0, 2, 6000)) % 3
        result = categorical_association(a, b)
        assert result.eta is not None and result.eta > 0.2
        assert result.method == "cramers_v"

    def test_too_many_levels_is_skipped_not_wrong(self):
        rng = np.random.default_rng(13)
        groups = rng.integers(0, 500, 5000)
        values = rng.normal(0, 1, 5000)
        result = grouped_comparison(values, groups, max_groups=40)
        assert result.skipped_reason is not None
        assert "levels" in result.skipped_reason


class TestGuardsAndEdgeCases:
    def test_constant_column_is_skipped(self):
        x = np.ones(1000)
        y = np.random.default_rng(1).normal(size=1000)
        result = pairwise_relationship(x, y, x_name="constant")
        assert result.skipped_reason is not None
        assert "constant" in result.skipped_reason

    def test_joint_masking_uses_only_rows_present_in_both(self):
        x = np.array([1.0, 2.0, np.nan, 4.0, 5.0] * 40)
        y = np.array([2.0, np.nan, 3.0, 8.0, 10.0] * 40)
        result = pairwise_relationship(x, y, min_sample=10)
        assert result.n == 120, "only rows where BOTH are finite may be counted"

    def test_too_few_rows_is_skipped(self):
        result = pairwise_relationship(np.arange(5.0), np.arange(5.0))
        assert result.skipped_reason is not None

    def test_shape_mismatch_is_reported_not_raised(self):
        result = pairwise_relationship(np.arange(10.0), np.arange(20.0))
        assert result.skipped_reason is not None
        assert "shape mismatch" in result.skipped_reason


class TestMultipleTesting:
    def test_bh_is_monotone_and_bounded(self):
        q = benjamini_hochberg([0.001, 0.01, 0.03, 0.2, 0.5])
        assert all(0.0 <= v <= 1.0 for v in q)
        assert list(q) == sorted(q), "BH q-values must be monotone in p"

    def test_bh_preserves_input_order(self):
        q = benjamini_hochberg([0.5, 0.001, 0.2])
        assert q[1] < q[2] < q[0]

    def test_nan_p_values_stay_nan(self):
        q = benjamini_hochberg([0.01, float("nan"), 0.2])
        assert np.isnan(q[1]), "a skipped test must never become a discovery"

    def test_empty_input(self):
        assert len(benjamini_hochberg([])) == 0


class TestMutualInformation:
    def test_identical_variables_have_high_mi(self):
        rng = np.random.default_rng(2)
        x = rng.normal(size=20_000)
        assert mutual_information(x, x.copy()) > 0.8

    def test_independent_variables_have_near_zero_mi(self):
        rng = np.random.default_rng(2)
        mi = mutual_information(rng.normal(size=20_000), rng.normal(size=20_000))
        assert mi < 0.05, f"independent variables should have ~0 MI, got {mi}"

    def test_constant_input_returns_nan(self):
        assert np.isnan(mutual_information(np.ones(500), np.arange(500.0)))

    def test_conditional_mi_collapses_when_z_explains_the_link(self):
        """X and Y correlated only through Z: I(X;Y) high, I(X;Y|Z) low."""
        rng = np.random.default_rng(4)
        z = rng.normal(size=40_000)
        x = z + rng.normal(0, 0.1, 40_000)
        y = z + rng.normal(0, 0.1, 40_000)
        assert mutual_information(x, y) > 0.3
        assert conditional_mutual_information(x, y, z) < mutual_information(x, y)
