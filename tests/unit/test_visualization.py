"""Graph selection and rendering.

Spec section 15.4.  The rule under test throughout: the chart form follows the
data's semantic types AND its scale.  Getting this wrong hands the VLM a
misleading picture, which it then interprets faithfully.
"""

from __future__ import annotations

import numpy as np
import pytest

from signal_engine.profiling.column_cards import CategorySummary, ColumnCard
from signal_engine.profiling.semantic_types import SemanticType
from signal_engine.profiling.units import CATEGORICAL, IDENTIFIER, MILES, TIMESTAMP, USD
from signal_engine.statistics.correlation import RelationshipResult
from signal_engine.visualization.render import render_plot, stats_caption
from signal_engine.visualization.selector import (
    PlotSpec,
    PlotType,
    select_plot,
    validate_override,
)


def card(name, semantic_type, unit=USD, distinct=None) -> ColumnCard:
    c = ColumnCard(
        name=name,
        physical_dtype="float64",
        semantic_type=semantic_type,
        unit=unit,
        type_confidence=1.0,
        type_rule="test",
        row_count=1_000_000,
    )
    if distinct is not None:
        c.categories = CategorySummary(distinct_count=distinct, approximate=False)
    return c


CONTINUOUS_X = card("trip_distance", SemanticType.CONTINUOUS_MEASUREMENT, MILES)
CONTINUOUS_Y = card("fare_amount", SemanticType.CURRENCY, USD)
CATEGORY = card("RatecodeID", SemanticType.CATEGORICAL, CATEGORICAL, distinct=6)
MANY_CATEGORIES = card("PULocationID", SemanticType.CATEGORICAL_IDENTIFIER, IDENTIFIER, distinct=263)
TIME = card("tpep_pickup_datetime", SemanticType.DATETIME, TIMESTAMP)


class TestSelectionByType:
    def test_continuous_pair_small_n_is_a_scatter(self):
        spec = select_plot(CONTINUOUS_X, CONTINUOUS_Y, n_rows=5_000)
        assert spec.plot_type is PlotType.SCATTER

    def test_continuous_pair_huge_n_is_a_density_plot(self):
        """3.5M points is a black rectangle, not a chart."""
        spec = select_plot(CONTINUOUS_X, CONTINUOUS_Y, n_rows=3_500_000)
        assert spec.plot_type is PlotType.HEXBIN
        assert "over-plot" in spec.rationale

    def test_continuous_pair_medium_n_samples(self):
        spec = select_plot(CONTINUOUS_X, CONTINUOUS_Y, n_rows=60_000, max_scatter_points=20_000)
        assert spec.plot_type is PlotType.SCATTER
        assert "sample" in spec.sample_strategy

    def test_time_axis_is_a_line(self):
        spec = select_plot(TIME, CONTINUOUS_Y, n_rows=1_000_000)
        assert spec.plot_type is PlotType.TIME_BINNED_LINE

    def test_time_on_the_y_axis_is_swapped_not_mishandled(self):
        spec = select_plot(CONTINUOUS_Y, TIME, n_rows=1_000_000)
        assert spec.plot_type is PlotType.TIME_BINNED_LINE
        assert spec.x == TIME.name

    def test_few_categories_get_a_box_plot(self):
        spec = select_plot(CATEGORY, CONTINUOUS_Y, n_rows=1_000_000)
        assert spec.plot_type is PlotType.BOX

    def test_many_categories_get_aggregated_bars(self):
        spec = select_plot(MANY_CATEGORIES, CONTINUOUS_Y, n_rows=1_000_000)
        assert spec.plot_type is PlotType.BAR_WITH_CI

    def test_categorical_pair_is_a_contingency_heatmap(self):
        other = card("payment_type", SemanticType.CATEGORICAL, CATEGORICAL, distinct=5)
        spec = select_plot(CATEGORY, other, n_rows=1_000_000)
        assert spec.plot_type is PlotType.CONTINGENCY_HEATMAP

    def test_single_variable_is_a_histogram(self):
        spec = select_plot(CONTINUOUS_X, None, n_rows=1_000_000)
        assert spec.plot_type is PlotType.HISTOGRAM

    def test_categorical_orientation_is_normalized(self):
        """Category on either axis must produce the same grouped plot."""
        a = select_plot(CATEGORY, CONTINUOUS_Y, n_rows=100_000)
        b = select_plot(CONTINUOUS_Y, CATEGORY, n_rows=100_000)
        assert a.plot_type is b.plot_type
        assert a.x == b.x == CATEGORY.name


class TestScaleHandling:
    def test_large_dataset_never_draws_every_row(self):
        """The explicit requirement: do not plot ten million points."""
        spec = select_plot(CONTINUOUS_X, CONTINUOUS_Y, n_rows=10_000_000)
        assert spec.plot_type is not PlotType.SCATTER
        assert spec.sample_strategy != "every row plotted"

    def test_identifier_axis_carries_an_ordering_warning(self):
        spec = select_plot(MANY_CATEGORIES, CONTINUOUS_Y, n_rows=1_000_000)
        assert any("no meaning" in a for a in spec.annotations)

    def test_vlm_description_states_the_sampling_strategy(self):
        """The VLM must be told colour encodes density, not a third variable."""
        spec = select_plot(CONTINUOUS_X, CONTINUOUS_Y, n_rows=3_500_000)
        description = spec.describe_for_vlm().lower()
        assert "hexbin" in description
        assert "how many records" in description
        assert "not the value of a third variable" in description


class TestOverrideValidation:
    def test_scatter_override_is_refused_at_scale(self):
        deterministic = select_plot(CONTINUOUS_X, CONTINUOUS_Y, n_rows=3_500_000)
        spec, reason = validate_override(
            "scatter", deterministic, CONTINUOUS_X, CONTINUOUS_Y, n_rows=3_500_000
        )
        assert spec.plot_type is PlotType.HEXBIN
        assert reason and "over-plot" in reason

    def test_line_override_on_a_non_time_axis_is_refused(self):
        deterministic = select_plot(CONTINUOUS_X, CONTINUOUS_Y, n_rows=5_000)
        spec, reason = validate_override(
            "line", deterministic, CONTINUOUS_X, CONTINUOUS_Y, n_rows=5_000
        )
        assert spec.plot_type is PlotType.SCATTER
        assert reason and "not a time axis" in reason

    def test_continuous_plot_on_a_category_is_refused(self):
        deterministic = select_plot(CATEGORY, CONTINUOUS_Y, n_rows=100_000)
        spec, reason = validate_override(
            "scatter", deterministic, CATEGORY, CONTINUOUS_Y, n_rows=100_000
        )
        assert spec.plot_type is PlotType.BOX
        assert reason is not None

    def test_unknown_plot_type_falls_back(self):
        deterministic = select_plot(CONTINUOUS_X, CONTINUOUS_Y, n_rows=5_000)
        spec, reason = validate_override(
            "pie_chart_3d", deterministic, CONTINUOUS_X, CONTINUOUS_Y, n_rows=5_000
        )
        assert spec.plot_type is deterministic.plot_type
        assert "unknown plot type" in reason

    def test_defensible_alternative_is_accepted(self):
        deterministic = select_plot(CATEGORY, CONTINUOUS_Y, n_rows=100_000)
        spec, reason = validate_override(
            "violin", deterministic, CATEGORY, CONTINUOUS_Y, n_rows=100_000
        )
        assert spec.plot_type is PlotType.VIOLIN
        assert reason is None


class TestRendering:
    @pytest.fixture
    def data(self):
        rng = np.random.default_rng(1)
        n = 30_000
        x = np.abs(rng.normal(3, 2, n)) + 0.1
        return {
            "trip_distance": x,
            "fare_amount": 3.0 + 2.5 * x + rng.normal(0, 2, n),
            "RatecodeID": rng.integers(1, 6, n).astype(float),
            "payment_type": rng.integers(1, 5, n).astype(float),
            "tpep_pickup_datetime": np.sort(rng.uniform(0, 1e6, n)),
        }

    @pytest.mark.parametrize(
        "plot_type",
        [
            PlotType.SCATTER,
            PlotType.HEXBIN,
            PlotType.BINNED_TREND,
            PlotType.HISTOGRAM,
            PlotType.BOX,
            PlotType.VIOLIN,
            PlotType.BAR_WITH_CI,
            PlotType.CONTINGENCY_HEATMAP,
            PlotType.TIME_BINNED_LINE,
        ],
    )
    def test_every_plot_type_renders(self, plot_type, data, tmp_path):
        if plot_type is PlotType.CONTINGENCY_HEATMAP:
            spec = PlotSpec(plot_type, "RatecodeID", "payment_type", title="t")
        elif plot_type in {PlotType.BOX, PlotType.VIOLIN, PlotType.BAR_WITH_CI}:
            spec = PlotSpec(plot_type, "RatecodeID", "fare_amount", title="t")
        elif plot_type is PlotType.HISTOGRAM:
            spec = PlotSpec(plot_type, "trip_distance", title="t")
        elif plot_type is PlotType.TIME_BINNED_LINE:
            spec = PlotSpec(plot_type, "tpep_pickup_datetime", "fare_amount", title="t")
        else:
            spec = PlotSpec(plot_type, "trip_distance", "fare_amount", title="t")

        artifact = render_plot(spec, data, output_dir=tmp_path, filename=f"{plot_type.value}.png")
        assert artifact.path is not None and artifact.path.exists()
        assert artifact.bytes_size > 1000
        assert artifact.data_uri and artifact.data_uri.startswith("data:image/png;base64,")

    def test_scatter_downsamples_and_says_so(self, data, tmp_path):
        spec = PlotSpec(PlotType.SCATTER, "trip_distance", "fare_amount", max_points=1000)
        artifact = render_plot(spec, data, output_dir=tmp_path, filename="s.png")
        assert artifact.points_drawn <= 1000
        assert artifact.rows_represented == 30_000
        assert any("sampled" in n for n in artifact.notes)

    def test_statistics_caption_is_burned_into_the_image(self, data, tmp_path):
        result = RelationshipResult(
            x_name="trip_distance", y_name="fare_amount", n=30_000, pearson_r=0.93, spearman_rho=0.92
        )
        caption = stats_caption(result)
        assert "r=+0.930" in caption
        assert "n=30,000" in caption
        spec = PlotSpec(PlotType.HEXBIN, "trip_distance", "fare_amount")
        artifact = render_plot(spec, data, output_dir=tmp_path, filename="c.png", stats_annotation=caption)
        assert artifact.path.exists()

    def test_empty_data_does_not_crash(self, tmp_path):
        spec = PlotSpec(PlotType.HEXBIN, "a", "b")
        data = {"a": np.array([np.nan] * 10), "b": np.array([np.nan] * 10)}
        artifact = render_plot(spec, data, output_dir=tmp_path, filename="e.png")
        assert artifact.notes

    def test_render_closes_figures(self, data, tmp_path):
        """A leaked figure per plot exhausts memory over a long run."""
        import matplotlib.pyplot as plt

        before = len(plt.get_fignums())
        for i in range(8):
            render_plot(
                PlotSpec(PlotType.SCATTER, "trip_distance", "fare_amount"),
                data,
                output_dir=tmp_path,
                filename=f"leak{i}.png",
            )
        assert len(plt.get_fignums()) == before
