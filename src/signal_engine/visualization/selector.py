"""Graph selection: the chart type follows the data's job, not convenience.

The rule the whole engine is built around is that ONE relationship gets the
visualization that most clearly exposes it.  A bar chart for everything is how
you end up showing a VLM a picture that cannot answer the question it is being
asked about.

Selection is deterministic and runs first.  The LLM may propose an override,
but `validate_override` has to accept it — a wrong plot type produces a
misleading image, and the VLM then faithfully interprets the misleading image.
Deterministic code wins on conflict.

Scale matters as much as type: continuous-vs-continuous at 3.5 million rows is
a density plot, not a scatter plot.  Drawing 3.5M points produces a black
rectangle that shows nothing and takes a minute to render.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from signal_engine.profiling.column_cards import ColumnCard
from signal_engine.profiling.semantic_types import SemanticType


class PlotType(str, Enum):
    SCATTER = "scatter"
    HEXBIN = "hexbin"
    # Density background + the conditional mean curve E[y|x] with confidence
    # bands and the best-fitting functional form.  This is the default for
    # continuous-vs-continuous at scale: a bare hexbin of 3.5M rows shows where
    # the mass is but not the SHAPE, and the shape is the finding.
    DENSITY_TREND = "density_trend"
    BINNED_TREND = "binned_trend"
    LINE = "line"
    TIME_BINNED_LINE = "time_binned_line"
    BOX = "box"
    VIOLIN = "violin"
    BAR_WITH_CI = "bar_with_ci"
    CONTINGENCY_HEATMAP = "contingency_heatmap"
    HISTOGRAM = "histogram"
    CORRELATION_HEATMAP = "correlation_heatmap"

    @property
    def needs_two_variables(self) -> bool:
        return self not in {PlotType.HISTOGRAM, PlotType.CORRELATION_HEATMAP}


# Above this row count a raw scatter is over-plotted to the point of being
# uninformative, so continuous-vs-continuous switches to a density form.
SCATTER_MAX_POINTS = 20_000
# Above this, even a sampled scatter is worse than hexbin at showing structure.
HEXBIN_THRESHOLD = 100_000
# A categorical axis past this many levels is unreadable as a box plot.
MAX_BOX_LEVELS = 15
MAX_BAR_LEVELS = 30
MAX_CONTINGENCY_LEVELS = 25


@dataclass
class PlotSpec:
    """A fully-resolved instruction for the renderer."""

    plot_type: PlotType
    x: str
    y: str | None = None
    color_by: str | None = None
    title: str = ""
    x_label: str = ""
    y_label: str = ""
    rationale: str = ""
    sample_strategy: str = "none"
    max_points: int = SCATTER_MAX_POINTS
    n_bins: int = 30
    overlay_trend: bool = True
    group_labels: dict = field(default_factory=dict)
    annotations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "plot_type": self.plot_type.value,
            "x": self.x,
            "y": self.y,
            "color_by": self.color_by,
            "title": self.title,
            "x_label": self.x_label,
            "y_label": self.y_label,
            "rationale": self.rationale,
            "sample_strategy": self.sample_strategy,
            "n_bins": self.n_bins,
            "overlay_trend": self.overlay_trend,
            "annotations": self.annotations,
        }

    def describe_for_vlm(self) -> str:
        """Plain description of the rendering, so the VLM knows what it is looking at.

        Critically this states the sampling strategy.  A VLM shown a hexbin of
        binned means must not describe it as if every trip were a point.
        """
        parts = [f"A {self.plot_type.value.replace('_', ' ')} of {self.y or self.x}"]
        if self.y:
            parts[0] += f" against {self.x}"
        if self.sample_strategy != "none":
            parts.append(f"Rendering strategy: {self.sample_strategy}.")
        if self.plot_type in {PlotType.HEXBIN}:
            parts.append(
                "Colour encodes how many records fall in each hexagonal cell (log scale), "
                "not the value of a third variable."
            )
        if self.plot_type is PlotType.DENSITY_TREND:
            parts.append(
                f"The shaded background is a density map: colour is how many records fall in "
                f"each cell (log scale), NOT the value of a third variable. The bold line is "
                f"the CONDITIONAL MEAN of {self.y} within {self.n_bins} quantile bins of "
                f"{self.x}, with a 95% confidence band. Read the bold line for the shape of "
                f"the relationship; read the background for where the data actually is."
            )
        if self.plot_type is PlotType.BINNED_TREND:
            parts.append(
                f"Points are conditional means of {self.y} within {self.n_bins} quantile bins of "
                f"{self.x}; vertical bars are 95% confidence intervals of those means. "
                "Individual records are not shown."
            )
        if self.plot_type in {PlotType.BOX, PlotType.VIOLIN}:
            parts.append(
                "Each box summarises the distribution within one category level "
                "(median, interquartile range, whiskers at 1.5 IQR)."
            )
        if self.plot_type is PlotType.BAR_WITH_CI:
            parts.append("Bar heights are group means; error bars are 95% confidence intervals.")
        if self.plot_type is PlotType.CONTINGENCY_HEATMAP:
            parts.append("Cell colour encodes the count of records in each category combination.")
        if self.overlay_trend and self.plot_type in {PlotType.SCATTER, PlotType.HEXBIN}:
            parts.append("The line is an ordinary least-squares fit.")
        if self.annotations:
            parts.extend(self.annotations)
        return " ".join(parts)


def select_plot(
    x_card: ColumnCard,
    y_card: ColumnCard | None,
    *,
    n_rows: int,
    max_scatter_points: int = SCATTER_MAX_POINTS,
    question: str = "",
) -> PlotSpec:
    """Choose the chart form from the semantic types and the data scale."""
    # ---- single variable -------------------------------------------------
    if y_card is None:
        return PlotSpec(
            plot_type=PlotType.HISTOGRAM,
            x=x_card.name,
            title=f"Distribution of {x_card.name}",
            x_label=_axis_label(x_card),
            y_label="count",
            rationale="A single continuous variable: a histogram shows its distribution.",
            n_bins=_histogram_bins(n_rows),
            sample_strategy="all rows aggregated into bins",
        )

    xt, yt = x_card.semantic_type, y_card.semantic_type

    # ---- datetime on either axis ----------------------------------------
    if xt is SemanticType.DATETIME and yt.is_numeric_quantity:
        return PlotSpec(
            plot_type=PlotType.TIME_BINNED_LINE,
            x=x_card.name,
            y=y_card.name,
            title=f"{y_card.name} over {x_card.name}",
            x_label=_axis_label(x_card),
            y_label=_axis_label(y_card),
            rationale="Time on the x-axis: a time-binned line shows how the value evolves.",
            n_bins=min(120, max(24, n_rows // 5000)),
            sample_strategy="rows aggregated into equal time bins (mean per bin)",
            overlay_trend=False,
        )
    if yt is SemanticType.DATETIME and xt.is_numeric_quantity:
        return select_plot(y_card, x_card, n_rows=n_rows, max_scatter_points=max_scatter_points)

    # ---- categorical vs numeric -----------------------------------------
    if xt.is_categorical and yt.is_numeric_quantity:
        return _categorical_numeric(x_card, y_card, n_rows)
    if yt.is_categorical and xt.is_numeric_quantity:
        return _categorical_numeric(y_card, x_card, n_rows)

    # ---- categorical vs categorical -------------------------------------
    if xt.is_categorical and yt.is_categorical:
        return PlotSpec(
            plot_type=PlotType.CONTINGENCY_HEATMAP,
            x=x_card.name,
            y=y_card.name,
            title=f"{x_card.name} vs {y_card.name}",
            x_label=x_card.name,
            y_label=y_card.name,
            rationale="Both variables are categorical: a contingency heatmap shows joint counts.",
            sample_strategy="all rows counted per category pair",
            overlay_trend=False,
            group_labels={
                x_card.name: x_card.category_labels,
                y_card.name: y_card.category_labels,
            },
        )

    # ---- continuous vs continuous ---------------------------------------
    if n_rows <= max_scatter_points:
        return PlotSpec(
            plot_type=PlotType.SCATTER,
            x=x_card.name,
            y=y_card.name,
            title=f"{y_card.name} vs {x_card.name}",
            x_label=_axis_label(x_card),
            y_label=_axis_label(y_card),
            rationale=f"{n_rows:,} rows fit in a scatter plot without over-plotting.",
            max_points=max_scatter_points,
            sample_strategy="every row plotted",
        )

    if n_rows >= HEXBIN_THRESHOLD:
        # At this scale a scatter is a solid block, and a bare hexbin is a
        # cloud: it shows where the mass sits but not how y moves with x.
        # Overlaying the conditional mean makes the shape readable.
        return PlotSpec(
            plot_type=PlotType.DENSITY_TREND,
            x=x_card.name,
            y=y_card.name,
            title=f"{y_card.name} vs {x_card.name}",
            x_label=_axis_label(x_card),
            y_label=_axis_label(y_card),
            rationale=(
                f"{n_rows:,} rows would over-plot a scatter into a solid block, and a bare "
                "density plot shows mass without shape; the conditional mean curve E[y|x] "
                "overlaid on the density shows both."
            ),
            sample_strategy=(
                "all rows aggregated into a density background, with the conditional mean "
                "of y within quantile bins of x overlaid as a curve with 95% confidence bands"
            ),
            n_bins=40,
        )

    return PlotSpec(
        plot_type=PlotType.SCATTER,
        x=x_card.name,
        y=y_card.name,
        title=f"{y_card.name} vs {x_card.name}",
        x_label=_axis_label(x_card),
        y_label=_axis_label(y_card),
        rationale=f"{n_rows:,} rows: a stratified sample keeps the scatter readable.",
        max_points=max_scatter_points,
        sample_strategy=f"stratified random sample of {max_scatter_points:,} rows",
    )


def _categorical_numeric(cat: ColumnCard, num: ColumnCard, n_rows: int) -> PlotSpec:
    levels = cat.categories.distinct_count or 0

    if levels <= MAX_BOX_LEVELS:
        plot_type = PlotType.BOX
        rationale = (
            f"{cat.name} is categorical with {levels} levels: a box plot compares the full "
            f"distribution of {num.name} across them, not just the means."
        )
        strategy = "all rows summarised per category (median, IQR, 1.5-IQR whiskers)"
    elif levels <= MAX_BAR_LEVELS:
        plot_type = PlotType.BAR_WITH_CI
        rationale = (
            f"{cat.name} has {levels} levels: too many for readable boxes, so group means "
            "with confidence intervals are shown."
        )
        strategy = "all rows aggregated to group means with 95% CIs"
    else:
        plot_type = PlotType.BAR_WITH_CI
        rationale = (
            f"{cat.name} has {levels} levels: showing the {MAX_BAR_LEVELS} most frequent."
        )
        strategy = f"top {MAX_BAR_LEVELS} levels by frequency, group means with 95% CIs"

    spec = PlotSpec(
        plot_type=plot_type,
        x=cat.name,
        y=num.name,
        title=f"{num.name} by {cat.name}",
        x_label=cat.name,
        y_label=_axis_label(num),
        rationale=rationale,
        sample_strategy=strategy,
        overlay_trend=False,
        group_labels={cat.name: cat.category_labels},
    )
    if cat.semantic_type is SemanticType.CATEGORICAL_IDENTIFIER:
        spec.annotations.append(
            f"{cat.name} holds arbitrary identifier codes; the order along the axis carries "
            "no meaning and no trend should be read across it."
        )
    return spec


def _axis_label(card: ColumnCard) -> str:
    if card.unit.label not in ("unitless", "unknown", "category", "id", "bool"):
        return f"{card.name} ({card.unit.label})"
    return card.name


def _histogram_bins(n_rows: int) -> int:
    if n_rows < 1_000:
        return 20
    if n_rows < 100_000:
        return 40
    return 60


# ---------------------------------------------------------------------------
# LLM override validation
# ---------------------------------------------------------------------------


def validate_override(
    proposed: str,
    deterministic: PlotSpec,
    x_card: ColumnCard,
    y_card: ColumnCard | None,
    *,
    n_rows: int,
) -> tuple[PlotSpec, str | None]:
    """Accept an LLM plot suggestion only when it is defensible.

    Returns (spec_to_use, rejection_reason).  Rejections are recorded rather
    than hidden, so the decision log shows where the model was overruled.
    """
    try:
        requested = PlotType(proposed.strip().lower().replace(" ", "_"))
    except ValueError:
        return deterministic, f"unknown plot type {proposed!r}"

    if requested is deterministic.plot_type:
        return deterministic, None

    xt = x_card.semantic_type
    yt = y_card.semantic_type if y_card else None

    # Hard rejections: forms that would misrepresent the data.
    if requested is PlotType.SCATTER and n_rows > HEXBIN_THRESHOLD:
        return deterministic, (
            f"scatter rejected: {n_rows:,} rows would over-plot into a solid block"
        )
    if requested in {PlotType.LINE, PlotType.TIME_BINNED_LINE} and xt is not SemanticType.DATETIME:
        return deterministic, (
            f"line plot rejected: {x_card.name} is not a time axis, so connecting points "
            "would imply an ordering that does not exist"
        )
    if requested in {PlotType.BOX, PlotType.VIOLIN, PlotType.BAR_WITH_CI}:
        if not (xt.is_categorical or (yt is not None and yt.is_categorical)):
            return deterministic, "grouped plot rejected: neither variable is categorical"
    if requested is PlotType.CONTINGENCY_HEATMAP:
        if not (xt.is_categorical and yt is not None and yt.is_categorical):
            return deterministic, "contingency heatmap rejected: both variables must be categorical"
    if requested in {PlotType.SCATTER, PlotType.HEXBIN, PlotType.BINNED_TREND, PlotType.DENSITY_TREND}:
        if xt.is_categorical or (yt is not None and yt.is_categorical):
            return deterministic, (
                "continuous plot rejected: a categorical code has no meaningful position "
                "on a continuous axis"
            )

    # Acceptable alternative: adopt it, keeping the resolved metadata.
    accepted = PlotSpec(
        plot_type=requested,
        x=deterministic.x,
        y=deterministic.y,
        color_by=deterministic.color_by,
        title=deterministic.title,
        x_label=deterministic.x_label,
        y_label=deterministic.y_label,
        rationale=f"model-recommended {requested.value}; validated against the semantic types",
        sample_strategy=deterministic.sample_strategy,
        max_points=deterministic.max_points,
        n_bins=deterministic.n_bins,
        overlay_trend=deterministic.overlay_trend,
        group_labels=deterministic.group_labels,
        annotations=list(deterministic.annotations),
    )
    return accepted, None
