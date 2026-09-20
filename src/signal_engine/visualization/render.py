"""Matplotlib rendering.

Design rules applied throughout (these follow a validated categorical palette;
see docs/decisions/0006-visualization-policy.md):

  * ONE hue for a single series; the categorical order is fixed, never cycled.
  * Sequential encoding (density, counts) uses a single hue light->dark.
    Never a rainbow colormap: rainbow implies ordering that the luminance does
    not carry, and it is unreadable under colour-vision deficiency.
  * Diverging encoding (correlation) uses blue<->red with a NEUTRAL GRAY
    midpoint, so "no correlation" reads as nothing.
  * Recessive grid and axes; thin marks; the data is the darkest thing.
  * Never a dual y-axis.
  * Identity is never colour-alone: labelled axes and direct annotation carry it.

Rendering is also a cost-control stage.  Nothing here draws 3.5 million marks:
each form aggregates or samples first, and the strategy used is recorded in the
PlotSpec so the VLM is told what it is looking at.
"""

from __future__ import annotations

import base64
import math
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless; must precede pyplot
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402

from signal_engine.visualization.selector import PlotSpec, PlotType  # noqa: E402

# ---- palette (light surface) ----------------------------------------------
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
TREND = "#e34948"

# Sequential blue ramp, light -> dark.
SEQUENTIAL_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue",
    ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)
# Diverging blue <-> red with a neutral gray midpoint.
DIVERGING = LinearSegmentedColormap.from_list(
    "div_blue_red",
    ["#0d366b", "#256abf", "#86b6ef", "#f0efec", "#f0a0a0", "#e34948", "#8f1f1f"],
)

DPI = 110
FIGSIZE = (7.2, 4.6)


@dataclass
class PlotArtifact:
    """A rendered graph plus everything needed to interpret it."""

    path: Path | None
    plot_type: str
    spec_description: str
    width_px: int = 0
    height_px: int = 0
    bytes_size: int = 0
    data_uri: str | None = field(default=None, repr=False)
    render_seconds: float = 0.0
    points_drawn: int = 0
    rows_represented: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "path": str(self.path) if self.path else None,
            "plot_type": self.plot_type,
            "description": self.spec_description,
            "width_px": self.width_px,
            "height_px": self.height_px,
            "bytes": self.bytes_size,
            "render_seconds": round(self.render_seconds, 3),
            "points_drawn": self.points_drawn,
            "rows_represented": self.rows_represented,
            "notes": self.notes,
        }


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9, length=0)


def _finish(fig, ax, spec: PlotSpec) -> None:
    ax.set_xlabel(spec.x_label or spec.x, color=TEXT_SECONDARY, fontsize=10)
    ax.set_ylabel(spec.y_label or (spec.y or ""), color=TEXT_SECONDARY, fontsize=10)
    ax.set_title(spec.title, color=TEXT_PRIMARY, fontsize=12, pad=12, loc="left", fontweight="medium")
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()


def render_plot(
    spec: PlotSpec,
    data: dict[str, np.ndarray],
    *,
    output_dir: Path | None = None,
    filename: str | None = None,
    stats_annotation: str = "",
    rng_seed: int = 7,
    embed_data_uri: bool = True,
) -> PlotArtifact:
    """Render `spec` against `data` and return the artifact."""
    import time

    started = time.time()
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    _style(ax)

    notes: list[str] = []
    points = 0
    rows = 0

    try:
        handler = {
            PlotType.SCATTER: _render_scatter,
            PlotType.HEXBIN: _render_hexbin,
            PlotType.BINNED_TREND: _render_binned_trend,
            PlotType.LINE: _render_time_line,
            PlotType.TIME_BINNED_LINE: _render_time_line,
            PlotType.BOX: _render_box,
            PlotType.VIOLIN: _render_violin,
            PlotType.BAR_WITH_CI: _render_bar_ci,
            PlotType.CONTINGENCY_HEATMAP: _render_contingency,
            PlotType.HISTOGRAM: _render_histogram,
            PlotType.CORRELATION_HEATMAP: _render_correlation_heatmap,
        }[spec.plot_type]
        points, rows, extra_notes = handler(ax, spec, data, rng_seed)
        notes.extend(extra_notes)

        if stats_annotation:
            ax.annotate(
                stats_annotation,
                xy=(0.985, 0.03),
                xycoords="axes fraction",
                ha="right",
                va="bottom",
                fontsize=8.5,
                color=TEXT_SECONDARY,
                bbox={"boxstyle": "round,pad=0.4", "facecolor": SURFACE, "edgecolor": GRID, "alpha": 0.95},
            )

        _finish(fig, ax, spec)

        path = None
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            path = output_dir / (filename or f"{spec.plot_type.value}_{spec.x}_{spec.y or ''}.png")
            fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")

        data_uri = None
        buf = BytesIO()
        fig.savefig(buf, format="png", facecolor=SURFACE, bbox_inches="tight")
        raw = buf.getvalue()
        if embed_data_uri:
            data_uri = "data:image/png;base64," + base64.b64encode(raw).decode()

        return PlotArtifact(
            path=path,
            plot_type=spec.plot_type.value,
            spec_description=spec.describe_for_vlm(),
            width_px=int(fig.get_size_inches()[0] * DPI),
            height_px=int(fig.get_size_inches()[1] * DPI),
            bytes_size=len(raw),
            data_uri=data_uri,
            render_seconds=time.time() - started,
            points_drawn=points,
            rows_represented=rows,
            notes=notes,
        )
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------


def _xy(spec: PlotSpec, data: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(data[spec.x], dtype=np.float64)
    y = np.asarray(data[spec.y], dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def _render_scatter(ax, spec, data, seed) -> tuple[int, int, list[str]]:
    x, y = _xy(spec, data)
    rows = x.size
    notes: list[str] = []

    if rows > spec.max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(rows, size=spec.max_points, replace=False)
        x, y = x[idx], y[idx]
        notes.append(f"sampled {spec.max_points:,} of {rows:,} rows for rendering")

    ax.scatter(x, y, s=7, alpha=0.28, color=SERIES[0], linewidths=0, rasterized=True)
    if spec.overlay_trend and x.size > 2 and x.std() > 0:
        _overlay_ols(ax, x, y)
    return x.size, rows, notes


def _render_hexbin(ax, spec, data, seed) -> tuple[int, int, list[str]]:
    x, y = _xy(spec, data)
    rows = x.size
    notes: list[str] = []
    if rows == 0:
        return 0, 0, ["no finite rows to plot"]

    # Clip the view to the 0.5-99.5 percentile range so a handful of extreme
    # values cannot compress all the real structure into one cell.
    xlo, xhi = np.percentile(x, [0.5, 99.5])
    ylo, yhi = np.percentile(y, [0.5, 99.5])
    if xhi > xlo and yhi > ylo:
        notes.append("axes clipped to the 0.5-99.5 percentile range to keep structure visible")
        extent = (xlo, xhi, ylo, yhi)
    else:
        extent = None

    hb = ax.hexbin(
        x, y, gridsize=spec.n_bins, cmap=SEQUENTIAL_BLUE, norm=LogNorm(), extent=extent,
        linewidths=0.15, edgecolors=SURFACE,
    )
    cbar = ax.figure.colorbar(hb, ax=ax, pad=0.015)
    cbar.set_label("records per cell (log)", color=TEXT_SECONDARY, fontsize=9)
    cbar.ax.tick_params(colors=TEXT_SECONDARY, labelsize=8, length=0)
    cbar.outline.set_edgecolor(GRID)

    if spec.overlay_trend and x.std() > 0:
        _overlay_ols(ax, x, y, clip=extent)
    return int(hb.get_array().size), rows, notes


def _render_binned_trend(ax, spec, data, seed) -> tuple[int, int, list[str]]:
    """Conditional mean of y within quantile bins of x, with 95% CIs.

    This is the single most informative form for a large noisy dataset: it
    shows the shape of E[y|x] without pretending to draw every record.
    """
    x, y = _xy(spec, data)
    rows = x.size
    if rows < 20:
        return 0, rows, ["too few rows for a binned trend"]

    bins = spec.n_bins
    edges = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)))
    if edges.size < 3:
        return 0, rows, ["x has too few distinct values to bin"]

    idx = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, edges.size - 2)
    centers, means, errs, counts = [], [], [], []
    for b in range(edges.size - 1):
        sel = idx == b
        n = int(sel.sum())
        if n < 5:
            continue
        centers.append(float((edges[b] + edges[b + 1]) / 2))
        m = float(y[sel].mean())
        means.append(m)
        errs.append(1.96 * float(y[sel].std(ddof=1)) / math.sqrt(n) if n > 1 else 0.0)
        counts.append(n)

    if not centers:
        return 0, rows, ["no bin had enough records"]

    ax.errorbar(
        centers, means, yerr=errs, fmt="o-", color=SERIES[0], ecolor=SERIES[0],
        elinewidth=1.2, capsize=3, markersize=5, linewidth=2, alpha=0.95,
    )
    return len(centers), rows, [f"{len(centers)} quantile bins, {rows:,} rows aggregated"]


def _render_time_line(ax, spec, data, seed) -> tuple[int, int, list[str]]:
    x, y = _xy(spec, data)
    rows = x.size
    if rows < 5:
        return 0, rows, ["too few rows for a time series"]

    bins = max(8, spec.n_bins)
    edges = np.linspace(x.min(), x.max(), bins + 1)
    idx = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, bins - 1)

    centers, means = [], []
    for b in range(bins):
        sel = idx == b
        if sel.sum() < 3:
            continue
        centers.append(float((edges[b] + edges[b + 1]) / 2))
        means.append(float(y[sel].mean()))

    if not centers:
        return 0, rows, ["no time bin had enough records"]

    ax.plot(centers, means, color=SERIES[0], linewidth=2, marker="o", markersize=4)
    return len(centers), rows, [f"{len(centers)} time bins over {rows:,} rows"]


def _group_arrays(spec, data) -> tuple[list, list[np.ndarray], list[int]]:
    groups = np.asarray(data[spec.x])
    values = np.asarray(data[spec.y], dtype=np.float64)
    mask = np.isfinite(values)
    if groups.dtype.kind == "f":
        mask &= np.isfinite(groups)
    groups, values = groups[mask], values[mask]

    levels, counts = np.unique(groups, return_counts=True)
    order = np.argsort(-counts)[:30]
    keep = levels[np.sort(order)] if order.size else levels

    series, labels, sizes = [], [], []
    label_map = spec.group_labels.get(spec.x, {})
    for level in keep:
        sel = groups == level
        if sel.sum() < 5:
            continue
        series.append(values[sel])
        labels.append(_level_label(level, label_map))
        sizes.append(int(sel.sum()))
    return labels, series, sizes


def _level_label(level, label_map: dict) -> str:
    try:
        key = int(level)
    except (TypeError, ValueError):
        key = level
    if label_map and key in label_map:
        text = str(label_map[key])
        return f"{key}: {text[:18]}"
    return str(level)


def _render_box(ax, spec, data, seed) -> tuple[int, int, list[str]]:
    labels, series, sizes = _group_arrays(spec, data)
    if not series:
        return 0, 0, ["no group had enough records"]

    bp = ax.boxplot(
        series, tick_labels=labels, showfliers=False, patch_artist=True, widths=0.6,
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(SERIES[0])
        patch.set_alpha(0.35)
        patch.set_edgecolor(SERIES[0])
        patch.set_linewidth(1.4)
    for key in ("whiskers", "caps"):
        for artist in bp[key]:
            artist.set_color(SERIES[0])
            artist.set_linewidth(1.2)
    for median in bp["medians"]:
        median.set_color(TEXT_PRIMARY)
        median.set_linewidth(1.8)

    _rotate_labels(ax, labels)
    for i, n in enumerate(sizes, start=1):
        ax.annotate(
            f"n={_compact(n)}", xy=(i, 1.0), xycoords=("data", "axes fraction"),
            ha="center", va="bottom", fontsize=7.5, color=TEXT_SECONDARY,
        )
    return len(series), int(sum(sizes)), [f"{len(series)} groups"]


def _render_violin(ax, spec, data, seed) -> tuple[int, int, list[str]]:
    labels, series, sizes = _group_arrays(spec, data)
    if not series:
        return 0, 0, ["no group had enough records"]

    parts = ax.violinplot(series, showmedians=True, widths=0.7)
    for body in parts["bodies"]:
        body.set_facecolor(SERIES[0])
        body.set_alpha(0.35)
        body.set_edgecolor(SERIES[0])
    for key in ("cbars", "cmins", "cmaxes", "cmedians"):
        if key in parts:
            parts[key].set_color(SERIES[0])
            parts[key].set_linewidth(1.2)

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    _rotate_labels(ax, labels)
    return len(series), int(sum(sizes)), [f"{len(series)} groups"]


def _render_bar_ci(ax, spec, data, seed) -> tuple[int, int, list[str]]:
    labels, series, sizes = _group_arrays(spec, data)
    if not series:
        return 0, 0, ["no group had enough records"]

    means = [float(s.mean()) for s in series]
    errs = [
        1.96 * float(s.std(ddof=1)) / math.sqrt(s.size) if s.size > 1 else 0.0 for s in series
    ]
    positions = np.arange(len(labels))

    ax.bar(
        positions, means, yerr=errs, color=SERIES[0], alpha=0.85, width=0.68,
        error_kw={"ecolor": TEXT_SECONDARY, "elinewidth": 1.2, "capsize": 3},
        edgecolor=SURFACE, linewidth=1.0,
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    _rotate_labels(ax, labels)

    # Direct-label the values; a number on every bar is fine at this count.
    if len(labels) <= 12:
        for pos, m, e in zip(positions, means, errs):
            ax.annotate(
                f"{m:,.4g}", xy=(pos, m + e), ha="center", va="bottom",
                fontsize=8, color=TEXT_SECONDARY, xytext=(0, 3), textcoords="offset points",
            )
    return len(labels), int(sum(sizes)), [f"{len(labels)} groups, means with 95% CI"]


def _render_contingency(ax, spec, data, seed) -> tuple[int, int, list[str]]:
    a = np.asarray(data[spec.x])
    b = np.asarray(data[spec.y])
    mask = np.ones(a.shape, dtype=bool)
    for arr in (a, b):
        if arr.dtype.kind == "f":
            mask &= np.isfinite(arr)
    a, b = a[mask], b[mask]

    la, ia = np.unique(a, return_inverse=True)
    lb, ib = np.unique(b, return_inverse=True)
    if la.size > 25 or lb.size > 25:
        return 0, int(a.size), ["too many levels for a readable contingency heatmap"]

    table = np.zeros((la.size, lb.size))
    np.add.at(table, (ia, ib), 1.0)

    im = ax.imshow(table, cmap=SEQUENTIAL_BLUE, aspect="auto", origin="lower")
    cbar = ax.figure.colorbar(im, ax=ax, pad=0.015)
    cbar.set_label("record count", color=TEXT_SECONDARY, fontsize=9)
    cbar.ax.tick_params(colors=TEXT_SECONDARY, labelsize=8, length=0)
    cbar.outline.set_edgecolor(GRID)

    amap = spec.group_labels.get(spec.x, {})
    bmap = spec.group_labels.get(spec.y, {})
    ax.set_xticks(range(lb.size))
    ax.set_xticklabels([_level_label(v, bmap) for v in lb])
    ax.set_yticks(range(la.size))
    ax.set_yticklabels([_level_label(v, amap) for v in la])
    _rotate_labels(ax, [str(v) for v in lb])
    ax.grid(False)
    return int(table.size), int(a.size), [f"{la.size} x {lb.size} contingency table"]


def _render_histogram(ax, spec, data, seed) -> tuple[int, int, list[str]]:
    x = np.asarray(data[spec.x], dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0, 0, ["no finite values"]

    lo, hi = np.percentile(x, [0.1, 99.9])
    notes = []
    if hi > lo:
        clipped = x[(x >= lo) & (x <= hi)]
        if clipped.size < x.size:
            notes.append(
                f"x-range clipped to the 0.1-99.9 percentile ({x.size - clipped.size:,} "
                "extreme values outside the drawn range)"
            )
        x_plot = clipped
    else:
        x_plot = x

    ax.hist(x_plot, bins=spec.n_bins, color=SERIES[0], alpha=0.85, edgecolor=SURFACE, linewidth=0.8)
    median = float(np.median(x))
    ax.axvline(median, color=TREND, linewidth=1.6, linestyle="--")
    ax.annotate(
        f"median {median:,.4g}", xy=(median, 0.95), xycoords=("data", "axes fraction"),
        ha="left", va="top", fontsize=8.5, color=TREND, xytext=(4, 0), textcoords="offset points",
    )
    return int(spec.n_bins), int(x.size), notes


def _render_correlation_heatmap(ax, spec, data, seed) -> tuple[int, int, list[str]]:
    names = [k for k in data if k != "__matrix__"]
    matrix = data.get("__matrix__")
    if matrix is None:
        cols = [np.asarray(data[n], dtype=np.float64) for n in names]
        stacked = np.column_stack(cols)
        complete = np.all(np.isfinite(stacked), axis=1)
        matrix = np.corrcoef(stacked[complete].T) if complete.sum() > 2 else np.eye(len(names))

    im = ax.imshow(matrix, cmap=DIVERGING, vmin=-1, vmax=1, aspect="auto", origin="lower")
    cbar = ax.figure.colorbar(im, ax=ax, pad=0.015)
    cbar.set_label("Pearson r", color=TEXT_SECONDARY, fontsize=9)
    cbar.ax.tick_params(colors=TEXT_SECONDARY, labelsize=8, length=0)
    cbar.outline.set_edgecolor(GRID)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.grid(False)

    if len(names) <= 14:
        for i in range(len(names)):
            for j in range(len(names)):
                v = matrix[i, j]
                if not np.isfinite(v):
                    continue
                ax.text(
                    j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color=TEXT_PRIMARY if abs(v) < 0.55 else SURFACE,
                )
    return int(matrix.size), 0, [f"{len(names)} variables"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _overlay_ols(ax, x: np.ndarray, y: np.ndarray, clip: tuple | None = None) -> None:
    if x.std() == 0:
        return
    slope, intercept = np.polyfit(x, y, 1)
    lo, hi = (clip[0], clip[1]) if clip else (x.min(), x.max())
    xs = np.linspace(lo, hi, 50)
    ax.plot(xs, slope * xs + intercept, color=TREND, linewidth=2, alpha=0.9, zorder=5)


def _rotate_labels(ax, labels: list[str]) -> None:
    longest = max((len(str(x)) for x in labels), default=0)
    if longest > 6 or len(labels) > 8:
        plt.setp(ax.get_xticklabels(), rotation=40, ha="right", fontsize=8)


def _compact(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.0f}k"
    return str(n)


def stats_caption(result) -> str:
    """Compact statistics caption burned into the image.

    Putting the numbers ON the graph means the VLM cannot read a direction off
    the picture that contradicts the arithmetic — the authoritative values are
    right there in its visual field.
    """
    bits = [f"n={result.n:,}"]
    if result.pearson_r is not None:
        bits.append(f"r={result.pearson_r:+.3f}")
    if result.spearman_rho is not None:
        bits.append(f"ρ={result.spearman_rho:+.3f}")
    if result.eta is not None:
        bits.append(f"η={result.eta:.3f}")
    if result.mutual_information is not None:
        bits.append(f"MI={result.mutual_information:.3f}")
    if result.stability is not None:
        bits.append(f"stab={result.stability:.2f}")
    return "  ".join(bits)
