#!/usr/bin/env python3
"""Build the presentation visualization suite for the NYC taxi analysis.

    python scripts/make_visuals.py --output nyc_taxi_analysis/08_visuals

These are the figures meant to be LOOKED at, as opposed to the engine's
internal evidence plots. Each one is chosen because it answers a question the
scatter/density plots cannot:

    1. choropleth       where in the city is the money?  (real zone geometry)
    2. hour x weekday   when is the city expensive?      (the commute signature)
    3. surface          fare as a joint function of distance and duration
    4. ridgeline        how do whole fare DISTRIBUTIONS differ by rate code?
    5. od_matrix        which borough pairs carry the flow?
    6. small multiples  does the distance->fare curve differ by rate code?
    7. residual map     which zones cost more than distance and time predict?

Geometry is the official TLC taxi-zone shapefile, in EPSG:2263 (NY State Plane,
feet). Nothing here reprojects: the whole map is one CRS, so plotting the raw
coordinates with an equal aspect ratio is correct and avoids a heavyweight
dependency.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import requests
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from signal_engine.features.derive import build_analysis_view  # noqa: E402
from signal_engine.ingestion.base import register_local_dataset  # noqa: E402
from signal_engine.ingestion.tlc import (  # noqa: E402
    YELLOW_ACCOUNTING_IDENTITIES,
    YELLOW_TAXI_DICTIONARY,
)
from signal_engine.profiling.profiler import profile_dataset  # noqa: E402

SHAPEFILE_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"

# ---- palette -------------------------------------------------------------
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_3 = "#86857f"
PAPER = "#fcfcfb"
GRID = "#e4e3df"
BLUE = "#2a78d6"
RED = "#e34948"
ORANGE = "#eb6834"

SEQ = LinearSegmentedColormap.from_list(
    "seq", ["#f2f7fe", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)
DIV = LinearSegmentedColormap.from_list(
    "div", ["#0d366b", "#256abf", "#86b6ef", "#eeedea", "#f0a0a0", "#e34948", "#8f1f1f"]
)
HEAT = LinearSegmentedColormap.from_list(
    "heat", ["#f7f6f3", "#cde2fb", "#6da7ec", "#3987e5", "#eda100", "#eb6834", "#e34948"]
)


def style(ax, *, grid=True):
    ax.set_facecolor(PAPER)
    if grid:
        ax.grid(True, color=GRID, linewidth=0.7, alpha=0.85)
        ax.set_axisbelow(True)
    else:
        ax.grid(False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)


def title(ax, text, subtitle="", *, gap=0.030):
    """Title with an optional subtitle stacked cleanly above the axes.

    The subtitle sits ABOVE the title (higher axes fraction) and the title's
    pad is sized from the subtitle's line count, so the two never collide on
    axes whose spines are hidden -- which is every map here.
    """
    lines = subtitle.count("\n") + 1 if subtitle else 0
    ax.set_title(text, color=INK, fontsize=14,
                 pad=10 + 13 * lines, loc="left", fontweight="semibold")
    if subtitle:
        ax.annotate(subtitle, xy=(0, 1.0 + gap * 0.35), xycoords="axes fraction",
                    fontsize=9.5, color=INK_2, va="bottom", linespacing=1.45)


def save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor(PAPER)
    fig.savefig(path, dpi=150, facecolor=PAPER, bbox_inches="tight")
    plt.close(fig)
    print(f"    {path.name}  ({path.stat().st_size // 1024} KB)")


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def ensure_shapefile(geo_dir: Path) -> Path:
    shp = geo_dir / "taxi_zones" / "taxi_zones.shp"
    if shp.exists():
        return shp
    geo_dir.mkdir(parents=True, exist_ok=True)
    archive = geo_dir / "taxi_zones.zip"
    if not archive.exists():
        print("  downloading the official TLC taxi-zone shapefile...")
        resp = requests.get(SHAPEFILE_URL, timeout=120)
        resp.raise_for_status()
        archive.write_bytes(resp.content)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(geo_dir)
    return shp


def load_zone_polygons(shp_path: Path) -> dict[int, list[np.ndarray]]:
    """LocationID -> list of polygon rings (multipart zones are common)."""
    import shapefile

    sf = shapefile.Reader(str(shp_path))
    fields = [f[0] for f in sf.fields[1:]]
    loc_idx = fields.index("LocationID")

    zones: dict[int, list[np.ndarray]] = {}
    for rec, shape in zip(sf.records(), sf.shapes()):
        loc = int(rec[loc_idx])
        pts = np.asarray(shape.points, dtype=np.float64)
        parts = list(shape.parts) + [len(pts)]
        rings = [pts[parts[i]:parts[i + 1]] for i in range(len(parts) - 1)]
        zones.setdefault(loc, []).extend(r for r in rings if len(r) >= 3)
    return zones


def draw_choropleth(ax, zones, values: dict[int, float], *, cmap, label,
                    norm=None, missing="#ecebe7"):
    """Fill each zone polygon by its value. One CRS, equal aspect, no reprojection."""
    polys, colors, plain = [], [], []
    finite = [v for v in values.values() if v is not None and np.isfinite(v)]
    if not finite:
        return None
    norm = norm or Normalize(vmin=float(np.min(finite)), vmax=float(np.max(finite)))

    for loc, rings in zones.items():
        v = values.get(loc)
        for ring in rings:
            if v is None or not np.isfinite(v):
                plain.append(ring)
            else:
                polys.append(ring)
                colors.append(cmap(norm(v)))

    if plain:
        ax.add_collection(PolyCollection(plain, facecolors=missing,
                                         edgecolors=PAPER, linewidths=0.35, zorder=1))
    pc = PolyCollection(polys, facecolors=colors, edgecolors=PAPER,
                        linewidths=0.35, zorder=2)
    ax.add_collection(pc)
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = ax.figure.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label(label, color=INK_2, fontsize=9.5)
    cbar.ax.tick_params(colors=INK_2, labelsize=8.5, length=0)
    cbar.outline.set_edgecolor(GRID)
    return norm


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def fig_choropleth_pair(df, zones, lookup, out: Path):
    """Where does the money come from, and where is it most expensive?"""
    agg = (
        df.group_by("PULocationID")
        .agg([
            pl.len().alias("trips"),
            pl.col("total_amount").mean().alias("avg_fare"),
            pl.col("fare_per_mile").median().alias("med_fpm"),
        ])
        .filter(pl.col("trips") >= 200)
    )
    trips = dict(zip(agg["PULocationID"].to_list(), agg["trips"].to_list()))
    fare = dict(zip(agg["PULocationID"].to_list(), agg["avg_fare"].to_list()))

    fig, axes = plt.subplots(1, 2, figsize=(17, 8.5))
    draw_choropleth(axes[0], zones, trips, cmap=SEQ, label="pickups (log scale)",
                    norm=LogNorm(vmin=max(1, min(trips.values())), vmax=max(trips.values())))
    title(axes[0], "Where trips begin",
          "Yellow-taxi pickups by TLC zone · January 2026 · log colour scale")

    draw_choropleth(axes[1], zones, fare, cmap=HEAT, label="mean total_amount (USD)")
    title(axes[1], "What a trip from here costs",
          "Mean total charged, by pickup zone · zones with <200 pickups left grey")

    # Name the extremes so the map is readable without a legend hunt.
    names = dict(zip(lookup["LocationID"].to_list(), lookup["Zone"].to_list()))
    top = sorted(fare.items(), key=lambda kv: -kv[1])[:3]
    lines = ["Priciest pickup zones:"] + [
        f"  {names.get(loc, loc)} — ${v:,.0f}" for loc, v in top
    ]
    axes[1].annotate("\n".join(lines), xy=(0.015, 0.02), xycoords="axes fraction",
                     fontsize=9, color=INK_2, va="bottom",
                     bbox=dict(boxstyle="round,pad=0.5", facecolor=PAPER,
                               edgecolor=GRID, alpha=0.95))
    fig.tight_layout()
    save(fig, out / "01_map_pickups_and_fares.png")


def fig_residual_map(df, zones, lookup, out: Path):
    """Which zones cost more than distance and duration alone predict?

    This is the map version of the engine's residual stage: fit fare from
    distance and duration, then colour each zone by its average leftover. Red
    means a trip from there costs more than its length and time explain.
    """
    sub = df.select(["PULocationID", "trip_distance", "trip_duration_minutes",
                     "total_amount"]).drop_nulls()
    dist = sub["trip_distance"].to_numpy()
    dur = sub["trip_duration_minutes"].to_numpy()
    fare = sub["total_amount"].to_numpy()
    loc = sub["PULocationID"].to_numpy()

    X = np.column_stack([np.ones(len(dist)), dist, dur])
    beta, *_ = np.linalg.lstsq(X, fare, rcond=None)
    resid = fare - X @ beta

    levels, inv = np.unique(loc, return_inverse=True)
    counts = np.bincount(inv).astype(float)
    sums = np.bincount(inv, weights=resid)
    means = sums / np.maximum(counts, 1)
    values = {int(l): m for l, m, c in zip(levels, means, counts) if c >= 200}

    lim = float(np.percentile(np.abs(list(values.values())), 97))
    fig, ax = plt.subplots(figsize=(10.5, 9.5))
    draw_choropleth(ax, zones, values, cmap=DIV,
                    label="mean residual fare (USD)",
                    norm=Normalize(vmin=-lim, vmax=lim))
    title(ax, "Which zones cost more than the trip itself explains",
          f"Baseline: total_amount ≈ {beta[0]:.2f} + {beta[1]:.2f}·distance + "
          f"{beta[2]:.2f}·duration\n"
          f"Red = costs MORE than distance and time predict · Blue = less",
          gap=0.055)

    names = dict(zip(lookup["LocationID"].to_list(), lookup["Zone"].to_list()))
    hi = sorted(values.items(), key=lambda kv: -kv[1])[:4]
    lo = sorted(values.items(), key=lambda kv: kv[1])[:3]
    txt = ["Most over baseline:"] + [f"  {names.get(k, k)[:26]}  {v:+.2f}" for k, v in hi]
    txt += ["", "Most under baseline:"] + [f"  {names.get(k, k)[:26]}  {v:+.2f}" for k, v in lo]
    ax.annotate("\n".join(txt), xy=(0.015, 0.02), xycoords="axes fraction",
                fontsize=8.8, color=INK_2, va="bottom", family="monospace",
                bbox=dict(boxstyle="round,pad=0.55", facecolor=PAPER,
                          edgecolor=GRID, alpha=0.96))
    fig.tight_layout()
    save(fig, out / "02_map_residual_fare.png")


def fig_time_heatmaps(df, out: Path):
    """The commute signature: hour x weekday, three ways."""
    agg = (
        df.group_by(["pickup_day_of_week", "pickup_hour"])
        .agg([
            pl.len().alias("trips"),
            pl.col("total_amount").mean().alias("fare"),
            pl.col("average_speed_mph").median().alias("speed"),
        ])
    )
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    def grid(col):
        g = np.full((7, 24), np.nan)
        for d, h, v in zip(agg["pickup_day_of_week"], agg["pickup_hour"], agg[col]):
            if d is not None and h is not None and 1 <= d <= 7:
                g[int(d) - 1, int(h)] = v
        return g

    panels = [
        ("trips", SEQ, "trips", "Demand", "When New York hails a cab"),
        ("fare", HEAT, "mean total_amount (USD)", "Price", "Mean fare by hour and weekday"),
        ("speed", DIV.reversed(), "median speed (mph)", "Congestion",
         "Median trip speed — dark = gridlock"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(14, 11.5))
    for ax, (col, cmap, label, tag, sub) in zip(axes, panels):
        g = grid(col)
        im = ax.imshow(g, cmap=cmap, aspect="auto", interpolation="nearest")
        ax.set_xticks(range(0, 24))
        ax.set_xticklabels([f"{h:02d}" for h in range(24)], fontsize=8.5)
        ax.set_yticks(range(7)); ax.set_yticklabels(days, fontsize=9.5)
        ax.set_xlabel("hour of pickup", color=INK_2, fontsize=10)
        style(ax, grid=False)
        title(ax, f"{tag} — {sub}")
        cbar = fig.colorbar(im, ax=ax, fraction=0.022, pad=0.012)
        cbar.set_label(label, color=INK_2, fontsize=9)
        cbar.ax.tick_params(colors=INK_2, labelsize=8, length=0)
        cbar.outline.set_edgecolor(GRID)

        # Mark the extreme cell so each panel states its own headline.
        if np.isfinite(g).any():
            flat = np.nanargmax(g) if col != "speed" else np.nanargmin(g)
            r, c = divmod(int(flat), 24)
            ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False,
                                       edgecolor=INK, linewidth=2.0, zorder=5))
            word = "peak" if col != "speed" else "slowest"
            # Place the callout INSIDE the panel, flipping away from whichever
            # edge the marked cell sits against -- a top-row callout otherwise
            # lands on top of the panel title.
            dy = 1.35 if r <= 1 else -1.35
            dx = -1.6 if c >= 18 else 1.6
            ax.annotate(f"{word}: {days[r]} {c:02d}:00 ({g[r, c]:,.1f})",
                        xy=(c, r), xytext=(c + dx, r + dy), fontsize=9,
                        color=INK, fontweight="semibold",
                        ha="right" if dx < 0 else "left",
                        va="top" if dy > 0 else "bottom",
                        bbox=dict(boxstyle="round,pad=0.32", facecolor=PAPER,
                                  edgecolor=GRID, alpha=0.93),
                        arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
    fig.tight_layout()
    save(fig, out / "03_heatmap_hour_weekday.png")


def fig_surface(df, out: Path):
    """Fare as a joint function of distance and duration.

    Two views of one surface: a 3D render for shape, and a filled contour for
    reading actual values off it.

    The cell-count floor matters more than it looks. At 40 trips a cell's mean
    is noisy enough to produce spikes that read as structure but are sampling
    error; the floor scales with the grid's median occupancy so it adapts to
    the data rather than being a magic number.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    sub = df.select(["trip_distance", "trip_duration_minutes", "total_amount"]).drop_nulls()
    d = sub["trip_distance"].to_numpy()
    t = sub["trip_duration_minutes"].to_numpy()
    f = sub["total_amount"].to_numpy()

    keep = (d <= np.percentile(d, 97)) & (t <= np.percentile(t, 97))
    d, t, f = d[keep], t[keep], f[keep]

    nb = 24
    de = np.unique(np.quantile(d, np.linspace(0, 1, nb + 1)))
    te = np.unique(np.quantile(t, np.linspace(0, 1, nb + 1)))
    di = np.clip(np.searchsorted(de, d, "right") - 1, 0, len(de) - 2)
    ti = np.clip(np.searchsorted(te, t, "right") - 1, 0, len(te) - 2)

    shape = (len(te) - 1, len(de) - 1)
    counts = np.zeros(shape)
    sums = np.zeros(shape)
    np.add.at(counts, (ti, di), 1.0)
    np.add.at(sums, (ti, di), f)

    occupied = counts[counts > 0]
    floor = max(60.0, float(np.median(occupied)) * 0.05)
    Z = np.where(counts >= floor, sums / np.maximum(counts, 1), np.nan)

    Dc = (de[:-1] + de[1:]) / 2
    Tc = (te[:-1] + te[1:]) / 2
    X, Y = np.meshgrid(Dc, Tc)

    fig = plt.figure(figsize=(16, 7.2))
    fig.suptitle("Fare is a plane in distance-duration space — until it isn't",
                 color=INK, fontsize=15, fontweight="semibold", x=0.055, ha="left", y=0.995)
    fig.text(0.055, 0.915,
             f"Mean total_amount over a {shape[1]}x{shape[0]} quantile grid. Cells with fewer "
             f"than {floor:.0f} trips are dropped as too noisy to plot.\n"
             "Distance sets the level; duration adds the surcharge that stopped traffic bills for.",
             color=INK_2, fontsize=9.5, ha="left", linespacing=1.5)

    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.plot_surface(X, Y, Z, cmap=HEAT, linewidth=0.2, antialiased=True,
                    edgecolor=PAPER, alpha=0.97, rstride=1, cstride=1)
    ax.set_xlabel("trip_distance (mi)", color=INK_2, fontsize=9.5, labelpad=8)
    ax.set_ylabel("duration (min)", color=INK_2, fontsize=9.5, labelpad=8)
    ax.set_zlabel("mean total_amount (USD)", color=INK_2, fontsize=9.5, labelpad=8)
    ax.view_init(elev=30, azim=-132)
    ax.set_facecolor(PAPER)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_facecolor(PAPER)
        pane.pane.set_edgecolor(GRID)
        pane.set_tick_params(colors=INK_2, labelsize=8)
    ax.set_title("the surface", color=INK, fontsize=11, pad=2, loc="left")

    ax2 = fig.add_subplot(1, 2, 2)
    masked = np.ma.masked_invalid(Z)
    cf = ax2.contourf(X, Y, masked, levels=16, cmap=HEAT)
    cs = ax2.contour(X, Y, masked, levels=8, colors=PAPER, linewidths=0.8, alpha=0.75)
    ax2.clabel(cs, inline=True, fontsize=7.5, fmt="$%.0f")
    ax2.set_xlabel("trip_distance (mi)", color=INK_2, fontsize=10)
    ax2.set_ylabel("duration (min)", color=INK_2, fontsize=10)
    style(ax2, grid=False)
    ax2.set_title("the same surface, read as a map", color=INK, fontsize=11,
                  pad=2, loc="left")
    cbar = fig.colorbar(cf, ax=ax2, fraction=0.04, pad=0.02)
    cbar.set_label("mean total_amount (USD)", color=INK_2, fontsize=9)
    cbar.ax.tick_params(colors=INK_2, labelsize=8, length=0)
    cbar.outline.set_edgecolor(GRID)

    fig.tight_layout(rect=[0, 0, 1, 0.885])
    save(fig, out / "04_surface_fare_distance_duration.png")


def fig_ridgeline(df, profile, out: Path):
    """Whole fare DISTRIBUTIONS by rate code — not just their means."""
    labels = profile.card("RatecodeID").category_labels
    sub = df.select(["RatecodeID", "total_amount"]).drop_nulls()
    codes = (
        sub.group_by("RatecodeID").agg(pl.len().alias("n"))
        .filter(pl.col("n") >= 2000).sort("n", descending=True)
    )
    order = codes["RatecodeID"].to_list()[:6]

    fig, ax = plt.subplots(figsize=(12.5, 7.5))
    style(ax)
    hi = float(np.percentile(sub["total_amount"].to_numpy(), 99))
    grid = np.linspace(0, hi, 400)
    cmap = plt.get_cmap("viridis")

    for i, code in enumerate(order[::-1]):
        vals = sub.filter(pl.col("RatecodeID") == code)["total_amount"].to_numpy()
        vals = vals[(vals >= 0) & (vals <= hi)]
        if vals.size < 500:
            continue
        # Histogram-based density: a KDE over 2M points is needlessly slow and
        # oversmooths the flat-fare spikes that are the whole point here.
        hist, edges = np.histogram(vals, bins=220, range=(0, hi), density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        dens = np.interp(grid, centers, hist)
        dens = dens / dens.max() * 0.92

        base = i * 1.0
        color = cmap(0.12 + 0.72 * i / max(len(order) - 1, 1))
        ax.fill_between(grid, base, base + dens, color=color, alpha=0.82,
                        linewidth=0, zorder=i * 2)
        ax.plot(grid, base + dens, color=PAPER, linewidth=1.3, zorder=i * 2 + 1)
        med = float(np.median(vals))
        ax.plot([med, med], [base, base + dens[np.argmin(abs(grid - med))]],
                color=PAPER, linewidth=1.6, linestyle=":", zorder=i * 2 + 1)
        ax.text(hi * 0.985, base + 0.12,
                f"{labels.get(int(code), code)}   n={vals.size:,}   median ${med:,.2f}",
                ha="right", fontsize=9.5, color=INK, zorder=99)

    ax.set_yticks([]); ax.set_xlim(0, hi)
    ax.set_xlabel("total_amount (USD)", color=INK_2, fontsize=10.5)
    title(ax, "Rate codes are different products, not different prices",
          "Fare distribution by RatecodeID. The JFK flat fare is a spike; "
          "standard-rate is a long right tail.")
    fig.tight_layout()
    save(fig, out / "05_ridgeline_fare_by_ratecode.png")


def fig_od_matrix(df, lookup, out: Path):
    """Borough-to-borough flow, and the busiest zone pairs."""
    boro = dict(zip(lookup["LocationID"].to_list(), lookup["Borough"].to_list()))
    names = dict(zip(lookup["LocationID"].to_list(), lookup["Zone"].to_list()))

    sub = df.select(["PULocationID", "DOLocationID", "total_amount"]).drop_nulls()
    pu = sub["PULocationID"].to_numpy()
    do = sub["DOLocationID"].to_numpy()

    bor = ["Manhattan", "Queens", "Brooklyn", "Bronx", "Staten Island", "EWR"]
    idx = {b: i for i, b in enumerate(bor)}
    M = np.zeros((len(bor), len(bor)))
    for p, dd in zip(pu, do):
        a, b = idx.get(boro.get(int(p))), idx.get(boro.get(int(dd)))
        if a is not None and b is not None:
            M[a, b] += 1

    fig, axes = plt.subplots(1, 2, figsize=(17, 7.2),
                             gridspec_kw={"width_ratios": [1, 1.25]})

    ax = axes[0]
    shown = np.where(M > 0, M, np.nan)
    im = ax.imshow(shown, cmap=SEQ, norm=LogNorm(vmin=max(1, np.nanmin(shown)),
                                                 vmax=np.nanmax(shown)))
    ax.set_xticks(range(len(bor))); ax.set_xticklabels(bor, rotation=35, ha="right", fontsize=9.5)
    ax.set_yticks(range(len(bor))); ax.set_yticklabels(bor, fontsize=9.5)
    ax.set_xlabel("drop-off borough", color=INK_2, fontsize=10)
    ax.set_ylabel("pickup borough", color=INK_2, fontsize=10)
    style(ax, grid=False)
    total = M.sum()
    for i in range(len(bor)):
        for j in range(len(bor)):
            if M[i, j] > 0:
                frac = M[i, j] / total
                ax.text(j, i, f"{frac:.1%}" if frac >= 0.001 else "·",
                        ha="center", va="center", fontsize=8.5,
                        color=PAPER if frac > 0.06 else INK)
    title(ax, "Borough-to-borough flow", "Share of all trips · log colour scale")
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("trips (log)", color=INK_2, fontsize=9)
    cbar.ax.tick_params(colors=INK_2, labelsize=8, length=0)
    cbar.outline.set_edgecolor(GRID)

    ax = axes[1]
    pairs = (
        sub.group_by(["PULocationID", "DOLocationID"])
        .agg([pl.len().alias("n"), pl.col("total_amount").mean().alias("fare")])
        .sort("n", descending=True).head(15)
    )
    lbl = [f"{names.get(int(a), a)[:20]} → {names.get(int(b), b)[:20]}"
           for a, b in zip(pairs["PULocationID"], pairs["DOLocationID"])]
    n = pairs["n"].to_numpy()
    fare = pairs["fare"].to_numpy()
    y = np.arange(len(lbl))[::-1]
    norm = Normalize(vmin=fare.min(), vmax=fare.max())
    ax.barh(y, n, color=[HEAT(norm(f)) for f in fare], height=0.74,
            edgecolor=PAPER, linewidth=0.8)
    ax.set_yticks(y); ax.set_yticklabels(lbl, fontsize=9)
    ax.set_xlabel("trips", color=INK_2, fontsize=10)
    style(ax)
    for yy, nn, ff in zip(y, n, fare):
        ax.text(nn * 1.012, yy, f"{nn:,}  ·  ${ff:,.0f}", va="center",
                fontsize=8.5, color=INK_2)
    ax.set_xlim(0, n.max() * 1.22)
    title(ax, "The 15 busiest zone pairs", "Bar colour is the mean fare on that route")
    fig.tight_layout()
    save(fig, out / "06_od_flow.png")


def fig_small_multiples(df, profile, out: Path):
    """Does the distance->fare curve have the same SHAPE for every rate code?"""
    from signal_engine.statistics.shape import conditional_mean_curve, fit_shape

    labels = profile.card("RatecodeID").category_labels
    sub = df.select(["RatecodeID", "trip_distance", "total_amount"]).drop_nulls()
    counts = (sub.group_by("RatecodeID").agg(pl.len().alias("n"))
              .filter(pl.col("n") >= 5000).sort("n", descending=True))
    codes = counts["RatecodeID"].to_list()[:6]
    if not codes:
        return

    cols = 3
    rows = int(np.ceil(len(codes) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.4 * cols, 4.0 * rows), squeeze=False)

    for ax, code in zip(axes.flat, codes):
        part = sub.filter(pl.col("RatecodeID") == code)
        d = part["trip_distance"].to_numpy()
        f = part["total_amount"].to_numpy()
        curve = conditional_mean_curve(d, f, bins=18)
        style(ax)
        if curve is None:
            ax.set_visible(False)
            continue
        fit = fit_shape(curve)
        lo = curve.y - 1.96 * curve.se
        hi = curve.y + 1.96 * curve.se
        ax.fill_between(curve.x, lo, hi, color=BLUE, alpha=0.22, linewidth=0)
        ax.plot(curve.x, curve.y, color=BLUE, linewidth=2.4, marker="o",
                markersize=3.8, markerfacecolor=PAPER, markeredgecolor=BLUE)
        ax.set_xlabel("trip_distance (mi)", color=INK_2, fontsize=9)
        ax.set_ylabel("mean total_amount (USD)", color=INK_2, fontsize=9)
        ax.set_title(f"{labels.get(int(code), code)}", color=INK,
                     fontsize=11, loc="left", fontweight="medium")
        ax.annotate(f"n={len(d):,}\n{fit.form}  R²={fit.r2:.3f}",
                    xy=(0.035, 0.95), xycoords="axes fraction", va="top",
                    fontsize=8.5, color=INK_2)

    for ax in axes.flat[len(codes):]:
        ax.set_visible(False)

    fig.suptitle("The same axes, six different pricing regimes",
                 color=INK, fontsize=14.5, fontweight="semibold", x=0.035, ha="left")
    fig.text(0.035, 0.955,
             "Conditional mean of fare given distance, fitted separately per rate code. "
             "A flat line is an administered fare; a rising line is a meter.",
             color=INK_2, fontsize=9.5, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, out / "07_small_multiples_by_ratecode.png")


# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", default="nyc_taxi_analysis/08_visuals")
    ap.add_argument("--sample", type=int, default=0,
                    help="row cap for speed; 0 uses every row")
    args = ap.parse_args(argv)

    out = REPO_ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    handle = register_local_dataset(
        REPO_ROOT / "data/raw/yellow_tripdata_2026-01.parquet",
        dataset_id="nyc_tlc_yellow_2026_01",
    )
    profile = profile_dataset(
        handle, dictionary=YELLOW_TAXI_DICTIONARY,
        accounting_identities=YELLOW_ACCOUNTING_IDENTITIES,
        cache_dir=REPO_ROOT / ".cache",
    )
    view = build_analysis_view(pl.scan_parquet(handle.path), profile)
    frame = view.frame
    if args.sample:
        frame = frame.head(args.sample)

    df = frame.select([
        "PULocationID", "DOLocationID", "RatecodeID", "trip_distance",
        "trip_duration_minutes", "total_amount", "fare_amount", "fare_per_mile",
        "average_speed_mph", "pickup_hour", "pickup_day_of_week",
    ]).collect()
    print(f"  {df.height:,} rows after validity filters")

    lookup = pl.read_csv(REPO_ROOT / "data/metadata/taxi_zone_lookup.csv")
    shp = ensure_shapefile(REPO_ROOT / "data/geo")
    zones = load_zone_polygons(shp)
    print(f"  {len(zones)} zone geometries loaded")

    print("\nRendering:")
    fig_choropleth_pair(df, zones, lookup, out)
    fig_residual_map(df, zones, lookup, out)
    fig_time_heatmaps(df, out)
    fig_surface(df, out)
    fig_ridgeline(df, profile, out)
    fig_od_matrix(df, lookup, out)
    fig_small_multiples(df, profile, out)

    print(f"\n{len(list(out.glob('*.png')))} figures written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
