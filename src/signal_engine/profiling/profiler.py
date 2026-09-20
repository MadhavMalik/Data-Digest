"""The cheap profiler.

Architecture decision (see docs/decisions/0002-profiler-single-pass.md):
the profiler builds ONE Polars lazy query containing every aggregation for
every column and collects it once.  A naive per-column loop over 3.7M rows
re-reads the Parquet file N times; the single-pass plan reads each column
exactly once and lets Polars parallelise across them.

Exactness policy: metrics that are cheap in a streaming pass (min, max, mean,
std, null counts, quantiles) are computed over ALL rows.  Distinct counts use
Polars' approximate counter and are explicitly flagged in
`approximate_metrics`, because an exact distinct count over millions of rows
costs far more than the decision it informs.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl

from signal_engine.ingestion.base import DatasetHandle, stable_hash
from signal_engine.profiling.column_cards import (
    CategorySummary,
    ColumnCard,
    DatasetCard,
    NumericSummary,
    TemporalSummary,
)
from signal_engine.profiling.semantic_types import (
    SemanticType,
    ValueEvidence,
    infer_semantic_type,
    unit_from_label,
)
from signal_engine.profiling.units import Unit

PROFILE_SCHEMA_VERSION = 3


@dataclass
class DatasetProfile:
    """Profiler output: the dataset card plus one card per column."""

    dataset: DatasetCard
    columns: dict[str, ColumnCard] = field(default_factory=dict)
    path: Path | None = None
    stats: dict = field(default_factory=dict)

    # ---- accessors ------------------------------------------------------
    def card(self, name: str) -> ColumnCard:
        return self.columns[name]

    def numeric_columns(self) -> list[str]:
        return [n for n, c in self.columns.items() if c.is_analyzable_numeric and not c.is_constant]

    def categorical_columns(self) -> list[str]:
        return [n for n, c in self.columns.items() if c.semantic_type.is_categorical]

    def datetime_columns(self) -> list[str]:
        return [n for n, c in self.columns.items() if c.semantic_type is SemanticType.DATETIME]

    def units(self) -> dict[str, Unit]:
        return {n: c.unit for n, c in self.columns.items()}

    # ---- LLM-facing rendering ------------------------------------------
    def to_prompt_text(self, columns: list[str] | None = None, *, max_columns: int = 60) -> str:
        """The compact dataset+column description sent to the model."""
        names = columns or list(self.columns)
        truncated = len(names) > max_columns
        names = names[:max_columns]
        body = "\n".join(f"- {self.columns[n].to_compact_text()}" for n in names if n in self.columns)
        out = f"{self.dataset.to_compact_text()}\n\nCOLUMNS:\n{body}"
        if truncated:
            out += f"\n(+{len(self.columns) - max_columns} more columns not shown)"
        return out

    def to_dict(self) -> dict:
        return {
            "schema_version": PROFILE_SCHEMA_VERSION,
            "dataset": self.dataset.to_dict(),
            "columns": {n: c.to_dict() for n, c in self.columns.items()},
            "path": str(self.path) if self.path else None,
            "stats": self.stats,
        }


# ---------------------------------------------------------------------------
# Profiling
# ---------------------------------------------------------------------------

_QUANTILES = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
_QUANTILE_FIELDS = ["p01", "p05", "p25", "median", "p75", "p95", "p99"]
_MAX_TOP_VALUES = 12
_TOP_VALUE_CARDINALITY_LIMIT = 200


def profile_dataset(
    handle: DatasetHandle,
    *,
    dictionary: dict[str, dict] | None = None,
    accounting_identities: list[dict] | None = None,
    dataset_notes: list[str] | None = None,
    cache_dir: Path | None = None,
    use_cache: bool = True,
    sample_rows: int = 5,
) -> DatasetProfile:
    """Profile a dataset once, then cache it against the dataset fingerprint."""
    dictionary = dictionary or {}
    cache_key = stable_hash(
        {
            "fingerprint": handle.fingerprint,
            "schema": PROFILE_SCHEMA_VERSION,
            "dict": sorted(dictionary),
        }
    )
    cache_path = (Path(cache_dir) / "profiles" / f"{cache_key}.json") if cache_dir else None

    if use_cache and cache_path and cache_path.exists():
        try:
            profile = _profile_from_dict(json.loads(cache_path.read_text()))
            profile.stats["cache_hit"] = True
            return profile
        except Exception:  # noqa: BLE001 - a corrupt cache entry just means recompute
            cache_path.unlink(missing_ok=True)

    started = time.time()
    lf = _scan(handle.path)
    schema = lf.collect_schema()
    names = list(schema.names())
    dtypes = {n: schema[n] for n in names}

    row_count = handle.row_count
    if row_count is None:
        row_count = int(lf.select(pl.len()).collect().item())

    # ---- pass 1: every aggregation for every column, one collect --------
    agg_exprs: list[pl.Expr] = []
    for name in names:
        dt = dtypes[name]
        agg_exprs.append(pl.col(name).null_count().alias(f"{name}|nulls"))
        agg_exprs.append(pl.col(name).approx_n_unique().alias(f"{name}|nuniq"))
        if dt.is_numeric():
            agg_exprs += [
                pl.col(name).min().alias(f"{name}|min"),
                pl.col(name).max().alias(f"{name}|max"),
                pl.col(name).mean().alias(f"{name}|mean"),
                pl.col(name).std().alias(f"{name}|std"),
                (pl.col(name) < 0).sum().alias(f"{name}|neg"),
                (pl.col(name) == 0).sum().alias(f"{name}|zero"),
            ]
            for q, field_name in zip(_QUANTILES, _QUANTILE_FIELDS):
                agg_exprs.append(pl.col(name).quantile(q).alias(f"{name}|{field_name}"))
            if dt.is_float():
                # Does the column actually use its fractional capacity?  A float
                # column holding only whole numbers is often an encoded code.
                agg_exprs.append(
                    (pl.col(name) != pl.col(name).floor()).sum().alias(f"{name}|frac")
                )
        elif dt.is_temporal():
            agg_exprs += [
                pl.col(name).min().alias(f"{name}|min"),
                pl.col(name).max().alias(f"{name}|max"),
            ]

    agg_row = lf.select(agg_exprs).collect().to_dicts()[0]

    # ---- pass 2: head sample (Parquet makes this nearly free) -----------
    head = lf.head(sample_rows).collect()

    # ---- build cards ----------------------------------------------------
    columns: dict[str, ColumnCard] = {}
    for name in names:
        dt = dtypes[name]
        meta = _lookup_metadata(name, dictionary)
        nulls = int(agg_row.get(f"{name}|nulls") or 0)
        nuniq = agg_row.get(f"{name}|nuniq")
        non_null = row_count - nulls

        evidence = ValueEvidence(
            dtype=str(dt),
            is_integral=dt.is_integer(),
            is_float=dt.is_float(),
            is_temporal=dt.is_temporal(),
            is_string=dt in (pl.Utf8, pl.String) or str(dt).lower().endswith("string"),
            is_boolean=dt == pl.Boolean,
            distinct_count=int(nuniq) if nuniq is not None else None,
            row_count=row_count,
            non_null_count=non_null,
            min_value=_as_float(agg_row.get(f"{name}|min")),
            max_value=_as_float(agg_row.get(f"{name}|max")),
            has_negative=bool(agg_row.get(f"{name}|neg") or 0),
            has_fractional=bool(agg_row.get(f"{name}|frac") or 0),
            distinct_ratio=(int(nuniq) / non_null) if (nuniq and non_null) else None,
        )

        inference = infer_semantic_type(name, evidence, meta)
        card = ColumnCard(
            name=name,
            physical_dtype=str(dt),
            semantic_type=inference.semantic_type,
            unit=inference.unit,
            type_confidence=inference.confidence,
            type_rule=inference.rule,
            row_count=row_count,
            null_count=nulls,
            non_null_count=non_null,
            description=(meta or {}).get("description", ""),
            sample_values=_head_values(head, name),
            caveats=list((meta or {}).get("caveats", [])),
            blocks_claims=list((meta or {}).get("blocks_claims", [])),
            is_fare_component=bool((meta or {}).get("fare_component")),
            approximate_metrics=["distinct_count"],
            provenance="official_metadata" if meta else "inferred",
        )
        card.warnings.extend(inference.notes)

        if dt.is_numeric():
            card.numeric = NumericSummary(
                min=_as_float(agg_row.get(f"{name}|min")),
                max=_as_float(agg_row.get(f"{name}|max")),
                mean=_as_float(agg_row.get(f"{name}|mean")),
                median=_as_float(agg_row.get(f"{name}|median")),
                std=_as_float(agg_row.get(f"{name}|std")),
                p01=_as_float(agg_row.get(f"{name}|p01")),
                p05=_as_float(agg_row.get(f"{name}|p05")),
                p25=_as_float(agg_row.get(f"{name}|p25")),
                p75=_as_float(agg_row.get(f"{name}|p75")),
                p95=_as_float(agg_row.get(f"{name}|p95")),
                p99=_as_float(agg_row.get(f"{name}|p99")),
                negative_count=int(agg_row.get(f"{name}|neg") or 0),
                zero_count=int(agg_row.get(f"{name}|zero") or 0),
            )
        if dt.is_temporal():
            tmin, tmax = agg_row.get(f"{name}|min"), agg_row.get(f"{name}|max")
            span = None
            if tmin is not None and tmax is not None:
                try:
                    span = (tmax - tmin).total_seconds() / 86400.0
                except Exception:  # noqa: BLE001 - date types differ across dialects
                    span = None
            card.temporal = TemporalSummary(
                min=str(tmin) if tmin is not None else None,
                max=str(tmax) if tmax is not None else None,
                span_days=round(span, 3) if span is not None else None,
            )

        card.categories = CategorySummary(
            distinct_count=int(nuniq) if nuniq is not None else None, approximate=True
        )
        if meta and meta.get("categories"):
            card.category_labels = {k: v for k, v in meta["categories"].items()}

        columns[name] = card

    # ---- pass 3: exact value counts for the few real categoricals -------
    cat_targets = [
        n
        for n, c in columns.items()
        if c.semantic_type.is_categorical
        and (c.categories.distinct_count or 0) <= _TOP_VALUE_CARDINALITY_LIMIT
    ]
    if cat_targets:
        counts = _value_counts(lf, cat_targets)
        for name, pairs in counts.items():
            columns[name].categories.top_values = pairs[:_MAX_TOP_VALUES]
            # Exact distinct count comes free with the exact value counts.
            columns[name].categories.distinct_count = len(pairs)
            columns[name].categories.approximate = False
            columns[name].approximate_metrics = []

    _add_cross_column_warnings(columns, accounting_identities or [])

    elapsed = time.time() - started
    dataset_card = DatasetCard(
        dataset_id=handle.dataset_id,
        fingerprint=handle.fingerprint,
        description=handle.description,
        row_count=row_count,
        column_count=len(columns),
        byte_size=handle.byte_size or 0,
        source_url=handle.source_url or "",
        dataset_notes=list(dataset_notes or []),
        accounting_identities=list(accounting_identities or []),
        profiled_seconds=round(elapsed, 3),
    )

    profile = DatasetProfile(
        dataset=dataset_card,
        columns=columns,
        path=handle.path,
        stats={
            "cache_hit": False,
            "elapsed_seconds": round(elapsed, 3),
            "aggregations_in_single_pass": len(agg_exprs),
            "value_count_columns": len(cat_targets),
            "rows_scanned": row_count,
            "columns_profiled": len(columns),
        },
    )

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(profile.to_dict(), indent=2, default=str))

    return profile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _scan(path: Path) -> pl.LazyFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pl.scan_parquet(path)
    if suffix in {".csv", ".tsv"}:
        return pl.scan_csv(path, separator="\t" if suffix == ".tsv" else ",", try_parse_dates=True)
    if suffix in {".ndjson", ".jsonl"}:
        return pl.scan_ndjson(path)
    raise ValueError(f"unsupported dataset format: {path.suffix}")


def _value_counts(lf: pl.LazyFrame, names: list[str]) -> dict[str, list[tuple[Any, int]]]:
    """Exact value counts for a handful of low-cardinality columns.

    Each column needs its own group-by, but Polars runs the collects in
    parallel via `collect_all`, so this is one scheduled batch rather than N
    sequential scans.
    """
    plans = [
        lf.group_by(name).agg(pl.len().alias("n")).sort("n", descending=True).head(_MAX_TOP_VALUES * 4)
        for name in names
    ]
    frames = pl.collect_all(plans)
    out: dict[str, list[tuple[Any, int]]] = {}
    for name, frame in zip(names, frames):
        out[name] = [(_jsonable(r[name]), int(r["n"])) for r in frame.to_dicts()]
    return out


def _lookup_metadata(name: str, dictionary: dict[str, dict]) -> dict | None:
    meta = dictionary.get(name)
    if meta is None:
        # TLC alternates `airport_fee` / `Airport_fee` between monthly files.
        for key, value in dictionary.items():
            if key.lower() == name.lower():
                meta = value
                break
    if meta and meta.get("alias_of"):
        canonical = dictionary.get(meta["alias_of"])
        if canonical:
            merged = dict(canonical)
            merged.update({k: v for k, v in meta.items() if k != "alias_of"})
            return merged
    return meta


def _add_cross_column_warnings(columns: dict[str, ColumnCard], identities: list[dict]) -> None:
    """Attach accounting-identity warnings to the columns they constrain."""
    # TLC alternates `airport_fee` / `Airport_fee` between monthly files, so
    # identity members are matched case-insensitively against real columns.
    by_lower = {n.lower(): n for n in columns}

    for ident in identities:
        target = by_lower.get(str(ident.get("target", "")).lower())
        comps = [by_lower[c.lower()] for c in ident.get("components", []) if c.lower() in by_lower]
        if target is None or not comps:
            continue
        note = (
            f"{target} is the {ident.get('relation', 'sum')} of "
            f"{', '.join(comps)}. Correlation with any of them is mechanical."
        )
        if note not in columns[target].caveats:
            columns[target].caveats.append(note)
        for comp in comps:
            comp_note = f"{comp} is a component of {target} (accounting identity)."
            if comp_note not in columns[comp].caveats:
                columns[comp].caveats.append(comp_note)


def _head_values(head: pl.DataFrame, name: str) -> list[Any]:
    if name not in head.columns:
        return []
    return [_jsonable(v) for v in head[name].to_list()]


def _as_float(v: Any) -> float | None:
    if v is None or isinstance(v, (str, bytes)):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _jsonable(v: Any) -> Any:
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    return str(v)


# ---------------------------------------------------------------------------
# Cache round-trip
# ---------------------------------------------------------------------------


def _profile_from_dict(payload: dict) -> DatasetProfile:
    if payload.get("schema_version") != PROFILE_SCHEMA_VERSION:
        raise ValueError("stale profile cache schema")

    ds = payload["dataset"]
    dataset = DatasetCard(**ds)

    columns: dict[str, ColumnCard] = {}
    for name, c in payload["columns"].items():
        unit_d = c["unit"]
        from signal_engine.profiling.units import Kind

        unit = unit_from_label(unit_d["label"], confidence=unit_d.get("confidence", 1.0))
        if unit.kind is Kind.UNKNOWN and unit_d.get("kind") != "unknown":
            unit = Unit(unit_d["label"], unit.dimension, Kind(unit_d["kind"]), unit_d.get("confidence", 1.0))

        card = ColumnCard(
            name=name,
            physical_dtype=c["physical_dtype"],
            semantic_type=SemanticType(c["semantic_type"]),
            unit=unit,
            type_confidence=c["type_confidence"],
            type_rule=c["type_rule"],
            row_count=c["row_count"],
            null_count=c["null_count"],
            non_null_count=c["non_null_count"],
            description=c.get("description", ""),
            numeric=NumericSummary(**c.get("numeric", {})),
            temporal=TemporalSummary(**c.get("temporal", {})),
            sample_values=c.get("sample_values", []),
            category_labels=_restore_labels(c.get("category_labels", {})),
            caveats=c.get("caveats", []),
            warnings=c.get("warnings", []),
            blocks_claims=c.get("blocks_claims", []),
            is_fare_component=c.get("is_fare_component", False),
            approximate_metrics=c.get("approximate_metrics", []),
            provenance=c.get("provenance", ""),
        )
        cats = c.get("categories", {})
        card.categories = CategorySummary(
            distinct_count=cats.get("distinct_count"),
            approximate=cats.get("approximate", False),
            top_values=[tuple(t) for t in cats.get("top_values", [])],
        )
        columns[name] = card

    return DatasetProfile(
        dataset=dataset,
        columns=columns,
        path=Path(payload["path"]) if payload.get("path") else None,
        stats=payload.get("stats", {}),
    )


def _restore_labels(raw: dict) -> dict:
    """Category codes round-trip through JSON as strings; restore ints."""
    out: dict = {}
    for k, v in raw.items():
        try:
            out[int(k)] = v
        except (TypeError, ValueError):
            out[k] = v
    return out
