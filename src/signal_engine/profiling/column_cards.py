"""Column cards and dataset cards — the ONLY dataset representation the LLM sees.

A card is a compact, structured description of a column: what it means, what
units it carries, how it is distributed, and what is known to be dangerous
about it.  The engine can hold 3.7 million rows in Parquet and still hand the
model roughly two kilobytes of text.

`to_compact_text()` is the token-efficient rendering used in prompts.
`to_dict()` is the full record used for caching, the API, and Elasticsearch.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from signal_engine.profiling.semantic_types import SemanticType
from signal_engine.profiling.units import Unit


@dataclass
class NumericSummary:
    min: float | None = None
    max: float | None = None
    mean: float | None = None
    median: float | None = None
    std: float | None = None
    p01: float | None = None
    p05: float | None = None
    p25: float | None = None
    p75: float | None = None
    p95: float | None = None
    p99: float | None = None
    negative_count: int | None = None
    zero_count: int | None = None

    @property
    def is_populated(self) -> bool:
        return self.min is not None or self.mean is not None

    def skew_hint(self) -> str | None:
        """Cheap distribution-shape label, useful for choosing transforms."""
        if self.mean is None or self.median is None or self.std in (None, 0):
            return None
        if self.std is None or self.std == 0:
            return "constant"
        gap = (self.mean - self.median) / self.std
        if gap > 0.5:
            return "right_skewed"
        if gap < -0.5:
            return "left_skewed"
        return "roughly_symmetric"


@dataclass
class CategorySummary:
    distinct_count: int | None = None
    approximate: bool = False
    top_values: list[tuple[Any, int]] = field(default_factory=list)

    def coverage(self, total: int) -> float | None:
        if not self.top_values or total <= 0:
            return None
        return sum(c for _, c in self.top_values) / total


@dataclass
class TemporalSummary:
    min: str | None = None
    max: str | None = None
    span_days: float | None = None


@dataclass
class ColumnCard:
    """Everything known about one column."""

    name: str
    physical_dtype: str
    semantic_type: SemanticType
    unit: Unit
    type_confidence: float
    type_rule: str

    row_count: int = 0
    null_count: int = 0
    non_null_count: int = 0

    description: str = ""
    numeric: NumericSummary = field(default_factory=NumericSummary)
    categories: CategorySummary = field(default_factory=CategorySummary)
    temporal: TemporalSummary = field(default_factory=TemporalSummary)

    sample_values: list[Any] = field(default_factory=list)
    category_labels: dict[Any, str] = field(default_factory=dict)

    caveats: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    blocks_claims: list[dict] = field(default_factory=list)
    is_fare_component: bool = False
    approximate_metrics: list[str] = field(default_factory=list)
    provenance: str = ""

    # ---- derived --------------------------------------------------------
    @property
    def null_fraction(self) -> float:
        return self.null_count / self.row_count if self.row_count else 0.0

    @property
    def is_analyzable_numeric(self) -> bool:
        """Can this column go straight into correlation maths?"""
        return self.semantic_type.is_numeric_quantity and self.non_null_count > 0

    @property
    def is_constant(self) -> bool:
        if self.numeric.is_populated and self.numeric.min is not None:
            return self.numeric.min == self.numeric.max
        return self.categories.distinct_count == 1

    # ---- rendering ------------------------------------------------------
    def to_compact_text(self) -> str:
        """One dense line per column.  This is what costs tokens, so it is terse."""
        parts = [f"{self.name} [{self.semantic_type.value}"]
        if self.unit.label not in ("unitless", "unknown"):
            parts.append(f", {self.unit.label}")
        parts.append("]")
        head = "".join(parts)

        bits: list[str] = []
        if self.description:
            bits.append(self.description.rstrip("."))
        if self.numeric.is_populated:
            bits.append(
                f"min={_fmt(self.numeric.min)} p25={_fmt(self.numeric.p25)} "
                f"med={_fmt(self.numeric.median)} p75={_fmt(self.numeric.p75)} "
                f"max={_fmt(self.numeric.max)} mean={_fmt(self.numeric.mean)} sd={_fmt(self.numeric.std)}"
            )
            hint = self.numeric.skew_hint()
            if hint and hint != "roughly_symmetric":
                bits.append(hint)
            if self.numeric.negative_count:
                bits.append(f"{self.numeric.negative_count:,} negative values")
        if self.categories.distinct_count is not None:
            approx = "~" if self.categories.approximate else ""
            bits.append(f"{approx}{self.categories.distinct_count:,} distinct")
            if self.category_labels:
                shown = list(self.category_labels.items())[:8]
                bits.append("codes: " + ", ".join(f"{k}={v}" for k, v in shown))
            elif self.categories.top_values:
                shown = self.categories.top_values[:5]
                bits.append("top: " + ", ".join(f"{k}({c:,})" for k, c in shown))
        if self.temporal.min:
            bits.append(f"range {self.temporal.min} .. {self.temporal.max}")
        if self.null_count:
            bits.append(f"{self.null_fraction:.1%} null")
        for caveat in self.caveats:
            bits.append(f"CAVEAT: {caveat}")
        for warning in self.warnings:
            bits.append(f"WARNING: {warning}")

        return f"{head}: " + "; ".join(bits)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["semantic_type"] = self.semantic_type.value
        d["unit"] = self.unit.to_dict()
        d["null_fraction"] = round(self.null_fraction, 6)
        d["categories"]["top_values"] = [list(t) for t in self.categories.top_values]
        d["category_labels"] = {str(k): v for k, v in self.category_labels.items()}
        d["sample_values"] = [_jsonable(v) for v in self.sample_values]
        return d


@dataclass
class DatasetCard:
    """Dataset-level context: identity, scale, provenance, global caveats."""

    dataset_id: str
    fingerprint: str
    description: str = ""
    row_count: int = 0
    column_count: int = 0
    byte_size: int = 0
    source_url: str = ""
    dataset_notes: list[str] = field(default_factory=list)
    accounting_identities: list[dict] = field(default_factory=list)
    profiled_seconds: float = 0.0
    scan_strategy: str = "polars_lazy_single_pass"

    def to_compact_text(self) -> str:
        lines = [
            f"DATASET {self.dataset_id}",
            f"{self.description}".strip(),
            f"{self.row_count:,} rows x {self.column_count} columns ({self.byte_size / (1 << 20):.1f} MiB on disk)",
        ]
        if self.dataset_notes:
            lines.append("Notes:")
            lines += [f"  - {n}" for n in self.dataset_notes]
        if self.accounting_identities:
            lines.append("Known accounting identities (mechanical, not discoveries):")
            for ident in self.accounting_identities:
                comps = " + ".join(ident.get("components", []))
                lines.append(f"  - {ident.get('target')} = {comps}")
        return "\n".join(x for x in lines if x)

    def to_dict(self) -> dict:
        return asdict(self)


def _fmt(v: float | None) -> str:
    if v is None:
        return "?"
    if isinstance(v, float):
        if v != v:  # NaN
            return "nan"
        av = abs(v)
        if av >= 1_000_000 or (0 < av < 0.001):
            return f"{v:.3g}"
        return f"{v:,.4g}"
    return str(v)


def _jsonable(v: Any) -> Any:
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    return str(v)
