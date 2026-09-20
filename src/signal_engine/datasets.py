"""Dataset registry: known-dataset metadata, with a working generic fallback.

The engine must run on any tabular dataset, but it does MUCH better when it
knows what the columns mean — the cash-tip trap is only catchable because the
TLC dictionary declares that `tip_amount` cannot support that comparison.

So metadata is a bonus layer, never a requirement:

    known dataset    -> curated dictionary, accounting identities, filters,
                        target hints, domain notes
    unknown dataset  -> semantic types inferred from dtype, name and
                        distribution; generic validity filters; target inferred
                        from the question

A dataset is matched on its filename, then on its column signature, so a file
renamed on upload still resolves. Adding support for another challenge dataset
means appending one `DatasetSpec` here — no engine changes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


@dataclass
class DatasetSpec:
    """What the engine knows about a dataset before it opens it."""

    key: str
    label: str
    description: str = ""
    dictionary: dict[str, dict] = field(default_factory=dict)
    accounting_identities: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    # Column names that make good analysis targets, most preferred first.
    target_hints: dict[str, list[str]] = field(default_factory=dict)
    preferred_targets: list[str] = field(default_factory=list)
    # None means "use the generic, semantics-derived filters".
    filters: list | None = None
    filename_patterns: list[str] = field(default_factory=list)
    # Columns that, if all present, identify this dataset regardless of filename.
    signature_columns: list[str] = field(default_factory=list)
    source_url: str = ""

    def matches_filename(self, name: str) -> bool:
        low = name.lower()
        return any(re.search(p, low) for p in self.filename_patterns)

    def matches_columns(self, columns: set[str]) -> bool:
        if not self.signature_columns:
            return False
        lower = {c.lower() for c in columns}
        return all(c.lower() in lower for c in self.signature_columns)


# ---------------------------------------------------------------------------
# Known datasets
# ---------------------------------------------------------------------------


def _tlc_yellow() -> DatasetSpec:
    from signal_engine.features.derive import TLC_ANALYSIS_FILTERS
    from signal_engine.ingestion.tlc import (
        YELLOW_ACCOUNTING_IDENTITIES,
        YELLOW_DATASET_NOTES,
        YELLOW_TAXI_DICTIONARY,
    )

    return DatasetSpec(
        key="nyc_tlc_yellow",
        label="NYC TLC Yellow Taxi",
        description="Yellow-taxi trip records published monthly by the NYC Taxi and "
                    "Limousine Commission.",
        dictionary=YELLOW_TAXI_DICTIONARY,
        accounting_identities=YELLOW_ACCOUNTING_IDENTITIES,
        notes=YELLOW_DATASET_NOTES,
        preferred_targets=["total_amount", "fare_amount", "tip_amount"],
        target_hints={
            "pay": ["total_amount", "fare_amount"],
            "paid": ["total_amount", "fare_amount"],
            "pays": ["total_amount", "fare_amount"],
            "charge": ["total_amount"], "charges": ["total_amount"],
            "charged": ["total_amount"], "cost": ["total_amount", "fare_amount"],
            "price": ["fare_amount", "total_amount"], "fare": ["fare_amount"],
            "tip": ["tip_amount"], "tips": ["tip_amount"], "tipping": ["tip_amount"],
            "duration": ["trip_duration_minutes"], "speed": ["average_speed_mph"],
            "distance": ["trip_distance"],
            "profitability": ["fare_per_mile", "total_per_mile"],
            "revenue": ["total_amount"],
        },
        filters=TLC_ANALYSIS_FILTERS,
        filename_patterns=[r"yellow_tripdata", r"yellow.*taxi"],
        signature_columns=["tpep_pickup_datetime", "trip_distance", "fare_amount"],
        source_url="https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page",
    )


def _tlc_green() -> DatasetSpec:
    from signal_engine.features.derive import TLC_ANALYSIS_FILTERS
    from signal_engine.ingestion.tlc import (
        TLCVehicle,
        accounting_identities_for,
        dictionary_for,
    )

    spec = _tlc_yellow()
    return DatasetSpec(
        key="nyc_tlc_green",
        label="NYC TLC Green Taxi",
        description="Green (boro) taxi trip records from the NYC TLC.",
        dictionary=dictionary_for(TLCVehicle.GREEN),
        accounting_identities=accounting_identities_for(TLCVehicle.GREEN),
        notes=spec.notes,
        preferred_targets=spec.preferred_targets,
        target_hints=spec.target_hints,
        filters=TLC_ANALYSIS_FILTERS,
        filename_patterns=[r"green_tripdata", r"green.*taxi"],
        signature_columns=["lpep_pickup_datetime", "trip_distance", "fare_amount"],
        source_url=spec.source_url,
    )


def _generic() -> DatasetSpec:
    return DatasetSpec(
        key="generic",
        label="Generic tabular dataset",
        description=(
            "No curated metadata. Semantic types are inferred from dtype, column name "
            "and value distribution; validity filters are derived from those semantics."
        ),
        filters=None,
    )


_BUILDERS: list[Callable[[], DatasetSpec]] = [_tlc_yellow, _tlc_green]
_CACHE: dict[str, DatasetSpec] = {}


def _known() -> list[DatasetSpec]:
    if not _CACHE:
        for build in _BUILDERS:
            try:
                spec = build()
                _CACHE[spec.key] = spec
            except Exception:  # noqa: BLE001 - a broken spec must not break the engine
                continue
    return list(_CACHE.values())


def resolve_spec(path: Path | str, columns: set[str] | None = None) -> DatasetSpec:
    """Identify a dataset from its filename, then from its columns.

    Filename first because it is free; columns second because an uploaded file
    is often renamed, and the column signature still identifies it.
    """
    name = Path(path).name
    for spec in _known():
        if spec.matches_filename(name):
            return spec
    if columns:
        for spec in _known():
            if spec.matches_columns(columns):
                return spec
    return _generic()


def spec_for_key(key: str) -> DatasetSpec:
    for spec in _known():
        if spec.key == key:
            return spec
    return _generic()


def list_specs() -> list[DatasetSpec]:
    return [*_known(), _generic()]


# ---------------------------------------------------------------------------
# Generic target inference
# ---------------------------------------------------------------------------

# Words that indicate the question is about a quantity of the given kind.
_KIND_WORDS: dict[str, tuple[str, ...]] = {
    "currency": ("pay", "paid", "pays", "cost", "costs", "price", "prices", "charge",
                 "charged", "charges", "fare", "revenue", "spend", "spending",
                 "amount", "money", "dollar", "dollars", "profit", "profitability",
                 "expensive", "cheap", "billed"),
    "duration": ("duration", "long", "length", "time", "delay", "wait", "elapsed"),
    "count": ("count", "number", "volume", "many", "frequency", "demand"),
    "rate": ("rate", "speed", "ratio", "per", "efficiency", "throughput"),
    "distance": ("distance", "far", "miles", "km", "travel"),
}


def infer_target_generic(question: str, cards: dict, spec: DatasetSpec | None = None) -> str | None:
    """Pick the dependent variable a question is about, for ANY dataset.

    Scoring, highest first:
      1. a curated hint for this dataset matched by a question word
      2. a column whose NAME shares a content word with the question
      3. a column whose DESCRIPTION shares a content word
      4. a column whose semantic KIND matches the question's subject
         ("what do people pay" -> a currency column)
      5. the highest-variance quantity, as a last resort

    Returns None only when the dataset has no analysable quantity at all.
    """
    from signal_engine.profiling.semantic_types import SemanticType
    from signal_engine.search.scorer import question_terms

    terms = question_terms(question)

    # 1. curated hints
    if spec is not None:
        for term in terms:
            for candidate in spec.target_hints.get(term, []):
                if candidate in cards:
                    return candidate

    numeric = {
        name: card for name, card in cards.items()
        if getattr(card, "semantic_type", None) is not None
        and card.semantic_type.is_numeric_quantity
    }
    if not numeric:
        return None

    kind_by_semantic = {
        SemanticType.CURRENCY: "currency",
        SemanticType.DURATION: "duration",
        SemanticType.COUNT: "count",
        SemanticType.RATE: "rate",
    }
    wanted_kinds = {
        kind for kind, words in _KIND_WORDS.items() if terms & set(words)
    }

    scored: list[tuple[float, str]] = []
    for name, card in numeric.items():
        score = 0.0
        tokens = {t for t in re.split(r"[_\-\s]+", name.lower()) if len(t) > 2}
        if tokens & terms:
            score += 6.0

        desc = (getattr(card, "description", "") or "").lower()
        if desc:
            desc_tokens = {t.strip(".,;:()") for t in desc.split() if len(t) > 3}
            score += 2.0 * len(desc_tokens & terms)

        kind = kind_by_semantic.get(card.semantic_type)
        if kind and kind in wanted_kinds:
            score += 3.0

        # A total is usually a better target than one of its parts.
        if getattr(card, "is_fare_component", False):
            score -= 1.0
        if any(w in name.lower() for w in ("total", "amount", "sum")):
            score += 0.75

        # Prefer something that actually varies.
        num = getattr(card, "numeric", None)
        if num is not None and num.std and num.mean:
            spread = abs(num.std / (abs(num.mean) + 1e-9))
            score += min(spread, 1.5) * 0.5

        scored.append((score, name))

    if spec is not None:
        for preferred in spec.preferred_targets:
            if preferred in numeric:
                scored.append((5.5, preferred))

    scored.sort(key=lambda s: (-s[0], s[1]))
    return scored[0][1] if scored else None
