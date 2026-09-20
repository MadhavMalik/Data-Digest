#!/usr/bin/env python3
"""Human / domain validation — checks the OUTPUT, not just that the code ran.

    python scripts/evaluate.py

Runs the engine on real NYC TLC data, then applies domain semantic checks drawn
from the official Yellow Taxi data dictionary and from physical plausibility,
and writes `artifacts/evaluation_report.md`.

The report deliberately includes what went wrong. A report that only shows
successes is marketing, not evaluation.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import numpy as np
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from signal_engine.config import load_settings  # noqa: E402
from signal_engine.features.dag import ExpressionDAG  # noqa: E402
from signal_engine.features.derive import build_analysis_view  # noqa: E402
from signal_engine.features.expressions import Col  # noqa: E402
from signal_engine.ingestion.base import register_local_dataset  # noqa: E402
from signal_engine.ingestion.tlc import (  # noqa: E402
    YELLOW_ACCOUNTING_IDENTITIES,
    YELLOW_DATASET_NOTES,
    YELLOW_TAXI_DICTIONARY,
)
from signal_engine.profiling.semantic_types import SemanticType  # noqa: E402
from signal_engine.reporting import write_evaluation_report  # noqa: E402
from signal_engine.search.orchestrator import AnalysisRequest, SignalEngine  # noqa: E402
from signal_engine.statistics.correlation import pairwise_relationship  # noqa: E402

QUESTIONS = [
    "What factors are associated with the amount passengers pay for NYC yellow taxi trips?",
    "What factors are associated with how long a taxi trip takes?",
]


def domain_checks(profile, dag) -> list[dict]:
    """Semantic checks a human analyst would insist on."""
    checks: list[dict] = []

    def add(name, expected, observed, passed):
        checks.append(
            {"name": name, "expected": expected, "observed": str(observed), "passed": bool(passed)}
        )

    # ---- typing ----------------------------------------------------------
    card = profile.card("trip_distance")
    add(
        "trip_distance is a continuous distance",
        "continuous_measurement / miles",
        f"{card.semantic_type.value} / {card.unit.label}",
        card.semantic_type is SemanticType.CONTINUOUS_MEASUREMENT and card.unit.label == "miles",
    )

    card = profile.card("fare_amount")
    add(
        "fare_amount is currency",
        "currency / USD",
        f"{card.semantic_type.value} / {card.unit.label}",
        card.semantic_type is SemanticType.CURRENCY,
    )

    for column in ("RatecodeID", "payment_type"):
        card = profile.card(column)
        add(
            f"{column} is categorical, not a quantity",
            "categorical",
            card.semantic_type.value,
            card.semantic_type.is_categorical,
        )

    for column in ("PULocationID", "DOLocationID"):
        card = profile.card(column)
        add(
            f"{column} is an arbitrary identifier",
            "categorical_identifier, excluded from numeric analysis",
            f"{card.semantic_type.value}, in numeric pool: {column in profile.numeric_columns()}",
            card.semantic_type is SemanticType.CATEGORICAL_IDENTIFIER
            and column not in profile.numeric_columns(),
        )

    # ---- dictionary caveats ---------------------------------------------
    tip = profile.card("tip_amount")
    add(
        "cash-tip caveat is attached to tip_amount",
        "a claim block naming payment_type",
        f"{len(tip.blocks_claims)} claim block(s), {len(tip.caveats)} caveat(s)",
        bool(tip.blocks_claims),
    )

    total = profile.card("total_amount")
    add(
        "total_amount is marked as the sum of its components",
        "accounting-identity caveat present",
        f"{len(total.caveats)} caveat(s)",
        any("sum" in c.lower() for c in total.caveats),
    )

    # ---- physical plausibility -------------------------------------------
    data = dag.materialize_columns(["trip_distance", "fare_amount", "average_speed_mph"])

    r = pairwise_relationship(data["trip_distance"], data["fare_amount"])
    add(
        "longer trips cost more (the sanity anchor)",
        "Pearson r > 0.5, positive",
        f"r = {r.pearson_r:+.4f} ({r.direction})",
        r.pearson_r > 0.5 and r.direction == "positive",
    )

    speed = data["average_speed_mph"]
    speed = speed[np.isfinite(speed)]
    median_speed = float(np.median(speed))
    add(
        "median NYC taxi speed is physically plausible",
        "between 3 and 30 mph",
        f"{median_speed:.2f} mph",
        3.0 < median_speed < 30.0,
    )

    fare_per_mile = dag.materialize(Col("fare_per_mile"))
    finite = fare_per_mile[~np.isnan(fare_per_mile)]
    add(
        "fare_per_mile contains no infinities after division guards",
        "all finite",
        f"{np.isfinite(finite).all()} over {finite.size:,} non-null values",
        bool(np.isfinite(finite).all()),
    )

    # ---- the cash-tip trap, stated explicitly ----------------------------
    payment = dag.materialize_columns(["payment_type", "tip_amount"])
    cash = payment["tip_amount"][payment["payment_type"] == 2]
    card_pay = payment["tip_amount"][payment["payment_type"] == 1]
    cash, card_pay = cash[np.isfinite(cash)], card_pay[np.isfinite(card_pay)]
    if cash.size and card_pay.size:
        add(
            "cash trips DO record lower tips (the raw pattern is real) ...",
            "cash mean < card mean",
            f"cash {cash.mean():.2f} vs card {card_pay.mean():.2f}",
            cash.mean() < card_pay.mean(),
        )
        add(
            "... and the engine blocks the naive conclusion from it",
            "a dictionary claim block prevents 'cash riders tip less'",
            f"blocked: {bool(tip.blocks_claims)}",
            bool(tip.blocks_claims),
        )

    return checks


async def run(args) -> int:
    settings = load_settings(REPO_ROOT)
    settings.paths.ensure()

    path = settings.paths.raw / "yellow_tripdata_2026-01.parquet"
    if not path.exists():
        print(f"Dataset not found at {path}")
        print("Run: python scripts/fetch_tlc.py --vehicle yellow --year 2026 --month 1")
        return 2

    handle = register_local_dataset(
        path,
        dataset_id="nyc_tlc_yellow_2026_01",
        description="NYC TLC yellow taxi trip records, 2026-01.",
    )

    engine = SignalEngine(settings)
    results = []
    try:
        for question in QUESTIONS[: args.questions]:
            print(f"\n=== {question}")
            result = await engine.analyze(
                AnalysisRequest(
                    question=question,
                    dataset_handle=handle,
                    dictionary=YELLOW_TAXI_DICTIONARY,
                    accounting_identities=YELLOW_ACCOUNTING_IDENTITIES,
                    dataset_notes=YELLOW_DATASET_NOTES,
                    max_rounds=args.rounds,
                    max_visualizations=args.max_visualizations,
                )
            )
            results.append(result)
            print(
                f"    {len(result.state.ranked_tests())} findings · "
                f"{len(result.evidence)} evidence · "
                f"{result.metrics.elapsed_seconds:.1f}s"
            )

        profile = results[0].profile
        view = build_analysis_view(pl.scan_parquet(handle.path), profile)
        dag = ExpressionDAG.from_path(
            handle.path,
            dataset_fingerprint=handle.fingerprint,
            cache_dir=settings.paths.cache,
            view_hash=view.view_hash,
            frame=view.frame,
        )
        print("\n=== domain semantic checks")
        checks = domain_checks(profile, dag)
    finally:
        await engine.aclose()

    passed = sum(1 for c in checks if c["passed"])
    for c in checks:
        mark = "PASS" if c["passed"] else "FAIL"
        print(f"  [{mark}] {c['name']}: {c['observed']}")
    print(f"\n{passed}/{len(checks)} domain checks passed")

    report = write_evaluation_report(results, settings.paths.artifacts, domain_checks=checks)
    print(f"Evaluation report: {report}")
    return 0 if passed == len(checks) else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--questions", type=int, default=2)
    p.add_argument("--max-visualizations", type=int, default=5)
    return asyncio.run(run(p.parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
