#!/usr/bin/env python3
"""Run the full research loop on the NYC TLC yellow-taxi dataset.

    python scripts/run_demo.py
    python scripts/run_demo.py --question "What drives tipping behaviour?" --rounds 2

Works with no credentials at all: the planner and interpreter fall back to
their deterministic paths and the run still produces real statistics, real
plots, and real evidence records.  Whatever was degraded is printed at the end.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from signal_engine.config import load_settings  # noqa: E402
from signal_engine.ingestion.base import register_local_dataset  # noqa: E402
from signal_engine.ingestion.tlc import (  # noqa: E402
    YELLOW_ACCOUNTING_IDENTITIES,
    YELLOW_DATASET_NOTES,
    YELLOW_TAXI_DICTIONARY,
    TLCSource,
    TLCVehicle,
)
from signal_engine.reporting import write_run_report  # noqa: E402
from signal_engine.search.orchestrator import AnalysisRequest, SignalEngine  # noqa: E402

DEFAULT_QUESTION = (
    "What factors are associated with the amount passengers pay for NYC yellow taxi trips?"
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--question", default=DEFAULT_QUESTION)
    p.add_argument("--vehicle", default="yellow", choices=[v.value for v in TLCVehicle])
    p.add_argument("--year", type=int, default=2026)
    p.add_argument("--month", type=int, default=1)
    p.add_argument("--data", default=None, help="path to a Parquet file (skips resolution by date)")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--max-visualizations", type=int, default=8)
    p.add_argument("--web-grounding", action="store_true", help="allow Brave grounding calls")
    p.add_argument("--json-out", default=None, help="write the full result JSON here")
    p.add_argument("--quiet", action="store_true")
    return p


async def run(args) -> int:
    settings = load_settings(REPO_ROOT)
    settings.paths.ensure()

    if args.data:
        path = Path(args.data)
        if not path.is_absolute():
            path = REPO_ROOT / path
    else:
        source = TLCSource(vehicle=TLCVehicle(args.vehicle), year=args.year, month=args.month)
        path = settings.paths.raw / source.local_filename()
        if not path.exists():
            print(f"Dataset not found at {path}.")
            print(
                f"Run: python scripts/fetch_tlc.py --vehicle {args.vehicle} "
                f"--year {args.year} --month {args.month}"
            )
            return 2

    handle = register_local_dataset(
        path,
        dataset_id=f"nyc_tlc_{args.vehicle}_{args.year:04d}_{args.month:02d}",
        description=(
            f"NYC TLC {args.vehicle} taxi trip records, {args.year:04d}-{args.month:02d}. "
            "Official monthly Parquet release."
        ),
        source_url="https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page",
    )

    if not args.quiet:
        print("=" * 78)
        print("SIGNAL ENGINE")
        print("=" * 78)
        print(f"Dataset : {handle.dataset_id}  ({handle.row_count:,} rows)")
        print(f"Question: {args.question}")
        missing = settings.missing_credentials()
        if missing:
            print("\nRunning with reduced capability. Missing:")
            for m in missing:
                print(f"  - {m}")
        print()

    engine = SignalEngine(settings)
    try:
        result = await engine.analyze(
            AnalysisRequest(
                question=args.question,
                dataset_handle=handle,
                dictionary=YELLOW_TAXI_DICTIONARY,
                accounting_identities=YELLOW_ACCOUNTING_IDENTITIES,
                dataset_notes=YELLOW_DATASET_NOTES,
                max_rounds=args.rounds,
                max_visualizations=args.max_visualizations,
                enable_web_grounding=args.web_grounding,
            )
        )
    finally:
        await engine.aclose()

    report_path = write_run_report(result, settings.paths.artifacts)

    if not args.quiet:
        _print_summary(result, report_path)

    if args.json_out:
        out = Path(args.json_out)
        if not out.is_absolute():
            out = REPO_ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result.to_dict(), indent=2, default=str))
        print(f"\nFull result JSON: {out}")

    return 0


def _print_summary(result, report_path: Path) -> None:
    metrics = result.metrics.to_dict() if result.metrics else {}
    derived = metrics.get("derived", {})
    counters = metrics.get("counters", {})

    print("-" * 78)
    print("TOP FINDINGS")
    print("-" * 78)
    for test in result.state.ranked_tests(limit=10):
        flag = "  [MECHANICAL]" if test.is_mechanical else ""
        print(f"  {test.result.to_compact_text()[:150]}{flag}")

    print()
    print("-" * 78)
    print("EVIDENCE")
    print("-" * 78)
    for e in result.evidence[:8]:
        print(f"  [{e.explanation_status.value:<12}] {e.canonical_expression[:48]:<50} "
              f"{e.statistical_metrics.to_compact_text()[:60]}")
        if e.plot_uri:
            print(f"      plot: {e.plot_uri}")

    print()
    print("-" * 78)
    print("DEMO METRICS (all measured, none estimated unless marked)")
    print("-" * 78)
    ds = result.profile.dataset
    view = result.view_report
    print(f"  Raw dataset rows            : {ds.row_count:,}")
    print(f"  Raw bytes on disk           : {ds.byte_size:,}")
    print(f"  Rows after validity filters : {view.get('rows_after', 0):,} "
          f"({view.get('exclusion_fraction', 0):.2%} excluded)")
    print(f"  Columns profiled            : {counters.get('columns_profiled', 0)}")
    print(f"  Candidate expressions       : {result.prune_stats.considered:,}")
    print(f"  Pruned by dimensional rules : {result.prune_stats.pruned_by_units:,}")
    print(f"  Pruned by policy            : {result.prune_stats.pruned_by_policy:,}")
    print(f"  Prune rate                  : {result.prune_stats.prune_rate:.1%}")
    print(f"  Statistical tests run       : {counters.get('statistical_tests', 0):,}")
    print(f"  Mutual-information tests    : {counters.get('mutual_information_tests', 0):,}")
    print(f"  Stability tests             : {counters.get('stability_tests', 0):,}")
    print(f"  Graphs rendered             : {counters.get('plots_rendered', 0)}")
    print(f"  LLM calls                   : {counters.get('llm_calls', 0)}")
    print(f"  VLM calls                   : {counters.get('vlm_calls', 0)}")
    print(f"  Model cache hit rate        : {derived.get('model_cache_hit_rate', 0):.1%}")
    print(f"  Total tokens                : {derived.get('total_tokens', 0):,}")
    print(f"  Tokens avoided by cache     : {derived.get('tokens_avoided_by_cache', 0):,}")
    print(f"  Evidence objects stored     : {counters.get('evidence_created', 0)}")
    print(f"  Prior evidence retrieved    : {counters.get('evidence_retrievals', 0)}")
    print(f"  Critic errors caught        : {counters.get('critic_errors', 0)}")
    print(f"  Critic warnings             : {counters.get('critic_warnings', 0)}")
    print(f"  Wall-clock                  : {metrics.get('elapsed_seconds', 0):.2f}s")

    if result.final_answer:
        print()
        print("-" * 78)
        print("ANSWER")
        print("-" * 78)
        print(f"  {result.final_answer.answer}")
        for finding in result.final_answer.key_findings[:6]:
            print(f"   - {finding[:160]}")
        if result.final_answer.caveats:
            print("  Caveats:")
            for c in result.final_answer.caveats[:5]:
                print(f"   ! {c[:160]}")

    if result.degradations:
        print()
        print("-" * 78)
        print("DEGRADATIONS (services unavailable or fallbacks used)")
        print("-" * 78)
        for d in result.degradations:
            print(f"  - {d}")

    print(f"\nRun report: {report_path}")
    print(f"Stop reason: {result.state.stop_reason.value if result.state.stop_reason else 'n/a'}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
