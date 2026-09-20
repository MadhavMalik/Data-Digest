#!/usr/bin/env python3
"""Run the full analysis on the NYC TLC data and export everything readable.

    python scripts/run_full_analysis.py

Downloads the dataset if it is not already present, runs the research loop with
the LLM and VLM live, and writes a complete human-readable export to
`nyc_taxi_analysis/` — profiler output, hypotheses, every combination tested,
every graph, and the full stage-2 interpretation exchange.
"""

from __future__ import annotations

import argparse
import asyncio
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from signal_engine.config import load_settings  # noqa: E402
from signal_engine.export import export_analysis  # noqa: E402
from signal_engine.ingestion.base import human_bytes, register_local_dataset  # noqa: E402
from signal_engine.ingestion.tlc import (  # noqa: E402
    YELLOW_ACCOUNTING_IDENTITIES,
    YELLOW_DATASET_NOTES,
    YELLOW_TAXI_DICTIONARY,
    TLCSource,
    TLCVehicle,
    write_dictionary_metadata,
)
from signal_engine.search.orchestrator import AnalysisRequest, SignalEngine  # noqa: E402

DEFAULT_QUESTION = (
    "What factors are associated with the amount passengers pay for NYC yellow taxi trips?"
)


async def run(args) -> int:
    settings = load_settings(REPO_ROOT)
    settings.paths.ensure()

    out_root = REPO_ROOT / args.output
    out_root.mkdir(parents=True, exist_ok=True)

    # ---- 1. ensure the data is present --------------------------------
    source = TLCSource(vehicle=TLCVehicle(args.vehicle), year=args.year, month=args.month)
    raw_path = settings.paths.raw / source.local_filename()

    print("=" * 78)
    print("SIGNAL ENGINE — full analysis")
    print("=" * 78)

    if not raw_path.exists():
        print(f"\nDataset not present. Downloading from the official TLC release...")
        print(f"  {source.resolve_url()}")
        handle, dl = source.fetch(settings.paths.raw, show_progress=True)
        print(f"  downloaded {human_bytes(dl.bytes_written)} in {dl.elapsed_seconds:.1f}s")
    else:
        handle = register_local_dataset(
            raw_path,
            dataset_id=source.dataset_id(),
            description=(
                f"NYC TLC {args.vehicle} taxi trip records, {args.year:04d}-{args.month:02d}. "
                "Official monthly Parquet release."
            ),
            source_url=source.resolve_url(),
        )
        print(f"\nDataset already present: {raw_path.name}")

    print(f"  {handle.row_count:,} rows · {human_bytes(handle.byte_size or 0)}")
    print(f"  fingerprint {handle.fingerprint}")

    # Keep a copy of the source data + dictionary with the analysis output.
    data_dir = out_root / "00_source_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    if args.copy_data and not (data_dir / raw_path.name).exists():
        print(f"  copying source data into {data_dir.name}/ ...")
        shutil.copy2(handle.path, data_dir / raw_path.name)
    write_dictionary_metadata(data_dir)
    zone_lookup = settings.paths.metadata / "taxi_zone_lookup.csv"
    if zone_lookup.exists():
        shutil.copy2(zone_lookup, data_dir / zone_lookup.name)

    (data_dir / "SOURCE.txt").write_text(
        f"""NYC Taxi and Limousine Commission (TLC) Trip Record Data
{'=' * 70}

File          : {raw_path.name}
Rows          : {handle.row_count:,}
Size          : {human_bytes(handle.byte_size or 0)}
Fingerprint   : {handle.fingerprint}
Downloaded from: {source.resolve_url()}

Official page : https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
AWS registry  : https://registry.opendata.aws/nyc-tlc-trip-records-pds/

This is the dataset listed in the Voloridge HackMIT 2026 "Signal in the Noise"
challenge. The official data dictionary is in nyc_tlc_yellow_dictionary.json;
taxi_zone_lookup.csv maps the zone IDs to borough and zone names.

The raw file is never modified.
"""
    )

    # ---- 2. report what is configured ---------------------------------
    cfg = settings.redacted_dump()
    print(f"\nLLM      : {'READY — ' + cfg['llm']['model'] if cfg['llm']['configured'] else 'not configured'}")
    print(f"VLM      : {cfg['llm']['vlm_model'] if cfg['llm']['configured'] else 'not configured'}")
    print(f"Elastic  : {'connected' if cfg['elastic']['configured'] else 'local evidence memory'}")
    print(f"Brave    : {'enabled' if cfg['brave']['configured'] else 'disabled (optional)'}")
    print(f"\nQuestion : {args.question}")
    print(f"Rounds   : {args.rounds} · max graphs: {args.max_visualizations}")
    print("\nRunning...\n")

    # ---- 3. run --------------------------------------------------------
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

    # ---- 4. export -----------------------------------------------------
    print("Exporting results...")
    export = export_analysis(result, out_root)

    _print_summary(result, export)
    return 0


def _print_summary(result, export) -> None:
    metrics = result.metrics.to_dict() if result.metrics else {}
    counters = metrics.get("counters", {})
    derived = metrics.get("derived", {})

    print()
    print("=" * 78)
    print("RESULTS")
    print("=" * 78)
    print(f"  Combinations tested      : {len(result.state.tested):,}")
    print(f"  Meaningful findings      : {len(result.state.ranked_tests()):,}")
    print(f"  Graphs rendered          : {counters.get('plots_rendered', 0)}")
    print(f"  Planning (LLM) calls     : {counters.get('llm_calls', 0)}")
    print(f"  Graph-reading (VLM) calls: {counters.get('vlm_calls', 0)}")
    print(f"  Total tokens             : {derived.get('total_tokens', 0):,}")
    print(f"  Critic errors caught     : {counters.get('critic_errors', 0)}")
    print(f"  Critic warnings          : {counters.get('critic_warnings', 0)}")
    print(f"  Evidence stored          : {counters.get('evidence_created', 0)}")
    print(f"  Wall-clock               : {metrics.get('elapsed_seconds', 0):.1f}s")

    if result.degradations:
        print("\n  Degradations:")
        for d in result.degradations:
            print(f"    - {d[:150]}")

    print()
    print("=" * 78)
    print("STAGE 2 — what the model read off each graph")
    print("=" * 78)
    for t in result.vlm_traces[:6]:
        interp = t.get("interpretation_after_critic") or t.get("interpretation") or {}
        print(f"\n  [{t['sequence']}] {t['expression']} vs {t['y']}  ({t.get('plot_type')})")
        print(f"      image sent: {t.get('image_sent')} · model: {t.get('model') or t.get('source')}")
        obs = (interp.get("observation") or "").strip()
        if obs:
            print(f"      observation    : {obs[:220]}")
        interp_text = (interp.get("interpretation") or "").strip()
        if interp_text:
            print(f"      interpretation : {interp_text[:220]}")
        mechs = interp.get("plausible_mechanisms") or []
        if mechs:
            print(f"      mechanism      : {str(mechs[0])[:220]}")
        print(f"      status: {interp.get('conclusion_status')} · confidence: {interp.get('confidence')}")
        critic = t.get("critic", {})
        if critic.get("violations"):
            for v in critic["violations"][:2]:
                print(f"      CRITIC [{v['severity']}] {v['check']}: {v['message'][:150]}")

    print()
    print("=" * 78)
    print(f"  {export.summary()}")
    print("=" * 78)
    print(f"  Start here : {export.root / 'README.md'}")
    print(f"  Stage 1    : {export.root / '01_profiling' / 'EXACT_TEXT_SENT_TO_LLM.txt'}")
    print(f"  Combos     : {export.root / '03_combinations' / 'ALL_COMBINATIONS_TESTED.xlsx'}")
    print(f"  Graphs     : {export.root / '04_graphs'}")
    print(f"  Stage 2    : {export.root / '05_vlm_reasoning' / 'VLM_REASONING.md'}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--question", default=DEFAULT_QUESTION)
    p.add_argument("--vehicle", default="yellow", choices=[v.value for v in TLCVehicle])
    p.add_argument("--year", type=int, default=2026)
    p.add_argument("--month", type=int, default=1)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--max-visualizations", type=int, default=8)
    p.add_argument("--output", default="nyc_taxi_analysis")
    p.add_argument("--web-grounding", action="store_true")
    p.add_argument("--copy-data", action="store_true", default=True)
    p.add_argument("--no-copy-data", dest="copy_data", action="store_false")
    return asyncio.run(run(p.parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
