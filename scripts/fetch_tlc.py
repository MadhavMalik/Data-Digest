#!/usr/bin/env python3
"""Download an official NYC TLC monthly trip-record Parquet file.

    python scripts/fetch_tlc.py --vehicle yellow --year 2026 --month 1 --output data/raw

Behaviour:
  * resolves the official TLC monthly Parquet URL (no hardcoded single file)
  * streams to disk (never loads the file into memory)
  * writes atomically (`.part` then rename) so a killed run leaves no half file
  * retries transient network errors with exponential backoff
  * validates the Parquet footer and prints row count / schema / size
  * skips the download when a valid identical file already exists (`--force` overrides)
  * also fetches the taxi-zone lookup and writes the official data dictionary
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from signal_engine.ingestion.base import human_bytes, validate_parquet  # noqa: E402
from signal_engine.ingestion.tlc import (  # noqa: E402
    TLCSource,
    TLCVehicle,
    fetch_zone_lookup,
    write_dictionary_metadata,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vehicle", default="yellow", choices=[v.value for v in TLCVehicle])
    p.add_argument("--year", type=int, default=2026)
    p.add_argument("--month", type=int, default=1)
    p.add_argument("--output", default="data/raw", help="directory for the raw Parquet file")
    p.add_argument("--metadata-dir", default="data/metadata")
    p.add_argument("--force", action="store_true", help="re-download even if the file exists")
    p.add_argument("--no-zones", action="store_true", help="skip the taxi-zone lookup download")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--json", action="store_true", help="emit the dataset handle as JSON")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    out_dir = (REPO_ROOT / args.output) if not Path(args.output).is_absolute() else Path(args.output)
    meta_dir = (
        (REPO_ROOT / args.metadata_dir)
        if not Path(args.metadata_dir).is_absolute()
        else Path(args.metadata_dir)
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    source = TLCSource(vehicle=TLCVehicle(args.vehicle), year=args.year, month=args.month)

    if not args.quiet:
        print(f"Source : {source.resolve_url()}")
        print(f"Target : {out_dir / source.local_filename()}")

    handle, result = source.fetch(out_dir, force=args.force, show_progress=not args.quiet)

    if not args.quiet:
        verb = "already present (skipped)" if result.skipped else "downloaded"
        print(f"\n{verb}: {human_bytes(result.bytes_written)} in {result.elapsed_seconds:.1f}s")

        info = validate_parquet(handle.path)
        print(f"Rows       : {info['row_count']:,}")
        print(f"Row groups : {info['row_groups']}")
        print(f"Size       : {human_bytes(info['byte_size'])}")
        print(f"Fingerprint: {handle.fingerprint}")
        print(f"Columns    : {len(info['columns'])}")
        for col in info["columns"]:
            print(f"  - {col['name']:<24} {col['arrow_type']}")

    if not args.no_zones:
        try:
            zones = fetch_zone_lookup(meta_dir, force=args.force)
            if not args.quiet:
                print(f"\nZone lookup: {zones}")
        except Exception as exc:  # noqa: BLE001 - zones are enrichment, not required
            print(f"\nWARNING: taxi-zone lookup failed ({exc}); continuing without it", file=sys.stderr)

    dict_path = write_dictionary_metadata(meta_dir)
    if not args.quiet:
        print(f"Dictionary : {dict_path}")

    # Persist the handle so the profiler/demo can pick the dataset up by id.
    registry = meta_dir / "datasets.json"
    existing = json.loads(registry.read_text()) if registry.exists() else {}
    existing[handle.dataset_id] = handle.to_dict()
    registry.write_text(json.dumps(existing, indent=2))

    if args.json:
        print(json.dumps(handle.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
