"""Dataset handles, fingerprints, and the download primitive.

Two rules drive this module:

1. Raw files are immutable.  Nothing here ever rewrites `data/raw/`.
2. A download is atomic: stream to `<name>.part`, verify, then rename.  A
   killed process can therefore never leave a truncated file that later looks
   like a valid cached dataset.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import requests

_CHUNK = 1 << 20  # 1 MiB
_USER_AGENT = "signal-engine/0.1 (HackMIT 2026 research prototype)"


@dataclass(frozen=True)
class DatasetHandle:
    """Everything downstream needs to identify a dataset version.

    `fingerprint` is what the profile cache keys on, so it must change whenever
    the bytes change.
    """

    dataset_id: str
    path: Path
    fingerprint: str
    row_count: int | None = None
    byte_size: int | None = None
    source_url: str | None = None
    metadata_path: Path | None = None
    description: str = ""
    extras: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "dataset_id": self.dataset_id,
            "path": str(self.path),
            "fingerprint": self.fingerprint,
            "row_count": self.row_count,
            "byte_size": self.byte_size,
            "source_url": self.source_url,
            "description": self.description,
            "extras": self.extras,
        }


@dataclass
class DownloadResult:
    path: Path
    bytes_written: int
    elapsed_seconds: float
    skipped: bool
    url: str

    @property
    def mib(self) -> float:
        return self.bytes_written / (1 << 20)


class DatasetSource(Protocol):
    """A resolvable source of a dataset file."""

    def resolve_url(self) -> str: ...
    def local_filename(self) -> str: ...
    def dataset_id(self) -> str: ...


# ---------------------------------------------------------------------------
# Fingerprinting
# ---------------------------------------------------------------------------


def fingerprint_file(path: Path, *, sample_bytes: int = 4 << 20) -> str:
    """Stable content fingerprint for a (possibly very large) file.

    Full SHA-256 of a multi-GB Parquet file costs minutes and buys nothing here:
    we combine size, mtime-independent head/tail samples, and a mid-file sample.
    Any realistic content change moves at least one of those.  The size is
    included verbatim so truncation is always caught.
    """
    size = path.stat().st_size
    h = hashlib.sha256()
    h.update(str(size).encode())
    h.update(path.name.encode())

    with path.open("rb") as fh:
        chunk = min(sample_bytes, size)
        h.update(fh.read(chunk))
        if size > chunk * 2:
            fh.seek(size // 2)
            h.update(fh.read(chunk))
            fh.seek(max(0, size - chunk))
            h.update(fh.read(chunk))
    return h.hexdigest()[:32]


def stable_hash(payload: object) -> str:
    """Deterministic hash of any JSON-serializable object (cache keys)."""
    blob = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:32]


# ---------------------------------------------------------------------------
# Streaming download
# ---------------------------------------------------------------------------


def stream_download(
    url: str,
    dest: Path,
    *,
    force: bool = False,
    max_retries: int = 4,
    backoff_base: float = 1.5,
    progress: Callable[[int, int | None], None] | None = None,
    timeout: int = 60,
) -> DownloadResult:
    """Download `url` to `dest` atomically, with retry + resume-free backoff.

    Skips the download when `dest` already exists and its size matches the
    server's Content-Length (cheap validity check; the caller additionally
    validates the Parquet footer).
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()

    remote_size = _head_content_length(url, timeout=timeout)
    if dest.exists() and not force:
        local_size = dest.stat().st_size
        if remote_size is None or local_size == remote_size:
            return DownloadResult(dest, local_size, time.time() - started, True, url)

    tmp = dest.with_suffix(dest.suffix + ".part")
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            written = 0
            with requests.get(
                url,
                stream=True,
                timeout=timeout,
                headers={"User-Agent": _USER_AGENT, "Accept-Encoding": "identity"},
            ) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get("Content-Length") or 0) or remote_size
                with tmp.open("wb") as fh:
                    for chunk in resp.iter_content(chunk_size=_CHUNK):
                        if not chunk:
                            continue
                        fh.write(chunk)
                        written += len(chunk)
                        if progress:
                            progress(written, total)
                    fh.flush()
                    os.fsync(fh.fileno())

            if total and written != total:
                raise OSError(f"incomplete download: got {written} of {total} bytes")

            tmp.replace(dest)  # atomic on POSIX
            return DownloadResult(dest, written, time.time() - started, False, url)

        except Exception as exc:  # noqa: BLE001 - retried below, re-raised at the end
            last_error = exc
            tmp.unlink(missing_ok=True)
            if attempt < max_retries:
                time.sleep(backoff_base**attempt)

    tmp.unlink(missing_ok=True)
    raise RuntimeError(f"failed to download {url} after {max_retries} attempts: {last_error}")


def _head_content_length(url: str, *, timeout: int = 30) -> int | None:
    try:
        resp = requests.head(
            url, timeout=timeout, allow_redirects=True, headers={"User-Agent": _USER_AGENT}
        )
        if resp.status_code >= 400:
            return None
        length = resp.headers.get("Content-Length")
        return int(length) if length else None
    except Exception:  # noqa: BLE001 - HEAD is an optimization, never fatal
        return None


# ---------------------------------------------------------------------------
# Parquet validation & local registration
# ---------------------------------------------------------------------------


def validate_parquet(path: Path) -> dict:
    """Confirm the file is readable Parquet and return schema facts.

    Reads only the footer metadata — no row groups — so this stays O(1) in the
    file size.
    """
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    schema = pf.schema_arrow
    return {
        "row_count": pf.metadata.num_rows,
        "row_groups": pf.metadata.num_row_groups,
        "columns": [
            {"name": name, "arrow_type": str(schema.field(name).type)} for name in schema.names
        ],
        "byte_size": path.stat().st_size,
    }


def register_local_dataset(
    path: Path,
    *,
    dataset_id: str | None = None,
    description: str = "",
    source_url: str | None = None,
    metadata_path: Path | None = None,
    extras: dict | None = None,
) -> DatasetHandle:
    """Wrap an existing local file in a DatasetHandle (fingerprint + row count)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    row_count: int | None = None
    if path.suffix == ".parquet":
        try:
            row_count = validate_parquet(path)["row_count"]
        except Exception:  # noqa: BLE001 - a non-Parquet file simply has no cheap row count
            row_count = None

    return DatasetHandle(
        dataset_id=dataset_id or path.stem,
        path=path,
        fingerprint=fingerprint_file(path),
        row_count=row_count,
        byte_size=path.stat().st_size,
        source_url=source_url,
        metadata_path=metadata_path,
        description=description,
        extras=extras or {},
    )


def human_bytes(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024 or unit == "TiB":
            return f"{n:,.1f} {unit}" if unit != "B" else f"{int(n):,} B"
        n /= 1024
    return f"{n:,.1f} TiB"


def iter_progress_bar(label: str) -> Callable[[int, int | None], None]:
    """Minimal stderr progress callback (no tqdm dependency at import time)."""
    state = {"last": 0.0}

    def _cb(written: int, total: int | None) -> None:
        now = time.time()
        if now - state["last"] < 0.4 and (not total or written < total):
            return
        state["last"] = now
        if total:
            pct = 100.0 * written / total
            print(
                f"\r  {label}: {human_bytes(written)} / {human_bytes(total)} ({pct:5.1f}%)",
                end="",
                flush=True,
            )
        else:
            print(f"\r  {label}: {human_bytes(written)}", end="", flush=True)

    return _cb


def flatten(iterable: Iterable[Iterable]) -> list:
    return [x for sub in iterable for x in sub]
