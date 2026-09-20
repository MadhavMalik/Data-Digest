"""Two-level cache for materialized feature vectors.

Layer 1: bounded in-memory LRU, sized in BYTES rather than entries.  A cap on
entry count is the wrong control here — one 3.7M-row float64 column is 30 MB,
so "100 entries" could mean 3 GB.

Layer 2: `.npy` files on disk, keyed by the expression's canonical hash.  These
survive process restarts, which matters because a hackathon demo gets run
dozens of times over the same dataset.

Cache keys combine the dataset fingerprint, the analysis-view filter hash, and
the canonical expression hash, so a cached vector can never be served for a
different dataset version or a different row filter.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DEFAULT_MEMORY_BUDGET = 512 * 1024 * 1024  # 512 MiB


@dataclass
class CacheStats:
    memory_hits: int = 0
    disk_hits: int = 0
    misses: int = 0
    stores: int = 0
    evictions: int = 0
    bytes_in_memory: int = 0
    bytes_written: int = 0

    @property
    def hits(self) -> int:
        return self.memory_hits + self.disk_hits

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def to_dict(self) -> dict:
        return {
            "memory_hits": self.memory_hits,
            "disk_hits": self.disk_hits,
            "hits": self.hits,
            "misses": self.misses,
            "stores": self.stores,
            "evictions": self.evictions,
            "hit_rate": round(self.hit_rate, 4),
            "memory_mib": round(self.bytes_in_memory / (1 << 20), 2),
            "written_mib": round(self.bytes_written / (1 << 20), 2),
        }


@dataclass
class ExpressionCache:
    """LRU memory cache in front of an optional on-disk `.npy` store."""

    directory: Path | None = None
    memory_budget_bytes: int = DEFAULT_MEMORY_BUDGET
    namespace: str = "default"
    stats: CacheStats = field(default_factory=CacheStats)
    _memory: OrderedDict[str, np.ndarray] = field(default_factory=OrderedDict, repr=False)

    def __post_init__(self) -> None:
        if self.directory is not None:
            self.directory = Path(self.directory)
            self.directory.mkdir(parents=True, exist_ok=True)

    # ---- keys -----------------------------------------------------------
    def key(self, expr_hash: str, *, dataset_fingerprint: str, view_hash: str = "") -> str:
        return f"{dataset_fingerprint[:12]}_{view_hash[:8]}_{expr_hash}"

    def _path(self, key: str) -> Path | None:
        if self.directory is None:
            return None
        return self.directory / f"{self.namespace}_{key}.npy"

    # ---- access ---------------------------------------------------------
    def get(self, key: str) -> np.ndarray | None:
        if key in self._memory:
            self._memory.move_to_end(key)
            self.stats.memory_hits += 1
            return self._memory[key]

        path = self._path(key)
        if path is not None and path.exists():
            try:
                arr = np.load(path, allow_pickle=False)
            except Exception:  # noqa: BLE001 - a corrupt entry is just a miss
                path.unlink(missing_ok=True)
                self.stats.misses += 1
                return None
            self.stats.disk_hits += 1
            self._admit(key, arr)
            return arr

        self.stats.misses += 1
        return None

    def put(self, key: str, array: np.ndarray, *, persist: bool = True) -> None:
        array = np.ascontiguousarray(array)
        self.stats.stores += 1
        self._admit(key, array)

        path = self._path(key)
        if persist and path is not None and not path.exists():
            tmp = path.with_suffix(".npy.part")
            try:
                # Write through an open handle: np.save(path_like) silently
                # appends `.npy` to any name that lacks it, which would make
                # the temp file `<name>.npy.part.npy` and break the rename.
                with tmp.open("wb") as fh:
                    np.save(fh, array, allow_pickle=False)
                tmp.replace(path)
                self.stats.bytes_written += array.nbytes
            except Exception:  # noqa: BLE001 - disk cache is best-effort
                tmp.unlink(missing_ok=True)

    def __contains__(self, key: str) -> bool:
        if key in self._memory:
            return True
        path = self._path(key)
        return path is not None and path.exists()

    # ---- eviction -------------------------------------------------------
    def _admit(self, key: str, array: np.ndarray) -> None:
        nbytes = array.nbytes
        if nbytes > self.memory_budget_bytes:
            # Too big to ever hold in memory; disk still serves it.
            return
        if key in self._memory:
            self.stats.bytes_in_memory -= self._memory[key].nbytes
        self._memory[key] = array
        self._memory.move_to_end(key)
        self.stats.bytes_in_memory += nbytes
        self._evict_to_budget()

    def _evict_to_budget(self) -> None:
        while self.stats.bytes_in_memory > self.memory_budget_bytes and self._memory:
            _, evicted = self._memory.popitem(last=False)
            self.stats.bytes_in_memory -= evicted.nbytes
            self.stats.evictions += 1

    def clear_memory(self) -> None:
        self._memory.clear()
        self.stats.bytes_in_memory = 0
