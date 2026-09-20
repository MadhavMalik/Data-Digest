"""The expression DAG: materialize derived features once, reuse everywhere.

Two things make this a DAG rather than a list of formulas:

  1. Shared subexpressions.  `fare_amount / trip_distance` may appear inside
     several deeper candidates.  Canonical hashing means they collide on one
     node and the vector is computed once.

  2. Batched evaluation.  Requesting 200 features one at a time means 200
     passes over the Parquet file.  `materialize_many` compiles them into ONE
     Polars `select` so the file is read once and Polars parallelises the
     columns internally.

Column projection is the other half of the win: only the base columns actually
referenced by the requested expressions are read off disk.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from signal_engine.features.cache import ExpressionCache
from signal_engine.features.canonicalize import canonicalize, expression_hash, suggest_name
from signal_engine.features.expressions import Col, Expr, is_safe_expression


@dataclass
class DAGNode:
    expr: Expr
    expr_hash: str
    name: str
    is_base: bool

    @property
    def columns(self) -> set[str]:
        return self.expr.columns()


@dataclass
class DAGStats:
    materializations: int = 0
    cache_hits: int = 0
    batches: int = 0
    rows_scanned: int = 0
    columns_projected: int = 0
    seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "materializations": self.materializations,
            "cache_hits": self.cache_hits,
            "batches": self.batches,
            "rows_scanned": self.rows_scanned,
            "columns_projected": self.columns_projected,
            "seconds": round(self.seconds, 3),
        }


@dataclass
class ExpressionDAG:
    """Evaluates expressions against a lazy frame, with caching and batching."""

    frame: pl.LazyFrame
    dataset_fingerprint: str
    allowed_columns: set[str]
    cache: ExpressionCache = field(default_factory=ExpressionCache)
    view_hash: str = ""
    stats: DAGStats = field(default_factory=DAGStats)
    nodes: dict[str, DAGNode] = field(default_factory=dict)

    # ---- construction ---------------------------------------------------
    @classmethod
    def from_path(
        cls,
        path: Path,
        *,
        dataset_fingerprint: str,
        cache_dir: Path | None = None,
        view_hash: str = "",
        frame: pl.LazyFrame | None = None,
    ) -> ExpressionDAG:
        lf = frame if frame is not None else pl.scan_parquet(path)
        columns = set(lf.collect_schema().names())
        cache = ExpressionCache(
            directory=(Path(cache_dir) / "features") if cache_dir else None,
            namespace=dataset_fingerprint[:8],
        )
        return cls(
            frame=lf,
            dataset_fingerprint=dataset_fingerprint,
            allowed_columns=columns,
            cache=cache,
            view_hash=view_hash,
        )

    # ---- registration ---------------------------------------------------
    def register(self, expr: Expr, name: str | None = None) -> DAGNode:
        canonical = canonicalize(expr)
        ok, reason = is_safe_expression(canonical, self.allowed_columns)
        if not ok:
            raise ValueError(f"unsafe expression {canonical.display()!r}: {reason}")

        h = expression_hash(canonical)
        if h in self.nodes:
            return self.nodes[h]

        node = DAGNode(
            expr=canonical,
            expr_hash=h,
            name=name or suggest_name(canonical),
            is_base=isinstance(canonical, Col),
        )
        self.nodes[h] = node
        return node

    def _cache_key(self, expr_hash: str) -> str:
        return self.cache.key(
            expr_hash, dataset_fingerprint=self.dataset_fingerprint, view_hash=self.view_hash
        )

    # ---- evaluation -----------------------------------------------------
    def materialize(self, expr: Expr) -> np.ndarray:
        return self.materialize_many([expr])[0]

    def materialize_many(self, exprs: list[Expr], *, names: list[str] | None = None) -> list[np.ndarray]:
        """Materialize several expressions in ONE pass over the data.

        Returns float64 arrays with nulls as NaN, aligned row-for-row across all
        requested expressions (so pairwise statistics can mask jointly).
        """
        started = time.time()
        nodes = [
            self.register(e, names[i] if names and i < len(names) else None)
            for i, e in enumerate(exprs)
        ]

        results: dict[str, np.ndarray] = {}
        pending: list[DAGNode] = []
        for node in nodes:
            cached = self.cache.get(self._cache_key(node.expr_hash))
            if cached is not None:
                results[node.expr_hash] = cached
                self.stats.cache_hits += 1
            else:
                pending.append(node)

        if pending:
            # Deduplicate by hash: the same expression requested twice in one
            # batch is computed once.
            unique: dict[str, DAGNode] = {n.expr_hash: n for n in pending}
            needed_columns = sorted({c for n in unique.values() for c in n.columns})

            select_exprs = [
                n.expr.to_polars().cast(pl.Float64).alias(h) for h, n in unique.items()
            ]
            frame = self.frame.select(needed_columns) if needed_columns else self.frame
            df = frame.select(select_exprs).collect()

            self.stats.batches += 1
            self.stats.rows_scanned += df.height
            self.stats.columns_projected += len(needed_columns)

            for h in unique:
                arr = df[h].to_numpy(allow_copy=True).astype(np.float64, copy=False)
                # Polars nulls arrive as NaN once cast to Float64 in numpy;
                # make that explicit so downstream masking is uniform.
                results[h] = arr
                self.cache.put(self._cache_key(h), arr)
                self.stats.materializations += 1

        self.stats.seconds += time.time() - started
        return [results[n.expr_hash] for n in nodes]

    # ---- convenience ----------------------------------------------------
    def materialize_columns(self, names: list[str]) -> dict[str, np.ndarray]:
        arrays = self.materialize_many([Col(n) for n in names])
        return dict(zip(names, arrays))

    def height(self) -> int:
        return int(self.frame.select(pl.len()).collect().item())

    def stats_dict(self) -> dict:
        return {**self.stats.to_dict(), "cache": self.cache.stats.to_dict(), "nodes": len(self.nodes)}
