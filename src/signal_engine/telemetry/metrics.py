"""Telemetry.

Every number in the demo metrics panel comes from here, and every one of them
is MEASURED.  Nothing in this module invents a figure: counters increment where
the work happens, and a metric that was never recorded reads as 0 or null
rather than as an estimate.

The two figures that need care, because they are the ones most easily
overstated:

  * `candidates_pruned_by_units` counts candidates the generator actually
    enumerated and rejected — not a theoretical space size.  The theoretical
    figure is reported separately and labelled as such.

  * `llm_tokens_avoided` counts tokens on requests that hit the cache, where
    the token count came from a real prior response.  Estimated counts are
    tracked separately so the panel can show what is measured vs estimated.
"""

from __future__ import annotations

import threading
import time
from collections import Counter, defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Timer:
    """Accumulating stopwatch for a named stage."""

    name: str
    total_seconds: float = 0.0
    calls: int = 0

    @contextmanager
    def __call__(self) -> Iterator[None]:
        started = time.time()
        try:
            yield
        finally:
            self.total_seconds += time.time() - started
            self.calls += 1

    def to_dict(self) -> dict:
        return {
            "seconds": round(self.total_seconds, 4),
            "calls": self.calls,
            "mean_ms": round(1000 * self.total_seconds / self.calls, 2) if self.calls else 0.0,
        }


@dataclass
class AnalysisMetrics:
    """All telemetry for one analysis run."""

    analysis_id: str = ""
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None

    counters: Counter = field(default_factory=Counter)
    timers: dict[str, Timer] = field(default_factory=dict)
    events: list[dict] = field(default_factory=list)
    gauges: dict[str, Any] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        # Seed the counters every consumer reads, so the reported shape is the
        # same whether or not a stage ran.  A Counter only grows keys it has
        # been touched with, which would otherwise make "no LLM configured"
        # look like "llm_calls field missing" to an API client.
        for name in (
            "statistical_tests",
            "mutual_information_tests",
            "stability_tests",
            "group_comparisons",
            "plots_rendered",
            "llm_calls",
            "vlm_calls",
            "llm_cache_hits",
            "vlm_cache_hits",
            "brave_calls",
            "evidence_created",
            "evidence_retrievals",
            "evidence_resolved",
            "joint_reinterpretations",
            "critic_errors",
            "critic_warnings",
            "hypotheses_generated",
            "rejected_transformations",
            "candidates_considered",
            "candidates_emitted",
            "candidates_pruned_by_units",
            "columns_profiled",
        ):
            self.counters.setdefault(name, 0)

    # ---- primitives -----------------------------------------------------
    def incr(self, name: str, amount: int = 1) -> None:
        with self._lock:
            self.counters[name] += amount

    def gauge(self, name: str, value: Any) -> None:
        with self._lock:
            self.gauges[name] = value

    def timer(self, name: str) -> Timer:
        with self._lock:
            if name not in self.timers:
                self.timers[name] = Timer(name)
            return self.timers[name]

    @contextmanager
    def time(self, name: str) -> Iterator[None]:
        with self.timer(name)():
            yield

    def event(self, kind: str, **fields: Any) -> None:
        """Structured event log (bounded, so a long run cannot exhaust memory)."""
        with self._lock:
            if len(self.events) < 5000:
                self.events.append(
                    {"t": round(time.time() - self.started_at, 3), "kind": kind, **fields}
                )

    def finish(self) -> None:
        self.finished_at = time.time()

    @property
    def elapsed_seconds(self) -> float:
        return (self.finished_at or time.time()) - self.started_at

    # ---- domain helpers -------------------------------------------------
    def record_llm_call(
        self,
        *,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency: float,
        cache_hit: bool,
        estimated: bool,
        kind: str = "llm",
    ) -> None:
        prefix = "vlm" if kind == "vlm" else "llm"
        if cache_hit:
            self.incr(f"{prefix}_cache_hits")
            self.incr(f"{prefix}_tokens_avoided", prompt_tokens + completion_tokens)
        else:
            self.incr(f"{prefix}_calls")
            self.incr(f"{prefix}_prompt_tokens", prompt_tokens)
            self.incr(f"{prefix}_completion_tokens", completion_tokens)
            if estimated:
                self.incr(f"{prefix}_estimated_token_calls")
            self.timer(f"{prefix}_latency").total_seconds += latency
            self.timer(f"{prefix}_latency").calls += 1
        self.event(f"{prefix}_call", model=model, cache_hit=cache_hit, tokens=prompt_tokens)

    def record_prune(self, stats) -> None:
        self.incr("candidates_considered", stats.considered)
        self.incr("candidates_emitted", stats.emitted)
        self.incr("candidates_pruned_by_units", stats.pruned_by_units)
        self.incr("candidates_pruned_by_dedup", stats.pruned_by_dedup)
        self.incr("candidates_pruned_by_policy", stats.pruned_by_policy)
        self.incr("candidates_pruned_by_budget", stats.pruned_by_budget)

    # ---- reporting ------------------------------------------------------
    def to_dict(self) -> dict:
        with self._lock:
            counters = dict(self.counters)
            timers = {k: v.to_dict() for k, v in self.timers.items()}
            gauges = dict(self.gauges)

        total_prompt = counters.get("llm_prompt_tokens", 0) + counters.get("vlm_prompt_tokens", 0)
        total_completion = counters.get("llm_completion_tokens", 0) + counters.get(
            "vlm_completion_tokens", 0
        )
        avoided = counters.get("llm_tokens_avoided", 0) + counters.get("vlm_tokens_avoided", 0)
        calls = counters.get("llm_calls", 0) + counters.get("vlm_calls", 0)
        hits = counters.get("llm_cache_hits", 0) + counters.get("vlm_cache_hits", 0)

        considered = counters.get("candidates_considered", 0)
        emitted = counters.get("candidates_emitted", 0)

        return {
            "analysis_id": self.analysis_id,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "counters": counters,
            "timers": timers,
            "gauges": gauges,
            "derived": {
                "total_prompt_tokens": total_prompt,
                "total_completion_tokens": total_completion,
                "total_tokens": total_prompt + total_completion,
                "tokens_avoided_by_cache": avoided,
                "model_calls": calls,
                "model_cache_hits": hits,
                "model_cache_hit_rate": round(hits / (hits + calls), 4) if (hits + calls) else 0.0,
                "candidate_prune_rate": (
                    round((considered - emitted) / considered, 4) if considered else 0.0
                ),
            },
            "event_count": len(self.events),
        }

    def recent_events(self, limit: int = 50) -> list[dict]:
        with self._lock:
            return self.events[-limit:]


class MetricsRegistry:
    """Process-wide registry of analysis metrics (bounded)."""

    def __init__(self, max_entries: int = 50) -> None:
        self._metrics: dict[str, AnalysisMetrics] = {}
        self._order: list[str] = []
        self._max = max_entries
        self._lock = threading.Lock()

    def create(self, analysis_id: str) -> AnalysisMetrics:
        metrics = AnalysisMetrics(analysis_id=analysis_id)
        with self._lock:
            self._metrics[analysis_id] = metrics
            self._order.append(analysis_id)
            while len(self._order) > self._max:
                self._metrics.pop(self._order.pop(0), None)
        return metrics

    def get(self, analysis_id: str) -> AnalysisMetrics | None:
        return self._metrics.get(analysis_id)

    def all_ids(self) -> list[str]:
        return list(self._order)

    def aggregate(self) -> dict:
        """Cross-run totals for the /metrics endpoint."""
        totals: dict[str, int] = defaultdict(int)
        for m in self._metrics.values():
            for k, v in m.counters.items():
                totals[k] += v
        return {
            "analyses": len(self._metrics),
            "totals": dict(totals),
            "analysis_ids": self.all_ids(),
        }


_REGISTRY = MetricsRegistry()


def get_registry() -> MetricsRegistry:
    return _REGISTRY
