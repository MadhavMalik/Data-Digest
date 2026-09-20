"""Live event stream for the two-agent UI.

The engine runs its CPU-heavy stages in worker threads (see the orchestrator's
concurrency note), so events are emitted from several threads while the SSE
endpoint consumes them from the event loop.

Rather than an asyncio.Queue plus `call_soon_threadsafe` plumbing, this uses an
append-only list behind a plain lock, and readers track their own cursor. That
makes the stream:

  * thread-safe with one primitive
  * replayable — a browser that connects late, or reconnects, gets the whole
    history and lands in the same state
  * impossible to deadlock, because emit never blocks on a consumer

The cost is polling, which at a 120 ms tick is invisible next to the seconds
each analysis stage takes.

Events are routed to one of two agent lanes:

    FINDER   proposes hypotheses, tests pairs, reports statistics
    ANALYST  reads the rendered graph and interprets it
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Literal

Lane = Literal["finder", "analyst", "system"]

MAX_EVENTS = 4000


@dataclass
class EventStream:
    """Append-only, thread-safe, replayable event log for one analysis."""

    analysis_id: str = ""
    started_at: float = field(default_factory=time.time)
    _events: list[dict] = field(default_factory=list, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _seq: int = 0
    finished: bool = False

    # ---- writing --------------------------------------------------------
    def emit(self, _kind: str, _lane: Lane = "system", **fields: Any) -> dict:
        """Record an event.  Never raises, never blocks on a reader.

        The parameters are underscore-prefixed so that an event's own DATA
        fields can never shadow them: `residual_driver` legitimately carries a
        field called `kind` (categorical vs numeric), and with a plain `kind`
        parameter that raised "got multiple values for argument 'kind'",
        silently killing the emitting stage.
        """
        with self._lock:
            self._seq += 1
            # Data fields are spread FIRST so the envelope keys always win. A
            # payload field named `kind` would otherwise overwrite the event
            # type and the consumer would route the event as whatever the data
            # happened to say.
            event = {**fields}
            event.update({
                "seq": self._seq,
                "t": round(time.time() - self.started_at, 3),
                "kind": _kind,
                "lane": _lane,
            })
            self._events.append(event)
            # Bound the log so a long run cannot exhaust memory. The head is
            # dropped rather than the tail: a late viewer cares about what is
            # happening now, and the run report holds the full record anyway.
            if len(self._events) > MAX_EVENTS:
                del self._events[: len(self._events) - MAX_EVENTS]
            return event

    def finish(self) -> None:
        self.emit("run_finished", "system")
        self.finished = True

    # ---- reading --------------------------------------------------------
    def since(self, cursor: int) -> tuple[list[dict], int]:
        """Events after `cursor`, plus the new cursor."""
        with self._lock:
            if not self._events:
                return [], cursor
            fresh = [e for e in self._events if e["seq"] > cursor]
            return fresh, self._events[-1]["seq"]

    def all(self) -> list[dict]:
        with self._lock:
            return list(self._events)

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._events)


# ---------------------------------------------------------------------------
# Convenience emitters
# ---------------------------------------------------------------------------
#
# Typed helpers rather than raw `emit` calls at the orchestrator's call sites:
# the UI's contract lives here, in one place, instead of being spread across a
# dozen string literals in the search loop.


class Emitter:
    """Thin façade so orchestrator code reads as intent, not as dict-building."""

    def __init__(self, stream: EventStream | None) -> None:
        self.stream = stream

    def _emit(self, _kind: str, _lane: Lane, **fields: Any) -> None:
        """Underscore-prefixed params so an event FIELD can never shadow them.

        `residual_driver` carries a data field called `kind` (categorical vs
        numeric), which collided with a positional named `kind` and raised
        "got multiple values for argument 'kind'" -- silently killing the whole
        residual stage at the emit call.
        """
        if self.stream is not None:
            self.stream.emit(_kind, _lane, **fields)

    # ---- system ---------------------------------------------------------
    def run_started(self, *, question: str, dataset: str, rows: int, columns: int, bytes_: int):
        self._emit("run_started", "system", question=question, dataset=dataset,
                   rows=rows, columns=columns, bytes=bytes_)

    def stage(self, name: str, detail: str = "", **extra):
        self._emit("stage", "system", name=name, detail=detail, **extra)

    def profiled(self, *, rows: int, columns: int, seconds: float,
                 prompt_chars: int, cache_hit: bool):
        self._emit("profiled", "system", rows=rows, columns=columns,
                   seconds=seconds, prompt_chars=prompt_chars, cache_hit=cache_hit)

    def view_built(self, *, rows_before: int, rows_after: int, excluded: int, filters: list):
        self._emit("view_built", "system", rows_before=rows_before, rows_after=rows_after,
                   excluded=excluded, filters=filters)

    def pruned(self, *, considered: int, emitted: int, by_units: int, rate: float, reasons: dict):
        self._emit("pruned", "finder", considered=considered, emitted=emitted,
                   by_units=by_units, rate=rate, reasons=reasons)

    # ---- finder ---------------------------------------------------------
    def planning(self, *, round_index: int, source: str):
        self._emit("planning", "finder", round=round_index, source=source)

    def hypothesis(self, *, round_index: int, source: str, base: list, target: list,
                   rationale: str, priority: str, transforms: list):
        self._emit("hypothesis", "finder", round=round_index, source=source,
                   base=base, target=target, rationale=rationale,
                   priority=priority, transforms=transforms)

    def screening(self, *, candidates: int, target: str):
        self._emit("screening", "finder", candidates=candidates, target=target)

    def result(self, *, x: str, y: str, n: int, r: float | None, rho: float | None,
               eta: float | None, mi: float | None, effect: float, direction: str,
               strength: str, stability: float | None, mechanical: bool, shape: str = ""):
        self._emit("result", "finder", x=x, y=y, n=n, r=r, rho=rho, eta=eta, mi=mi,
                   effect=effect, direction=direction, strength=strength,
                   stability=stability, mechanical=mechanical, shape=shape)

    def suppressed(self, *, count: int, reason: str):
        self._emit("suppressed", "finder", count=count, reason=reason)

    def residual_baseline(self, *, predictors: list, target: str, r2: float,
                          residual_std: float, target_std: float, n: int):
        self._emit("residual_baseline", "finder", predictors=predictors, target=target,
                   r2=r2, residual_std=residual_std, target_std=target_std, n=n)

    def residual_driver(self, *, name: str, kind: str, effect: float,
                        spread: float | None, levels: list):
        # Sent as `driver_kind`: `kind` is reserved for the event type.
        self._emit("residual_driver", "finder", name=name, driver_kind=kind,
                   effect=effect, spread=spread, levels=levels)

    # ---- handoff --------------------------------------------------------
    def handoff(self, *, x: str, y: str, expression: str, plot_type: str,
                plot_url: str, stats: str, stage: str = "marginal"):
        self._emit("handoff", "analyst", x=x, y=y, expression=expression,
                   plot_type=plot_type, plot_url=plot_url, stats=stats, stage=stage)

    # ---- analyst --------------------------------------------------------
    def interpreting(self, *, x: str, y: str, model: str):
        self._emit("interpreting", "analyst", x=x, y=y, model=model)

    def interpretation(self, *, x: str, y: str, expression: str, plot_url: str,
                       observation: str, interpretation: str, mechanisms: list,
                       confounders: list, status: str, confidence: float,
                       model: str, tokens: int, seconds: float, stage: str = "marginal"):
        self._emit("interpretation", "analyst", x=x, y=y, expression=expression,
                   plot_url=plot_url, observation=observation,
                   interpretation=interpretation, mechanisms=mechanisms,
                   confounders=confounders, status=status, confidence=confidence,
                   model=model, tokens=tokens, seconds=seconds, stage=stage)

    def critic(self, *, x: str, y: str, passed: bool, violations: list):
        self._emit("critic", "analyst", x=x, y=y, passed=passed, violations=violations)

    def memory(self, *, retrieved: int, resolved: int, store: str):
        self._emit("memory", "analyst", retrieved=retrieved, resolved=resolved, store=store)

    # ---- end ------------------------------------------------------------
    def answer(self, *, text: str, findings: list, caveats: list):
        self._emit("answer", "system", text=text, findings=findings, caveats=caveats)

    def finished(self, *, metrics: dict, stop_reason: str):
        self._emit("run_finished", "system", metrics=metrics, stop_reason=stop_reason)
