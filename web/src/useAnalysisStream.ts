import { useCallback, useEffect, useRef, useState } from "react";
import type { EngineEvent, RunStats } from "./types";
import { EMPTY_STATS } from "./types";

/** Cards kept per lane. Older ones are dropped; the run report has them all. */
const MAX_PER_LANE = 80;

export interface StreamState {
  finder: EngineEvent[];
  analyst: EngineEvent[];
  stage: string | null;
  stagesDone: Set<string>;
  stats: RunStats;
  answer: Extract<EngineEvent, { kind: "answer" }> | null;
  error: string | null;
  running: boolean;
  dataset: { rows: number; name: string } | null;
}

const INITIAL: StreamState = {
  finder: [], analyst: [], stage: null, stagesDone: new Set(),
  stats: EMPTY_STATS, answer: null, error: null, running: false, dataset: null,
};

export const STAGES = ["profiling", "view", "planning", "screening", "residual", "answer"] as const;

/** Map an event to the stage it implies, for the progress rail. */
function stageOf(e: EngineEvent): string | null {
  switch (e.kind) {
    case "profiled": return "profiling";
    case "view_built": return "view";
    case "planning": return "planning";
    case "screening": return "screening";
    case "residual_baseline": return "residual";
    case "answer": return "answer";
    case "stage": return e.name === "view_built" ? "view" : e.name;
    default: return null;
  }
}

function reduce(state: StreamState, e: EngineEvent): StreamState {
  const next: StreamState = {
    ...state,
    finder: state.finder,
    analyst: state.analyst,
    stagesDone: state.stagesDone,
  };

  const stage = stageOf(e);
  if (stage) {
    const idx = STAGES.indexOf(stage as (typeof STAGES)[number]);
    if (idx >= 0) {
      const done = new Set(state.stagesDone);
      STAGES.slice(0, idx).forEach((s) => done.add(s));
      next.stagesDone = done;
      next.stage = stage;
    }
  }

  switch (e.kind) {
    case "run_started":
      next.dataset = { rows: e.rows, name: e.dataset };
      next.running = true;
      break;
    case "result":
      next.stats = { ...next.stats, tests: next.stats.tests + 1 };
      break;
    case "handoff":
      next.stats = { ...next.stats, graphs: next.stats.graphs + 1 };
      break;
    case "planning":
      if (e.source === "llm") next.stats = { ...next.stats, llm: next.stats.llm + 1 };
      break;
    case "interpretation":
      next.stats = {
        ...next.stats,
        vlm: next.stats.vlm + 1,
        tokens: next.stats.tokens + (e.tokens || 0),
        evidence: next.stats.evidence + 1,
      };
      break;
    case "answer":
      next.answer = e;
      break;
    case "run_failed":
      next.error = e.error;
      next.running = false;
      break;
    case "run_finished": {
      const c = e.metrics?.counters;
      const d = e.metrics?.derived;
      if (c) {
        next.stats = {
          tests: c.statistical_tests ?? next.stats.tests,
          graphs: c.plots_rendered ?? next.stats.graphs,
          llm: c.llm_calls ?? next.stats.llm,
          vlm: c.vlm_calls ?? next.stats.vlm,
          tokens: d?.total_tokens ?? next.stats.tokens,
          evidence: c.evidence_created ?? next.stats.evidence,
        };
      }
      next.running = false;
      next.stagesDone = new Set(STAGES);
      next.stage = null;
      break;
    }
  }

  // Newest first: the lane reads top-down as most-recent-first, which is what
  // the age animation depends on.
  if (e.lane === "finder") {
    next.finder = [e, ...state.finder].slice(0, MAX_PER_LANE);
  } else if (e.lane === "analyst") {
    next.analyst = [e, ...state.analyst].slice(0, MAX_PER_LANE);
  } else if (e.kind === "profiled" || e.kind === "view_built") {
    // System events that describe the data belong in the finder's narrative.
    next.finder = [e, ...state.finder].slice(0, MAX_PER_LANE);
  }

  return next;
}

export function useAnalysisStream() {
  const [state, setState] = useState<StreamState>(INITIAL);
  const source = useRef<EventSource | null>(null);

  const stop = useCallback(() => {
    source.current?.close();
    source.current = null;
  }, []);

  const connect = useCallback(
    (analysisId: string) => {
      stop();
      setState({ ...INITIAL, running: true, stagesDone: new Set() });

      // cursor=0 replays the run from the beginning, so a reconnect or a late
      // tab lands in exactly the same state rather than missing the start.
      const es = new EventSource(`/analyses/${analysisId}/stream?cursor=0`);
      source.current = es;

      es.onmessage = (msg) => {
        let event: EngineEvent;
        try {
          event = JSON.parse(msg.data) as EngineEvent;
        } catch {
          return;
        }
        if (!event || typeof event.kind !== "string") return;
        setState((prev) => reduce(prev, event));
      };

      es.addEventListener("end", () => stop());
      es.onerror = () => {
        // EventSource reconnects on its own and the stream replays, so a
        // dropped connection is self-healing; only a closed one is terminal.
        if (es.readyState === EventSource.CLOSED) {
          setState((prev) => ({ ...prev, running: false }));
        }
      };
    },
    [stop],
  );

  useEffect(() => stop, [stop]);

  return { state, connect, stop, setState };
}
