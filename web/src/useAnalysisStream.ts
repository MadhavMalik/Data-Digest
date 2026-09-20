import { useCallback, useEffect, useRef, useState } from "react";
import { apiUrl } from "./config";
import type { EngineEvent, ResultEvent, RunStats } from "./types";
import { EMPTY_STATS } from "./types";

/** Cards kept in a scrolling lane. The table and the gallery keep everything. */
const MAX_PER_LANE = 80;

/** A rendered chart plus, once it arrives, the reading of it. */
export interface ChartRow {
  seq: number;
  x: string;
  y: string;
  expression: string;
  plotType: string;
  url: string;
  stats: string;
  stage: "marginal" | "residual";
  status?: string;
  confidence?: number;
  observation?: string;
  interpretation?: string;
}

export interface StreamState {
  left: EngineEvent[];
  right: EngineEvent[];
  /** Every relationship measured, in the order it was measured. */
  measurements: ResultEvent[];
  /** Every chart rendered, newest first. */
  charts: ChartRow[];
  stage: string | null;
  stagesDone: Set<string>;
  stats: RunStats;
  answer: Extract<EngineEvent, { kind: "answer" }> | null;
  error: string | null;
  running: boolean;
  dataset: { rows: number; name: string } | null;
}

const INITIAL: StreamState = {
  left: [], right: [], measurements: [], charts: [],
  stage: null, stagesDone: new Set(),
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
  const next: StreamState = { ...state };

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
      // Kept in full: this is the table, and a table you can sort is worth more
      // than a card that scrolled away.
      next.measurements = [...state.measurements, e];
      next.stats = { ...next.stats, tests: next.stats.tests + 1 };
      break;

    case "handoff":
      if (e.plot_url) {
        next.charts = [
          {
            seq: e.seq, x: e.x, y: e.y, expression: e.expression,
            plotType: e.plot_type, url: e.plot_url, stats: e.stats, stage: e.stage,
          },
          ...state.charts,
        ];
      }
      next.stats = { ...next.stats, graphs: next.stats.graphs + 1 };
      break;

    case "planning":
      if (e.source === "llm") next.stats = { ...next.stats, llm: next.stats.llm + 1 };
      break;

    case "interpretation":
      // Fold the reading back onto the chart it read, so the gallery shows a
      // conclusion under each image rather than a bare thumbnail.
      next.charts = state.charts.map((c) =>
        c.url === e.plot_url && c.status === undefined
          ? {
              ...c,
              status: e.status,
              confidence: e.confidence,
              observation: e.observation,
              interpretation: e.interpretation,
            }
          : c,
      );
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

  // Newest first: the column reads top-down as most-recent-first, which is what
  // the depth-of-field animation depends on.
  if (e.lane === "finder") {
    next.left = [e, ...state.left].slice(0, MAX_PER_LANE);
  } else if (e.lane === "analyst") {
    next.right = [e, ...state.right].slice(0, MAX_PER_LANE);
  } else if (e.kind === "profiled" || e.kind === "view_built") {
    next.left = [e, ...state.left].slice(0, MAX_PER_LANE);
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
      const es = new EventSource(apiUrl(`/analyses/${analysisId}/stream?cursor=0`));
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
