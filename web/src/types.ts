/**
 * The event contract, mirroring src/signal_engine/events.py.
 *
 * Every event carries the same envelope (seq, t, kind, lane) plus its own
 * fields. A discriminated union on `kind` means the compiler catches a handler
 * reading a field the event does not have -- which is how the earlier JS
 * version silently dropped `residual_driver` payloads.
 */

export type Lane = "finder" | "analyst" | "system";

interface Envelope {
  seq: number;
  /** Seconds since the run started. */
  t: number;
  lane: Lane;
}

export interface RunStarted extends Envelope {
  kind: "run_started";
  question: string;
  dataset: string;
  rows: number;
  columns: number;
  bytes: number;
}

export interface Stage extends Envelope {
  kind: "stage";
  name: string;
  detail: string;
}

export interface Profiled extends Envelope {
  kind: "profiled";
  rows: number;
  columns: number;
  seconds: number;
  prompt_chars: number;
  cache_hit: boolean;
}

export interface ViewBuilt extends Envelope {
  kind: "view_built";
  rows_before: number;
  rows_after: number;
  excluded: number;
  filters: string[];
}

export interface Planning extends Envelope {
  kind: "planning";
  round: number;
  source: "llm" | "deterministic";
}

export interface HypothesisEvent extends Envelope {
  kind: "hypothesis";
  round: number;
  source: string;
  base: string[];
  target: string[];
  rationale: string;
  priority: "high" | "medium" | "low";
  transforms: string[];
}

export interface Pruned extends Envelope {
  kind: "pruned";
  considered: number;
  emitted: number;
  by_units: number;
  rate: number;
  reasons: Record<string, number>;
}

export interface Screening extends Envelope {
  kind: "screening";
  candidates: number;
  target: string;
}

export interface ResultEvent extends Envelope {
  kind: "result";
  x: string;
  y: string;
  n: number;
  r: number | null;
  rho: number | null;
  eta: number | null;
  mi: number | null;
  effect: number;
  direction: string;
  strength: string;
  stability: number | null;
  mechanical: boolean;
  shape: string;
}

export interface Suppressed extends Envelope {
  kind: "suppressed";
  count: number;
  reason: string;
}

export interface ResidualBaseline extends Envelope {
  kind: "residual_baseline";
  predictors: string[];
  target: string;
  r2: number;
  residual_std: number;
  target_std: number;
  n: number;
}

export interface ResidualLevel {
  label: string;
  value: number;
  n: number;
}

export interface ResidualDriver extends Envelope {
  kind: "residual_driver";
  name: string;
  /** Named `driver_kind` on the wire: `kind` is the event type. */
  driver_kind: "categorical" | "numeric";
  effect: number;
  spread: number | null;
  levels: ResidualLevel[];
}

export interface Handoff extends Envelope {
  kind: "handoff";
  x: string;
  y: string;
  expression: string;
  plot_type: string;
  plot_url: string;
  stats: string;
  stage: "marginal" | "residual";
}

export interface Interpreting extends Envelope {
  kind: "interpreting";
  x: string;
  y: string;
  model: string;
}

export interface Interpretation extends Envelope {
  kind: "interpretation";
  x: string;
  y: string;
  expression: string;
  plot_url: string;
  observation: string;
  interpretation: string;
  mechanisms: string[];
  confounders: string[];
  status: string;
  confidence: number;
  model: string;
  tokens: number;
  seconds: number;
  stage: "marginal" | "residual";
}

export interface Violation {
  severity: "error" | "warning" | "note";
  check: string;
  message: string;
}

export interface CriticEvent extends Envelope {
  kind: "critic";
  x: string;
  y: string;
  passed: boolean;
  violations: Violation[];
}

export interface MemoryEvent extends Envelope {
  kind: "memory";
  retrieved: number;
  resolved: number;
  store: string;
}

export interface AnswerEvent extends Envelope {
  kind: "answer";
  text: string;
  findings: string[];
  caveats: string[];
}

export interface RunFinished extends Envelope {
  kind: "run_finished";
  metrics?: { counters?: Record<string, number>; derived?: Record<string, number> };
  stop_reason?: string;
}

export interface RunFailed extends Envelope {
  kind: "run_failed";
  error: string;
}

export type EngineEvent =
  | RunStarted | Stage | Profiled | ViewBuilt
  | Planning | HypothesisEvent | Pruned | Screening | ResultEvent | Suppressed
  | ResidualBaseline | ResidualDriver
  | Handoff | Interpreting | Interpretation | CriticEvent | MemoryEvent
  | AnswerEvent | RunFinished | RunFailed;

export interface DatasetRow {
  path: string;
  name: string;
  rows: number | null;
  bytes: number;
}

export interface RunStats {
  tests: number;
  graphs: number;
  llm: number;
  vlm: number;
  tokens: number;
  evidence: number;
}

export const EMPTY_STATS: RunStats = {
  tests: 0, graphs: 0, llm: 0, vlm: 0, tokens: 0, evidence: 0,
};
