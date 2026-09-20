import { assetUrl } from "../config";
import type { EngineEvent, ResidualLevel, Violation } from "../types";

const fmt = (v: number, d = 0) =>
  v.toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d });
const sig = (v: number | null | undefined) =>
  v == null ? "\u2014" : `${v >= 0 ? "" : "\u2212"}${Math.abs(v).toFixed(3)}`;

const STAGE_COPY: Record<string, string> = {
  marginal: "raw relationship",
  residual: "after removing the obvious",
};

function Head({ kind, t }: { kind: string; t: number }) {
  return (
    <div className="c-hd">
      <span className="c-kind">{kind}</span>
      <span className="c-t">{t.toFixed(1)}s</span>
    </div>
  );
}

function Working() {
  return <span className="dots" aria-label="working"><i /><i /><i /></span>;
}

function Levels({ levels }: { levels: ResidualLevel[] }) {
  const max = Math.max(...levels.map((l) => Math.abs(l.value)), 1);
  return (
    <div className="levels">
      {levels.map((l) => {
        const width = (Math.abs(l.value) / max) * 50;
        const positive = l.value >= 0;
        return (
          <div className="lvl" key={l.label}>
            <span className="name">{l.label}</span>
            <span className="bar">
              <i
                style={{
                  [positive ? "left" : "right"]: "50%",
                  width: `${width}%`,
                  background: positive ? "var(--pos)" : "var(--neg)",
                }}
              />
            </span>
            <span className="v">{l.value >= 0 ? "+" : "\u2212"}{Math.abs(l.value).toFixed(2)}</span>
          </div>
        );
      })}
    </div>
  );
}

function Violations({ items }: { items: Violation[] }) {
  return (
    <>
      {items.map((v, i) => (
        <div className="viol" key={`${v.check}-${i}`}>
          <span className={`pill ${v.severity === "error" ? "bad" : v.severity === "warning" ? "warn" : "mech"}`}>
            {v.severity}
          </span>
          <span>{v.message}</span>
        </div>
      ))}
    </>
  );
}

interface Props {
  event: EngineEvent;
  age: number;
  live: boolean;
  onZoom: (src: string) => void;
}

export function EventCard({ event: e, age, live, onZoom }: Props) {
  const style = { "--age": age.toFixed(3) } as React.CSSProperties;
  const cls = `card${live ? " live" : ""}`;

  const wrap = (kind: string, body: React.ReactNode) => (
    <div className={cls} style={style}>
      <Head kind={kind} t={e.t} />
      {body}
    </div>
  );

  switch (e.kind) {
    case "profiled":
      return wrap("Profile", (
        <>
          <div className="c-title">{fmt(e.columns)} columns profiled in {e.seconds.toFixed(2)}s</div>
          <div className="c-body">
            The model reads <b>{fmt(e.prompt_chars)} characters</b> of column summaries standing
            in for <b>{fmt(e.rows)} rows</b>. It never sees a row of your data.
          </div>
          {e.cache_hit && <div className="nums"><span className="num">served from cache</span></div>}
        </>
      ));

    case "view_built":
      return wrap("Validity filter", (
        <>
          <div className="c-title">{fmt(e.rows_after)} of {fmt(e.rows_before)} rows kept</div>
          <div className="c-body">
            {fmt(e.excluded)} excluded by {e.filters.length} recorded rules. Nothing is dropped silently.
          </div>
        </>
      ));

    case "planning":
      return wrap(`Round ${e.round + 1}`, (
        <div className="c-title">
          {e.source === "llm" ? "Proposing what to test next" : "Falling back to the deterministic plan"}
          <Working />
        </div>
      ));

    case "hypothesis":
      return wrap("Candidate", (
        <>
          <div className="c-title">
            {e.base.join(", ")} <span className="arrow">&rarr;</span> {e.target.join(", ")}{" "}
            <span className={`pill ${e.priority === "high" ? "hi" : "mech"}`}>{e.priority}</span>
          </div>
          {e.rationale && <div className="c-body">{e.rationale}</div>}
          {e.transforms.length > 0 && (
            <div className="nums">{e.transforms.map((t) => <span className="num" key={t}>{t}</span>)}</div>
          )}
        </>
      ));

    case "pruned":
      return wrap("Dimensional pruning", (
        <>
          <div className="c-title">
            {fmt(e.by_units)} of {fmt(e.considered)} candidates rejected on units alone
          </div>
          <div className="c-body">
            Before reading a single row &mdash; <b>{(e.rate * 100).toFixed(0)}%</b> of the search space,
            removed by arithmetic.
          </div>
          <div className="levels">
            {Object.entries(e.reasons).slice(0, 4).map(([reason, count]) => (
              <div className="lvl" key={reason}>
                <span className="name">{reason}</span>
                <span className="v">{fmt(count)}</span>
              </div>
            ))}
          </div>
        </>
      ));

    case "screening":
      return wrap("Measuring", (
        <div className="c-title">
          {fmt(e.candidates)} candidates against {e.target}
          <Working />
        </div>
      ));

    case "result":
      return wrap("Measured", (
        <>
          <div className="c-title">
            {e.x} <span className="vs">vs</span> {e.y}{" "}
            {e.mechanical && <span className="pill mech">definitional</span>}{" "}
            <span className={`pill ${e.effect >= 0.5 ? "ok" : e.effect >= 0.25 ? "warn" : "mech"}`}>
              {e.strength}
            </span>
          </div>
          <div className="nums">
            <span className="num">n <b>{fmt(e.n)}</b></span>
            {e.r != null && <span className={`num ${e.r >= 0 ? "pos" : "neg"}`}>r <b>{sig(e.r)}</b></span>}
            {e.rho != null && <span className={`num ${e.rho >= 0 ? "pos" : "neg"}`}>&rho; <b>{sig(e.rho)}</b></span>}
            {e.eta != null && <span className="num">&eta; <b>{e.eta.toFixed(3)}</b></span>}
            {e.mi != null && <span className="num">MI <b>{e.mi.toFixed(3)}</b></span>}
            {e.stability != null && <span className="num">stability <b>{e.stability.toFixed(2)}</b></span>}
          </div>
        </>
      ));

    case "suppressed":
      return wrap("Held back", (
        <>
          <div className="c-title">{fmt(e.count)} result(s) kept out of the write-up</div>
          <div className="c-body">{e.reason}</div>
        </>
      ));

    case "residual_baseline":
      return wrap("Baseline removed", (
        <>
          <div className="c-title">{e.predictors.join(" + ")} <span className="arrow">&rarr;</span> {e.target}</div>
          <div className="c-body">
            The obvious part accounts for <b>{(e.r2 * 100).toFixed(1)}%</b>. Spread falls from{" "}
            {e.target_std.toFixed(2)} to <b>{e.residual_std.toFixed(2)}</b> &mdash; what is left is
            where the interesting structure lives.
          </div>
        </>
      ));

    case "residual_driver":
      return wrap("Drives the remainder", (
        <>
          <div className="c-title">
            {e.name}{" "}
            <span className={`pill ${e.effect >= 0.4 ? "ok" : "warn"}`}>&eta; {e.effect.toFixed(3)}</span>
          </div>
          {e.spread != null && (
            <div className="c-body">
              Spread across levels: <b>{e.spread.toFixed(2)}</b>, in the target&rsquo;s own units.
            </div>
          )}
          {e.levels.length > 0 && <Levels levels={e.levels} />}
        </>
      ));

    case "handoff":
      return wrap(`Chart \u00b7 ${STAGE_COPY[e.stage] ?? e.stage}`, (
        <>
          <div className="c-title">{e.expression}</div>
          <div className="nums">
            <span className="num">{e.plot_type}</span>
            <span className="num">{e.stats}</span>
          </div>
          {e.plot_url && (
            <button
              type="button"
              className="shot"
              onClick={() => onZoom(e.plot_url)}
              title="Enlarge this chart"
            >
              <img src={assetUrl(e.plot_url)} loading="lazy" alt={`${e.plot_type} of ${e.expression}`} />
            </button>
          )}
        </>
      ));

    case "interpreting":
      return wrap("Reading", (
        <div className="c-title">{e.x} vs {e.y}<Working /></div>
      ));

    case "interpretation":
      return wrap("Reading", (
        <>
          <div className="c-title">
            {e.expression}{" "}
            <span className={`pill ${e.status === "explained" ? "ok" : e.status === "mechanical" ? "mech" : "warn"}`}>
              {e.status}
            </span>
          </div>
          <div className="c-body">{e.observation}</div>
          {e.interpretation && <div className="c-body strong">{e.interpretation}</div>}
          {e.mechanisms.length > 0 && (
            <div className="c-body">
              <b>Possible mechanisms</b>
              <ul>{e.mechanisms.slice(0, 2).map((m) => <li key={m}>{m}</li>)}</ul>
            </div>
          )}
          <div className="nums">
            <span className="num">confidence <b>{(e.confidence * 100).toFixed(0)}%</b></span>
            <span className="num">{fmt(e.tokens)} tokens</span>
            <span className="num">{e.seconds.toFixed(2)}s</span>
          </div>
        </>
      ));

    case "critic":
      if (e.violations.length === 0) return null;
      return wrap("Cross-check", (
        <>
          <div className="c-title">
            {e.passed ? "Reading is consistent with the numbers" : "Reading contradicts the numbers"}
          </div>
          <Violations items={e.violations} />
        </>
      ));

    case "memory":
      return wrap("Prior evidence", (
        <>
          <div className="c-title">{fmt(e.retrieved)} earlier finding(s) recalled</div>
          <div className="c-body">From <b>{e.store}</b>, and fed into this reading.</div>
        </>
      ));

    case "run_failed":
      return wrap("Failed", <div className="c-title err">{e.error}</div>);

    default:
      return null;
  }
}
