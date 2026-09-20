import { assetUrl } from "../config";
import type { EngineEvent, ResidualLevel, Violation } from "../types";

const fmt = (v: number, d = 0) =>
  v.toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d });
const sig = (v: number | null | undefined) =>
  v == null ? "—" : `${v >= 0 ? "+" : ""}${v.toFixed(3)}`;

function Head({ kind, t }: { kind: string; t: number }) {
  return (
    <div className="c-hd">
      <span className="c-kind">{kind}</span>
      <span className="c-t">{t.toFixed(1)}s</span>
    </div>
  );
}

function Dots() {
  return (
    <span className="dots" aria-label="working">
      <i /><i /><i />
    </span>
  );
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
                  background: positive ? "var(--bad)" : "var(--finder)",
                }}
              />
            </span>
            <span className="v">
              {l.value >= 0 ? "+" : ""}
              {l.value.toFixed(2)}
            </span>
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
      return wrap(
        "profile",
        <>
          <div className="c-title">
            {fmt(e.columns)} columns profiled in {e.seconds.toFixed(2)}s
          </div>
          <div className="c-body">
            The model will see <b>{fmt(e.prompt_chars)} characters</b> of column cards
            standing in for <b>{fmt(e.rows)} rows</b>. It never sees a row.
          </div>
          {e.cache_hit && (
            <div className="nums">
              <span className="num">cache <b>hit</b></span>
            </div>
          )}
        </>,
      );

    case "view_built":
      return wrap(
        "validity filter",
        <>
          <div className="c-title">
            {fmt(e.rows_after)} of {fmt(e.rows_before)} rows retained
          </div>
          <div className="c-body">
            {fmt(e.excluded)} excluded by {e.filters.length} documented filters. Nothing is
            silently dropped.
          </div>
        </>,
      );

    case "planning":
      return wrap(
        `round ${e.round + 1} · ${e.source}`,
        <div className="c-title">
          Planning the next hypotheses
          <Dots />
        </div>,
      );

    case "hypothesis":
      return wrap(
        "hypothesis",
        <>
          <div className="c-title">
            {e.base.join(", ")} → {e.target.join(", ")}{" "}
            <span className={`pill ${e.priority === "high" ? "hi" : "mech"}`}>{e.priority}</span>
          </div>
          {e.rationale && <div className="c-body">{e.rationale}</div>}
          {e.transforms.length > 0 && (
            <div className="nums">
              {e.transforms.map((t) => (
                <span className="num" key={t}>{t}</span>
              ))}
            </div>
          )}
        </>,
      );

    case "pruned":
      return wrap(
        "dimensional pruning",
        <>
          <div className="c-title">
            {fmt(e.by_units)} of {fmt(e.considered)} candidates rejected on units
          </div>
          <div className="c-body">
            Before reading a single row. <b>{(e.rate * 100).toFixed(0)}%</b> pruned.
          </div>
          <div className="levels">
            {Object.entries(e.reasons).slice(0, 4).map(([reason, count]) => (
              <div className="lvl" key={reason}>
                <span className="name">{reason}</span>
                <span className="v">{fmt(count)}</span>
              </div>
            ))}
          </div>
        </>,
      );

    case "screening":
      return wrap(
        "measuring",
        <div className="c-title">
          Screening {fmt(e.candidates)} candidates against {e.target}
          <Dots />
        </div>,
      );

    case "result":
      return wrap(
        "result",
        <>
          <div className="c-title">
            {e.x} <span style={{ color: "var(--ink-3)" }}>~</span> {e.y}{" "}
            {e.mechanical && <span className="pill mech">definitional</span>}{" "}
            <span className={`pill ${e.effect >= 0.5 ? "ok" : e.effect >= 0.25 ? "warn" : "mech"}`}>
              {e.strength}
            </span>
          </div>
          <div className="nums">
            <span className="num">n <b>{fmt(e.n)}</b></span>
            {e.r != null && (
              <span className={`num ${e.r >= 0 ? "pos" : "neg"}`}>r <b>{sig(e.r)}</b></span>
            )}
            {e.rho != null && (
              <span className={`num ${e.rho >= 0 ? "pos" : "neg"}`}>ρ <b>{sig(e.rho)}</b></span>
            )}
            {e.eta != null && <span className="num">η <b>{e.eta.toFixed(3)}</b></span>}
            {e.mi != null && <span className="num">MI <b>{e.mi.toFixed(3)}</b></span>}
            {e.stability != null && (
              <span className="num">stab <b>{e.stability.toFixed(2)}</b></span>
            )}
          </div>
        </>,
      );

    case "suppressed":
      return wrap(
        "suppressed",
        <>
          <div className="c-title">{fmt(e.count)} finding(s) held back from interpretation</div>
          <div className="c-body">{e.reason}</div>
        </>,
      );

    case "residual_baseline":
      return wrap(
        "residual baseline",
        <>
          <div className="c-title">
            {e.predictors.join(" + ")} → {e.target}
          </div>
          <div className="c-body">
            Explains <b>{(e.r2 * 100).toFixed(1)}%</b>. Now searching what is left: spread falls{" "}
            {e.target_std.toFixed(2)} → <b>{e.residual_std.toFixed(2)}</b>.
          </div>
        </>,
      );

    case "residual_driver":
      return wrap(
        "residual driver",
        <>
          <div className="c-title">
            {e.name} explains the remainder{" "}
            <span className={`pill ${e.effect >= 0.4 ? "ok" : "warn"}`}>
              η {e.effect.toFixed(3)}
            </span>
          </div>
          {e.spread != null && (
            <div className="c-body">
              Spread across levels: <b>{e.spread.toFixed(2)}</b> in the target's own units.
            </div>
          )}
          {e.levels.length > 0 && <Levels levels={e.levels} />}
        </>,
      );

    case "handoff":
      return wrap(
        `handoff · ${e.stage}`,
        <>
          <div className="c-title">
            {e.expression} <span style={{ color: "var(--ink-3)" }}>vs</span> {e.y}
          </div>
          <div className="nums">
            <span className="num">{e.plot_type}</span>
            <span className="num">{e.stats}</span>
          </div>
          {e.plot_url && (
            <div className="shot">
              <img
                src={assetUrl(e.plot_url)}
                loading="lazy"
                alt={`${e.plot_type} of ${e.x} against ${e.y}`}
                onClick={() => onZoom(assetUrl(e.plot_url))}
              />
            </div>
          )}
        </>,
      );

    case "interpreting":
      return wrap(
        "reading",
        <div className="c-title">
          Interpreting {e.x} ~ {e.y}
          <Dots />
        </div>,
      );

    case "interpretation":
      return wrap(
        `interpretation · ${e.model}`,
        <>
          <div className="c-title">
            {e.expression}{" "}
            <span
              className={`pill ${
                e.status === "explained" ? "ok" : e.status === "mechanical" ? "mech" : "warn"
              }`}
            >
              {e.status}
            </span>
          </div>
          <div className="c-body">{e.observation}</div>
          {e.interpretation && (
            <div className="c-body" style={{ color: "var(--ink)" }}>{e.interpretation}</div>
          )}
          {e.mechanisms.length > 0 && (
            <div className="c-body">
              <b>Mechanisms</b>
              <ul>
                {e.mechanisms.slice(0, 2).map((m) => (
                  <li key={m}>{m}</li>
                ))}
              </ul>
            </div>
          )}
          <div className="nums">
            <span className="num">conf <b>{(e.confidence * 100).toFixed(0)}%</b></span>
            <span className="num">{fmt(e.tokens)} tok</span>
            <span className="num">{e.seconds.toFixed(2)}s</span>
          </div>
        </>,
      );

    case "critic":
      if (e.violations.length === 0) return null;
      return wrap(
        "critic",
        <>
          <div className="c-title">
            {e.passed ? "Claim consistent with the numbers" : "Claim contradicts the evidence"}
          </div>
          <Violations items={e.violations} />
        </>,
      );

    case "memory":
      return wrap(
        "evidence memory",
        <>
          <div className="c-title">{fmt(e.retrieved)} prior finding(s) retrieved</div>
          <div className="c-body">
            From <b>{e.store}</b>, fed into this interpretation.
          </div>
        </>,
      );

    case "run_failed":
      return wrap(
        "failed",
        <div className="c-title" style={{ color: "var(--bad)" }}>{e.error}</div>,
      );

    default:
      return null;
  }
}
