import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { getHealth, listDatasets, startAnalysis, uploadDataset } from "./api";
import { assetUrl } from "./config";
import type { DatasetRow, EngineEvent } from "./types";
import { STAGES, useAnalysisStream } from "./useAnalysisStream";
import type { ChartRow } from "./useAnalysisStream";
import { EventCard } from "./components/EventCard";
import { Measurements } from "./components/Measurements";
import { Gallery } from "./components/Gallery";

const STAGE_LABEL: Record<string, string> = {
  profiling: "Profile",
  view: "Filter",
  planning: "Propose",
  screening: "Measure",
  residual: "Residualise",
  answer: "Conclude",
};

const fmt = (v: number) => v.toLocaleString();
const bytes = (b: number) =>
  b > 1e9 ? `${(b / 1e9).toFixed(1)} GB` : `${Math.max(1, Math.round(b / 1e6))} MB`;

/** How quickly a card falls out of focus as newer ones push it down. */
const AGE_DEPTH = 7;

type Tab = "live" | "table" | "charts" | "answer";

function Column({
  index, title, subtitle, events, emptyText, onZoom,
}: {
  index: string;
  title: string;
  subtitle: string;
  events: EngineEvent[];
  emptyText: string;
  onZoom: (src: string) => void;
}) {
  const feed = useRef<HTMLDivElement>(null);

  // Snap to the top when something new arrives; the newest card is the subject.
  useEffect(() => {
    if (feed.current) feed.current.scrollTop = 0;
  }, [events.length]);

  return (
    <section className="col">
      <header className="col-hd">
        <span className="ix">{index}</span>
        <h2>{title}</h2>
        <p>{subtitle}</p>
      </header>
      <div className="feed" ref={feed}>
        {events.length === 0 ? (
          <div className="blank"><p>{emptyText}</p></div>
        ) : (
          events.map((e, i) => (
            <EventCard
              key={e.seq}
              event={e}
              age={Math.min(i / AGE_DEPTH, 1)}
              live={i === 0}
              onZoom={onZoom}
            />
          ))
        )}
      </div>
    </section>
  );
}

export default function App() {
  const { state, connect } = useAnalysisStream();
  const [datasets, setDatasets] = useState<DatasetRow[]>([]);
  const [selected, setSelected] = useState("");
  const [question, setQuestion] = useState(
    "What is associated with the amount passengers pay for a New York yellow-taxi trip?",
  );
  const [rounds, setRounds] = useState(3);
  const [busy, setBusy] = useState(false);
  const [uploadPct, setUploadPct] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [zoom, setZoom] = useState<ChartRow | { url: string } | null>(null);
  const [model, setModel] = useState<string>("");
  const [tab, setTab] = useState<Tab>("live");
  const [setupOpen, setSetupOpen] = useState(true);
  const [elapsed, setElapsed] = useState(0);

  const refreshDatasets = useCallback(async (prefer?: string) => {
    try {
      const rows = await listDatasets();
      setDatasets(rows);
      setSelected((cur) => prefer ?? (cur || rows[0]?.path || ""));
    } catch {
      setError("Cannot reach the API. Is the server running?");
    }
  }, []);

  useEffect(() => {
    void refreshDatasets();
    void getHealth()
      .then((h) => setModel(h.config.llm.configured ? h.config.llm.model : "no model configured"))
      .catch(() => undefined);
  }, [refreshDatasets]);

  // A visible clock beats a spinner: a run takes minutes and the number is the
  // difference between "thinking" and "hung".
  useEffect(() => {
    if (!state.running) return;
    const started = Date.now();
    const id = setInterval(() => setElapsed((Date.now() - started) / 1000), 200);
    return () => clearInterval(id);
  }, [state.running]);

  useEffect(() => {
    if (!state.running) setBusy(false);
    else setSetupOpen(false);
  }, [state.running]);

  const onUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setError(null);
    setUploadPct(0);
    try {
      const row = await uploadDataset(file, (f) => setUploadPct(f));
      await refreshDatasets(row.path);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setUploadPct(null);
      e.target.value = "";
    }
  };

  const run = async () => {
    setError(null);
    setBusy(true);
    setTab("live");
    try {
      const id = await startAnalysis({
        question, path: selected || null, max_rounds: rounds, max_visualizations: 8,
      });
      connect(id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not start the run");
      setBusy(false);
    }
  };

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") setZoom(null); };
    addEventListener("keydown", onKey);
    return () => removeEventListener("keydown", onKey);
  }, []);

  const active = datasets.find((d) => d.path === selected);
  const rowCount = state.dataset?.rows ?? active?.rows ?? null;

  const tabs: { id: Tab; label: string; count?: number; dot?: boolean }[] = useMemo(
    () => [
      { id: "live", label: "Live" },
      { id: "table", label: "Measurements", count: state.measurements.length },
      { id: "charts", label: "Charts", count: state.charts.length },
      { id: "answer", label: "Conclusion", dot: !!state.answer },
    ],
    [state.measurements.length, state.charts.length, state.answer],
  );

  return (
    <div className="app">
      <header className="top">
        <div className="brand">
          <span className={`dot${state.running ? " on" : ""}`} />
          <b>Signal</b>
          <span className="rule" />
          <span className="what">finds structure in a table that a correlation matrix misses</span>
        </div>
        <div className="top-r">
          {state.running && <span className="clock">{elapsed.toFixed(1)}s</span>}
          {model && <span className="chiplet">{model}</span>}
          {rowCount != null && <span className="chiplet">{fmt(rowCount)} rows</span>}
        </div>
      </header>

      {setupOpen ? (
        <div className="setup">
          <label className="field">
            <span>Dataset</span>
            <select value={selected} onChange={(e) => setSelected(e.target.value)}>
              {datasets.length === 0 && <option value="">No datasets found</option>}
              {datasets.map((d) => (
                <option key={d.path} value={d.path}>
                  {d.name}{d.rows ? ` — ${fmt(d.rows)} rows` : ""} · {bytes(d.bytes)}
                </option>
              ))}
            </select>
          </label>

          <label className="field file">
            <span>Or upload</span>
            <span className="fake-btn">
              {uploadPct !== null ? `Uploading ${(uploadPct * 100).toFixed(0)}%` : "Choose .parquet or .csv"}
              <input type="file" accept=".parquet,.csv,.tsv" onChange={onUpload} />
            </span>
          </label>

          <label className="field grow">
            <span>Question</span>
            <input
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter" && !busy) void run(); }}
            />
          </label>

          <label className="field narrow">
            <span>Rounds</span>
            <input
              type="number" min={1} max={6} value={rounds}
              onChange={(e) => setRounds(Number(e.target.value))}
            />
          </label>

          <button className="go" onClick={() => void run()} disabled={busy || !selected}>
            {busy ? "Starting\u2026" : "Analyse"}
          </button>
          {error && <span className="err">{error}</span>}
        </div>
      ) : (
        <div className="setup collapsed">
          <span className="sum">
            <b>{active?.name ?? state.dataset?.name ?? "dataset"}</b>
            <span className="q">{question}</span>
          </span>
          <button className="ghost" onClick={() => setSetupOpen(true)} disabled={state.running}>
            {state.running ? "Running\u2026" : "Change"}
          </button>
          {!state.running && (
            <button className="go" onClick={() => void run()} disabled={busy}>Run again</button>
          )}
          {error && <span className="err">{error}</span>}
        </div>
      )}

      <nav className="rail" aria-label="progress">
        {STAGES.map((s) => (
          <span
            key={s}
            className={`step${state.stagesDone.has(s) ? " done" : ""}${state.stage === s ? " active" : ""}`}
          >
            <i />{STAGE_LABEL[s]}
          </span>
        ))}
      </nav>

      <div className="tabs" role="tablist">
        {tabs.map((t) => (
          <button
            key={t.id}
            role="tab"
            aria-selected={tab === t.id}
            className={`tab${tab === t.id ? " on" : ""}`}
            onClick={() => setTab(t.id)}
          >
            {t.label}
            {t.count != null && t.count > 0 && <b>{t.count}</b>}
            {t.dot && tab !== t.id && <i className="new" />}
          </button>
        ))}
      </div>

      <main className="body">
        {tab === "live" && (
          <div className="cols">
            <Column
              index="01"
              title="Measurement"
              subtitle="proposes pairs, prunes on units, measures what survives"
              events={state.left}
              emptyText="Candidate relationships appear here as they are proposed, pruned on dimensional grounds, and measured against the data."
              onZoom={(url) => setZoom({ url })}
            />
            <Column
              index="02"
              title="Interpretation"
              subtitle="reads each chart against its own statistics"
              events={state.right}
              emptyText="Once a relationship survives screening it is plotted, read back against its exact statistics, and checked for claims the numbers do not support."
              onZoom={(url) => setZoom({ url })}
            />
          </div>
        )}

        {tab === "table" && (
          <Measurements rows={state.measurements} charts={state.charts} onOpen={setZoom} />
        )}

        {tab === "charts" && <Gallery charts={state.charts} onOpen={setZoom} />}

        {tab === "answer" && (
          <div className="answer">
            {state.answer ? (
              <>
                <h3>Conclusion</h3>
                <p className="lede">{state.answer.text}</p>
                {state.answer.findings.length > 0 && (
                  <>
                    <h4>What held up</h4>
                    <ol>{state.answer.findings.map((f) => <li key={f}>{f}</li>)}</ol>
                  </>
                )}
                {state.answer.caveats.length > 0 && (
                  <>
                    <h4>What to be careful about</h4>
                    <ul>{state.answer.caveats.map((c) => <li key={c}>{c}</li>)}</ul>
                  </>
                )}
              </>
            ) : (
              <div className="blank">
                <p>The conclusion is written once every surviving relationship has been
                measured, plotted and checked. It lands here.</p>
              </div>
            )}
          </div>
        )}
      </main>

      {state.error && <div className="banner">{state.error}</div>}

      {zoom && (
        <div className="light" onClick={() => setZoom(null)}>
          <div className="light-in" onClick={(e) => e.stopPropagation()}>
            <img src={assetUrl(zoom.url)} alt="chart" />
            {"stats" in zoom && (
              <div className="light-meta">
                <div className="cap-hd">
                  <span className="expr">{zoom.expression}</span>
                  <span className="cap-st">{zoom.stats}</span>
                </div>
                {zoom.observation && <p>{zoom.observation}</p>}
                {zoom.interpretation && <p className="strong">{zoom.interpretation}</p>}
              </div>
            )}
            <button className="ghost x" onClick={() => setZoom(null)}>Close</button>
          </div>
        </div>
      )}
    </div>
  );
}
