import { useCallback, useEffect, useRef, useState } from "react";
import { getHealth, listDatasets, startAnalysis, uploadDataset } from "./api";
import type { DatasetRow, EngineEvent } from "./types";
import { STAGES, useAnalysisStream } from "./useAnalysisStream";
import { EventCard } from "./components/EventCard";

const STAGE_LABEL: Record<string, string> = {
  profiling: "profile",
  view: "filter",
  planning: "hypothesize",
  screening: "measure",
  residual: "residualise",
  answer: "synthesise",
};

const fmt = (v: number) => v.toLocaleString();

/** How quickly a card fades as newer ones push it down. */
const AGE_DEPTH = 7;

function Lane({
  tag,
  title,
  meta,
  events,
  emptyKey,
  emptyText,
  onZoom,
  className,
}: {
  tag: string;
  title: string;
  meta: string;
  events: EngineEvent[];
  emptyKey: string;
  emptyText: string;
  onZoom: (src: string) => void;
  className: string;
}) {
  const feed = useRef<HTMLDivElement>(null);

  // Snap to the top when a new event arrives; the newest card is the subject.
  useEffect(() => {
    if (feed.current) feed.current.scrollTop = 0;
  }, [events.length]);

  return (
    <section className={`lane ${className}`}>
      <div className="lane-hd">
        <span className="tag">{tag}</span>
        <h2>{title}</h2>
        <p>{meta}</p>
      </div>
      <div className="feed" ref={feed}>
        {events.length === 0 ? (
          <div className="empty">
            <div className="k">{emptyKey}</div>
            <p>{emptyText}</p>
          </div>
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
    "What factors are associated with the amount passengers pay for NYC yellow taxi trips?",
  );
  const [rounds, setRounds] = useState(3);
  const [busy, setBusy] = useState(false);
  const [uploadPct, setUploadPct] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [zoom, setZoom] = useState<string | null>(null);
  const [model, setModel] = useState<string>("");
  const [answerOpen, setAnswerOpen] = useState(false);

  const refreshDatasets = useCallback(async (prefer?: string) => {
    try {
      const rows = await listDatasets();
      setDatasets(rows);
      setSelected((cur) => prefer ?? (cur || rows[0]?.path || ""));
    } catch {
      setError("API unreachable — is the server running?");
    }
  }, []);

  useEffect(() => {
    void refreshDatasets();
    void getHealth()
      .then((h) => setModel(h.config.llm.configured ? h.config.llm.model : "no model configured"))
      .catch(() => undefined);
  }, [refreshDatasets]);

  useEffect(() => {
    if (state.answer) setAnswerOpen(true);
  }, [state.answer]);

  useEffect(() => {
    if (!state.running) setBusy(false);
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
      setError(err instanceof Error ? err.message : "upload failed");
    } finally {
      setUploadPct(null);
      e.target.value = "";
    }
  };

  const run = async () => {
    setError(null);
    setBusy(true);
    setAnswerOpen(false);
    try {
      const id = await startAnalysis({
        question,
        path: selected || null,
        max_rounds: rounds,
        max_visualizations: 8,
      });
      connect(id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "failed to start");
      setBusy(false);
    }
  };

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setZoom(null);
        setAnswerOpen(false);
      }
    };
    addEventListener("keydown", onKey);
    return () => removeEventListener("keydown", onKey);
  }, []);

  const stats: [string, number][] = [
    ["tests", state.stats.tests],
    ["graphs", state.stats.graphs],
    ["plan", state.stats.llm],
    ["read", state.stats.vlm],
    ["tokens", state.stats.tokens],
    ["evidence", state.stats.evidence],
  ];

  return (
    <div className="app">
      <header className="hdr">
        <div className="brand">
          <span className={`dot${state.running ? "" : " idle"}`} />
          Signal Engine
          <small>dual-agent</small>
        </div>
        {model && <span className="hdr-note">{model}</span>}
        <div className="stats">
          {stats.map(([k, v]) => (
            <div className="stat" key={k}>
              <b>{fmt(v)}</b>
              <span>{k}</span>
            </div>
          ))}
        </div>
      </header>

      <div className={`setup${state.running ? " gone" : ""}`}>
        <span className="lbl">Dataset</span>
        <select value={selected} onChange={(e) => setSelected(e.target.value)}>
          {datasets.length === 0 && <option value="">no datasets found</option>}
          {datasets.map((d) => (
            <option key={d.path} value={d.path}>
              {d.name}
              {d.rows ? ` · ${fmt(d.rows)} rows` : ""}
            </option>
          ))}
        </select>

        <button className="ghost upload" disabled={uploadPct !== null}>
          {uploadPct !== null ? `${(uploadPct * 100).toFixed(0)}%` : "upload"}
          <input type="file" accept=".parquet,.csv,.tsv" onChange={onUpload} />
        </button>

        <span className="lbl">Question</span>
        <input
          className="q"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !busy) void run();
          }}
        />

        <span className="lbl">Rounds</span>
        <input
          className="num"
          type="number"
          min={1}
          max={6}
          value={rounds}
          onChange={(e) => setRounds(Number(e.target.value))}
        />

        <button onClick={() => void run()} disabled={busy || !selected}>
          {busy ? "running" : "Run"}
        </button>
        {error && <span className="err">{error}</span>}
      </div>

      <div className="rail">
        {STAGES.map((s) => (
          <div
            key={s}
            className={`step${state.stagesDone.has(s) ? " done" : ""}${
              state.stage === s ? " active" : ""
            }`}
          >
            <i />
            {STAGE_LABEL[s]}
          </div>
        ))}
      </div>

      <div className="lanes">
        <Lane
          className="lane-f"
          tag="01"
          title="Finding agent"
          meta={state.dataset ? `${fmt(state.dataset.rows)} rows` : "idle"}
          events={state.finder}
          emptyKey="awaiting run"
          emptyText="Proposes hypotheses, prunes them on unit algebra, and measures what survives."
          onZoom={setZoom}
        />
        <Lane
          className="lane-a"
          tag="02"
          title="Analysis agent"
          meta={state.running ? "reading graphs" : "idle"}
          events={state.analyst}
          emptyKey="awaiting handoff"
          emptyText="Reads each rendered graph alongside its exact statistics, then is checked by the critic."
          onZoom={setZoom}
        />
      </div>

      <div className={`answer${answerOpen && state.answer ? " show" : ""}`}>
        <button className="ghost close" onClick={() => setAnswerOpen(false)}>
          close
        </button>
        <h3>Answer</h3>
        <div className="txt">{state.answer?.text}</div>
        {state.answer && state.answer.findings.length > 0 && (
          <>
            <h3 style={{ marginTop: 12 }}>Key findings</h3>
            <ul>
              {state.answer.findings.slice(0, 6).map((f) => (
                <li key={f}>{f}</li>
              ))}
            </ul>
          </>
        )}
      </div>

      {zoom && (
        <div className="box" onClick={() => setZoom(null)}>
          <img src={zoom} alt="enlarged chart" />
        </div>
      )}
    </div>
  );
}
