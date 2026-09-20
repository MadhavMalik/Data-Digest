import { useMemo, useState } from "react";
import type { ResultEvent } from "../types";
import type { ChartRow } from "../useAnalysisStream";

type Key = "pair" | "n" | "r" | "rho" | "eta" | "mi" | "stability" | "effect";

const COLUMNS: { key: Key; label: string; help: string; numeric: boolean }[] = [
  { key: "pair", label: "Relationship", help: "The pair of columns compared", numeric: false },
  { key: "n", label: "n", help: "Complete cases used", numeric: true },
  { key: "r", label: "r", help: "Pearson — straight-line association", numeric: true },
  { key: "rho", label: "\u03C1", help: "Spearman — monotone association, rank-based", numeric: true },
  { key: "eta", label: "\u03B7", help: "Correlation ratio — how much a category explains", numeric: true },
  { key: "mi", label: "MI", help: "Mutual information — any shape, including non-monotone", numeric: true },
  { key: "stability", label: "stab", help: "Agreement across held-out folds", numeric: true },
  { key: "effect", label: "effect", help: "The headline effect size used for ranking", numeric: true },
];

function value(row: ResultEvent, key: Key): number | string {
  switch (key) {
    case "pair": return `${row.x} ${row.y}`;
    case "n": return row.n;
    case "r": return row.r == null ? -Infinity : Math.abs(row.r);
    case "rho": return row.rho == null ? -Infinity : Math.abs(row.rho);
    case "eta": return row.eta ?? -Infinity;
    case "mi": return row.mi ?? -Infinity;
    case "stability": return row.stability ?? -Infinity;
    case "effect": return row.effect;
  }
}

const num = (v: number | null | undefined, digits = 3) =>
  v == null ? "" : `${v >= 0 ? "" : "\u2212"}${Math.abs(v).toFixed(digits)}`;

interface Props {
  rows: ResultEvent[];
  charts: ChartRow[];
  onOpen: (chart: ChartRow) => void;
}

export function Measurements({ rows, charts, onOpen }: Props) {
  const [sort, setSort] = useState<{ key: Key; desc: boolean }>({ key: "effect", desc: true });
  const [hideDefinitional, setHideDefinitional] = useState(true);
  const [needle, setNeedle] = useState("");

  const chartFor = useMemo(() => {
    const index = new Map<string, ChartRow>();
    for (const c of charts) if (!index.has(`${c.x}|${c.y}`)) index.set(`${c.x}|${c.y}`, c);
    return index;
  }, [charts]);

  const visible = useMemo(() => {
    const q = needle.trim().toLowerCase();
    const filtered = rows.filter(
      (r) =>
        (!hideDefinitional || !r.mechanical) &&
        (!q || `${r.x} ${r.y}`.toLowerCase().includes(q)),
    );
    const dir = sort.desc ? -1 : 1;
    return [...filtered].sort((a, b) => {
      const av = value(a, sort.key);
      const bv = value(b, sort.key);
      if (typeof av === "string" || typeof bv === "string") {
        return String(av).localeCompare(String(bv)) * dir;
      }
      return (av - bv) * dir;
    });
  }, [rows, sort, hideDefinitional, needle]);

  const definitional = rows.filter((r) => r.mechanical).length;

  if (rows.length === 0) {
    return (
      <div className="blank">
        <p>Nothing measured yet. Every relationship the engine tests lands here, with its
        full statistics, whether it survived or not.</p>
      </div>
    );
  }

  return (
    <div className="sheet">
      <div className="sheet-bar">
        <input
          className="find"
          placeholder="Filter by column name"
          value={needle}
          onChange={(e) => setNeedle(e.target.value)}
        />
        <label className="check">
          <input
            type="checkbox"
            checked={hideDefinitional}
            onChange={(e) => setHideDefinitional(e.target.checked)}
          />
          Hide definitional ({definitional})
        </label>
        <span className="count">
          {visible.length} of {rows.length} shown
        </span>
      </div>

      <div className="sheet-scroll">
        <table>
          <thead>
            <tr>
              {COLUMNS.map((c) => (
                <th
                  key={c.key}
                  title={c.help}
                  className={`${c.numeric ? "r" : ""}${sort.key === c.key ? " sorted" : ""}`}
                >
                  <button
                    type="button"
                    onClick={() =>
                      setSort((s) =>
                        s.key === c.key ? { key: c.key, desc: !s.desc } : { key: c.key, desc: true },
                      )
                    }
                  >
                    {c.label}
                    <i className={sort.key === c.key ? (sort.desc ? "dn" : "up") : ""} />
                  </button>
                </th>
              ))}
              <th className="r">shape</th>
            </tr>
          </thead>
          <tbody>
            {visible.map((row) => {
              const chart = chartFor.get(`${row.x}|${row.y}`);
              return (
                <tr
                  key={`${row.seq}`}
                  className={chart ? "has-chart" : ""}
                  onClick={chart ? () => onOpen(chart) : undefined}
                  title={chart ? "Open the chart for this pair" : undefined}
                >
                  <td className="pair">
                    <span className="x">{row.x}</span>
                    <span className="vs">vs</span>
                    <span className="y">{row.y}</span>
                    {row.mechanical && <span className="tagx">definitional</span>}
                    {chart && <span className="tagx chartable">chart</span>}
                  </td>
                  <td className="r dim">{row.n.toLocaleString()}</td>
                  <td className={`r ${row.r == null ? "dim" : row.r >= 0 ? "pos" : "neg"}`}>
                    {num(row.r)}
                  </td>
                  <td className={`r ${row.rho == null ? "dim" : row.rho >= 0 ? "pos" : "neg"}`}>
                    {num(row.rho)}
                  </td>
                  <td className="r">{num(row.eta)}</td>
                  <td className="r">{num(row.mi)}</td>
                  <td className="r">{num(row.stability, 2)}</td>
                  <td className="r strong">{num(row.effect, 3)}</td>
                  <td className="r dim shape">{row.shape}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
