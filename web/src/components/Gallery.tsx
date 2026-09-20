import { useState } from "react";
import { assetUrl } from "../config";
import type { ChartRow } from "../useAnalysisStream";

interface Props {
  charts: ChartRow[];
  onOpen: (chart: ChartRow) => void;
}

const STAGE_COPY: Record<string, string> = {
  marginal: "raw relationship",
  residual: "after removing the obvious",
};

export function Gallery({ charts, onOpen }: Props) {
  const [only, setOnly] = useState<"all" | "marginal" | "residual">("all");

  if (charts.length === 0) {
    return (
      <div className="blank">
        <p>No charts yet. Each one is drawn from the filtered data, then read back
        against the exact statistics that produced it.</p>
      </div>
    );
  }

  const shown = charts.filter((c) => only === "all" || c.stage === only);
  const counts = {
    all: charts.length,
    marginal: charts.filter((c) => c.stage === "marginal").length,
    residual: charts.filter((c) => c.stage === "residual").length,
  };

  return (
    <div className="gallery">
      <div className="sheet-bar">
        {(["all", "marginal", "residual"] as const).map((k) => (
          <button
            key={k}
            type="button"
            className={`chip${only === k ? " on" : ""}`}
            onClick={() => setOnly(k)}
          >
            {k === "all" ? "All" : STAGE_COPY[k]} <b>{counts[k]}</b>
          </button>
        ))}
      </div>

      <div className="grid">
        {shown.map((c) => (
          <figure key={c.seq} className="plate" onClick={() => onOpen(c)}>
            <img src={assetUrl(c.url)} loading="lazy" alt={`${c.plotType} of ${c.expression}`} />
            <figcaption>
              <div className="cap-hd">
                <span className="expr">{c.expression}</span>
                {c.status && (
                  <span className={`pill ${c.status === "explained" ? "ok" : c.status === "mechanical" ? "mech" : "warn"}`}>
                    {c.status}
                  </span>
                )}
              </div>
              <div className="cap-st">{c.stats}</div>
              {c.interpretation && <p>{c.interpretation}</p>}
            </figcaption>
          </figure>
        ))}
      </div>
    </div>
  );
}
