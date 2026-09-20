# Analysis output

**Question:** What factors are associated with the amount passengers pay for NYC yellow taxi trips?
**Dataset:** nyc_tlc_yellow_2026_01 — 3,724,889 rows × 20 columns (61.2 MiB)
**Analysis ID:** `an_d32e2ca9b8dd`
**Generated:** 2026-09-20T13:25:22+00:00

## Where to look

| Folder | What's in it |
|---|---|
| `01_profiling/` | What the profiler found. **`EXACT_TEXT_SENT_TO_LLM.txt`** is the complete dataset description the model receives — it never sees a row. |
| `02_hypotheses/` | What the planner proposed each round, and any hallucinated column names that were stripped. |
| `03_combinations/` | **`ALL_COMBINATIONS_TESTED.xlsx`** — every variable combination tried with its measured result. Plus what was rejected and why. |
| `04_graphs/` | Every rendered graph, with an index. |
| `05_vlm_reasoning/` | **`VLM_REASONING.md`** — stage 2. What the model was shown and what it concluded, per graph, with the critic's verdict. |
| `06_evidence/` | The stored evidence objects. |
| `07_summary/` | Final answer, telemetry, filter accounting. |

## Headline numbers

- Candidate expressions considered: **314**
- Pruned by dimensional analysis: **250** (88% overall prune rate)
- Statistical tests run: **61**
- Graphs rendered: **14**
- Planning (LLM) calls: **4**
- Graph-reading (VLM) calls: **14**
- Critic errors caught: **0**
- Wall-clock: **109.3s**

19 files written.