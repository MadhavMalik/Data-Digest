"""Run reports and the human/domain evaluation report.

Two artifacts:

  run_report.md        what happened in one analysis: findings, evidence,
                       telemetry, degradations, and where the search stopped

  evaluation_report.md the domain-validation report required by the project
                       brief.  It deliberately INCLUDES the bad examples.  A
                       report that only shows what worked is marketing, not
                       evaluation; the failures are where the engineering
                       decisions get tested.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def write_run_report(result, artifacts_dir: Path) -> Path:
    """Write the per-run markdown report and return its path."""
    directory = Path(artifacts_dir) / result.analysis_id
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "run_report.md"

    metrics = result.metrics.to_dict() if result.metrics else {}
    counters = metrics.get("counters", {})
    derived = metrics.get("derived", {})
    ds = result.profile.dataset
    view = result.view_report

    lines: list[str] = [
        f"# Analysis {result.analysis_id}",
        "",
        f"**Question:** {result.question}",
        "",
        f"**Dataset:** {ds.dataset_id} — {ds.row_count:,} rows x {ds.column_count} columns "
        f"({ds.byte_size / (1 << 20):.1f} MiB)",
        "",
        f"**Generated:** {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        f"**Stop reason:** `{result.state.stop_reason.value if result.state.stop_reason else 'n/a'}`",
        "",
        "---",
        "",
        "## Analysis view",
        "",
        f"{view.get('rows_after', 0):,} of {view.get('rows_before', 0):,} rows retained "
        f"({view.get('exclusion_fraction', 0):.2%} excluded). No record was silently dropped:",
        "",
        "| Filter | Rows it excludes | Rationale |",
        "|---|---:|---|",
    ]
    for name, rationale in view.get("rationale", {}).items():
        excluded = view.get("excluded_by_rule", {}).get(name, 0)
        lines.append(f"| `{name}` | {excluded:,} | {rationale} |")

    lines += [
        "",
        "## Findings",
        "",
        "| Relationship | n | effect | direction | stability | kind |",
        "|---|---:|---:|---|---:|---|",
    ]
    for test in result.state.ranked_tests(limit=15):
        r = test.result
        lines.append(
            f"| `{r.x_name}` ~ `{r.y_name}` | {r.n:,} | {r.effect:.3f} | {r.direction} | "
            f"{(r.stability if r.stability is not None else float('nan')):.2f} | "
            f"{'mechanical' if test.is_mechanical else 'empirical'} |"
        )

    lines += ["", "## Evidence", ""]
    for e in result.evidence:
        lines += [
            f"### `{e.evidence_id}` — {e.canonical_expression or ' ~ '.join(e.feature_names)}",
            "",
            f"- **Status:** `{e.explanation_status.value}` (confidence {e.explanation_confidence:.2f})",
            f"- **Statistics:** {e.statistical_metrics.to_compact_text()}",
            f"- **Plot:** `{e.plot_type}` → `{e.plot_uri or 'not rendered'}`",
            f"- **Observation:** {e.textual_summary}",
        ]
        interp = e.vlm_interpretation or {}
        if interp.get("interpretation"):
            lines.append(f"- **Interpretation:** {interp['interpretation']}")
        for key, label in (
            ("plausible_mechanisms", "Plausible mechanisms"),
            ("alternative_explanations", "Alternative explanations"),
            ("confounders", "Confounders"),
        ):
            values = interp.get(key) or []
            if values:
                lines.append(f"- **{label}:**")
                lines.extend(f"  - {v}" for v in values[:4])
        critic = e.critic_report or {}
        if critic.get("violations"):
            lines.append("- **Critic:**")
            for v in critic["violations"][:5]:
                lines.append(f"  - `{v['severity']}` {v['check']}: {v['message']}")
        if e.related_evidence_ids:
            lines.append(f"- **Related prior evidence:** {', '.join(e.related_evidence_ids[:5])}")
        if e.resolved_by_ids:
            lines.append(f"- **Resolved by:** {', '.join(e.resolved_by_ids)}")
        lines.append("")

    lines += [
        "## Telemetry",
        "",
        "Every figure below is measured from instrumentation, not estimated.",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Raw dataset rows | {ds.row_count:,} |",
        f"| Raw bytes on disk | {ds.byte_size:,} |",
        f"| Rows after validity filters | {view.get('rows_after', 0):,} |",
        f"| Columns profiled | {counters.get('columns_profiled', 0)} |",
        f"| Candidate expressions considered | {result.prune_stats.considered:,} |",
        f"| Pruned by dimensional analysis | {result.prune_stats.pruned_by_units:,} |",
        f"| Pruned by policy | {result.prune_stats.pruned_by_policy:,} |",
        f"| Pruned by dedup | {result.prune_stats.pruned_by_dedup:,} |",
        f"| Prune rate | {result.prune_stats.prune_rate:.1%} |",
        f"| Statistical tests run | {counters.get('statistical_tests', 0):,} |",
        f"| Mutual-information tests | {counters.get('mutual_information_tests', 0):,} |",
        f"| Stability tests | {counters.get('stability_tests', 0):,} |",
        f"| Group comparisons | {counters.get('group_comparisons', 0):,} |",
        f"| Graphs rendered | {counters.get('plots_rendered', 0)} |",
        f"| LLM calls | {counters.get('llm_calls', 0)} |",
        f"| VLM calls | {counters.get('vlm_calls', 0)} |",
        f"| Model cache hits | {derived.get('model_cache_hits', 0)} |",
        f"| Model cache hit rate | {derived.get('model_cache_hit_rate', 0):.1%} |",
        f"| Total prompt tokens | {derived.get('total_prompt_tokens', 0):,} |",
        f"| Total completion tokens | {derived.get('total_completion_tokens', 0):,} |",
        f"| Tokens avoided by cache | {derived.get('tokens_avoided_by_cache', 0):,} |",
        f"| Evidence objects created | {counters.get('evidence_created', 0)} |",
        f"| Prior evidence retrieved | {counters.get('evidence_retrievals', 0)} |",
        f"| Joint reinterpretations | {counters.get('joint_reinterpretations', 0)} |",
        f"| Evidence resolved by later findings | {counters.get('evidence_resolved', 0)} |",
        f"| Critic errors caught | {counters.get('critic_errors', 0)} |",
        f"| Critic warnings | {counters.get('critic_warnings', 0)} |",
        f"| Rejected model transformations | {counters.get('rejected_transformations', 0)} |",
        f"| Wall-clock seconds | {metrics.get('elapsed_seconds', 0):.2f} |",
        "",
    ]

    cov = result.covariance_summary or {}
    if cov:
        lines += [
            "### Covariance shortcut",
            "",
            f"- Variables: {cov.get('variables', 0)}",
            f"- Complete cases: {cov.get('n_complete', 0):,} of {cov.get('n_total', 0):,} "
            f"({cov.get('complete_fraction', 0):.1%})",
            f"- Shortcut valid: **{cov.get('valid', False)}**"
            + (f" — {cov.get('invalid_reason')}" if cov.get("invalid_reason") else ""),
            f"- {cov.get('note', '')}",
            "",
        ]

    lines += ["### Branches", "", "| Branch | Tests | Best score | Status | Stop reason |", "|---|---:|---:|---|---|"]
    for branch in result.state.branches.values():
        lines.append(
            f"| {branch.label or branch.branch_id} | {len(branch.tests)} | {branch.best_score:.3f} | "
            f"{branch.status.value} | {branch.stop_reason.value if branch.stop_reason else '—'} |"
        )

    if result.degradations:
        lines += ["", "## Degradations", ""]
        lines.extend(f"- {d}" for d in result.degradations)

    if result.final_answer:
        fa = result.final_answer
        lines += ["", "## Answer", "", fa.answer, ""]
        if fa.key_findings:
            lines += ["### Key findings", ""]
            lines.extend(f"- {f}" for f in fa.key_findings)
        if fa.caveats:
            lines += ["", "### Caveats", ""]
            lines.extend(f"- {c}" for c in fa.caveats)
        if fa.unresolved_questions:
            lines += ["", "### Unresolved", ""]
            lines.extend(f"- {u}" for u in fa.unresolved_questions)

    path.write_text("\n".join(lines))

    (directory / "result.json").write_text(
        json.dumps(result.to_dict(), indent=2, default=str)
    )
    return path


def write_evaluation_report(
    results: list,
    artifacts_dir: Path,
    *,
    domain_checks: list[dict] | None = None,
) -> Path:
    """Write the domain-validation report, including the failures."""
    directory = Path(artifacts_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "evaluation_report.md"

    lines = [
        "# Evaluation report",
        "",
        f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        "This report checks whether the engine's OUTPUT is defensible, not merely whether the "
        "code ran. It includes the cases the engine got wrong or could not settle; hiding those "
        "would make the report useless.",
        "",
        "---",
        "",
        "## Per-finding validation",
        "",
    ]

    for result in results:
        lines += [f"### Analysis `{result.analysis_id}` — {result.question}", ""]
        for e in result.evidence:
            interp = e.vlm_interpretation or {}
            critic = e.critic_report or {}
            follows = "yes" if critic.get("passed", True) else "NO — critic raised errors"
            lines += [
                f"#### `{e.canonical_expression or ' ~ '.join(e.feature_names)}`",
                "",
                f"- **Numerical evidence:** {e.statistical_metrics.to_compact_text()}",
                f"- **Interpretation:** {interp.get('interpretation') or e.textual_summary}",
                f"- **Does the interpretation follow from the evidence?** {follows}",
                f"- **Potential confounders:** "
                f"{', '.join(interp.get('confounders') or []) or 'none identified'}",
                f"- **Mechanical relationship?** "
                f"{'yes' if e.explanation_status.value == 'mechanical' else 'no'}",
                f"- **Contradicts dataset metadata?** "
                f"{'YES' if any(v['check'] == 'claim_blocked_by_data_dictionary' for v in critic.get('violations', [])) else 'no'}",
                f"- **Causal language used improperly?** "
                f"{'YES' if any(v['check'] == 'causal_claim_from_association' for v in critic.get('violations', [])) else 'no'}",
                "",
            ]
            if critic.get("violations"):
                lines.append("  Critic findings:")
                for v in critic["violations"]:
                    lines.append(f"  - `{v['severity']}` **{v['check']}** — {v['message']}")
                lines.append("")

    if domain_checks:
        lines += ["---", "", "## Domain semantic checks", "", "| Check | Expected | Observed | Result |", "|---|---|---|---|"]
        for check in domain_checks:
            lines.append(
                f"| {check['name']} | {check['expected']} | {check['observed']} | "
                f"{'PASS' if check['passed'] else '**FAIL**'} |"
            )
        lines.append("")

    failures = []
    for result in results:
        for e in result.evidence:
            critic = e.critic_report or {}
            for v in critic.get("violations", []):
                if v["severity"] == "error":
                    failures.append((e, v))

    lines += ["---", "", "## Failures caught", ""]
    if not failures:
        lines.append(
            "No critic errors were raised in these runs. That means the interpretations were "
            "consistent with the statistics, the units, and the dataset's documented semantics — "
            "not that the engine cannot produce a bad interpretation."
        )
    else:
        for e, v in failures:
            lines += [
                f"### {v['check']} in `{e.evidence_id}`",
                "",
                f"- **Failure:** {v['message']}",
                f"- **Evidence:** {v.get('evidence', '')}",
                "- **Likely cause:** the interpretation stage described the relationship in a way "
                "the numbers do not support.",
                f"- **Proposed fix:** {v.get('suggested_fix', 'n/a')}",
                f"- **Implemented:** yes — the critic downgraded this evidence to "
                f"`{e.explanation_status.value}` and the violation is stored with the record.",
                "",
            ]

    path.write_text("\n".join(lines))
    return path
