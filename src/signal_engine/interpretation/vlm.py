"""Multimodal interpretation of an evidence graph.

The VLM sees the image AND everything needed to read it honestly: the exact
statistics, the column definitions and units, the transformation applied, the
row filters, the dataset caveats, and the top related prior findings.

A naked graph is not enough context.  A scatter plot of two unlabelled columns
invites the model to invent a story; the same plot with "tip_amount records
credit-card tips only; cash tips are structurally absent" attached does not.

When no provider is configured, `interpret_evidence` returns a DETERMINISTIC
interpretation built from the statistics alone.  It is deliberately modest —
it states the observation and marks the conclusion UNRESOLVED — so a run
without credentials still produces complete, honest evidence records.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from signal_engine.llm.base import LLMProvider, LLMResponse, LLMUnavailable, ModelUsage
from signal_engine.llm.prompts import build_interpretation_prompt
from signal_engine.llm.schemas import ConclusionStatus, Direction, VisualInterpretation
from signal_engine.profiling.column_cards import ColumnCard
from signal_engine.statistics.correlation import RelationshipResult
from signal_engine.visualization.render import PlotArtifact


@dataclass
class InterpretationRequest:
    question: str
    result: RelationshipResult
    artifact: PlotArtifact | None
    x_card: ColumnCard | None = None
    y_card: ColumnCard | None = None
    transformation_description: str = ""
    filters_text: str = ""
    related_evidence: list[str] = field(default_factory=list)
    extra_caveats: list[str] = field(default_factory=list)
    # The fitted functional form of the conditional mean. A correlation says
    # "these move together"; the shape says HOW, which is usually the finding.
    shape_text: str = ""
    # Present when this evidence came from residual analysis: states what was
    # already accounted for, so the model interprets the REMAINDER rather than
    # re-describing the baseline.
    residual_context: str = ""

    # ---- context assembly ----------------------------------------------
    def statistics_text(self) -> str:
        """Exact numbers as compact structured lines — never prose.

        Protected from context compression: these are the authoritative values
        and paraphrasing them would defeat the purpose.
        """
        r = self.result
        lines = [
            f"variables: {r.x_name} vs {r.y_name}",
            f"n (jointly present rows): {r.n:,}",
            f"method: {r.method}",
        ]
        if r.pearson_r is not None:
            lines.append(f"pearson r: {r.pearson_r:+.4f}  (r^2 = {(r.r_squared or 0):.4f})")
        if r.spearman_rho is not None:
            lines.append(f"spearman rho: {r.spearman_rho:+.4f}")
        if r.eta is not None:
            lines.append(f"eta (variance explained by group): {r.eta:.4f}")
        if r.mutual_information is not None:
            lines.append(f"normalized mutual information: {r.mutual_information:.4f}")
        if r.slope is not None:
            lines.append(f"OLS slope: {r.slope:+.5g} per unit of {r.x_name}")
        if r.stability is not None:
            lines.append(f"cross-fold stability: {r.stability:.3f} (1.0 = identical across folds)")
        if r.q_value is not None:
            lines.append(f"BH-FDR q-value: {r.q_value:.3g}")
        lines.append(f"DIRECTION AS MEASURED: {r.direction}")
        lines.append(f"EFFECT MAGNITUDE: {r.effect:.4f} ({r.strength_label()})")
        if self.shape_text:
            lines.append("")
            lines.append("FUNCTIONAL FORM (fitted to the conditional mean E[y|x], not to raw points):")
            lines.extend(f"  {line}" for line in self.shape_text.splitlines())
        if r.extra.get("group_means"):
            lines.append("group means:")
            for g in r.extra["group_means"][:12]:
                lines.append(f"  {g['level']}: mean={g['mean']:,.4g} (n={g['n']:,})")
        for w in r.warnings:
            lines.append(f"warning: {w}")
        return "\n".join(lines)

    def column_context(self) -> str:
        parts: list[str] = []
        for card in (self.x_card, self.y_card):
            if card is None:
                continue
            parts.append(f"- {card.to_compact_text()}")
        if self.transformation_description:
            parts.append(f"- transformation applied: {self.transformation_description}")
        return "\n".join(parts) or "(no column metadata available)"

    def caveats(self) -> list[str]:
        out: list[str] = list(self.extra_caveats)
        for card in (self.x_card, self.y_card):
            if card is not None:
                out.extend(card.caveats)
        # Preserve order, drop duplicates.
        seen: set[str] = set()
        unique = []
        for c in out:
            if c not in seen:
                seen.add(c)
                unique.append(c)
        return unique


async def interpret_evidence(
    provider: LLMProvider,
    request: InterpretationRequest,
    *,
    model: str | None = None,
    send_image: bool = True,
) -> tuple[VisualInterpretation, LLMResponse | None, str | None]:
    """Interpret one piece of evidence.

    Returns (interpretation, response_or_None, fallback_reason_or_None).
    Never raises for a provider problem: an unavailable model degrades to the
    deterministic interpretation and the analysis continues.
    """
    if not provider.available:
        return _deterministic(request), None, "no LLM provider configured"

    image_uri = None
    if send_image and request.artifact is not None:
        image_uri = request.artifact.data_uri

    messages = build_interpretation_prompt(
        question=request.question,
        statistics_text=request.statistics_text(),
        column_context=request.column_context(),
        plot_description=(
            request.artifact.spec_description if request.artifact else "(no graph rendered)"
        ),
        image_data_uri=image_uri,
        filters_text=request.filters_text,
        caveats=request.caveats(),
        related_evidence=request.related_evidence,
        residual_context=request.residual_context,
    )

    try:
        interpretation, response = await provider.complete_structured(
            messages, VisualInterpretation, model=model
        )
        return interpretation, response, None
    except LLMUnavailable as exc:
        return _deterministic(request), None, f"provider unavailable: {exc}"
    except Exception as exc:  # noqa: BLE001 - never fail the analysis over interpretation
        return _deterministic(request), None, f"interpretation failed: {type(exc).__name__}: {exc}"


def _deterministic(request: InterpretationRequest) -> VisualInterpretation:
    """Build an interpretation from the statistics alone.

    Used when no model is available.  It states only what the numbers say and
    stops there — the conclusion is UNRESOLVED because a deterministic
    template genuinely cannot supply a mechanism, and claiming otherwise would
    be exactly the failure mode this engine exists to avoid.
    """
    r = request.result
    direction_word = {"positive": "increases with", "negative": "decreases with"}.get(
        r.direction, "shows no consistent monotonic relationship with"
    )

    coefficient = ""
    if r.pearson_r is not None:
        coefficient = f" (Pearson r = {r.pearson_r:+.3f}"
        if r.spearman_rho is not None:
            coefficient += f", Spearman rho = {r.spearman_rho:+.3f}"
        coefficient += f", n = {r.n:,})"
    elif r.eta is not None:
        coefficient = f" (eta = {r.eta:.3f}, n = {r.n:,})"

    observation = (
        f"Over {r.n:,} rows, {r.y_name} {direction_word} {r.x_name}{coefficient}. "
        f"The measured effect magnitude is {r.effect:.3f}, which is {r.strength_label()}."
    )

    status = ConclusionStatus.UNRESOLVED
    mechanisms: list[str] = []
    if r.extra.get("group_means"):
        best = max(r.extra["group_means"], key=lambda g: g["mean"])
        worst = min(r.extra["group_means"], key=lambda g: g["mean"])
        observation += (
            f" The highest group mean is {best['mean']:,.4g} at level {best['level']} and the "
            f"lowest is {worst['mean']:,.4g} at level {worst['level']}."
        )

    caveats = request.caveats()
    alternatives = list(caveats[:3])
    if r.stability is not None and r.stability < 0.5:
        alternatives.append(
            f"cross-fold stability is only {r.stability:.2f}, so the association is not "
            "consistent across independent subsamples"
        )

    return VisualInterpretation(
        observation=observation,
        strength=r.strength_label(),
        interpretation=(
            "No language model was available to propose a mechanism, so this record carries the "
            "measured association only."
        ),
        stated_direction=(
            Direction(r.direction) if r.direction in ("positive", "negative", "none")
            else Direction.UNKNOWN
        ),
        plausible_mechanisms=mechanisms,
        alternative_explanations=alternatives,
        confounders=[],
        conclusion_status=status,
        confidence=0.35,
        additional_tests=[
            f"stratify {r.x_name} vs {r.y_name} by a plausible confounder",
            "test a nonlinear form if the monotonic measure exceeds the linear one",
        ],
    )


def estimate_request_tokens(request: InterpretationRequest) -> ModelUsage:
    """Token estimate for the context this request would send (telemetry)."""
    from signal_engine.llm.base import estimate_tokens

    text = "\n".join(
        [
            request.question,
            request.statistics_text(),
            request.column_context(),
            request.filters_text,
            "\n".join(request.caveats()),
            "\n".join(request.related_evidence),
        ]
    )
    return ModelUsage(prompt_tokens=estimate_tokens(text), estimated=True)
