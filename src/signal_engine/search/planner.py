"""Hypothesis planning: the LLM, with a deterministic fallback that always works.

`DeterministicPlanner` is not a stub.  It encodes the same reasoning a careful
analyst applies before any model is involved:

  * prefer columns whose name or description matches the question
  * pair quantities with the question's target quantity
  * route categoricals to group comparisons, never to correlations
  * propose the ratio and duration features the domain obviously needs
  * skip pairs already tested

It means the engine produces real findings with zero credentials, and it gives
the LLM path something to be measured against — if model-generated hypotheses
do not beat this, the model is not earning its cost.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from signal_engine.llm.base import LLMProvider, LLMResponse, LLMUnavailable
from signal_engine.llm.prompts import budget_note, build_hypothesis_prompt
from signal_engine.llm.schemas import (
    Direction,
    Hypothesis,
    HypothesisBatch,
    Priority,
    RelationshipType,
)
from signal_engine.profiling.profiler import DatasetProfile
from signal_engine.profiling.semantic_types import SemanticType
from signal_engine.search.scorer import question_terms


@dataclass
class PlanningOutcome:
    batch: HypothesisBatch
    response: LLMResponse | None
    source: str  # "llm" | "deterministic"
    dropped_columns: list[str]
    fallback_reason: str | None = None


async def plan_round(
    provider: LLMProvider,
    *,
    question: str,
    profile: DatasetProfile,
    dataset_text: str,
    prior_results: list[str],
    retrieved_evidence: list[str],
    tested_pairs: list[str],
    round_index: int,
    budget,
    max_hypotheses: int = 6,
    extra_columns: dict | None = None,
) -> PlanningOutcome:
    """Get this round's hypotheses, from the model or the deterministic planner."""
    known_columns = set(profile.columns) | set(extra_columns or {})

    if not provider.available or not budget.may_call_llm():
        reason = (
            "no LLM provider configured"
            if not provider.available
            else "LLM call budget exhausted"
        )
        batch = deterministic_hypotheses(
            question=question,
            profile=profile,
            tested_pairs=set(tested_pairs),
            extra_columns=extra_columns or {},
            max_hypotheses=max_hypotheses,
            round_index=round_index,
        )
        return PlanningOutcome(batch, None, "deterministic", [], reason)

    messages = build_hypothesis_prompt(
        question=question,
        dataset_text=dataset_text,
        prior_results=prior_results,
        retrieved_evidence=retrieved_evidence,
        tested_pairs=tested_pairs,
        budget_note=budget_note(
            llm_calls_left=budget.llm_calls_left,
            vlm_calls_left=budget.vlm_calls_left,
            seconds_left=budget.seconds_left,
            tests_left=budget.tests_left,
        ),
        round_index=round_index,
        max_hypotheses=max_hypotheses,
    )

    try:
        batch, response = await provider.complete_structured(messages, HypothesisBatch)
    except (LLMUnavailable, Exception) as exc:  # noqa: BLE001
        fallback = deterministic_hypotheses(
            question=question,
            profile=profile,
            tested_pairs=set(tested_pairs),
            extra_columns=extra_columns or {},
            max_hypotheses=max_hypotheses,
            round_index=round_index,
        )
        return PlanningOutcome(
            fallback, None, "deterministic", [], f"{type(exc).__name__}: {str(exc)[:160]}"
        )

    # Strip hallucinated column names before anything reaches the engine.
    cleaned: list[Hypothesis] = []
    dropped: list[str] = []
    for i, h in enumerate(batch.hypotheses):
        fixed, gone = h.resolve_columns(known_columns)
        dropped.extend(gone)
        if not fixed.base_features and not fixed.transformation_candidates:
            continue
        if not fixed.id:
            fixed.id = f"h{round_index}_{i}"
        cleaned.append(fixed)

    if not cleaned:
        fallback = deterministic_hypotheses(
            question=question,
            profile=profile,
            tested_pairs=set(tested_pairs),
            extra_columns=extra_columns or {},
            max_hypotheses=max_hypotheses,
            round_index=round_index,
        )
        return PlanningOutcome(
            fallback, response, "deterministic", dropped,
            "every model hypothesis referenced unknown columns",
        )

    batch.hypotheses = cleaned
    return PlanningOutcome(batch, response, "llm", dropped)


# ---------------------------------------------------------------------------
# Deterministic planner
# ---------------------------------------------------------------------------

# Words that identify the quantity a question is about, mapped to what they
# most plausibly refer to in a trip-record schema.
def infer_target(question: str, available: dict, spec=None) -> str | None:
    """Pick the dependent variable the question is about.

    Delegates to the dataset registry's generic scorer, which uses curated
    hints when the dataset is recognised and falls back to matching the
    question against column names, descriptions and semantic kinds otherwise.
    """
    from signal_engine.datasets import infer_target_generic

    return infer_target_generic(question, available, spec)


def deterministic_hypotheses(
    *,
    question: str,
    profile: DatasetProfile,
    tested_pairs: set[str],
    extra_columns: dict,
    max_hypotheses: int = 6,
    round_index: int = 0,
) -> HypothesisBatch:
    """Build hypotheses from semantics alone — no model involved."""
    cards = dict(profile.columns)
    available = {**cards, **extra_columns}
    terms = question_terms(question)

    target = infer_target(question, available)
    hypotheses: list[Hypothesis] = []

    def already(a: str, b: str) -> bool:
        x, y = sorted([a, b])
        return f"{x}~{y}" in tested_pairs

    numeric = [
        n
        for n, c in available.items()
        if getattr(c, "semantic_type", None) is not None
        and c.semantic_type.is_numeric_quantity
        and n != target
    ]
    categorical = [
        n
        for n, c in available.items()
        if getattr(c, "semantic_type", None) is not None and c.semantic_type.is_categorical
    ]

    def relevance(name: str) -> int:
        tokens = {t for t in name.lower().split("_") if len(t) > 2}
        score = 3 if tokens & terms else 0
        card = available.get(name)
        desc = (getattr(card, "description", "") or "").lower()
        if terms & {w.strip(".,;:") for w in desc.split() if len(w) > 3}:
            score += 1
        # Mechanical components of the target are real but uninformative, so
        # they sort last rather than being excluded.
        if getattr(card, "is_fare_component", False) and target == "total_amount":
            score -= 2
        return score

    numeric.sort(key=lambda n: -relevance(n))
    categorical.sort(key=lambda n: -relevance(n))

    # Round 0 covers the obvious drivers; later rounds go wider.
    offset = round_index * 3

    if target:
        for name in numeric[offset : offset + max_hypotheses]:
            if already(name, target):
                continue
            card = available.get(name)
            mechanical = bool(getattr(card, "is_fare_component", False))
            hypotheses.append(
                Hypothesis(
                    id=f"det{round_index}_{uuid.uuid4().hex[:6]}",
                    base_features=[name],
                    target_features=[target],
                    relationship_types=[RelationshipType.LINEAR, RelationshipType.MONOTONIC],
                    expected_direction=Direction.UNKNOWN,
                    rationale=(
                        f"{name} is a quantity that may co-vary with {target}."
                        + (
                            f" Note: {name} is a component of {target}, so any relationship is "
                            "partly definitional."
                            if mechanical
                            else ""
                        )
                    ),
                    priority=Priority.LOW if mechanical else Priority.HIGH,
                    estimated_information_gain=0.2 if mechanical else 0.7,
                    estimated_compute_cost=0.2,
                )
            )

        for name in categorical[: max(2, max_hypotheses // 2)]:
            if already(name, target):
                continue
            hypotheses.append(
                Hypothesis(
                    id=f"det{round_index}_{uuid.uuid4().hex[:6]}",
                    base_features=[name],
                    target_features=[target],
                    relationship_types=[RelationshipType.GROUP_DIFFERENCE],
                    expected_direction=Direction.UNKNOWN,
                    rationale=(
                        f"{name} is categorical; compare {target} ACROSS its levels rather "
                        "than correlating against its codes."
                    ),
                    proposed_stratifications=[name],
                    priority=Priority.MEDIUM,
                    estimated_information_gain=0.6,
                    estimated_compute_cost=0.3,
                )
            )

    # Domain ratio features, proposed only when their inputs exist.
    ratio_ideas = [
        ("fare_amount / trip_distance", ["fare_amount", "trip_distance"], "cost per mile"),
        (
            "trip_distance / trip_duration_seconds",
            ["trip_distance", "trip_duration_seconds"],
            "average speed",
        ),
        ("tip_amount / fare_amount", ["tip_amount", "fare_amount"], "tip as a share of fare"),
    ]
    for expression, needed, description in ratio_ideas:
        if all(n in available for n in needed) and len(hypotheses) < max_hypotheses + 3:
            hypotheses.append(
                Hypothesis(
                    id=f"det{round_index}_{uuid.uuid4().hex[:6]}",
                    base_features=needed,
                    target_features=[target] if target else [],
                    transformation_candidates=[expression],
                    relationship_types=[RelationshipType.LINEAR],
                    rationale=f"{description} is a dimensionally coherent derived quantity.",
                    priority=Priority.MEDIUM,
                    estimated_information_gain=0.6,
                    estimated_compute_cost=0.3,
                )
            )

    return HypothesisBatch(
        hypotheses=hypotheses[: max_hypotheses + 3],
        reasoning_summary=(
            f"Deterministic plan for round {round_index + 1}: pair candidate quantities with "
            f"{target or 'the most relevant target'}, route categoricals to group comparisons, "
            "and propose dimensionally coherent ratios."
        ),
    )
