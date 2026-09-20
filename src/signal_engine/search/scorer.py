"""Candidate scoring — what makes a relationship worth more compute.

The conceptual objective is

    utility = strength * stability * relevance * novelty / cost

but shipping that formula literally would be a mistake, for three reasons the
implementation addresses:

1. **A zero factor annihilates everything.**  Pure multiplication means a
   candidate with unknown stability (not yet measured) scores 0 and is never
   expanded — so it never gets measured.  Unknown factors default to a neutral
   value, not zero.

2. **Cost division is unstable.**  Dividing by a near-zero cost estimate sends
   the score to infinity.  Cost enters as a bounded penalty instead.

3. **The factors are not equally important.**  Effect strength and stability
   are measurements; relevance and novelty are heuristics.  Weighting them
   equally would let a keyword match outrank a real finding.

So the implementation is a weighted geometric mean over clamped factors with
an additive cost penalty, and `explain()` returns the breakdown so a ranking
can always be justified.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from signal_engine.statistics.correlation import RelationshipResult

# Weights over the factors.  Measurements outweigh heuristics.
W_STRENGTH = 0.40
W_STABILITY = 0.25
W_RELEVANCE = 0.20
W_NOVELTY = 0.15

NEUTRAL = 0.6  # default for a factor that has not been measured yet


@dataclass
class CandidateScore:
    total: float = 0.0
    strength: float = 0.0
    stability: float = NEUTRAL
    relevance: float = NEUTRAL
    novelty: float = 1.0
    cost_penalty: float = 0.0
    priority_boost: float = 1.0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total": round(self.total, 5),
            "strength": round(self.strength, 4),
            "stability": round(self.stability, 4),
            "relevance": round(self.relevance, 4),
            "novelty": round(self.novelty, 4),
            "cost_penalty": round(self.cost_penalty, 4),
            "priority_boost": round(self.priority_boost, 3),
            "notes": self.notes,
        }

    def explain(self) -> str:
        return (
            f"score={self.total:.3f} = strength {self.strength:.2f}^{W_STRENGTH} x "
            f"stability {self.stability:.2f}^{W_STABILITY} x relevance {self.relevance:.2f}^{W_RELEVANCE} x "
            f"novelty {self.novelty:.2f}^{W_NOVELTY}, cost penalty {self.cost_penalty:.2f}, "
            f"priority x{self.priority_boost:.2f}"
        )


def _clamp(v: float, lo: float = 0.01, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def score_candidate(
    result: RelationshipResult | None,
    *,
    question_terms: set[str] | None = None,
    feature_names: list[str] | None = None,
    seen_expression_hashes: set[str] | None = None,
    expression_hash: str | None = None,
    depth: int = 0,
    llm_priority: float = 1.0,
    estimated_cost: float = 0.3,
    column_descriptions: dict[str, str] | None = None,
    is_mechanical: bool = False,
) -> CandidateScore:
    """Score one candidate for beam selection."""
    score = CandidateScore()

    # ---- strength: measured effect --------------------------------------
    if result is not None and result.skipped_reason is None:
        score.strength = _clamp(result.effect, lo=0.0)
    else:
        score.strength = NEUTRAL if result is None else 0.0
        if result is not None and result.skipped_reason:
            score.notes.append(f"skipped: {result.skipped_reason}")

    # ---- stability: measured, else neutral ------------------------------
    if result is not None and result.stability is not None and not math.isnan(result.stability):
        score.stability = _clamp(result.stability, lo=0.0)
    else:
        score.stability = NEUTRAL

    # ---- relevance: overlap with the question ---------------------------
    score.relevance = _relevance(question_terms, feature_names, column_descriptions)

    # ---- novelty: have we effectively seen this before? -----------------
    if expression_hash and seen_expression_hashes and expression_hash in seen_expression_hashes:
        score.novelty = 0.05
        score.notes.append("expression already evaluated")
    else:
        # Deeper transformations are less novel per unit of complexity: a
        # depth-3 feature must earn its interpretability cost.
        score.novelty = _clamp(1.0 / (1.0 + 0.35 * depth))

    # ---- mechanical relationships are real but not discoveries ----------
    if is_mechanical:
        score.novelty *= 0.25
        score.notes.append("accounting identity: down-weighted as mechanical")

    # ---- cost: bounded additive penalty, never a divisor ----------------
    score.cost_penalty = _clamp(estimated_cost, lo=0.0) * 0.15
    score.priority_boost = _clamp(llm_priority, lo=0.3, hi=1.3)

    # ---- combine: weighted geometric mean -------------------------------
    factors = [
        (_clamp(score.strength), W_STRENGTH),
        (_clamp(score.stability), W_STABILITY),
        (_clamp(score.relevance), W_RELEVANCE),
        (_clamp(score.novelty), W_NOVELTY),
    ]
    log_sum = sum(w * math.log(v) for v, w in factors)
    geometric = math.exp(log_sum / sum(w for _, w in factors))

    score.total = max(0.0, geometric * score.priority_boost - score.cost_penalty)
    return score


def _relevance(
    question_terms: set[str] | None,
    feature_names: list[str] | None,
    column_descriptions: dict[str, str] | None,
) -> float:
    """How related is this candidate to what the user actually asked?

    Matches on the column NAME first, then on its description — a question
    about "what passengers pay" should reach `total_amount` even though the
    words differ, because the description says "total amount charged to
    passengers".
    """
    if not question_terms or not feature_names:
        return NEUTRAL

    hits = 0
    for name in feature_names:
        tokens = {t for t in name.lower().replace("-", "_").split("_") if len(t) > 2}
        if tokens & question_terms:
            hits += 2
            continue
        description = (column_descriptions or {}).get(name, "").lower()
        if description:
            desc_tokens = {t.strip(".,;:()") for t in description.split() if len(t) > 3}
            if desc_tokens & question_terms:
                hits += 1

    if hits == 0:
        return 0.35  # not obviously relevant, but not disqualified
    return _clamp(0.5 + 0.25 * hits)


def question_terms(question: str) -> set[str]:
    """Content words from the research question, for relevance matching."""
    stop = {
        "what", "which", "how", "why", "when", "where", "does", "do", "did", "is", "are",
        "was", "were", "the", "a", "an", "of", "in", "on", "at", "to", "for", "with", "by",
        "from", "and", "or", "not", "that", "this", "these", "those", "there", "about",
        "influence", "affect", "factors", "appear", "appears", "amount", "variables",
        "associated", "relationship", "between", "data", "dataset", "analysis",
    }
    words = {
        w.strip(".,;:!?()'\"").lower()
        for w in question.split()
    }
    return {w for w in words if len(w) > 2 and w not in stop}


def select_beam(
    scored: list[tuple[object, CandidateScore]], width: int
) -> list[tuple[object, CandidateScore]]:
    """Take the top `width` candidates by score."""
    return sorted(scored, key=lambda pair: -pair[1].total)[:width]
