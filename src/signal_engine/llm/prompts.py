"""Prompt construction.

Two principles govern everything here.

**Context minimization.**  The model gets column cards, exact statistics, and
the few retrieved evidence items that matter — never rows, never the full
evidence history.  Numbers are passed as compact structured lines rather than
narrative, because "pearson=+0.834 n=3,509,466" costs a fraction of the tokens
of a sentence saying the same thing and is less ambiguous.

**The model proposes; it does not conclude.**  Prompts repeatedly separate
observation from mechanism, forbid causal language, and instruct the model to
answer UNRESOLVED rather than invent an explanation.  An honest "I cannot
justify this" is a first-class output that gets stored as evidence and
revisited later.
"""

from __future__ import annotations

from signal_engine.llm.base import ChatMessage

PROMPT_VERSION = "v1"

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

HYPOTHESIS_SYSTEM = """You are the hypothesis-generation stage of an automated data-research engine.

You never see raw data. You see compact column descriptions and exact statistics computed by a deterministic engine. Your job is to decide WHAT IS WORTH TESTING NEXT, not to state what is true.

Rules you must follow:
1. Only reference column names that appear in the COLUMNS list, spelled exactly.
2. Never treat an identifier or categorical code as a quantity. Correlating an arbitrary zone ID or payment code with a number is meaningless; propose a group comparison instead.
3. Respect stated CAVEATS. If a caveat says a field cannot support a comparison, do not propose that comparison.
4. Do not propose a relationship that is an accounting identity (a total against its own components) as if it were a discovery. If it is worth testing to quantify, say so explicitly in the rationale.
5. Prefer hypotheses that are (a) relevant to the user's question, (b) not already tested, and (c) capable of surprising you.
6. Transformations must be written in this grammar only:
      column names, numeric literals, + - * /, and log(), log1p(), sqrt(), abs()
   Anything else is rejected by the parser.
7. Units must be coherent. Dollars plus miles is meaningless; dollars per mile is not.

Respond with a single JSON object and nothing else."""


VLM_SYSTEM = """You are the interpretation stage of an automated data-research engine.

You receive a graph image together with the exact statistics behind it, the column definitions, the units, the filters applied, and any related prior findings.

Absolute rules:
1. The statistics are authoritative. The image illustrates them. If your reading of the image disagrees with the numbers, the numbers win, and you should say the image is hard to read.
2. Never describe a positive coefficient as an inverse/negative relationship, or vice versa. State the direction exactly as the statistics report it.
3. Never turn correlation into causation. Use "is associated with", "co-varies with", "may be partly explained by". Never "causes", "drives", "leads to", "because of" unless you are describing a documented mechanical/definitional relationship.
4. Distinguish clearly between:
      observation           - what the numbers and picture show
      interpretation        - what that might mean
      plausible_mechanisms  - candidate explanations, each labelled as a hypothesis
      confounders           - what else could produce this pattern
5. If you cannot justify an explanation with the evidence in front of you, set conclusion_status to "unresolved" and say what additional test would help. This is a correct and valued answer. Do NOT invent a mechanism to fill the space.
6. If the relationship follows from how the fields are defined (a total against its components, a ratio against its own denominator), set conclusion_status to "mechanical".
7. Respect the CAVEATS. They describe how the data was collected and they override what the pattern appears to show.

Respond with a single JSON object and nothing else."""


JOINT_SYSTEM = """You are the evidence-synthesis stage of an automated data-research engine.

You are given one NEW finding and several PRIOR findings retrieved from the engine's persistent evidence memory. Some prior findings were previously marked UNRESOLVED because they could not be explained in isolation.

Your job: decide whether the new finding, taken together with the prior ones, explains anything that could not be explained before.

This is how the engine imitates how a researcher actually works: an unexplained observation becomes explainable once a related observation arrives.

Rules:
1. Only claim to resolve a prior finding if the new evidence genuinely bears on it. List those evidence_ids in resolves_evidence_ids.
2. An explanation is a hypothesis about mechanism, not proof. Say "may be partly explained by", not "is caused by".
3. Quantities must stay consistent with the statistics you were given.
4. If the pieces do not combine into anything, say so and keep conclusion_status "unresolved". Do not manufacture a connection.

Respond with a single JSON object and nothing else."""


FINAL_ANSWER_SYSTEM = """You are the reporting stage of an automated data-research engine.

You are given the user's research question and the ranked findings the engine verified, each with exact statistics and an interpretation status.

Write the answer for a technical reader. Requirements:
1. Lead with what the evidence supports, with the actual numbers.
2. Mark mechanical/definitional relationships as such. They are not discoveries.
3. Carry forward every caveat that limits a finding, especially data-collection caveats.
4. List what remains unresolved. An honest unresolved list is more valuable than a confident wrong answer.
5. No causal language for associational evidence.

Respond with a single JSON object and nothing else."""


# ---------------------------------------------------------------------------
# Schema hints (kept terse: they are paid for on every call)
# ---------------------------------------------------------------------------

HYPOTHESIS_SCHEMA_HINT = """{
  "hypotheses": [
    {
      "id": "h1",
      "base_features": ["<exact column name>"],
      "target_features": ["<exact column name>"],
      "transformation_candidates": ["fare_amount / trip_distance"],
      "relationship_types": ["linear"|"monotonic"|"nonlinear"|"group_difference"|"threshold"|"interaction"],
      "expected_direction": "positive"|"negative"|"none"|"unknown",
      "rationale": "<one or two sentences>",
      "proposed_controls": ["<column>"],
      "proposed_stratifications": ["<column>"],
      "priority": "high"|"medium"|"low",
      "estimated_information_gain": 0.0-1.0,
      "estimated_compute_cost": 0.0-1.0
    }
  ],
  "reasoning_summary": "<one sentence on the strategy for this round>",
  "abandon_branches": ["<branch id worth dropping>"]
}"""

VLM_SCHEMA_HINT = """{
  "observation": "<what the statistics and image show, with numbers>",
  "strength": "negligible"|"weak"|"moderate"|"strong"|"very strong",
  "interpretation": "<what it may mean, associational language only>",
  "stated_direction": "positive"|"negative"|"none"|"unknown",
  "plausible_mechanisms": ["<candidate explanation>"],
  "alternative_explanations": ["<other way this pattern could arise>"],
  "confounders": ["<variable that could produce this>"],
  "conclusion_status": "explained"|"unresolved"|"mechanical"|"contradicted"|"invalid",
  "confidence": 0.0-1.0,
  "additional_tests": ["<what would settle it>"]
}"""

JOINT_SCHEMA_HINT = """{
  "combined_explanation": "<what the pieces together suggest>",
  "resolves_evidence_ids": ["<evidence_id>"],
  "supporting_evidence_ids": ["<evidence_id>"],
  "contradicting_evidence_ids": ["<evidence_id>"],
  "conclusion_status": "explained"|"unresolved"|"mechanical"|"contradicted",
  "confidence": 0.0-1.0,
  "remaining_uncertainty": "<what is still unknown>"
}"""

FINAL_SCHEMA_HINT = """{
  "answer": "<the direct answer to the question>",
  "key_findings": ["<finding with its statistics>"],
  "caveats": ["<limitation>"],
  "unresolved_questions": ["<what the engine could not settle>"],
  "confidence": 0.0-1.0
}"""


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def build_hypothesis_prompt(
    *,
    question: str,
    dataset_text: str,
    prior_results: list[str] | None = None,
    retrieved_evidence: list[str] | None = None,
    tested_pairs: list[str] | None = None,
    budget_note: str = "",
    round_index: int = 0,
    max_hypotheses: int = 6,
) -> list[ChatMessage]:
    sections = [
        f"RESEARCH QUESTION:\n{question}",
        f"\n{dataset_text}",
    ]

    if prior_results:
        sections.append(
            "\nRESULTS FROM THE PREVIOUS ROUND (computed by the deterministic engine):\n"
            + "\n".join(f"  {line}" for line in prior_results[:25])
        )

    if retrieved_evidence:
        sections.append(
            "\nRELATED PRIOR FINDINGS retrieved from evidence memory:\n"
            + "\n".join(f"  {line}" for line in retrieved_evidence[:8])
        )

    if tested_pairs:
        sections.append(
            "\nALREADY TESTED (do not propose these again):\n  "
            + "; ".join(tested_pairs[:60])
        )

    if budget_note:
        sections.append(f"\nBUDGET: {budget_note}")

    sections.append(
        f"\nPropose at most {max_hypotheses} hypotheses for round {round_index + 1}. "
        "Prioritise what is most likely to change the answer to the research question. "
        "If the previous round showed a branch is exhausted, list it in abandon_branches.\n"
        f"\nRespond with exactly this JSON shape:\n{HYPOTHESIS_SCHEMA_HINT}"
    )

    return [
        ChatMessage("system", HYPOTHESIS_SYSTEM),
        ChatMessage("user", "\n".join(sections)),
    ]


def build_interpretation_prompt(
    *,
    question: str,
    statistics_text: str,
    column_context: str,
    plot_description: str,
    image_data_uri: str | None,
    filters_text: str = "",
    caveats: list[str] | None = None,
    related_evidence: list[str] | None = None,
) -> list[ChatMessage]:
    sections = [
        f"RESEARCH QUESTION:\n{question}",
        f"\nEXACT STATISTICS (authoritative):\n{statistics_text}",
        f"\nCOLUMN DEFINITIONS:\n{column_context}",
        f"\nWHAT THE GRAPH SHOWS:\n{plot_description}",
    ]
    if filters_text:
        sections.append(f"\nROWS INCLUDED:\n{filters_text}")
    if caveats:
        sections.append(
            "\nCAVEATS (these override what the pattern appears to show):\n"
            + "\n".join(f"  - {c}" for c in caveats)
        )
    if related_evidence:
        sections.append(
            "\nRELATED PRIOR FINDINGS:\n" + "\n".join(f"  - {e}" for e in related_evidence[:5])
        )
    sections.append(f"\nRespond with exactly this JSON shape:\n{VLM_SCHEMA_HINT}")

    user = ChatMessage("user", "\n".join(sections))
    if image_data_uri:
        user.images.append(image_data_uri)

    return [ChatMessage("system", VLM_SYSTEM), user]


def build_joint_reinterpretation_prompt(
    *,
    question: str,
    new_evidence_text: str,
    prior_evidence_texts: list[str],
) -> list[ChatMessage]:
    body = "\n".join(
        [
            f"RESEARCH QUESTION:\n{question}",
            f"\nNEW FINDING:\n{new_evidence_text}",
            "\nPRIOR FINDINGS FROM EVIDENCE MEMORY:",
            *[f"\n[{i + 1}] {t}" for i, t in enumerate(prior_evidence_texts[:6])],
            f"\nRespond with exactly this JSON shape:\n{JOINT_SCHEMA_HINT}",
        ]
    )
    return [ChatMessage("system", JOINT_SYSTEM), ChatMessage("user", body)]


def build_final_answer_prompt(
    *,
    question: str,
    findings_text: str,
    unresolved_text: str = "",
    dataset_summary: str = "",
) -> list[ChatMessage]:
    sections = [f"RESEARCH QUESTION:\n{question}"]
    if dataset_summary:
        sections.append(f"\nDATASET:\n{dataset_summary}")
    sections.append(f"\nVERIFIED FINDINGS (ranked):\n{findings_text}")
    if unresolved_text:
        sections.append(f"\nUNRESOLVED:\n{unresolved_text}")
    sections.append(f"\nRespond with exactly this JSON shape:\n{FINAL_SCHEMA_HINT}")
    return [ChatMessage("system", FINAL_ANSWER_SYSTEM), ChatMessage("user", "\n".join(sections))]


def budget_note(
    *, llm_calls_left: int, vlm_calls_left: int, seconds_left: float, tests_left: int
) -> str:
    return (
        f"{llm_calls_left} planning calls, {vlm_calls_left} graph interpretations, "
        f"{tests_left:,} statistical tests, and {seconds_left:.0f}s remain. "
        "Prioritise accordingly."
    )
