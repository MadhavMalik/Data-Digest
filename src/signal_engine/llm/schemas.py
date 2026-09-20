"""Pydantic schemas for every structured model interaction.

Nothing the model returns is parsed as prose when it could be parsed as data.
Each schema is permissive about what it ACCEPTS (models emit "HIGH", "high",
3, stray prose) and strict about what it PRODUCES, so the engine downstream
always sees normalized values.

The validators here are the second line of defence after the expression
parser: a hallucinated column name, an unsupported transformation, or an
out-of-range confidence is corrected or rejected at the schema boundary.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class Priority(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

    @property
    def weight(self) -> float:
        return {"high": 1.0, "medium": 0.6, "low": 0.3}[self.value]


class RelationshipType(str, Enum):
    LINEAR = "linear"
    MONOTONIC = "monotonic"
    NONLINEAR = "nonlinear"
    GROUP_DIFFERENCE = "group_difference"
    THRESHOLD = "threshold"
    INTERACTION = "interaction"


class Direction(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NONE = "none"
    UNKNOWN = "unknown"


class ConclusionStatus(str, Enum):
    EXPLAINED = "explained"
    UNRESOLVED = "unresolved"
    CONTRADICTED = "contradicted"
    MECHANICAL = "mechanical"
    INVALID = "invalid"


def _coerce_enum(value: Any, enum_cls: type[Enum], default: Enum) -> Enum:
    if isinstance(value, enum_cls):
        return value
    if isinstance(value, str):
        try:
            return enum_cls(value.strip().lower())
        except ValueError:
            return default
    return default


class Hypothesis(BaseModel):
    """One testable proposition from the model."""

    id: str = ""
    base_features: list[str] = Field(default_factory=list)
    target_features: list[str] = Field(default_factory=list)
    transformation_candidates: list[str] = Field(default_factory=list)
    relationship_types: list[RelationshipType] = Field(default_factory=list)
    expected_direction: Direction = Direction.UNKNOWN
    rationale: str = ""
    proposed_controls: list[str] = Field(default_factory=list)
    proposed_stratifications: list[str] = Field(default_factory=list)
    priority: Priority = Priority.MEDIUM
    estimated_information_gain: float = 0.5
    estimated_compute_cost: float = 0.5

    @field_validator("relationship_types", mode="before")
    @classmethod
    def _rel_types(cls, v: Any) -> list:
        if v is None:
            return []
        if isinstance(v, str):
            v = [v]
        out = []
        for item in v:
            coerced = _coerce_enum(item, RelationshipType, RelationshipType.LINEAR)
            if coerced not in out:
                out.append(coerced)
        return out

    @field_validator("expected_direction", mode="before")
    @classmethod
    def _direction(cls, v: Any) -> Any:
        return _coerce_enum(v, Direction, Direction.UNKNOWN)

    @field_validator("priority", mode="before")
    @classmethod
    def _priority(cls, v: Any) -> Any:
        if isinstance(v, (int, float)):
            return Priority.HIGH if v >= 0.7 else (Priority.MEDIUM if v >= 0.4 else Priority.LOW)
        return _coerce_enum(v, Priority, Priority.MEDIUM)

    @field_validator("estimated_information_gain", "estimated_compute_cost", mode="before")
    @classmethod
    def _unit_interval(cls, v: Any) -> float:
        try:
            f = float(v)
        except (TypeError, ValueError):
            return 0.5
        return min(1.0, max(0.0, f))

    @field_validator(
        "base_features",
        "target_features",
        "transformation_candidates",
        "proposed_controls",
        "proposed_stratifications",
        mode="before",
    )
    @classmethod
    def _string_list(cls, v: Any) -> list[str]:
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        return [str(x).strip() for x in v if str(x).strip()]

    @model_validator(mode="after")
    def _require_features(self) -> Hypothesis:
        if not self.base_features and not self.transformation_candidates:
            raise ValueError("hypothesis names no features to test")
        return self

    # ---- validation against the real dataset ----------------------------
    def resolve_columns(self, known_columns: set[str]) -> tuple[Hypothesis, list[str]]:
        """Drop hallucinated column names; report what was dropped.

        Case-insensitive resolution first, because a model will write
        `airport_fee` when the file says `Airport_fee`.  Anything that still
        does not resolve is removed rather than allowed to reach the engine.
        """
        lower = {c.lower(): c for c in known_columns}
        dropped: list[str] = []

        def fix(names: list[str]) -> list[str]:
            out = []
            for n in names:
                resolved = lower.get(n.strip().lower())
                if resolved:
                    out.append(resolved)
                else:
                    dropped.append(n)
            return out

        clean = self.model_copy(
            update={
                "base_features": fix(self.base_features),
                "target_features": fix(self.target_features),
                "proposed_controls": fix(self.proposed_controls),
                "proposed_stratifications": fix(self.proposed_stratifications),
            }
        )
        return clean, dropped


class HypothesisBatch(BaseModel):
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    reasoning_summary: str = ""
    abandon_branches: list[str] = Field(default_factory=list)

    @field_validator("hypotheses", mode="before")
    @classmethod
    def _tolerate_shapes(cls, v: Any) -> Any:
        # Models return {"hypotheses": [...]}, a bare list, or a single object.
        if v is None:
            return []
        if isinstance(v, dict):
            return [v]
        return v


class PlotChoice(BaseModel):
    """An LLM recommendation for how to visualize a relationship.

    Advisory only — the deterministic GraphSelector validates it and wins on
    conflict, because a wrong plot type produces a misleading picture and the
    VLM then interprets the misleading picture.
    """

    plot_type: str = "scatter"
    x: str = ""
    y: str = ""
    color_by: str | None = None
    rationale: str = ""

    @field_validator("plot_type", mode="before")
    @classmethod
    def _norm(cls, v: Any) -> str:
        return str(v or "scatter").strip().lower().replace(" ", "_")


class VisualInterpretation(BaseModel):
    """The VLM's reading of one evidence graph.

    The separation between `observation`, `interpretation`, and
    `plausible_mechanisms` is deliberate and enforced by the critic: the model
    must not collapse "these move together" into "this causes that".
    """

    observation: str = ""
    strength: str = ""
    interpretation: str = ""
    plausible_mechanisms: list[str] = Field(default_factory=list)
    alternative_explanations: list[str] = Field(default_factory=list)
    confounders: list[str] = Field(default_factory=list)
    conclusion_status: ConclusionStatus = ConclusionStatus.UNRESOLVED
    confidence: float = 0.5
    additional_tests: list[str] = Field(default_factory=list)
    stated_direction: Direction = Direction.UNKNOWN

    @field_validator("conclusion_status", mode="before")
    @classmethod
    def _status(cls, v: Any) -> Any:
        return _coerce_enum(v, ConclusionStatus, ConclusionStatus.UNRESOLVED)

    @field_validator("stated_direction", mode="before")
    @classmethod
    def _stated(cls, v: Any) -> Any:
        return _coerce_enum(v, Direction, Direction.UNKNOWN)

    @field_validator("confidence", mode="before")
    @classmethod
    def _conf(cls, v: Any) -> float:
        try:
            f = float(v)
        except (TypeError, ValueError):
            return 0.5
        if f > 1.0:  # a model that answered on a 0-100 scale
            f = f / 100.0
        return min(1.0, max(0.0, f))

    @field_validator(
        "plausible_mechanisms", "alternative_explanations", "confounders", "additional_tests",
        mode="before",
    )
    @classmethod
    def _lists(cls, v: Any) -> list[str]:
        if v is None:
            return []
        if isinstance(v, str):
            return [v] if v.strip() else []
        return [str(x).strip() for x in v if str(x).strip()]


class JointReinterpretation(BaseModel):
    """The output of reconsidering new evidence together with retrieved old evidence.

    This is the payoff of the evidence-memory design: an unresolved finding
    becomes explainable once a later, related finding arrives.
    """

    combined_explanation: str = ""
    resolves_evidence_ids: list[str] = Field(default_factory=list)
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    contradicting_evidence_ids: list[str] = Field(default_factory=list)
    confidence: float = 0.5
    remaining_uncertainty: str = ""
    conclusion_status: ConclusionStatus = ConclusionStatus.UNRESOLVED

    @field_validator("conclusion_status", mode="before")
    @classmethod
    def _status(cls, v: Any) -> Any:
        return _coerce_enum(v, ConclusionStatus, ConclusionStatus.UNRESOLVED)

    @field_validator("confidence", mode="before")
    @classmethod
    def _conf(cls, v: Any) -> float:
        try:
            f = float(v)
        except (TypeError, ValueError):
            return 0.5
        return min(1.0, max(0.0, f / 100.0 if f > 1.0 else f))

    @field_validator(
        "resolves_evidence_ids", "supporting_evidence_ids", "contradicting_evidence_ids",
        mode="before",
    )
    @classmethod
    def _ids(cls, v: Any) -> list[str]:
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        return [str(x) for x in v]


class FinalAnswer(BaseModel):
    """The synthesized answer to the user's research question."""

    answer: str = ""
    key_findings: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    confidence: float = 0.5

    @field_validator("key_findings", "caveats", "unresolved_questions", mode="before")
    @classmethod
    def _lists(cls, v: Any) -> list[str]:
        if v is None:
            return []
        if isinstance(v, str):
            return [v] if v.strip() else []
        return [str(x).strip() for x in v if str(x).strip()]

    @field_validator("confidence", mode="before")
    @classmethod
    def _conf(cls, v: Any) -> float:
        try:
            f = float(v)
        except (TypeError, ValueError):
            return 0.5
        return min(1.0, max(0.0, f / 100.0 if f > 1.0 else f))
