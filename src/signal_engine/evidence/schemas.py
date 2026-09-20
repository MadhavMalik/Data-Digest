"""The EvidenceObject — the unit of the engine's memory.

Every analyzed relationship becomes one of these, INCLUDING the ones nothing
could explain.  That is the point: an unexplained-but-robust finding is not a
failure to discard, it is a standing question the engine can answer later when
a related finding arrives.

    Evidence 1  airport pickup -> unusually high fare per mile    UNRESOLVED
    ... later ...
    Evidence 2  airport pickup -> fixed airport fee
    Evidence 3  airport routes -> toll charges

    retrieval surfaces 1 alongside 2 and 3
    -> "the apparent airport premium may be partly explained by fixed fees and
        tolls rather than distance itself"

`semantic_search_text` is the field that makes this work: a dense natural-
language rendering of the finding that BM25 and vector search both bite on.
"""

from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum


class ExplanationStatus(str, Enum):
    EXPLAINED = "explained"
    UNRESOLVED = "unresolved"
    CONTRADICTED = "contradicted"
    MECHANICAL = "mechanical"
    REJECTED = "rejected"
    NEEDS_MORE_EVIDENCE = "needs_more_evidence"

    @property
    def is_open_question(self) -> bool:
        """Statuses that future evidence should be given a chance to resolve."""
        return self in {ExplanationStatus.UNRESOLVED, ExplanationStatus.NEEDS_MORE_EVIDENCE}


@dataclass
class StatisticalMetrics:
    n: int = 0
    pearson_r: float | None = None
    pearson_p: float | None = None
    spearman_rho: float | None = None
    mutual_information: float | None = None
    eta: float | None = None
    r_squared: float | None = None
    slope: float | None = None
    covariance: float | None = None
    effect: float = 0.0
    direction: str = "undetermined"
    strength: str = "negligible"
    stability: float | None = None
    q_value: float | None = None
    method: str = ""

    @classmethod
    def from_result(cls, result) -> StatisticalMetrics:
        return cls(
            n=result.n,
            pearson_r=result.pearson_r,
            pearson_p=result.pearson_p,
            spearman_rho=result.spearman_rho,
            mutual_information=result.mutual_information,
            eta=result.eta,
            r_squared=result.r_squared,
            slope=result.slope,
            covariance=result.covariance,
            effect=result.effect,
            direction=result.direction,
            strength=result.strength_label(),
            stability=result.stability,
            q_value=result.q_value,
            method=result.method,
        )

    def to_dict(self) -> dict:
        return asdict(self)

    def to_compact_text(self) -> str:
        bits = [f"n={self.n:,}"]
        if self.pearson_r is not None:
            bits.append(f"r={self.pearson_r:+.3f}")
        if self.spearman_rho is not None:
            bits.append(f"rho={self.spearman_rho:+.3f}")
        if self.eta is not None:
            bits.append(f"eta={self.eta:.3f}")
        if self.mutual_information is not None:
            bits.append(f"MI={self.mutual_information:.3f}")
        if self.stability is not None:
            bits.append(f"stability={self.stability:.2f}")
        return " ".join(bits) + f" dir={self.direction} ({self.strength})"


@dataclass
class EvidenceObject:
    """One analyzed relationship, with everything needed to reconsider it later."""

    # ---- identity -------------------------------------------------------
    evidence_id: str = field(default_factory=lambda: f"ev_{uuid.uuid4().hex[:16]}")
    analysis_id: str = ""
    hypothesis_id: str = ""

    # ---- what was analysed ---------------------------------------------
    dataset_id: str = ""
    dataset_fingerprint: str = ""
    user_question: str = ""
    feature_names: list[str] = field(default_factory=list)
    canonical_expression: str = ""
    expression_hash: str = ""
    transformation_description: str = ""
    units: dict[str, str] = field(default_factory=dict)

    # ---- what was found -------------------------------------------------
    statistical_metrics: StatisticalMetrics = field(default_factory=StatisticalMetrics)
    sample_size: int = 0
    filters: list[str] = field(default_factory=list)
    stratification: str = ""

    # ---- how it was shown ----------------------------------------------
    plot_type: str = ""
    plot_uri: str = ""
    plot_description: str = ""

    # ---- what it means --------------------------------------------------
    textual_summary: str = ""
    vlm_interpretation: dict = field(default_factory=dict)
    explanation_status: ExplanationStatus = ExplanationStatus.UNRESOLVED
    explanation_confidence: float = 0.0
    critic_report: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    # ---- graph of evidence ----------------------------------------------
    parent_evidence_ids: list[str] = field(default_factory=list)
    related_evidence_ids: list[str] = field(default_factory=list)
    supports_ids: list[str] = field(default_factory=list)
    contradicts_ids: list[str] = field(default_factory=list)
    resolved_by_ids: list[str] = field(default_factory=list)

    # ---- bookkeeping ----------------------------------------------------
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    updated_at: str = ""
    tags: list[str] = field(default_factory=list)
    provenance: list[str] = field(default_factory=list)
    semantic_search_text: str = ""
    embedding: list[float] | None = field(default=None, repr=False)

    # ---- derived --------------------------------------------------------
    def __post_init__(self) -> None:
        if not self.semantic_search_text:
            self.semantic_search_text = self.build_search_text()
        if not self.sample_size:
            self.sample_size = self.statistical_metrics.n

    def build_search_text(self) -> str:
        """Dense natural-language rendering used for lexical + vector retrieval.

        Deliberately includes the column names, units, direction, strength,
        status, and the interpretation text.  Retrieval has to be able to find
        "airport pickup -> high fare per mile" from a later query about airport
        fees, and that only works if the words are actually in the document.
        """
        parts: list[str] = []
        if self.feature_names:
            parts.append(" and ".join(self.feature_names))
        if self.canonical_expression:
            parts.append(f"expression {self.canonical_expression}")
        if self.transformation_description:
            parts.append(self.transformation_description)
        m = self.statistical_metrics
        parts.append(f"{m.direction} {m.strength} relationship")
        if self.units:
            parts.append("units " + ", ".join(f"{k} in {v}" for k, v in self.units.items()))
        if self.stratification:
            parts.append(f"stratified by {self.stratification}")
        parts.append(f"status {self.explanation_status.value}")
        if self.textual_summary:
            parts.append(self.textual_summary)
        interp = self.vlm_interpretation or {}
        for key in ("observation", "interpretation"):
            if interp.get(key):
                parts.append(str(interp[key]))
        for key in ("plausible_mechanisms", "confounders", "alternative_explanations"):
            vals = interp.get(key) or []
            if vals:
                parts.append(f"{key.replace('_', ' ')}: " + "; ".join(str(v) for v in vals))
        if self.tags:
            parts.append("tags " + " ".join(self.tags))
        return ". ".join(p for p in parts if p)

    # ---- mutation -------------------------------------------------------
    def mark_resolved(
        self, *, by_ids: list[str], explanation: str, confidence: float
    ) -> EvidenceObject:
        """Promote an unresolved finding once later evidence explains it."""
        self.explanation_status = ExplanationStatus.EXPLAINED
        self.explanation_confidence = confidence
        self.resolved_by_ids = list(dict.fromkeys(self.resolved_by_ids + by_ids))
        self.related_evidence_ids = list(dict.fromkeys(self.related_evidence_ids + by_ids))
        self.textual_summary = explanation
        self.updated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.semantic_search_text = self.build_search_text()
        self.provenance.append(f"resolved at {self.updated_at} by {', '.join(by_ids)}")
        return self

    # ---- serialization --------------------------------------------------
    def to_dict(self, *, include_embedding: bool = False) -> dict:
        d = asdict(self)
        d["explanation_status"] = self.explanation_status.value
        d["statistical_metrics"] = self.statistical_metrics.to_dict()
        if not include_embedding:
            d.pop("embedding", None)
        return d

    @classmethod
    def from_dict(cls, payload: dict) -> EvidenceObject:
        data = dict(payload)
        metrics = data.pop("statistical_metrics", {}) or {}
        status = data.pop("explanation_status", "unresolved")
        known = {f for f in cls.__dataclass_fields__}
        data = {k: v for k, v in data.items() if k in known}
        obj = cls(**data)
        obj.statistical_metrics = StatisticalMetrics(
            **{k: v for k, v in metrics.items() if k in StatisticalMetrics.__dataclass_fields__}
        )
        try:
            obj.explanation_status = ExplanationStatus(status)
        except ValueError:
            obj.explanation_status = ExplanationStatus.UNRESOLVED
        return obj

    def to_compact_text(self, *, max_chars: int = 420) -> str:
        """Token-efficient rendering for LLM context."""
        head = self.canonical_expression or " ~ ".join(self.feature_names[:2])
        line = (
            f"[{self.evidence_id}] {head}: {self.statistical_metrics.to_compact_text()} "
            f"status={self.explanation_status.value}"
        )
        summary = self.textual_summary or (self.vlm_interpretation or {}).get("observation", "")
        if summary:
            line += f" | {summary}"
        if self.stratification:
            line += f" | stratified by {self.stratification}"
        return line[:max_chars]

    def content_hash(self) -> str:
        """Identity of the FINDING (not the record), for dedup across runs."""
        payload = "|".join(
            [
                self.dataset_fingerprint,
                self.expression_hash or self.canonical_expression,
                ",".join(sorted(self.feature_names)),
                self.stratification,
                ",".join(sorted(self.filters)),
            ]
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:24]


def evidence_from_analysis(
    *,
    analysis_id: str,
    hypothesis_id: str,
    dataset_id: str,
    dataset_fingerprint: str,
    question: str,
    result,
    interpretation,
    critic_report,
    plot_artifact=None,
    canonical_expression: str = "",
    expression_hash: str = "",
    transformation_description: str = "",
    units: dict[str, str] | None = None,
    filters: list[str] | None = None,
    stratification: str = "",
    tags: list[str] | None = None,
) -> EvidenceObject:
    """Assemble an EvidenceObject from one completed analysis step."""
    from signal_engine.llm.schemas import ConclusionStatus

    status_map = {
        ConclusionStatus.EXPLAINED: ExplanationStatus.EXPLAINED,
        ConclusionStatus.UNRESOLVED: ExplanationStatus.UNRESOLVED,
        ConclusionStatus.CONTRADICTED: ExplanationStatus.CONTRADICTED,
        ConclusionStatus.MECHANICAL: ExplanationStatus.MECHANICAL,
        ConclusionStatus.INVALID: ExplanationStatus.REJECTED,
    }
    status = status_map.get(interpretation.conclusion_status, ExplanationStatus.UNRESOLVED)

    warnings = list(result.warnings)
    if critic_report is not None:
        warnings.extend(v.message for v in critic_report.violations if v.severity.value != "note")

    evidence = EvidenceObject(
        analysis_id=analysis_id,
        hypothesis_id=hypothesis_id,
        dataset_id=dataset_id,
        dataset_fingerprint=dataset_fingerprint,
        user_question=question,
        feature_names=[result.x_name, result.y_name],
        canonical_expression=canonical_expression,
        expression_hash=expression_hash,
        transformation_description=transformation_description,
        units=units or {},
        statistical_metrics=StatisticalMetrics.from_result(result),
        sample_size=result.n,
        filters=filters or [],
        stratification=stratification,
        plot_type=plot_artifact.plot_type if plot_artifact else "",
        plot_uri=str(plot_artifact.path) if (plot_artifact and plot_artifact.path) else "",
        plot_description=plot_artifact.spec_description if plot_artifact else "",
        textual_summary=interpretation.observation,
        vlm_interpretation=interpretation.model_dump(mode="json"),
        explanation_status=status,
        explanation_confidence=interpretation.confidence,
        critic_report=critic_report.to_dict() if critic_report else {},
        warnings=warnings,
        tags=tags or [],
        provenance=[f"analysis {analysis_id}", f"created {time.strftime('%Y-%m-%dT%H:%M:%S')}"],
    )
    return evidence
