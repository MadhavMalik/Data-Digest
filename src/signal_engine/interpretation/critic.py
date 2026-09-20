"""The semantic critic.

This runs AFTER the model interprets a graph and BEFORE the interpretation is
stored as evidence.  It is deterministic: no model call, no judgement about
whether an idea is interesting — only whether a stated claim is consistent with
the numbers, the units, the dataset's documented semantics, and the rules of
inference.

The checks:

  1. DIRECTION      claimed direction vs the sign of the coefficient
  2. STRENGTH       claimed strength vs the measured effect size
  3. IDENTIFIER     did it treat an arbitrary code as a magnitude?
  4. CAUSATION      causal verbs attached to associational evidence
  5. BLOCKED CLAIM  a claim the data dictionary says the data cannot support
  6. MECHANICAL     an accounting identity presented as a discovery
  7. LEAKAGE        the "predictor" is a component of the target
  8. SAMPLING       filters that shaped the result went unmentioned

One rule governs the whole module and is worth stating plainly:

    WORLD KNOWLEDGE NEVER OVERRIDES THE DATA.

If a relationship is surprising, the critic flags it as
`semantically_surprising` for investigation.  It does not delete it, downgrade
it, or "correct" it toward what we expected.  Surprising empirical results are
the entire point of the exercise; the failure mode we guard against is a model
narrating the numbers wrongly, not the numbers being inconvenient.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

from signal_engine.llm.schemas import ConclusionStatus, Direction, VisualInterpretation
from signal_engine.profiling.column_cards import ColumnCard
from signal_engine.profiling.semantic_types import SemanticType
from signal_engine.statistics.correlation import RelationshipResult


class Severity(str, Enum):
    ERROR = "error"      # the claim contradicts the evidence; must not stand
    WARNING = "warning"  # the claim overreaches or omits a limit
    NOTE = "note"        # worth recording, not a defect

    @property
    def rank(self) -> int:
        return {"error": 3, "warning": 2, "note": 1}[self.value]


@dataclass
class Violation:
    check: str
    severity: Severity
    message: str
    evidence: str = ""
    suggested_fix: str = ""

    def to_dict(self) -> dict:
        return {
            "check": self.check,
            "severity": self.severity.value,
            "message": self.message,
            "evidence": self.evidence,
            "suggested_fix": self.suggested_fix,
        }


@dataclass
class CriticReport:
    violations: list[Violation] = field(default_factory=list)
    corrected_status: ConclusionStatus | None = None
    confidence_penalty: float = 0.0
    semantically_surprising: bool = False

    @property
    def passed(self) -> bool:
        return not any(v.severity is Severity.ERROR for v in self.violations)

    @property
    def errors(self) -> list[Violation]:
        return [v for v in self.violations if v.severity is Severity.ERROR]

    @property
    def warnings(self) -> list[Violation]:
        return [v for v in self.violations if v.severity is Severity.WARNING]

    def add(self, *args, **kwargs) -> None:
        self.violations.append(Violation(*args, **kwargs))

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "violations": [v.to_dict() for v in self.violations],
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "corrected_status": self.corrected_status.value if self.corrected_status else None,
            "confidence_penalty": round(self.confidence_penalty, 3),
            "semantically_surprising": self.semantically_surprising,
        }

    def summary(self) -> str:
        if not self.violations:
            return "critic: no issues"
        parts = [f"{len(self.errors)} error(s), {len(self.warnings)} warning(s)"]
        for v in sorted(self.violations, key=lambda x: -x.severity.rank)[:4]:
            parts.append(f"[{v.severity.value}] {v.message}")
        return "critic: " + "; ".join(parts)


# ---------------------------------------------------------------------------
# Language detectors
# ---------------------------------------------------------------------------

_CAUSAL_VERBS = re.compile(
    r"\b(causes?|caused|causing|drives?|driven|driving|leads? to|"
    r"results? in|produces?|makes? .{0,20}(higher|lower|increase|decrease)|"
    r"due to|because of|as a result of|the reason (for|why)|"
    r"increases? the|decreases? the|determines?)\b",
    re.I,
)

# Hedges that legitimize a mechanism sentence.
_HEDGES = re.compile(
    r"\b(may|might|could|appears?|suggests?|consistent with|possibly|likely|"
    r"plausibl\w+|hypothes\w+|associated with|correlat\w+|partly explained|"
    r"potential\w*|seems?)\b",
    re.I,
)

_POSITIVE_WORDS = re.compile(
    r"\b(positive(ly)?|increas\w+|rises?|rising|grows?|higher .{0,25}(higher|more|greater)|"
    r"direct(ly)? (proportional|related)|more .{0,25}more)\b",
    re.I,
)
_NEGATIVE_WORDS = re.compile(
    r"\b(negative(ly)?|invers(e|ely)|decreas\w+|declin\w+|falls?|falling|drops?|"
    r"reduc\w+|lower .{0,25}(higher|more)|higher .{0,25}(lower|less|fewer)|"
    r"anti-?correlat\w+)\b",
    re.I,
)

_STRENGTH_WORDS = {
    "very strong": 0.8,
    "strong": 0.5,
    "moderate": 0.3,
    "weak": 0.1,
    "negligible": 0.0,
}


def critique(
    interpretation: VisualInterpretation,
    result: RelationshipResult,
    *,
    x_card: ColumnCard | None = None,
    y_card: ColumnCard | None = None,
    accounting_identities: list[dict] | None = None,
    filters_applied: list[str] | None = None,
    expression_columns: set[str] | None = None,
) -> CriticReport:
    """Validate an interpretation against the evidence that produced it."""
    report = CriticReport()
    text = " ".join(
        [
            interpretation.observation,
            interpretation.interpretation,
            " ".join(interpretation.plausible_mechanisms),
        ]
    ).strip()

    _check_direction(report, interpretation, result, text)
    _check_strength(report, interpretation, result)
    _check_identifier_treatment(report, result, x_card, y_card)
    _check_causal_language(report, interpretation, result, text)
    _check_blocked_claims(report, text, x_card, y_card, result)
    _check_mechanical(report, interpretation, result, x_card, y_card, accounting_identities, expression_columns)
    _check_sampling_disclosure(report, interpretation, result, filters_applied)
    _check_surprise(report, result, x_card, y_card)

    for v in report.violations:
        report.confidence_penalty += {Severity.ERROR: 0.4, Severity.WARNING: 0.15, Severity.NOTE: 0.0}[
            v.severity
        ]
    report.confidence_penalty = min(0.9, report.confidence_penalty)
    return report


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def _check_direction(report, interp, result, text: str) -> None:
    """The headline check: a claim must not invert the sign of the coefficient."""
    actual = result.direction
    if actual in {"undetermined", "none"}:
        return

    # Prefer the model's explicit structured field; fall back to prose.
    stated = None
    if interp.stated_direction in (Direction.POSITIVE, Direction.NEGATIVE):
        stated = interp.stated_direction.value
    else:
        has_pos = bool(_POSITIVE_WORDS.search(text))
        has_neg = bool(_NEGATIVE_WORDS.search(text))
        if has_pos != has_neg:
            stated = "positive" if has_pos else "negative"

    if stated and stated != actual:
        coefficient = result.pearson_r if result.pearson_r is not None else result.spearman_rho
        report.add(
            "direction_mismatch",
            Severity.ERROR,
            f"the interpretation describes a {stated} relationship but the coefficient is "
            f"{coefficient:+.3f}, which is {actual}",
            evidence=f"r={coefficient:+.4f}, n={result.n:,}",
            suggested_fix=f"restate the relationship as {actual}",
        )
        report.corrected_status = ConclusionStatus.INVALID


def _check_strength(report, interp, result) -> None:
    claimed = (interp.strength or "").strip().lower()
    if claimed not in _STRENGTH_WORDS:
        return
    floor = _STRENGTH_WORDS[claimed]
    actual = result.effect
    if floor >= 0.5 and actual < 0.3:
        report.add(
            "strength_overstated",
            Severity.WARNING,
            f"the interpretation calls this '{claimed}' but the measured effect is "
            f"{actual:.3f} ({result.strength_label()})",
            evidence=f"effect={actual:.4f}",
            suggested_fix=f"describe it as {result.strength_label()}",
        )
    elif floor == 0.0 and actual >= 0.5:
        report.add(
            "strength_understated",
            Severity.WARNING,
            f"the interpretation calls this '{claimed}' but the measured effect is "
            f"{actual:.3f} ({result.strength_label()})",
            evidence=f"effect={actual:.4f}",
        )


def _check_identifier_treatment(report, result, x_card, y_card) -> None:
    """Catch a correlation computed against arbitrary integer labels."""
    for card in (x_card, y_card):
        if card is None:
            continue
        if card.semantic_type is not SemanticType.CATEGORICAL_IDENTIFIER:
            continue
        if result.method in {"grouped_eta_squared", "cramers_v"}:
            continue  # handled correctly
        if result.pearson_r is not None or result.spearman_rho is not None:
            report.add(
                "identifier_as_quantity",
                Severity.ERROR,
                f"{card.name} holds arbitrary identifier codes, so a correlation coefficient "
                f"against it has no meaning — its numeric ordering is not a magnitude",
                evidence=f"{card.name} semantic_type={card.semantic_type.value}",
                suggested_fix="compare group means across levels instead of correlating",
            )
            report.corrected_status = ConclusionStatus.INVALID


def _check_causal_language(report, interp, result, text: str) -> None:
    """Associational evidence must not be narrated causally."""
    if interp.conclusion_status is ConclusionStatus.MECHANICAL:
        return  # a definitional relationship may be stated as such

    match = _CAUSAL_VERBS.search(text)
    if not match:
        return

    # A causal verb inside a hedged sentence is acceptable ("may be driven by").
    sentence = _sentence_containing(text, match.start())
    if _HEDGES.search(sentence):
        report.add(
            "hedged_causal_language",
            Severity.NOTE,
            f"causal phrasing '{match.group(0)}' appears, but hedged as a hypothesis",
            evidence=sentence[:180],
        )
        return

    report.add(
        "causal_claim_from_association",
        Severity.WARNING,
        f"'{match.group(0)}' asserts causation, but the evidence is an observational "
        f"association with no identification strategy",
        evidence=sentence[:180],
        suggested_fix="rephrase as 'is associated with' or 'may be partly explained by'",
    )


def _check_blocked_claims(report, text: str, x_card, y_card, result) -> None:
    """Enforce the data dictionary's explicit 'this field cannot support X' rules.

    This is what catches the cash-tip trap: the TLC dictionary states that
    cash tips are absent from `tip_amount`, so any comparison of tipping
    behaviour across payment types is measuring the recording mechanism, not
    the behaviour.
    """
    involved = {c.name for c in (x_card, y_card) if c is not None}
    involved |= {result.x_name, result.y_name}

    for card in (x_card, y_card):
        if card is None:
            continue
        for rule in card.blocks_claims:
            required = set(rule.get("involves_columns", []))
            if required and not required & involved:
                continue
            report.add(
                "claim_blocked_by_data_dictionary",
                Severity.ERROR,
                f"this comparison cannot be supported by {card.name}: {rule.get('reason', '')}",
                evidence=f"forbidden claim: {rule.get('forbidden_claim', '')}",
                suggested_fix=(
                    "state that the field does not capture the quantity being compared, "
                    "rather than reporting the apparent difference as behaviour"
                ),
            )
            report.corrected_status = ConclusionStatus.INVALID


def _check_mechanical(
    report, interp, result, x_card, y_card, identities, expression_columns
) -> None:
    """Flag accounting identities and component-of-target leakage."""
    identities = identities or []
    names = {result.x_name, result.y_name}
    if expression_columns:
        names |= expression_columns
    for card in (x_card, y_card):
        if card is not None:
            names.add(card.name)

    lower = {n.lower() for n in names}

    for ident in identities:
        target = str(ident.get("target", "")).lower()
        comps = {str(c).lower() for c in ident.get("components", [])}
        if target in lower and (comps & lower):
            overlap = sorted(comps & lower)
            if interp.conclusion_status is not ConclusionStatus.MECHANICAL:
                report.add(
                    "accounting_identity_presented_as_discovery",
                    Severity.WARNING,
                    f"{ident.get('target')} is defined as the {ident.get('relation', 'sum')} of "
                    f"{', '.join(ident.get('components', []))}; its relationship with "
                    f"{', '.join(overlap)} is mechanical, not an empirical finding",
                    evidence=ident.get("note", ""),
                    suggested_fix="label this relationship as definitional",
                )
                report.corrected_status = ConclusionStatus.MECHANICAL
            else:
                report.add(
                    "accounting_identity_correctly_labelled",
                    Severity.NOTE,
                    "correctly identified as a definitional relationship",
                )
            return

    # Ratio-against-its-own-denominator leakage (fare_per_mile vs trip_distance).
    for a, b in ((result.x_name, result.y_name), (result.y_name, result.x_name)):
        if b and a and b.lower() in a.lower() and a != b and result.effect > 0.3:
            report.add(
                "derived_feature_shares_a_term",
                Severity.WARNING,
                f"'{a}' is derived from '{b}', so part of their association is structural "
                f"rather than empirical",
                evidence=f"effect={result.effect:.3f}",
                suggested_fix="compare against an independent variable, or state the dependency",
            )
            return


def _check_sampling_disclosure(report, interp, result, filters_applied) -> None:
    if not filters_applied:
        return
    text = " ".join([interp.observation, interp.interpretation]).lower()
    mentions = any(w in text for w in ("filter", "exclud", "subset", "restrict", "only trips", "sample"))
    if not mentions:
        report.add(
            "filters_not_disclosed",
            Severity.NOTE,
            f"the interpretation does not mention that {len(filters_applied)} row filters shaped "
            f"the analysed set",
            evidence=", ".join(filters_applied[:5]),
        )


def _check_surprise(report, result, x_card, y_card) -> None:
    """Record a surprising-but-real result WITHOUT downgrading it.

    Deliberately a NOTE, never an error: the engine's job is to find
    unexpected structure, and 'this contradicts my prior' is not evidence
    against a measurement.
    """
    if not result.is_meaningful or x_card is None or y_card is None:
        return

    surprising = False
    reason = ""

    # A strong NEGATIVE association between two quantities that normally move
    # together (a cost and the amount of service delivered) is worth a look.
    cost_like = {SemanticType.CURRENCY}
    amount_like = {SemanticType.CONTINUOUS_MEASUREMENT, SemanticType.DURATION, SemanticType.COUNT}
    pair = {x_card.semantic_type, y_card.semantic_type}
    if pair & cost_like and pair & amount_like and result.direction == "negative" and result.effect > 0.3:
        surprising = True
        reason = (
            f"a cost quantity ({'/'.join(sorted(c.name for c in (x_card, y_card)))}) moves "
            f"opposite to a quantity of service delivered"
        )

    if surprising:
        report.semantically_surprising = True
        report.add(
            "semantically_surprising",
            Severity.NOTE,
            f"statistically observed but semantically surprising; investigate rather than dismiss: {reason}",
            evidence=f"r={result.pearson_r}, n={result.n:,}, stability={result.stability}",
            suggested_fix="check for a confounder, a subgroup reversal, or a data-collection artifact",
        )


def _sentence_containing(text: str, index: int) -> str:
    start = max(text.rfind(".", 0, index), text.rfind("\n", 0, index)) + 1
    end = text.find(".", index)
    end = len(text) if end == -1 else end + 1
    return text[start:end].strip()


def apply_report(
    interpretation: VisualInterpretation, report: CriticReport
) -> VisualInterpretation:
    """Return the interpretation with the critic's corrections applied.

    Errors force the status down and cut confidence.  Nothing is deleted: the
    original text stays, and the violations travel with it into evidence, so
    the failure is visible in the record rather than quietly patched.
    """
    updates: dict = {}
    if report.corrected_status is not None:
        updates["conclusion_status"] = report.corrected_status
    if report.confidence_penalty:
        updates["confidence"] = max(0.0, interpretation.confidence - report.confidence_penalty)
    if not updates:
        return interpretation
    return interpretation.model_copy(update=updates)
