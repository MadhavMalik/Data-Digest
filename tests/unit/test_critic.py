"""Semantic validation.

Spec sections 15.5 and 6.  These are the tests that distinguish this engine
from "paste the data into an LLM": a model can produce fluent prose that
contradicts the arithmetic, and the critic has to catch it deterministically.
"""

from __future__ import annotations

from signal_engine.interpretation.critic import Severity, apply_report, critique
from signal_engine.llm.schemas import ConclusionStatus, Direction, VisualInterpretation
from signal_engine.profiling.column_cards import ColumnCard
from signal_engine.profiling.semantic_types import SemanticType
from signal_engine.profiling.units import IDENTIFIER, MILES, USD
from signal_engine.statistics.correlation import RelationshipResult


def make_result(**kwargs) -> RelationshipResult:
    defaults = {
        "x_name": "trip_distance",
        "y_name": "fare_amount",
        "n": 1_000_000,
        "pearson_r": 0.8,
        "spearman_rho": 0.79,
        "stability": 0.95,
    }
    defaults.update(kwargs)
    return RelationshipResult(**defaults)


def make_card(name: str, semantic_type: SemanticType, unit=USD, **kwargs) -> ColumnCard:
    return ColumnCard(
        name=name,
        physical_dtype="float64",
        semantic_type=semantic_type,
        unit=unit,
        type_confidence=1.0,
        type_rule="test",
        row_count=1_000_000,
        **kwargs,
    )


def make_interpretation(**kwargs) -> VisualInterpretation:
    defaults = {
        "observation": "The two variables move together.",
        "strength": "strong",
        "interpretation": "They are associated.",
        "conclusion_status": ConclusionStatus.EXPLAINED,
        "confidence": 0.8,
    }
    defaults.update(kwargs)
    return VisualInterpretation(**defaults)


class TestDirectionConsistency:
    """The headline check: prose must not invert the sign of the coefficient."""

    def test_positive_described_as_inverse_is_an_error(self):
        result = make_result(pearson_r=0.8)
        interpretation = make_interpretation(
            observation="There is a clear inverse relationship between the two.",
            stated_direction=Direction.NEGATIVE,
        )
        report = critique(interpretation, result)
        assert not report.passed
        errors = [v for v in report.errors if v.check == "direction_mismatch"]
        assert errors, "describing r=+0.8 as inverse must be a hard error"
        assert "+0.800" in errors[0].evidence

    def test_negative_described_as_positive_is_an_error(self):
        result = make_result(pearson_r=-0.8, spearman_rho=-0.79)
        interpretation = make_interpretation(
            observation="As one increases the other increases as well.",
            stated_direction=Direction.POSITIVE,
        )
        report = critique(interpretation, result)
        assert not report.passed
        assert any(v.check == "direction_mismatch" for v in report.errors)

    def test_correct_direction_passes(self):
        result = make_result(pearson_r=0.8)
        interpretation = make_interpretation(
            observation="Fare increases with distance.", stated_direction=Direction.POSITIVE
        )
        report = critique(interpretation, result)
        assert not [v for v in report.errors if v.check == "direction_mismatch"]

    def test_prose_direction_is_detected_without_the_structured_field(self):
        result = make_result(pearson_r=-0.75, spearman_rho=-0.7)
        interpretation = make_interpretation(
            observation="Longer trips show a positive increase in cost per mile.",
            stated_direction=Direction.UNKNOWN,
        )
        report = critique(interpretation, result)
        assert any(v.check == "direction_mismatch" for v in report.errors)

    def test_error_forces_the_status_to_invalid(self):
        result = make_result(pearson_r=0.8)
        interpretation = make_interpretation(stated_direction=Direction.NEGATIVE)
        report = critique(interpretation, result)
        corrected = apply_report(interpretation, report)
        assert corrected.conclusion_status is ConclusionStatus.INVALID
        assert corrected.confidence < interpretation.confidence


class TestIdentifierMisuse:
    def test_correlating_against_a_zone_id_is_an_error(self):
        """Zone IDs are arbitrary labels; a correlation against them is noise."""
        result = make_result(x_name="PULocationID", pearson_r=0.3)
        card = make_card("PULocationID", SemanticType.CATEGORICAL_IDENTIFIER, unit=IDENTIFIER)
        report = critique(make_interpretation(), result, x_card=card)
        assert not report.passed
        violation = next(v for v in report.errors if v.check == "identifier_as_quantity")
        assert "arbitrary" in violation.message
        assert "group means" in violation.suggested_fix

    def test_a_proper_group_comparison_is_accepted(self):
        result = RelationshipResult(
            x_name="PULocationID",
            y_name="fare_amount",
            n=1_000_000,
            eta=0.4,
            method="grouped_eta_squared",
        )
        card = make_card("PULocationID", SemanticType.CATEGORICAL_IDENTIFIER, unit=IDENTIFIER)
        report = critique(make_interpretation(), result, x_card=card)
        assert not [v for v in report.errors if v.check == "identifier_as_quantity"]


class TestCausalLanguage:
    def test_unhedged_causal_claim_is_a_warning(self):
        interpretation = make_interpretation(
            interpretation="Longer distance causes higher fares.",
            stated_direction=Direction.POSITIVE,
        )
        report = critique(interpretation, make_result())
        assert any(v.check == "causal_claim_from_association" for v in report.warnings)

    def test_hedged_causal_language_is_only_a_note(self):
        interpretation = make_interpretation(
            interpretation="The premium may be partly driven by fixed airport fees.",
            stated_direction=Direction.POSITIVE,
        )
        report = critique(interpretation, make_result())
        causal = [v for v in report.violations if "causal" in v.check]
        assert all(v.severity is Severity.NOTE for v in causal)

    def test_associational_language_raises_nothing(self):
        interpretation = make_interpretation(
            interpretation="Fare is associated with distance.", stated_direction=Direction.POSITIVE
        )
        report = critique(interpretation, make_result())
        assert not [v for v in report.violations if "causal" in v.check]

    def test_mechanical_status_may_state_the_definition(self):
        interpretation = make_interpretation(
            interpretation="total_amount is the sum of its components, so fare_amount determines much of it.",
            conclusion_status=ConclusionStatus.MECHANICAL,
            stated_direction=Direction.POSITIVE,
        )
        report = critique(interpretation, make_result())
        assert not [v for v in report.violations if v.check == "causal_claim_from_association"]


class TestDataDictionaryBlocks:
    """The cash-tip trap — the headline domain test from the brief."""

    def test_cash_tip_conclusion_is_blocked(self):
        tip_card = make_card(
            "tip_amount",
            SemanticType.CURRENCY,
            caveats=["Cash tips are not included in this field."],
            blocks_claims=[
                {
                    "involves_columns": ["payment_type"],
                    "forbidden_claim": "cash passengers tip less than card passengers",
                    "reason": "tip_amount only captures credit-card tips.",
                }
            ],
        )
        payment_card = make_card("payment_type", SemanticType.CATEGORICAL)
        result = RelationshipResult(
            x_name="payment_type", y_name="tip_amount", n=1_000_000, eta=0.7,
            method="grouped_eta_squared",
        )
        interpretation = make_interpretation(
            observation="Cash trips record far lower tips than card trips.",
            interpretation="Cash passengers tip less.",
        )
        report = critique(interpretation, result, x_card=payment_card, y_card=tip_card)

        assert not report.passed
        violation = next(
            v for v in report.errors if v.check == "claim_blocked_by_data_dictionary"
        )
        assert "credit-card tips" in violation.message
        assert apply_report(interpretation, report).conclusion_status is ConclusionStatus.INVALID

    def test_block_does_not_fire_without_the_named_column(self):
        tip_card = make_card(
            "tip_amount",
            SemanticType.CURRENCY,
            blocks_claims=[
                {
                    "involves_columns": ["payment_type"],
                    "forbidden_claim": "cash passengers tip less",
                    "reason": "cash tips absent",
                }
            ],
        )
        result = make_result(x_name="trip_distance", y_name="tip_amount")
        distance_card = make_card("trip_distance", SemanticType.CONTINUOUS_MEASUREMENT, unit=MILES)
        report = critique(
            make_interpretation(stated_direction=Direction.POSITIVE),
            result,
            x_card=distance_card,
            y_card=tip_card,
        )
        assert not [v for v in report.errors if v.check == "claim_blocked_by_data_dictionary"]


class TestAccountingIdentities:
    IDENTITIES = [
        {
            "target": "total_amount",
            "components": ["fare_amount", "tip_amount", "tolls_amount"],
            "relation": "sum",
            "note": "total_amount is the sum of its components.",
        }
    ]

    def test_identity_presented_as_discovery_is_flagged(self):
        result = make_result(x_name="fare_amount", y_name="total_amount", pearson_r=0.97)
        interpretation = make_interpretation(
            interpretation="Fare amount strongly predicts the total charged — a key discovery.",
            stated_direction=Direction.POSITIVE,
        )
        report = critique(
            interpretation,
            result,
            accounting_identities=self.IDENTITIES,
            expression_columns={"fare_amount", "total_amount"},
        )
        assert any(v.check == "accounting_identity_presented_as_discovery" for v in report.warnings)
        assert apply_report(interpretation, report).conclusion_status is ConclusionStatus.MECHANICAL

    def test_correctly_labelled_identity_is_only_a_note(self):
        result = make_result(x_name="fare_amount", y_name="total_amount", pearson_r=0.97)
        interpretation = make_interpretation(
            interpretation="This is definitional: fare_amount is a component of total_amount.",
            conclusion_status=ConclusionStatus.MECHANICAL,
            stated_direction=Direction.POSITIVE,
        )
        report = critique(
            interpretation,
            result,
            accounting_identities=self.IDENTITIES,
            expression_columns={"fare_amount", "total_amount"},
        )
        assert report.passed
        assert any(v.check == "accounting_identity_correctly_labelled" for v in report.violations)

    def test_identity_is_detected_from_base_columns_not_the_display_name(self):
        """A composite like `fare_amount + tolls_amount` must still be caught."""
        result = make_result(
            x_name="fare_amount_plus_tolls_amount", y_name="total_amount", pearson_r=0.98
        )
        report = critique(
            make_interpretation(stated_direction=Direction.POSITIVE),
            result,
            accounting_identities=self.IDENTITIES,
            expression_columns={"fare_amount", "tolls_amount", "total_amount"},
        )
        assert any("accounting_identity" in v.check for v in report.violations)


class TestSurprisingResults:
    def test_surprising_result_is_noted_not_deleted(self):
        """World knowledge must never override a measurement."""
        result = make_result(
            x_name="fare_amount", y_name="trip_distance", pearson_r=-0.6, spearman_rho=-0.58
        )
        fare = make_card("fare_amount", SemanticType.CURRENCY)
        distance = make_card("trip_distance", SemanticType.CONTINUOUS_MEASUREMENT, unit=MILES)
        interpretation = make_interpretation(stated_direction=Direction.NEGATIVE)

        report = critique(interpretation, result, x_card=fare, y_card=distance)

        assert report.semantically_surprising
        surprise = next(v for v in report.violations if v.check == "semantically_surprising")
        assert surprise.severity is Severity.NOTE, "a surprise is not an error"
        assert "investigate rather than dismiss" in surprise.message
        # It must NOT be downgraded to invalid.
        assert apply_report(interpretation, report).conclusion_status is not ConclusionStatus.INVALID


class TestStrengthConsistency:
    def test_overstated_strength_is_warned(self):
        result = make_result(pearson_r=0.15, spearman_rho=0.14)
        interpretation = make_interpretation(strength="very strong", stated_direction=Direction.POSITIVE)
        report = critique(interpretation, result)
        assert any(v.check == "strength_overstated" for v in report.warnings)

    def test_understated_strength_is_warned(self):
        result = make_result(pearson_r=0.9, spearman_rho=0.89)
        interpretation = make_interpretation(strength="negligible", stated_direction=Direction.POSITIVE)
        report = critique(interpretation, result)
        assert any(v.check == "strength_understated" for v in report.warnings)

    def test_accurate_strength_passes(self):
        result = make_result(pearson_r=0.85, spearman_rho=0.84)
        interpretation = make_interpretation(strength="very strong", stated_direction=Direction.POSITIVE)
        report = critique(interpretation, result)
        assert not [v for v in report.violations if "strength" in v.check]


class TestDisclosure:
    def test_undisclosed_filters_are_noted(self):
        report = critique(
            make_interpretation(stated_direction=Direction.POSITIVE),
            make_result(),
            filters_applied=["positive_distance", "plausible_speed"],
        )
        notes = [v for v in report.violations if v.check == "filters_not_disclosed"]
        assert notes and notes[0].severity is Severity.NOTE

    def test_disclosed_filters_are_not_noted(self):
        interpretation = make_interpretation(
            observation="Among the subset of trips with positive distance, fare rises with distance.",
            stated_direction=Direction.POSITIVE,
        )
        report = critique(interpretation, make_result(), filters_applied=["positive_distance"])
        assert not [v for v in report.violations if v.check == "filters_not_disclosed"]


class TestReportAggregation:
    def test_confidence_penalty_scales_with_severity(self):
        clean = critique(make_interpretation(stated_direction=Direction.POSITIVE), make_result())
        broken = critique(make_interpretation(stated_direction=Direction.NEGATIVE), make_result())
        assert broken.confidence_penalty > clean.confidence_penalty

    def test_penalty_is_capped(self):
        result = make_result(x_name="PULocationID", pearson_r=-0.9, spearman_rho=-0.9)
        card = make_card("PULocationID", SemanticType.CATEGORICAL_IDENTIFIER, unit=IDENTIFIER)
        interpretation = make_interpretation(
            interpretation="The zone id causes higher fares, a positive increase.",
            strength="negligible",
            stated_direction=Direction.POSITIVE,
        )
        report = critique(interpretation, result, x_card=card)
        assert report.confidence_penalty <= 0.9
        assert apply_report(interpretation, report).confidence >= 0.0

    def test_summary_is_human_readable(self):
        report = critique(make_interpretation(stated_direction=Direction.NEGATIVE), make_result())
        assert "error" in report.summary()
