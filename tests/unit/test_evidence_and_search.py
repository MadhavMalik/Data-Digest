"""Evidence memory, retrieval, and search-budget enforcement.

Spec sections 15.7, 15.8 (unit half), and 10.  The headline test here is
`test_unresolved_finding_is_retrieved_by_later_related_evidence` — the airport
scenario, which is the product's central idea.
"""

from __future__ import annotations

import asyncio

import pytest

from signal_engine.config import SearchBudget
from signal_engine.evidence.base import (
    HashingEmbedder,
    RetrievalQuery,
    bm25_scores,
    cosine_similarity,
    reciprocal_rank_fusion,
    tokenize_for_bm25,
)
from signal_engine.evidence.memory import MemoryEvidenceStore
from signal_engine.evidence.retrieval import EvidenceMemory
from signal_engine.evidence.schemas import (
    EvidenceObject,
    ExplanationStatus,
    StatisticalMetrics,
)
from signal_engine.search.budgets import (
    BudgetTracker,
    ImprovementTracker,
    StopReason,
)
from signal_engine.search.scorer import question_terms, score_candidate
from signal_engine.statistics.correlation import RelationshipResult


def make_evidence(**kwargs) -> EvidenceObject:
    defaults = {
        "dataset_id": "nyc",
        "analysis_id": "an1",
        "feature_names": ["a", "b"],
        "statistical_metrics": StatisticalMetrics(n=1000, pearson_r=0.5, effect=0.5, direction="positive"),
    }
    defaults.update(kwargs)
    return EvidenceObject(**defaults)


def run(coro):
    return asyncio.run(coro)


class TestEvidenceObject:
    def test_search_text_is_built_automatically(self):
        e = make_evidence(feature_names=["airport_fee", "fare_per_mile"], textual_summary="Airport premium")
        assert "airport_fee" in e.semantic_search_text
        assert "Airport premium" in e.semantic_search_text

    def test_search_text_includes_the_interpretation(self):
        e = make_evidence(
            vlm_interpretation={
                "observation": "Airport pickups cost more per mile.",
                "plausible_mechanisms": ["fixed airport surcharge"],
            }
        )
        assert "surcharge" in e.semantic_search_text

    def test_round_trips_through_a_dict(self):
        e = make_evidence(tags=["surprising"], warnings=["w"])
        restored = EvidenceObject.from_dict(e.to_dict())
        assert restored.evidence_id == e.evidence_id
        assert restored.explanation_status is e.explanation_status
        assert restored.statistical_metrics.pearson_r == e.statistical_metrics.pearson_r
        assert restored.tags == ["surprising"]

    def test_unknown_fields_are_ignored_on_load(self):
        payload = make_evidence().to_dict()
        payload["a_field_from_the_future"] = 1
        assert EvidenceObject.from_dict(payload) is not None

    def test_content_hash_identifies_the_finding_not_the_record(self):
        a = make_evidence(dataset_fingerprint="fp", expression_hash="h1")
        b = make_evidence(dataset_fingerprint="fp", expression_hash="h1")
        assert a.evidence_id != b.evidence_id
        assert a.content_hash() == b.content_hash()

    def test_open_question_statuses(self):
        assert ExplanationStatus.UNRESOLVED.is_open_question
        assert ExplanationStatus.NEEDS_MORE_EVIDENCE.is_open_question
        assert not ExplanationStatus.EXPLAINED.is_open_question

    def test_mark_resolved_records_what_resolved_it(self):
        e = make_evidence(explanation_status=ExplanationStatus.UNRESOLVED)
        e.mark_resolved(by_ids=["ev_x"], explanation="Explained by fees.", confidence=0.8)
        assert e.explanation_status is ExplanationStatus.EXPLAINED
        assert "ev_x" in e.resolved_by_ids
        assert e.updated_at
        assert any("resolved" in p for p in e.provenance)

    def test_compact_text_is_bounded(self):
        e = make_evidence(textual_summary="x" * 5000)
        assert len(e.to_compact_text(max_chars=200)) <= 200


class TestHybridRetrieval:
    def test_bm25_ranks_matching_documents_first(self):
        docs = {
            "d1": tokenize_for_bm25("airport pickup fee surcharge"),
            "d2": tokenize_for_bm25("trip distance and duration"),
        }
        scores = bm25_scores(tokenize_for_bm25("airport fee"), docs)
        assert scores.get("d1", 0) > scores.get("d2", 0)

    def test_embedder_is_deterministic_and_normalized(self):
        e = HashingEmbedder(dims=128)
        a, b = e.embed("airport fee surcharge"), e.embed("airport fee surcharge")
        assert a == b
        assert cosine_similarity(a, b) == pytest.approx(1.0, abs=1e-9)

    def test_related_text_scores_higher_than_unrelated(self):
        e = HashingEmbedder(dims=384)
        query = e.embed("airport pickup surcharge fee")
        related = e.embed("the airport fee is a fixed surcharge on pickups")
        unrelated = e.embed("passenger count distribution by vendor")
        assert cosine_similarity(query, related) > cosine_similarity(query, unrelated)

    def test_subtoken_matching_links_related_column_names(self):
        e = HashingEmbedder(dims=384)
        assert cosine_similarity(e.embed("fare_per_mile"), e.embed("fare_amount")) > 0

    def test_rrf_rewards_appearing_in_both_streams(self):
        scores = reciprocal_rank_fusion([["a", "b", "c"], ["c", "a", "d"]])
        assert scores["a"] > scores["b"]
        assert scores["c"] > scores["d"]

    def test_rrf_weights_are_applied(self):
        equal = reciprocal_rank_fusion([["a"], ["b"]], weights=[1.0, 1.0])
        assert equal["a"] == equal["b"]
        skewed = reciprocal_rank_fusion([["a"], ["b"]], weights=[2.0, 1.0])
        assert skewed["a"] > skewed["b"]


class TestMemoryEvidenceStore:
    def test_put_and_get(self):
        store = MemoryEvidenceStore()
        e = make_evidence()
        run(store.put(e))
        assert run(store.get(e.evidence_id)).evidence_id == e.evidence_id

    def test_search_finds_by_feature_name(self):
        store = MemoryEvidenceStore()
        run(store.put(make_evidence(feature_names=["airport_fee", "fare_per_mile"])))
        run(store.put(make_evidence(feature_names=["passenger_count", "vendor"])))
        results = run(store.search(RetrievalQuery(feature_names=["airport_fee"], limit=5)))
        assert results
        assert "airport_fee" in results[0].evidence.feature_names

    def test_status_filter(self):
        store = MemoryEvidenceStore()
        run(store.put(make_evidence(
            explanation_status=ExplanationStatus.UNRESOLVED, feature_names=["airport_fee"]
        )))
        run(store.put(make_evidence(
            explanation_status=ExplanationStatus.EXPLAINED, feature_names=["trip_distance"]
        )))
        results = run(store.search(RetrievalQuery(
            text="airport trip", statuses=[ExplanationStatus.UNRESOLVED], limit=10
        )))
        assert len(results) == 1
        assert results[0].evidence.explanation_status is ExplanationStatus.UNRESOLVED

    def test_filter_only_query_returns_the_filtered_set(self):
        """"Show me every open question" has no text to rank on."""
        store = MemoryEvidenceStore()
        run(store.put(make_evidence(
            explanation_status=ExplanationStatus.UNRESOLVED, feature_names=["airport_fee"]
        )))
        run(store.put(make_evidence(
            explanation_status=ExplanationStatus.EXPLAINED, feature_names=["trip_distance"]
        )))
        results = run(store.search(RetrievalQuery(
            statuses=[ExplanationStatus.UNRESOLVED], limit=10
        )))
        assert len(results) == 1
        assert results[0].evidence.explanation_status is ExplanationStatus.UNRESOLVED

    def test_dataset_filter_isolates_datasets(self):
        store = MemoryEvidenceStore()
        run(store.put(make_evidence(dataset_id="a", feature_names=["shared"])))
        run(store.put(make_evidence(dataset_id="b", feature_names=["shared"])))
        results = run(store.search(RetrievalQuery(feature_names=["shared"], dataset_id="a", limit=10)))
        assert all(r.evidence.dataset_id == "a" for r in results)

    def test_exclusion_list_is_honoured(self):
        store = MemoryEvidenceStore()
        e = make_evidence(feature_names=["x"])
        run(store.put(e))
        results = run(
            store.search(RetrievalQuery(feature_names=["x"], exclude_ids=[e.evidence_id], limit=5))
        )
        assert not results

    def test_min_effect_filter(self):
        store = MemoryEvidenceStore()
        run(store.put(make_evidence(
            feature_names=["weak"], statistical_metrics=StatisticalMetrics(effect=0.05)
        )))
        run(store.put(make_evidence(
            feature_names=["strong"], statistical_metrics=StatisticalMetrics(effect=0.9)
        )))
        results = run(store.search(RetrievalQuery(text="weak strong", min_effect=0.5, limit=10)))
        assert len(results) == 1

    def test_status_update_reindexes_search_text(self):
        store = MemoryEvidenceStore()
        e = make_evidence(explanation_status=ExplanationStatus.UNRESOLVED)
        run(store.put(e))
        run(store.update_status(
            e.evidence_id, ExplanationStatus.EXPLAINED,
            explanation="Explained by tolls.", confidence=0.9, resolved_by=["other"],
        ))
        updated = run(store.get(e.evidence_id))
        assert updated.explanation_status is ExplanationStatus.EXPLAINED
        assert "tolls" in updated.semantic_search_text

    def test_jsonl_persistence_round_trips(self, tmp_path):
        path = tmp_path / "evidence.jsonl"
        store = MemoryEvidenceStore(persist_path=path)
        e = make_evidence(feature_names=["airport_fee"])
        run(store.put(e))

        reloaded = MemoryEvidenceStore(persist_path=path)
        assert run(reloaded.count()) == 1
        assert run(reloaded.get(e.evidence_id)) is not None

    def test_corrupt_persistence_line_is_skipped(self, tmp_path):
        path = tmp_path / "evidence.jsonl"
        store = MemoryEvidenceStore(persist_path=path)
        run(store.put(make_evidence()))
        with path.open("a") as fh:
            fh.write("{not valid json\n")
        assert run(MemoryEvidenceStore(persist_path=path).count()) == 1


class TestEvidenceMemoryProduct:
    """The airport scenario — the product's central idea."""

    def test_unresolved_finding_is_retrieved_by_later_related_evidence(self):
        store = MemoryEvidenceStore()
        memory = EvidenceMemory(store=store)

        # 1. A robust but unexplainable observation is STORED, not discarded.
        unresolved = make_evidence(
            feature_names=["is_airport_pickup", "fare_per_mile"],
            textual_summary="Airport pickups show an unusually high fare per mile.",
            explanation_status=ExplanationStatus.UNRESOLVED,
            statistical_metrics=StatisticalMetrics(n=50_000, pearson_r=0.45, effect=0.45, direction="positive"),
        )
        run(memory.remember(unresolved))

        # 2. Unrelated evidence accumulates.
        run(memory.remember(make_evidence(
            feature_names=["passenger_count", "vendor_id"],
            textual_summary="Passenger count barely varies by vendor.",
        )))

        # 3. Later, a related finding arrives.
        new = make_evidence(
            feature_names=["is_airport_pickup", "airport_fee"],
            textual_summary="Airport pickups carry a fixed airport surcharge.",
        )

        open_questions = run(memory.open_questions(
            feature_names=new.feature_names,
            text=new.semantic_search_text,
            exclude_ids=[new.evidence_id],
        ))

        ids = [r.evidence.evidence_id for r in open_questions]
        assert unresolved.evidence_id in ids, (
            "the earlier unexplained airport finding must resurface when related "
            "evidence arrives -- this is the whole point of evidence memory"
        )
        assert all(r.evidence.explanation_status.is_open_question for r in open_questions)

    def test_write_failure_is_queued_not_lost(self):
        class BrokenStore(MemoryEvidenceStore):
            async def put(self, evidence):
                raise RuntimeError("elasticsearch is down")

        memory = EvidenceMemory(store=BrokenStore())
        evidence_id = run(memory.remember(make_evidence()))
        assert evidence_id
        assert len(memory.pending) == 1
        assert memory.stats.write_failures == 1
        assert memory.errors, "a degradation must be recorded, never silent"

    def test_search_failure_returns_empty_not_an_exception(self):
        class BrokenStore(MemoryEvidenceStore):
            async def search(self, query):
                raise RuntimeError("connection reset")

        memory = EvidenceMemory(store=BrokenStore())
        assert run(memory.search(RetrievalQuery(text="anything"))) == []
        assert memory.errors

    def test_telemetry_is_reported(self):
        memory = EvidenceMemory(store=MemoryEvidenceStore())
        run(memory.remember(make_evidence()))
        telemetry = memory.telemetry()
        assert telemetry["stored"] == 1
        assert telemetry["store"] == "memory"


class TestBudgets:
    """15.7 — no infinite loops, and every stop is attributable."""

    def test_wall_clock_budget_stops_the_search(self):
        tracker = BudgetTracker(budget=SearchBudget(wall_clock_seconds=0))
        assert tracker.check() is StopReason.WALL_CLOCK

    def test_test_budget_stops_the_search(self):
        tracker = BudgetTracker(budget=SearchBudget(max_tests=10))
        tracker.spend_tests(10)
        assert tracker.check() is StopReason.TEST_BUDGET

    def test_candidate_budget_stops_the_search(self):
        tracker = BudgetTracker(budget=SearchBudget(max_candidates=5))
        tracker.spend_candidates(5)
        assert tracker.check() is StopReason.CANDIDATE_BUDGET

    def test_llm_budget_blocks_further_calls(self):
        tracker = BudgetTracker(budget=SearchBudget(max_llm_calls=2))
        assert tracker.may_call_llm()
        tracker.spend_llm(2)
        assert not tracker.may_call_llm()
        assert StopReason.LLM_BUDGET in tracker.exhausted

    def test_vlm_budget_is_tracked_separately(self):
        tracker = BudgetTracker(budget=SearchBudget(max_llm_calls=10, max_vlm_calls=1))
        tracker.spend_vlm(1)
        assert not tracker.may_call_vlm()
        assert tracker.may_call_llm()

    def test_cancellation_stops_everything(self):
        tracker = BudgetTracker()
        tracker.cancel()
        assert tracker.check() is StopReason.CANCELLED
        assert not tracker.may_call_llm()
        assert not tracker.may_test()

    def test_exhaustion_reasons_are_recorded_once(self):
        tracker = BudgetTracker(budget=SearchBudget(max_tests=1))
        tracker.spend_tests(1)
        tracker.spend_tests(1)
        assert tracker.exhausted.count(StopReason.TEST_BUDGET) == 1

    def test_report_includes_the_limits(self):
        report = BudgetTracker(budget=SearchBudget(max_tests=99)).to_dict()
        assert report["limits"]["max_tests"] == 99


class TestMarginalImprovement:
    def test_improving_scores_reset_the_counter(self):
        tracker = ImprovementTracker(min_improvement=0.05, patience=3)
        for score in (0.1, 0.3, 0.5):
            assert tracker.observe(score)
        assert not tracker.should_stop

    def test_stagnation_stops_the_branch(self):
        tracker = ImprovementTracker(min_improvement=0.05, patience=3)
        tracker.observe(0.5)
        for _ in range(3):
            assert not tracker.observe(0.51)
        assert tracker.should_stop, "no improvement for `patience` rounds must stop the branch"

    def test_tiny_improvements_do_not_count(self):
        tracker = ImprovementTracker(min_improvement=0.1, patience=2)
        tracker.observe(0.5)
        assert not tracker.observe(0.55)

    def test_best_score_never_decreases(self):
        tracker = ImprovementTracker(min_improvement=0.01, patience=5)
        tracker.observe(0.9)
        tracker.observe(0.1)
        assert tracker.best_score == 0.9


class TestScoring:
    def _result(self, **kwargs) -> RelationshipResult:
        defaults = {"x_name": "a", "y_name": "b", "n": 10000, "pearson_r": 0.8, "stability": 0.9}
        defaults.update(kwargs)
        return RelationshipResult(**defaults)

    def test_stronger_effect_scores_higher(self):
        strong = score_candidate(self._result(pearson_r=0.9))
        weak = score_candidate(self._result(pearson_r=0.15))
        assert strong.total > weak.total

    def test_unknown_stability_does_not_zero_the_score(self):
        """A pure product would annihilate any unmeasured factor."""
        score = score_candidate(self._result(stability=None))
        assert score.total > 0, "an unmeasured factor must not make a candidate unrankable"

    def test_unstable_relationships_are_penalized(self):
        stable = score_candidate(self._result(stability=0.95))
        unstable = score_candidate(self._result(stability=0.05))
        assert stable.total > unstable.total

    def test_relevance_to_the_question_raises_the_score(self):
        terms = question_terms("What affects the fare passengers pay?")
        relevant = score_candidate(self._result(), question_terms=terms, feature_names=["fare_amount"])
        irrelevant = score_candidate(self._result(), question_terms=terms, feature_names=["vendor_id"])
        assert relevant.total > irrelevant.total

    def test_already_seen_expressions_lose_novelty(self):
        fresh = score_candidate(self._result(), expression_hash="h1", seen_expression_hashes=set())
        seen = score_candidate(self._result(), expression_hash="h1", seen_expression_hashes={"h1"})
        assert seen.total < fresh.total

    def test_mechanical_relationships_are_down_weighted(self):
        empirical = score_candidate(self._result(), is_mechanical=False)
        mechanical = score_candidate(self._result(), is_mechanical=True)
        assert mechanical.total < empirical.total
        assert any("mechanical" in n for n in mechanical.notes)

    def test_deeper_transformations_are_down_weighted(self):
        shallow = score_candidate(self._result(), depth=0)
        deep = score_candidate(self._result(), depth=4)
        assert deep.total < shallow.total

    def test_high_cost_never_produces_infinity(self):
        """The formula must not divide by a near-zero cost."""
        for cost in (0.0, 1e-12, 1.0, 1e6):
            score = score_candidate(self._result(), estimated_cost=cost)
            assert 0.0 <= score.total < 10.0

    def test_skipped_results_score_zero_strength(self):
        score = score_candidate(self._result(skipped_reason="constant column"))
        assert score.strength == 0.0
        assert any("skipped" in n for n in score.notes)

    def test_score_is_explainable(self):
        assert "strength" in score_candidate(self._result()).explain()

    def test_question_terms_drop_stopwords(self):
        terms = question_terms("What factors are associated with the amount passengers pay?")
        assert "passengers" in terms
        assert "the" not in terms
        assert "what" not in terms
