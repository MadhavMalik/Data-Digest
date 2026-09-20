"""End-to-end pipeline tests.

Spec section 15.10.  The contract being proven: the FULL loop runs to
completion with NO credentials configured, producing real statistics, real
plots, and honest evidence — and it says out loud what it degraded.
"""

from __future__ import annotations

import asyncio

import numpy as np
import polars as pl
import pytest

from signal_engine.evidence.memory import MemoryEvidenceStore
from signal_engine.evidence.retrieval import EvidenceMemory
from signal_engine.evidence.schemas import ExplanationStatus
from signal_engine.ingestion.base import register_local_dataset
from signal_engine.ingestion.tlc import (
    YELLOW_ACCOUNTING_IDENTITIES,
    YELLOW_DATASET_NOTES,
    YELLOW_TAXI_DICTIONARY,
)
from signal_engine.reporting import write_evaluation_report, write_run_report
from signal_engine.search.budgets import StopReason
from signal_engine.search.orchestrator import AnalysisRequest, SignalEngine
from tests.conftest import requires_tlc


def run(coro):
    return asyncio.run(coro)


@pytest.fixture
def engine(offline_settings):
    memory = EvidenceMemory(store=MemoryEvidenceStore())
    return SignalEngine(offline_settings, memory=memory)


class TestSyntheticEndToEnd:
    """Runs on a small synthetic file, so it is fast and always available."""

    @pytest.fixture(scope="class")
    def small_dataset(self, tmp_path_factory):
        rng = np.random.default_rng(99)
        n = 40_000
        distance = np.abs(rng.normal(3.0, 2.0, n)) + 0.2
        duration = distance * 240 + rng.normal(0, 120, n) + 200
        fare = 3.0 + 2.6 * distance + 0.004 * duration + rng.normal(0, 1.5, n)
        tip = np.where(rng.random(n) < 0.6, fare * rng.uniform(0.1, 0.3, n), 0.0)
        total = fare + tip + 1.0

        path = tmp_path_factory.mktemp("e2e") / "trips.parquet"
        pl.DataFrame(
            {
                "trip_distance": distance,
                "trip_duration_seconds": duration,
                "fare_amount": fare,
                "tip_amount": tip,
                "total_amount": total,
                "payment_type": rng.integers(1, 3, n),
                "PULocationID": rng.integers(1, 264, n),
            }
        ).write_parquet(path)
        return path

    def test_full_loop_completes_without_credentials(self, engine, small_dataset):
        handle = register_local_dataset(small_dataset, dataset_id="e2e_synthetic")
        result = run(
            engine.analyze(
                AnalysisRequest(
                    question="What factors are associated with total passenger charges?",
                    dataset_handle=handle,
                    accounting_identities=[
                        {
                            "target": "total_amount",
                            "components": ["fare_amount", "tip_amount"],
                            "relation": "sum",
                        }
                    ],
                    max_rounds=2,
                    max_visualizations=3,
                )
            )
        )
        run(engine.aclose())

        assert result.status == "completed"
        assert result.state.stop_reason is not None
        assert result.state.rounds_completed >= 1

        # Real statistics were computed.
        assert len(result.state.tested) > 0
        assert result.metrics.counters["statistical_tests"] > 0

        # The engine found the relationship that was built into the data.
        findings = {(t.result.x_name, t.result.y_name) for t in result.state.ranked_tests()}
        assert any("trip_distance" in x for x, _ in findings), (
            "the engine must recover the distance->total relationship it was given"
        )

        # Evidence was stored with plots.
        assert result.evidence
        assert any(e.plot_uri for e in result.evidence)

        # Degradations are declared, not hidden.
        assert any("deterministic" in d for d in result.degradations)

    def test_identifier_is_never_correlated(self, engine, small_dataset):
        handle = register_local_dataset(small_dataset, dataset_id="e2e_ids")
        result = run(
            engine.analyze(
                AnalysisRequest(
                    question="What affects the total amount paid?",
                    dataset_handle=handle,
                    max_rounds=1,
                    max_visualizations=2,
                )
            )
        )
        run(engine.aclose())

        for test in result.state.all_tests():
            if test.result.x_name == "PULocationID":
                assert test.result.method in {"grouped_eta_squared", "cramers_v"}, (
                    "a zone identifier must be compared by group, never correlated"
                )

    def test_accounting_identities_are_labelled_mechanical(self, engine, small_dataset):
        handle = register_local_dataset(small_dataset, dataset_id="e2e_mech")
        result = run(
            engine.analyze(
                AnalysisRequest(
                    question="What factors are associated with total passenger charges?",
                    dataset_handle=handle,
                    accounting_identities=[
                        {
                            "target": "total_amount",
                            "components": ["fare_amount", "tip_amount"],
                            "relation": "sum",
                        }
                    ],
                    max_rounds=1,
                    max_visualizations=4,
                )
            )
        )
        run(engine.aclose())

        mechanical = [t for t in result.state.all_tests() if t.is_mechanical]
        assert mechanical, "fare_amount vs total_amount must be recognised as definitional"

    def test_budgets_are_enforced(self, offline_settings, small_dataset):
        offline_settings = offline_settings.__class__(
            **{
                **offline_settings.__dict__,
                "budget": offline_settings.budget.__class__(
                    max_depth=1, beam_width=2, max_candidates=30, max_tests=20,
                    max_llm_calls=1, max_vlm_calls=1, wall_clock_seconds=60,
                    no_improvement_rounds=1,
                ),
            }
        )
        engine = SignalEngine(offline_settings, memory=EvidenceMemory(store=MemoryEvidenceStore()))
        handle = register_local_dataset(small_dataset, dataset_id="e2e_budget")
        result = run(
            engine.analyze(
                AnalysisRequest(question="What drives cost?", dataset_handle=handle, max_rounds=5)
            )
        )
        run(engine.aclose())

        assert result.budget.tests_run <= 20 + 50, "the test budget must bound the work"
        assert result.state.stop_reason is not None
        assert result.metrics.elapsed_seconds < 120

    def test_no_infinite_loop_when_nothing_improves(self, engine, tmp_path):
        """Pure noise: the search must give up, not grind forever."""
        rng = np.random.default_rng(3)
        path = tmp_path / "noise.parquet"
        pl.DataFrame(
            {f"noise_{i}": rng.normal(size=3000) for i in range(6)}
            | {"total_amount": rng.normal(size=3000)}
        ).write_parquet(path)

        handle = register_local_dataset(path, dataset_id="e2e_noise")
        result = run(
            engine.analyze(
                AnalysisRequest(
                    question="What affects the total amount?", dataset_handle=handle, max_rounds=4
                )
            )
        )
        run(engine.aclose())

        assert result.state.stop_reason in {
            StopReason.COMPLETED,
            StopReason.NO_IMPROVEMENT,
            StopReason.NO_CANDIDATES,
        }
        assert not result.state.ranked_tests(), "pure noise must yield no meaningful findings"

    def test_reports_are_written(self, engine, small_dataset, offline_settings):
        handle = register_local_dataset(small_dataset, dataset_id="e2e_report")
        result = run(
            engine.analyze(
                AnalysisRequest(
                    question="What affects the total amount paid?",
                    dataset_handle=handle,
                    max_rounds=1,
                    max_visualizations=2,
                )
            )
        )
        run(engine.aclose())

        run_report = write_run_report(result, offline_settings.paths.artifacts)
        assert run_report.exists()
        text = run_report.read_text()
        assert "Analysis view" in text
        assert "Telemetry" in text
        assert "Findings" in text

        evaluation = write_evaluation_report([result], offline_settings.paths.artifacts)
        assert evaluation.exists()
        assert "Per-finding validation" in evaluation.read_text()


@requires_tlc
@pytest.mark.slow
class TestRealDataEndToEnd:
    def test_nyc_taxi_full_pipeline(self, engine, tlc_handle):
        result = run(
            engine.analyze(
                AnalysisRequest(
                    question="What factors are associated with total passenger charges?",
                    dataset_handle=tlc_handle,
                    dictionary=YELLOW_TAXI_DICTIONARY,
                    accounting_identities=YELLOW_ACCOUNTING_IDENTITIES,
                    dataset_notes=YELLOW_DATASET_NOTES,
                    max_rounds=2,
                    max_visualizations=4,
                )
            )
        )
        run(engine.aclose())

        assert result.status == "completed"
        assert result.profile.dataset.row_count > 3_000_000
        assert result.view_report["rows_after"] > 3_000_000

        # Pruning actually happened.
        assert result.prune_stats.pruned_by_units > 0
        assert result.prune_stats.prune_rate > 0.3

        # Real findings, with real plots.
        assert result.state.ranked_tests()
        assert any(e.plot_uri for e in result.evidence)

        # The engine did NOT reconstruct the target from its own derivatives.
        for test in result.state.ranked_tests():
            if test.result.pearson_r is not None:
                assert abs(test.result.pearson_r) < 0.9999, (
                    f"{test.result.x_name} correlates {test.result.pearson_r} with the target; "
                    "that is an algebraic identity, not a finding"
                )

        # Distance must be among the discovered drivers.
        names = {t.result.x_name for t in result.state.ranked_tests()}
        assert any("distance" in n or "duration" in n for n in names)

    def test_evidence_memory_accumulates_across_runs(self, offline_settings, tlc_handle):
        """Two analyses sharing one memory: the second must see the first."""
        memory = EvidenceMemory(store=MemoryEvidenceStore())
        engine = SignalEngine(offline_settings, memory=memory)

        run(
            engine.analyze(
                AnalysisRequest(
                    question="What affects the fare amount?",
                    dataset_handle=tlc_handle,
                    dictionary=YELLOW_TAXI_DICTIONARY,
                    accounting_identities=YELLOW_ACCOUNTING_IDENTITIES,
                    max_rounds=1,
                    max_visualizations=3,
                )
            )
        )
        stored_after_first = run(memory.store.count())
        assert stored_after_first > 0

        second = run(
            engine.analyze(
                AnalysisRequest(
                    question="What affects the total amount charged?",
                    dataset_handle=tlc_handle,
                    dictionary=YELLOW_TAXI_DICTIONARY,
                    accounting_identities=YELLOW_ACCOUNTING_IDENTITIES,
                    max_rounds=1,
                    max_visualizations=3,
                )
            )
        )
        run(engine.aclose())

        assert run(memory.store.count()) > stored_after_first
        assert second.metrics.counters.get("evidence_retrievals", 0) > 0, (
            "the second analysis must retrieve evidence the first one stored"
        )

    def test_unresolved_findings_are_kept_as_open_questions(self, engine, tlc_handle):
        result = run(
            engine.analyze(
                AnalysisRequest(
                    question="What factors are associated with total passenger charges?",
                    dataset_handle=tlc_handle,
                    dictionary=YELLOW_TAXI_DICTIONARY,
                    accounting_identities=YELLOW_ACCOUNTING_IDENTITIES,
                    max_rounds=1,
                    max_visualizations=4,
                )
            )
        )
        run(engine.aclose())

        statuses = {e.explanation_status for e in result.evidence}
        assert statuses, "evidence must be produced"
        # Without an LLM the interpreter cannot supply mechanisms, so findings
        # are honestly recorded as open questions rather than fabricated.
        assert ExplanationStatus.UNRESOLVED in statuses or ExplanationStatus.MECHANICAL in statuses
