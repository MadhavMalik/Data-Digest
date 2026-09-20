"""The research loop.

    HYPOTHESIZE -> COMPUTE -> OBSERVE -> REFINE -> TRANSFORM -> COMPUTE
    -> VISUALIZE -> INTERPRET -> REMEMBER -> RETRIEVE -> REINTERPRET

Ordering principles, in priority order:

1. The LLM is the LAST expensive operation, never the data processor.  By the
   time a model sees anything, deterministic code has already profiled the
   data, pruned the candidate space on units, screened thousands of candidates,
   and selected a handful worth explaining.

2. Every external dependency is optional.  No LLM, no Elasticsearch, no Brave:
   the run still completes with real statistics, real plots, and honest
   evidence records.  Degradations are recorded, never silent.

3. Nothing is claimed that was not measured.  Skipped stages, hit budgets, and
   failed services all appear in the result.

Concurrency policy: `analyze()` is a coroutine, but the heavy stages are
SYNCHRONOUS CPU work in Polars/NumPy/Matplotlib.  Running those directly on the
event loop would block it for tens of seconds, freezing the API that is meant
to be reporting progress.  Each such stage is therefore dispatched with
`asyncio.to_thread`: Polars and NumPy release the GIL for their vectorized
kernels, so this genuinely overlaps with request handling rather than merely
deferring it.  Network I/O (LLM, Elasticsearch, Brave) stays on the loop where
it belongs.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field

import polars as pl

from signal_engine.config import Settings, get_settings
from signal_engine.evidence.retrieval import EvidenceMemory, build_memory
from signal_engine.evidence.schemas import (
    EvidenceObject,
    evidence_from_analysis,
)
from signal_engine.external.brave import BraveSearchClient
from signal_engine.external.compression import build_compressor
from signal_engine.features.canonicalize import expression_hash
from signal_engine.features.dag import ExpressionDAG
from signal_engine.features.derive import (
    build_analysis_view,
    derivation_closure,
    materializable_numeric_columns,
    positive_columns,
    reconstructs_target,
)
from signal_engine.features.expressions import Col
from signal_engine.features.parser import ExpressionParseError, parse_expression
from signal_engine.features.transforms import (
    CandidateFeature,
    PruneStats,
    generate_candidates,
    theoretical_space_size,
)
from signal_engine.ingestion.base import DatasetHandle
from signal_engine.interpretation.critic import apply_report, critique
from signal_engine.interpretation.vlm import InterpretationRequest, interpret_evidence
from signal_engine.llm.base import LLMProvider
from signal_engine.llm.llama import build_provider
from signal_engine.llm.prompts import build_final_answer_prompt
from signal_engine.llm.schemas import ConclusionStatus, FinalAnswer, Hypothesis
from signal_engine.profiling.column_cards import ColumnCard
from signal_engine.profiling.profiler import DatasetProfile, profile_dataset
from signal_engine.profiling.semantic_types import SemanticType
from signal_engine.profiling.units import Kind, Unit
from signal_engine.search.beam import (
    deduplicate_findings,
    screen_candidates,
    select_for_visualization,
)
from signal_engine.search.budgets import BudgetTracker, ImprovementTracker, StopReason
from signal_engine.search.planner import infer_target, plan_round
from signal_engine.search.scorer import question_terms, score_candidate
from signal_engine.search.state import Branch, SearchState, TestedPair
from signal_engine.statistics.correlation import (
    categorical_association,
    grouped_comparison,
)
from signal_engine.statistics.covariance import build_covariance_model
from signal_engine.statistics.multiple_testing import significance_note
from signal_engine.statistics.stability import subgroup_consistency
from signal_engine.telemetry.metrics import AnalysisMetrics, get_registry
from signal_engine.visualization.render import render_plot, stats_caption
from signal_engine.visualization.selector import select_plot


@dataclass
class AnalysisRequest:
    question: str
    dataset_handle: DatasetHandle
    dictionary: dict = field(default_factory=dict)
    accounting_identities: list[dict] = field(default_factory=list)
    dataset_notes: list[str] = field(default_factory=list)
    max_rounds: int = 3
    max_visualizations: int = 8
    enable_web_grounding: bool = False
    analysis_id: str = ""


@dataclass
class AnalysisResult:
    analysis_id: str
    question: str
    state: SearchState
    profile: DatasetProfile
    evidence: list[EvidenceObject] = field(default_factory=list)
    final_answer: FinalAnswer | None = None
    metrics: AnalysisMetrics | None = None
    budget: BudgetTracker | None = None
    view_report: dict = field(default_factory=dict)
    prune_stats: PruneStats = field(default_factory=PruneStats)
    degradations: list[str] = field(default_factory=list)
    covariance_summary: dict = field(default_factory=dict)
    status: str = "completed"

    def to_dict(self, *, include_evidence: bool = True) -> dict:
        return {
            "analysis_id": self.analysis_id,
            "question": self.question,
            "status": self.status,
            "state": self.state.to_dict(),
            "dataset": self.profile.dataset.to_dict(),
            "final_answer": self.final_answer.model_dump() if self.final_answer else None,
            "evidence": [e.to_dict() for e in self.evidence] if include_evidence else [],
            "evidence_count": len(self.evidence),
            "metrics": self.metrics.to_dict() if self.metrics else {},
            "budget": self.budget.to_dict() if self.budget else {},
            "analysis_view": self.view_report,
            "pruning": self.prune_stats.to_dict(),
            "covariance_model": self.covariance_summary,
            "degradations": self.degradations,
        }


class SignalEngine:
    """Runs the research loop over one dataset."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        provider: LLMProvider | None = None,
        memory: EvidenceMemory | None = None,
        brave: BraveSearchClient | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.settings.paths.ensure()
        self.provider = provider or build_provider(
            self.settings.llm, cache_dir=self.settings.paths.cache
        )
        self.memory = memory or build_memory(self.settings, cache_dir=self.settings.paths.cache)
        self.brave = brave or BraveSearchClient(
            config=self.settings.brave, cache_dir=self.settings.paths.cache
        )
        self.compressor = build_compressor(self.settings.compression)

    # =====================================================================
    # Entry point
    # =====================================================================
    async def analyze(self, request: AnalysisRequest) -> AnalysisResult:
        analysis_id = request.analysis_id or f"an_{uuid.uuid4().hex[:12]}"
        metrics = get_registry().create(analysis_id)
        budget = BudgetTracker(budget=self.settings.budget)
        degradations: list[str] = []

        handle = request.dataset_handle
        state = SearchState(
            analysis_id=analysis_id,
            question=request.question,
            dataset_id=handle.dataset_id,
            dataset_fingerprint=handle.fingerprint,
        )

        await self.memory.ensure_ready()

        # ---- stage A: profile (cached on the dataset fingerprint) -------
        with metrics.time("profile"):
            profile = await asyncio.to_thread(
                profile_dataset,
                handle,
                dictionary=request.dictionary,
                accounting_identities=request.accounting_identities,
                dataset_notes=request.dataset_notes,
                cache_dir=self.settings.paths.cache,
            )
        metrics.gauge("profile_cache_hit", profile.stats.get("cache_hit", False))
        metrics.gauge("rows_in_dataset", profile.dataset.row_count)
        metrics.gauge("bytes_on_disk", profile.dataset.byte_size)
        metrics.incr("columns_profiled", len(profile.columns))

        # ---- analysis view ----------------------------------------------
        with metrics.time("analysis_view"):
            lazy = pl.scan_parquet(handle.path)
            view = await asyncio.to_thread(build_analysis_view, lazy, profile)
        metrics.gauge("rows_after_filters", view.report.get("rows_after", 0))
        metrics.gauge("rows_excluded", view.report.get("rows_excluded", 0))

        units: dict[str, Unit] = dict(profile.units())
        units.update(view.derived_units)
        derived_cards = self._derived_cards(view, profile)
        all_cards: dict[str, ColumnCard] = {**profile.columns, **derived_cards}

        dag = ExpressionDAG.from_path(
            handle.path,
            dataset_fingerprint=handle.fingerprint,
            cache_dir=self.settings.paths.cache,
            view_hash=view.view_hash,
            frame=view.frame,
        )
        n_rows = view.report.get("rows_after") or profile.dataset.row_count

        numeric_columns = [
            c
            for c in materializable_numeric_columns(view.frame, units)
            if c not in view.redundant_derived
        ]
        positives = await asyncio.to_thread(positive_columns, view.frame, numeric_columns)
        if view.redundant_derived:
            state.note(
                "excluded exact unit-rescalings from the candidate pool: "
                + ", ".join(sorted(view.redundant_derived))
            )

        # ---- covariance model (sufficient statistics, computed once) ----
        covariance_summary = await self._build_covariance(
            dag, numeric_columns, units, metrics, n_rows
        )

        result = AnalysisResult(
            analysis_id=analysis_id,
            question=request.question,
            state=state,
            profile=profile,
            metrics=metrics,
            budget=budget,
            view_report=view.report,
            covariance_summary=covariance_summary,
        )

        target = infer_target(request.question, all_cards)
        if target is None:
            state.note("no target quantity could be inferred from the question")
        else:
            state.note(f"inferred target quantity: {target}")
        metrics.gauge("target_column", target)

        terms = question_terms(request.question)
        descriptions = {n: c.description for n, c in all_cards.items()}

        # =================================================================
        # Round loop
        # =================================================================
        for round_index in range(request.max_rounds):
            stop = budget.check()
            if stop is not None:
                state.stop_reason = stop
                state.note(f"search stopped: {stop.value}")
                break

            with metrics.time("planning"):
                outcome = await plan_round(
                    self.provider,
                    question=request.question,
                    profile=profile,
                    dataset_text=self._dataset_text(profile, derived_cards, view),
                    prior_results=state.summary_lines(limit=20),
                    retrieved_evidence=await self._retrieved_context(state, request, target),
                    tested_pairs=state.tested_pair_labels(),
                    round_index=round_index,
                    budget=budget,
                    extra_columns=derived_cards,
                )

            if outcome.source == "llm":
                budget.spend_llm()
                if outcome.response:
                    metrics.record_llm_call(
                        model=outcome.response.model,
                        prompt_tokens=outcome.response.usage.prompt_tokens,
                        completion_tokens=outcome.response.usage.completion_tokens,
                        latency=outcome.response.latency_seconds,
                        cache_hit=outcome.response.cache_hit,
                        estimated=outcome.response.usage.estimated,
                    )
            elif outcome.fallback_reason:
                message = f"planner fell back to deterministic: {outcome.fallback_reason}"
                if message not in degradations:
                    degradations.append(message)

            state.dropped_hypothesis_columns.extend(outcome.dropped_columns)
            state.hypotheses.append(
                {
                    "round": round_index,
                    "source": outcome.source,
                    "summary": outcome.batch.reasoning_summary,
                    "count": len(outcome.batch.hypotheses),
                    "hypotheses": [h.model_dump(mode="json") for h in outcome.batch.hypotheses],
                }
            )
            metrics.incr("hypotheses_generated", len(outcome.batch.hypotheses))

            if not outcome.batch.hypotheses:
                state.stop_reason = StopReason.NO_CANDIDATES
                break

            improved = await self._run_round(
                round_index=round_index,
                hypotheses=outcome.batch.hypotheses,
                abandon=set(outcome.batch.abandon_branches),
                request=request,
                state=state,
                profile=profile,
                all_cards=all_cards,
                units=units,
                dag=dag,
                view=view,
                numeric_columns=numeric_columns,
                positives=positives,
                target=target,
                terms=terms,
                descriptions=descriptions,
                n_rows=n_rows,
                budget=budget,
                metrics=metrics,
                result=result,
                degradations=degradations,
            )
            state.rounds_completed = round_index + 1

            if not improved:
                tracker = self._global_improvement(state)
                if tracker.should_stop:
                    state.stop_reason = StopReason.NO_IMPROVEMENT
                    state.note(
                        f"stopped after {tracker.rounds_without_improvement} rounds with no "
                        f"score improvement above {tracker.min_improvement}"
                    )
                    break

        if state.stop_reason is None:
            state.stop_reason = StopReason.COMPLETED

        # ---- final synthesis --------------------------------------------
        result.final_answer = await self._synthesize(request, state, profile, budget, metrics)
        await self.memory.flush_pending()

        metrics.gauge("evidence_stored", len(result.evidence))
        metrics.gauge(
            "significance_note", significance_note(n_rows, metrics.counters.get("statistical_tests", 0))
        )
        metrics.gauge("dag", dag.stats_dict())
        metrics.gauge("evidence_memory", self.memory.telemetry())
        metrics.gauge("brave", self.brave.telemetry())
        metrics.finish()

        degradations.extend(m for m in self.memory.errors if m not in degradations)
        result.degradations = degradations
        result.status = "completed"
        return result

    # =====================================================================
    # One round
    # =====================================================================
    async def _run_round(
        self,
        *,
        round_index: int,
        hypotheses: list[Hypothesis],
        abandon: set[str],
        request: AnalysisRequest,
        state: SearchState,
        profile: DatasetProfile,
        all_cards: dict[str, ColumnCard],
        units: dict[str, Unit],
        dag: ExpressionDAG,
        view,
        numeric_columns: list[str],
        positives: set[str],
        target: str | None,
        terms: set[str],
        descriptions: dict[str, str],
        n_rows: int,
        budget: BudgetTracker,
        metrics: AnalysisMetrics,
        result: AnalysisResult,
        degradations: list[str],
    ) -> bool:
        """Execute one hypothesize->compute->interpret cycle.  Returns True if
        the round improved on the best score so far."""
        best_before = max((t.score.total for t in state.all_tests()), default=0.0)

        for branch_id in abandon:
            branch = state.branches.get(branch_id)
            if branch is not None:
                branch.stop(StopReason.ABANDONED_BY_PLANNER, "planner marked this branch exhausted")

        # ---- build the candidate pool -----------------------------------
        candidates: list[CandidateFeature] = []
        categorical_jobs: list[tuple[str, str, Hypothesis]] = []

        for hypothesis in hypotheses:
            branch = state.add_branch(
                Branch(
                    label=" + ".join(hypothesis.base_features[:2]) or hypothesis.id,
                    hypothesis_id=hypothesis.id,
                    seed_features=hypothesis.base_features,
                    target=(hypothesis.target_features or [target or ""])[0],
                    depth=round_index,
                    improvement=ImprovementTracker(
                        min_improvement=self.settings.budget.min_improvement,
                        patience=self.settings.budget.no_improvement_rounds,
                    ),
                )
            )

            hypothesis_target = (hypothesis.target_features or [target] or [None])[0]

            # Categorical features route to group comparisons, never to a
            # correlation against their integer codes.
            for name in hypothesis.base_features:
                card = all_cards.get(name)
                if card is None:
                    continue
                if card.semantic_type.is_categorical and hypothesis_target:
                    categorical_jobs.append((name, hypothesis_target, hypothesis))
                elif name in numeric_columns:
                    candidates.append(
                        CandidateFeature(
                            expr=Col(name),
                            unit=units.get(name, Unit()),
                            name=name,
                            expr_hash=expression_hash(Col(name)),
                            depth=0,
                            origin=f"hypothesis:{hypothesis.id}",
                            rationale=hypothesis.rationale,
                        )
                    )

            # Model-proposed transformations go through the whitelisted parser.
            for expression in hypothesis.transformation_candidates:
                try:
                    parsed = parse_expression(expression, set(dag.allowed_columns))
                except ExpressionParseError as exc:
                    metrics.incr("rejected_transformations")
                    state.note(f"rejected transformation {expression!r}: {exc}")
                    continue
                derived_unit, unit_error = parsed.unit(units)
                if unit_error or derived_unit is None:
                    metrics.incr("rejected_transformations_units")
                    state.note(f"rejected transformation {expression!r}: {unit_error}")
                    continue
                candidates.append(
                    CandidateFeature(
                        expr=parsed,
                        unit=derived_unit,
                        name=expression,
                        expr_hash=expression_hash(parsed),
                        depth=parsed.depth - 1,
                        origin=f"hypothesis:{hypothesis.id}",
                        rationale=hypothesis.rationale,
                    )
                )

        # ---- dimensional pruning over the generated space ---------------
        seed = {c for h in hypotheses for c in h.base_features if c in numeric_columns}
        pool = sorted(seed) if seed else numeric_columns[:16]
        with metrics.time("candidate_generation"):
            generated, prune_stats = await asyncio.to_thread(
                generate_candidates,
                pool,
                units,
                max_candidates=min(budget.candidates_left, 600),
                exclude_hashes=state.seen_expression_hashes,
                positive_only=positives,
            )
        metrics.record_prune(prune_stats)
        result.prune_stats.merge(prune_stats)
        budget.spend_candidates(prune_stats.considered)
        metrics.gauge(
            "theoretical_candidate_space_depth1", theoretical_space_size(len(pool), 1)
        )
        candidates.extend(generated)

        # Drop anything already evaluated.
        candidates = [c for c in candidates if c.expr_hash not in state.seen_expression_hashes]

        # Drop anything that algebraically rebuilds the target.  Without this,
        # `(total_amount / trip_distance) * trip_distance` scores a perfect
        # 1.000 against total_amount -- a restatement of the target, not a
        # finding about the world.
        if target:
            kept: list[CandidateFeature] = []
            for candidate in candidates:
                if reconstructs_target(
                    candidate.expr.columns(), target, view.derived_provenance
                ):
                    metrics.incr("candidates_dropped_target_leakage")
                    continue
                kept.append(candidate)
            if len(kept) != len(candidates):
                state.note(
                    f"dropped {len(candidates) - len(kept)} candidate(s) that reconstruct "
                    f"'{target}' from its own derived columns"
                )
            candidates = kept

        if not candidates and not categorical_jobs:
            return False

        # ---- screening ladder -------------------------------------------
        screened = []
        if target and candidates and budget.may_test():
            try:
                target_vector = await asyncio.to_thread(dag.materialize, Col(target))
            except Exception as exc:  # noqa: BLE001
                degradations.append(f"could not materialize target {target}: {exc}")
                target_vector = None

            if target_vector is not None:
                with metrics.time("screening"):
                    screened, screen_stats = await asyncio.to_thread(
                        screen_candidates,
                        dag,
                        candidates,
                        target,
                        target_vector,
                        max_tests=min(budget.tests_left, 400),
                        mi_sample=self.settings.stats.mi_sample_rows,
                        stability_folds=self.settings.stats.stability_folds,
                        min_sample=self.settings.stats.min_sample_size,
                        metrics=metrics,
                        keep_vectors=True,
                    )
                budget.spend_tests(screen_stats.linear_tested)
                screened, redundant = deduplicate_findings(screened)
                metrics.incr("findings_deduplicated", redundant)
                metrics.gauge(f"screen_round_{round_index}", screen_stats.to_dict())

        # ---- categorical group comparisons ------------------------------
        categorical_results = await self._run_categorical_jobs(
            categorical_jobs, dag, all_cards, state, metrics, budget
        )

        # ---- score and record -------------------------------------------
        identity_pairs = self._identity_pairs(request.accounting_identities)

        for item in screened:
            is_mechanical = self._is_mechanical(
                item.candidate.expr.columns() | {target or ""}, identity_pairs
            )
            score = score_candidate(
                item.result,
                question_terms=terms,
                feature_names=sorted(item.candidate.expr.columns() | {target or ""}),
                seen_expression_hashes=state.seen_expression_hashes,
                expression_hash=item.candidate.expr_hash,
                depth=item.candidate.depth,
                column_descriptions=descriptions,
                is_mechanical=is_mechanical,
            )
            branch_id = self._branch_for(state, item.candidate.origin)
            state.record_test(
                TestedPair(
                    x=item.candidate.name,
                    y=target or "",
                    expression_hash=item.candidate.expr_hash,
                    result=item.result,
                    score=score,
                    depth=item.candidate.depth,
                    branch_id=branch_id,
                    is_mechanical=is_mechanical,
                )
            )

        for name, result_obj, hypothesis in categorical_results:
            score = score_candidate(
                result_obj,
                question_terms=terms,
                feature_names=[name, result_obj.y_name],
                column_descriptions=descriptions,
                expression_hash=f"grp:{name}:{result_obj.y_name}",
                seen_expression_hashes=state.seen_expression_hashes,
            )
            state.record_test(
                TestedPair(
                    x=name,
                    y=result_obj.y_name,
                    expression_hash=f"grp:{name}:{result_obj.y_name}",
                    result=result_obj,
                    score=score,
                    depth=0,
                    branch_id=self._branch_for(state, f"hypothesis:{hypothesis.id}"),
                    stratification=name,
                )
            )

        # ---- expensive stage: plots + interpretation --------------------
        to_visualize = select_for_visualization(
            screened,
            limit=min(request.max_visualizations, budget.vlm_calls_left or request.max_visualizations),
            min_effect=0.1,
        )
        await self._visualize_and_interpret(
            to_visualize,
            categorical_results,
            request=request,
            state=state,
            all_cards=all_cards,
            dag=dag,
            view=view,
            units=units,
            target=target,
            n_rows=n_rows,
            budget=budget,
            metrics=metrics,
            result=result,
            degradations=degradations,
        )

        # ---- branch improvement bookkeeping -----------------------------
        for branch in state.branches.values():
            if branch.improvement is not None and branch.is_active:
                branch.improvement.observe(branch.best_score)
                if branch.improvement.should_stop:
                    branch.stop(
                        StopReason.NO_IMPROVEMENT,
                        f"no improvement above {branch.improvement.min_improvement} for "
                        f"{branch.improvement.rounds_without_improvement} rounds",
                    )

        best_after = max((t.score.total for t in state.all_tests()), default=0.0)
        return best_after > best_before + self.settings.budget.min_improvement

    # =====================================================================
    # Helpers
    # =====================================================================
    async def _run_categorical_jobs(
        self, jobs, dag, all_cards, state, metrics, budget
    ) -> list[tuple]:
        out: list[tuple] = []
        for name, target_name, hypothesis in jobs:
            if not budget.may_test() or state.already_tested(name, target_name, name):
                continue
            card = all_cards.get(name)
            target_card = all_cards.get(target_name)
            if card is None or target_card is None:
                continue
            try:
                # Bind the loop variables explicitly rather than closing over
                # them: a closure over a loop variable reads whatever the
                # variable holds when it RUNS, which is a real hazard the
                # moment this stops being awaited immediately.
                group_vec, value_vec = await asyncio.to_thread(
                    _materialize_pair, dag, name, target_name
                )
            except Exception:  # noqa: BLE001 - a string-backed category is not castable
                continue

            if target_card.semantic_type.is_categorical:
                result_obj = await asyncio.to_thread(
                    categorical_association, group_vec, value_vec,
                    a_name=name, b_name=target_name,
                )
            else:
                result_obj = await asyncio.to_thread(
                    grouped_comparison, value_vec, group_vec,
                    value_name=target_name, group_name=name,
                    labels=card.category_labels,
                )
            budget.spend_tests()
            metrics.incr("statistical_tests")
            metrics.incr("group_comparisons")
            if result_obj.skipped_reason is None:
                out.append((name, result_obj, hypothesis))
        return out

    async def _visualize_and_interpret(
        self,
        screened,
        categorical_results,
        *,
        request,
        state,
        all_cards,
        dag,
        view,
        units,
        target,
        n_rows,
        budget,
        metrics,
        result,
        degradations,
    ) -> None:
        """Render, interpret, critique, store, and reinterpret with history."""
        filters_text = self._filters_text(view)
        jobs: list[tuple] = []

        for item in screened:
            jobs.append(("numeric", item))
        # Categorical findings face the same effect floor as numeric ones: a
        # box plot of an eta=0.03 grouping costs a VLM call and shows nothing.
        meaningful_categorical = [
            job for job in categorical_results if job[1].effect >= 0.1
        ]
        for name, res, hypothesis in meaningful_categorical[:3]:
            jobs.append(("categorical", (name, res, hypothesis)))

        for kind, payload in jobs:
            if not budget.may_call_vlm() and self.provider.available:
                break

            if kind == "numeric":
                item = payload
                x_name = item.candidate.name
                x_card = all_cards.get(x_name) or self._synthetic_card(
                    x_name, item.candidate.unit, n_rows
                )
                y_card = all_cards.get(target) if target else None
                vectors = {x_name: item.vector, target: None}
                try:
                    vectors[target] = dag.materialize(Col(target))
                except Exception:  # noqa: BLE001
                    continue
                if item.vector is None:
                    try:
                        vectors[x_name] = dag.materialize(item.candidate.expr)
                    except Exception:  # noqa: BLE001
                        continue
                res = item.result
                base_columns = derivation_closure(
                    item.candidate.expr.columns() | ({target} if target else set()),
                    view.derived_provenance,
                )
                transformation = (
                    item.candidate.expr.display()
                    if item.candidate.expr.size > 1
                    else ""
                )
                expr_hash = item.candidate.expr_hash
                unit_map = {
                    x_name: item.candidate.unit.label,
                    target: units.get(target, Unit()).label if target else "",
                }
                stratification = ""
            else:
                name, res, hypothesis = payload
                x_card = all_cards.get(name)
                y_card = all_cards.get(res.y_name)
                try:
                    vectors = {
                        name: dag.materialize(Col(name)),
                        res.y_name: dag.materialize(Col(res.y_name)),
                    }
                except Exception:  # noqa: BLE001
                    continue
                x_name = name
                base_columns = derivation_closure(
                    {name, res.y_name}, view.derived_provenance
                )
                transformation = ""
                expr_hash = f"grp:{name}:{res.y_name}"
                unit_map = {
                    name: units.get(name, Unit()).label,
                    res.y_name: units.get(res.y_name, Unit()).label,
                }
                stratification = name

            if x_card is None:
                continue

            # ---- select + render ----------------------------------------
            spec = select_plot(
                x_card,
                y_card,
                n_rows=n_rows,
                max_scatter_points=self.settings.stats.plot_max_scatter_points,
                question=request.question,
            )
            plot_data = {k: v for k, v in vectors.items() if v is not None}
            if spec.x not in plot_data:
                plot_data[spec.x] = vectors.get(x_name)
            try:
                with metrics.time("rendering"):
                    artifact = await asyncio.to_thread(
                        render_plot,
                        spec,
                        plot_data,
                        output_dir=self.settings.paths.artifacts / state.analysis_id,
                        filename=f"{expr_hash[:10]}_{spec.plot_type.value}.png",
                        stats_annotation=stats_caption(res),
                    )
                budget.spend_plot()
                metrics.incr("plots_rendered")
            except Exception as exc:  # noqa: BLE001 - keep the statistics either way
                degradations.append(f"plot failed for {x_name}: {type(exc).__name__}")
                artifact = None

            # ---- retrieve related prior evidence ------------------------
            feature_names = [x_name, target or res.y_name]
            related = await self.memory.related_to(
                feature_names=feature_names,
                text=f"{x_name} {target or res.y_name} {request.question}",
                dataset_id=state.dataset_id,
                exclude_ids=state.evidence_ids,
                limit=4,
            )
            metrics.incr("evidence_retrievals", len(related))

            # ---- interpret ----------------------------------------------
            interp_request = InterpretationRequest(
                question=request.question,
                result=res,
                artifact=artifact,
                x_card=x_card,
                y_card=y_card,
                transformation_description=transformation,
                filters_text=filters_text,
                related_evidence=[r.evidence.to_compact_text() for r in related],
            )
            with metrics.time("interpretation"):
                interpretation, response, fallback = await interpret_evidence(
                    self.provider, interp_request, model=self.settings.llm.effective_vlm_model
                )
            if response is not None:
                budget.spend_vlm()
                metrics.record_llm_call(
                    model=response.model,
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    latency=response.latency_seconds,
                    cache_hit=response.cache_hit,
                    estimated=response.usage.estimated,
                    kind="vlm",
                )
            elif fallback:
                message = f"interpretation fell back to deterministic: {fallback}"
                if message not in degradations:
                    degradations.append(message)

            # ---- critique ------------------------------------------------
            # The critic needs the expression's BASE columns (and what those
            # derive from), not its display name: an accounting identity
            # between `fare_amount + improvement_surcharge` and `total_amount`
            # is invisible if it only sees the composite label.
            critic_columns = set(feature_names) | base_columns
            report = critique(
                interpretation,
                res,
                x_card=x_card,
                y_card=y_card,
                accounting_identities=request.accounting_identities,
                filters_applied=[r.name for r in view.applied],
                expression_columns=critic_columns,
            )
            interpretation = apply_report(interpretation, report)
            metrics.incr("critic_errors", len(report.errors))
            metrics.incr("critic_warnings", len(report.warnings))
            if report.semantically_surprising:
                metrics.incr("semantically_surprising_findings")

            # ---- subgroup / Simpson check --------------------------------
            if kind == "numeric" and target and "RatecodeID" in all_cards:
                try:
                    groups = dag.materialize(Col("RatecodeID"))
                    consistency = subgroup_consistency(
                        vectors[x_name], vectors[target], groups
                    )
                    if consistency.get("simpson_reversal_suspected"):
                        res.warnings.append(consistency["note"])
                        metrics.incr("simpson_reversals_detected")
                    res.extra["subgroup_consistency"] = consistency
                except Exception:  # noqa: BLE001 - diagnostic only
                    pass

            # ---- optional web grounding ----------------------------------
            provenance: list[str] = []
            if request.enable_web_grounding and budget.may_call_brave():
                should, reason = self.brave.should_ground(
                    effect=res.effect,
                    explanation_status=interpretation.conclusion_status.value,
                    user_requested=False,
                )
                if should:
                    results, query = await self.brave.ground_relationship(
                        x_name=x_name,
                        y_name=target or res.y_name,
                        domain_hint="NYC taxi fare",
                        direction=res.direction,
                    )
                    budget.spend_brave()
                    metrics.incr("brave_calls")
                    provenance.extend(f"brave: {r.url}" for r in results[:3])
                    if results:
                        interpretation.alternative_explanations.extend(
                            r.to_compact_text() for r in results[:2]
                        )

            # ---- store evidence ------------------------------------------
            evidence = evidence_from_analysis(
                analysis_id=state.analysis_id,
                hypothesis_id="",
                dataset_id=state.dataset_id,
                dataset_fingerprint=state.dataset_fingerprint,
                question=request.question,
                result=res,
                interpretation=interpretation,
                critic_report=report,
                plot_artifact=artifact,
                canonical_expression=(
                    item.candidate.expr.display() if kind == "numeric" else f"{x_name} groups"
                ),
                expression_hash=expr_hash,
                transformation_description=transformation,
                units=unit_map,
                filters=[r.name for r in view.applied],
                stratification=stratification,
                tags=self._tags(res, interpretation, report),
            )
            evidence.provenance.extend(provenance)
            evidence.related_evidence_ids = [r.evidence.evidence_id for r in related]

            # ---- reinterpret jointly with open questions ------------------
            open_questions = await self.memory.open_questions(
                feature_names=feature_names,
                text=evidence.semantic_search_text[:400],
                dataset_id=state.dataset_id,
                exclude_ids=state.evidence_ids + [evidence.evidence_id],
                limit=3,
            )
            if open_questions and budget.may_call_llm():
                joint = await self.memory.reinterpret_with_history(
                    self.provider,
                    question=request.question,
                    new_evidence=evidence,
                    prior=[r.evidence for r in open_questions],
                )
                if joint is not None:
                    budget.spend_llm()
                    metrics.incr("joint_reinterpretations")
                    if joint.resolves_evidence_ids:
                        metrics.incr("evidence_resolved", len(joint.resolves_evidence_ids))
                        evidence.provenance.append(
                            f"resolved {len(joint.resolves_evidence_ids)} prior open question(s)"
                        )

            evidence.semantic_search_text = evidence.build_search_text()
            await self.memory.remember(evidence)
            state.evidence_ids.append(evidence.evidence_id)
            result.evidence.append(evidence)
            metrics.incr("evidence_created")

    async def _build_covariance(self, dag, numeric_columns, units, metrics, n_rows) -> dict:
        """Compute Σ once over the base numeric variables.

        This is the sufficient-statistics optimization: with Σ in hand, the
        correlation of ANY pair of linear combinations of these variables is
        k×k arithmetic instead of an N-row scan.
        """
        subset = [
            n
            for n in numeric_columns
            if units.get(n, Unit()).kind not in {Kind.CATEGORICAL, Kind.IDENTIFIER, Kind.TEXT}
        ][:24]
        if len(subset) < 2:
            return {"available": False, "reason": "fewer than 2 numeric columns"}

        try:
            with metrics.time("covariance_model"):
                data = await asyncio.to_thread(dag.materialize_columns, subset)
                model = await asyncio.to_thread(build_covariance_model, data, n_total=n_rows)
        except Exception as exc:  # noqa: BLE001
            return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}

        summary = model.to_dict()
        summary["available"] = model.is_valid()
        summary["variables"] = len(subset)
        summary["note"] = (
            "Sigma computed once on complete cases. Any linear combination of these variables "
            "can now be correlated with k x k arithmetic instead of an N-row scan."
        )
        if not model.is_valid():
            summary["note"] += (
                f" Shortcut currently DISABLED: {model.invalid_reason()}. Direct computation "
                "is used instead."
            )
        metrics.gauge("covariance_model", summary)
        return summary

    def _derived_cards(self, view, profile) -> dict[str, ColumnCard]:
        """Column cards for the derived columns, so they carry the same
        metadata (units, descriptions, caveats) as real ones."""
        cards: dict[str, ColumnCard] = {}
        for name, unit in view.derived_units.items():
            semantic = _semantic_for_kind(unit)
            card = ColumnCard(
                name=name,
                physical_dtype="derived",
                semantic_type=semantic,
                unit=unit,
                type_confidence=1.0,
                type_rule="derived_feature",
                row_count=profile.dataset.row_count,
                description=view.derived_descriptions.get(name, ""),
                provenance="derived",
            )
            if "tip" in name:
                card.caveats.append(
                    "Inherits the tip_amount caveat: credit-card tips only, cash tips absent."
                )
            if name in {"fare_per_mile", "total_per_mile"}:
                card.caveats.append(
                    "Airport pickups carry a fixed fee regardless of distance, which inflates "
                    "cost per mile independently of trip length."
                )
            cards[name] = card
        return cards

    @staticmethod
    def _synthetic_card(name: str, unit: Unit, n_rows: int) -> ColumnCard:
        return ColumnCard(
            name=name,
            physical_dtype="derived",
            semantic_type=_semantic_for_kind(unit),
            unit=unit,
            type_confidence=unit.confidence,
            type_rule="generated_expression",
            row_count=n_rows,
            description=f"Derived feature: {name}",
            provenance="generated",
        )

    @staticmethod
    def _filters_text(view) -> str:
        report = view.report
        if not report:
            return ""
        lines = [
            f"{report.get('rows_after', 0):,} of {report.get('rows_before', 0):,} rows "
            f"({report.get('exclusion_fraction', 0):.2%} excluded)."
        ]
        for name, rationale in list(report.get("rationale", {}).items())[:8]:
            excluded = report.get("excluded_by_rule", {}).get(name, 0)
            lines.append(f"  - {name} (excludes {excluded:,}): {rationale}")
        return "\n".join(lines)

    @staticmethod
    def _dataset_text(profile, derived_cards, view) -> str:
        base = profile.to_prompt_text()
        if derived_cards:
            derived = "\n".join(f"- {c.to_compact_text()}" for c in derived_cards.values())
            base += f"\n\nDERIVED COLUMNS (available for use):\n{derived}"
        if view.report:
            base += (
                f"\n\nANALYSIS VIEW: {view.report.get('rows_after', 0):,} rows after "
                f"{len(view.applied)} documented validity filters "
                f"({view.report.get('exclusion_fraction', 0):.2%} excluded)."
            )
        return base

    async def _retrieved_context(self, state, request, target) -> list[str]:
        if not state.evidence_ids:
            return []
        related = await self.memory.related_to(
            feature_names=[target] if target else [],
            text=request.question,
            dataset_id=state.dataset_id,
            limit=5,
        )
        return [r.evidence.to_compact_text(max_chars=260) for r in related]

    @staticmethod
    def _identity_pairs(identities: list[dict]) -> list[tuple[str, set[str]]]:
        out: list[tuple[str, set[str]]] = []
        for ident in identities or []:
            out.append(
                (
                    str(ident.get("target", "")).lower(),
                    {str(c).lower() for c in ident.get("components", [])},
                )
            )
        return out

    @staticmethod
    def _is_mechanical(names: set[str], identity_pairs) -> bool:
        lower = {n.lower() for n in names if n}
        for target, comps in identity_pairs:
            if target in lower and (comps & lower):
                return True
        return False

    @staticmethod
    def _branch_for(state: SearchState, origin: str) -> str:
        if origin.startswith("hypothesis:"):
            hid = origin.split(":", 1)[1]
            for branch in state.branches.values():
                if branch.hypothesis_id == hid:
                    return branch.branch_id
        active = state.active_branches()
        return active[0].branch_id if active else ""

    @staticmethod
    def _tags(result, interpretation, report) -> list[str]:
        tags = [result.direction, result.strength_label().replace(" ", "_"), result.method]
        if report.semantically_surprising:
            tags.append("surprising")
        if interpretation.conclusion_status is ConclusionStatus.MECHANICAL:
            tags.append("mechanical")
        if not report.passed:
            tags.append("critic_error")
        if result.stability is not None and result.stability > 0.8:
            tags.append("stable")
        return tags

    def _global_improvement(self, state: SearchState) -> ImprovementTracker:
        tracker = ImprovementTracker(
            min_improvement=self.settings.budget.min_improvement,
            patience=self.settings.budget.no_improvement_rounds,
        )
        by_round: dict[int, float] = {}
        for test in state.all_tests():
            by_round[test.depth] = max(by_round.get(test.depth, 0.0), test.score.total)
        for _, score in sorted(by_round.items()):
            tracker.observe(score)
        return tracker

    async def _synthesize(self, request, state, profile, budget, metrics) -> FinalAnswer | None:
        """Write the final answer, from the model or deterministically."""
        findings = state.ranked_tests(limit=12)
        findings_text = "\n".join(f"- {t.to_compact_text()}" for t in findings)
        unresolved = [
            e for e in state.evidence_ids
        ]

        if not self.provider.available or not budget.may_call_llm():
            return self._deterministic_answer(state, profile, findings)

        messages = build_final_answer_prompt(
            question=request.question,
            findings_text=findings_text or "(no findings passed the effect threshold)",
            unresolved_text=f"{len(unresolved)} evidence objects stored.",
            dataset_summary=profile.dataset.to_compact_text()[:600],
        )
        try:
            answer, response = await self.provider.complete_structured(messages, FinalAnswer)
            budget.spend_llm()
            metrics.record_llm_call(
                model=response.model,
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                latency=response.latency_seconds,
                cache_hit=response.cache_hit,
                estimated=response.usage.estimated,
            )
            return answer
        except Exception:  # noqa: BLE001
            return self._deterministic_answer(state, profile, findings)

    @staticmethod
    def _deterministic_answer(state, profile, findings) -> FinalAnswer:
        key: list[str] = []
        caveats: list[str] = []
        for test in findings[:8]:
            label = "definitional" if test.is_mechanical else "empirical"
            key.append(f"{test.to_compact_text()} [{label}]")
        seen: set[str] = set()
        for card in profile.columns.values():
            for caveat in card.caveats:
                if caveat not in seen:
                    seen.add(caveat)
                    caveats.append(f"{card.name}: {caveat}")
        return FinalAnswer(
            answer=(
                f"The engine tested {len(state.tested)} relationships across "
                f"{len(state.branches)} branches and retained "
                f"{len(state.ranked_tests())} with a meaningful effect size. "
                "This summary was generated deterministically from the statistics; no language "
                "model was available to synthesize it."
            ),
            key_findings=key,
            caveats=caveats[:8],
            unresolved_questions=[
                f"{len(state.evidence_ids)} evidence objects were stored; those marked "
                "unresolved await related evidence."
            ],
            confidence=0.4,
        )

    async def aclose(self) -> None:
        await self.memory.close()
        await self.brave.aclose()
        closer = getattr(self.provider, "aclose", None)
        if closer is not None:
            await closer()
        await self.compressor.aclose()


def _materialize_pair(dag: ExpressionDAG, a: str, b: str):
    """Materialize two columns in one worker call."""
    return dag.materialize(Col(a)), dag.materialize(Col(b))


def _semantic_for_kind(unit: Unit) -> SemanticType:
    from signal_engine.profiling.units import Dimension

    if unit.kind is Kind.CATEGORICAL:
        return SemanticType.CATEGORICAL
    if unit.kind is Kind.IDENTIFIER:
        return SemanticType.CATEGORICAL_IDENTIFIER
    if unit.kind is Kind.BOOLEAN:
        return SemanticType.BOOLEAN
    if unit.kind is Kind.DATETIME:
        return SemanticType.DATETIME
    if unit.dimension == Dimension.of(currency=1):
        return SemanticType.CURRENCY
    if unit.dimension == Dimension.of(time=1):
        return SemanticType.DURATION
    if unit.dimension == Dimension.of(count=1):
        return SemanticType.COUNT
    if not unit.dimension.is_dimensionless:
        return SemanticType.CONTINUOUS_MEASUREMENT
    return SemanticType.CONTINUOUS_MEASUREMENT
