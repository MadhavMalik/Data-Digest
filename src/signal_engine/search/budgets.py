"""Budget enforcement and stopping rules.

The candidate space is combinatorial.  The honest position is not "we will
search it exhaustively" — that is impossible once transformations compose —
but "we will spend a bounded amount of compute and say exactly where we
stopped and why."

Two kinds of stopping:

  HARD BUDGETS   depth, beam width, candidate count, test count, model calls,
                 wall clock.  Checked before every expensive operation.

  MARGINAL       a branch that has not improved its best score by more than
  IMPROVEMENT    `min_improvement` for `no_improvement_rounds` consecutive
                 rounds is closed.  This is what stops the search grinding on
                 a branch that has already given up everything it has.

Every stop is recorded with its reason, and the reasons surface in the API and
the report.  A search that quietly hit a cap and reported its partial results
as complete would be worse than useless.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

from signal_engine.config import SearchBudget


class StopReason(str, Enum):
    COMPLETED = "completed"
    MAX_DEPTH = "max_depth_reached"
    NO_IMPROVEMENT = "no_marginal_improvement"
    CANDIDATE_BUDGET = "candidate_budget_exhausted"
    TEST_BUDGET = "test_budget_exhausted"
    LLM_BUDGET = "llm_call_budget_exhausted"
    VLM_BUDGET = "vlm_call_budget_exhausted"
    WALL_CLOCK = "wall_clock_budget_exhausted"
    NO_CANDIDATES = "no_viable_candidates_remain"
    ABANDONED_BY_PLANNER = "abandoned_by_planner"
    CANCELLED = "cancelled"
    ERROR = "error"

    @property
    def is_budget(self) -> bool:
        return self in {
            StopReason.CANDIDATE_BUDGET,
            StopReason.TEST_BUDGET,
            StopReason.LLM_BUDGET,
            StopReason.VLM_BUDGET,
            StopReason.WALL_CLOCK,
        }


@dataclass
class BudgetTracker:
    """Tracks consumption against `SearchBudget` and answers 'may I?'."""

    budget: SearchBudget = field(default_factory=SearchBudget)
    started_at: float = field(default_factory=time.time)

    candidates_generated: int = 0
    tests_run: int = 0
    llm_calls: int = 0
    vlm_calls: int = 0
    brave_calls: int = 0
    plots_rendered: int = 0
    cancelled: bool = False

    exhausted: list[StopReason] = field(default_factory=list)

    # ---- remaining ------------------------------------------------------
    @property
    def seconds_elapsed(self) -> float:
        return time.time() - self.started_at

    @property
    def seconds_left(self) -> float:
        return max(0.0, self.budget.wall_clock_seconds - self.seconds_elapsed)

    @property
    def tests_left(self) -> int:
        return max(0, self.budget.max_tests - self.tests_run)

    @property
    def candidates_left(self) -> int:
        return max(0, self.budget.max_candidates - self.candidates_generated)

    @property
    def llm_calls_left(self) -> int:
        return max(0, self.budget.max_llm_calls - self.llm_calls)

    @property
    def vlm_calls_left(self) -> int:
        return max(0, self.budget.max_vlm_calls - self.vlm_calls)

    @property
    def brave_calls_left(self) -> int:
        return max(0, self.budget.max_brave_calls - self.brave_calls)

    # ---- permissions ----------------------------------------------------
    def check(self) -> StopReason | None:
        """Global stop check, run at the top of every round."""
        if self.cancelled:
            return StopReason.CANCELLED
        if self.seconds_left <= 0:
            return self._exhaust(StopReason.WALL_CLOCK)
        if self.tests_left <= 0:
            return self._exhaust(StopReason.TEST_BUDGET)
        if self.candidates_left <= 0:
            return self._exhaust(StopReason.CANDIDATE_BUDGET)
        return None

    def may_call_llm(self) -> bool:
        return self.llm_calls_left > 0 and self.seconds_left > 5 and not self.cancelled

    def may_call_vlm(self) -> bool:
        return self.vlm_calls_left > 0 and self.seconds_left > 5 and not self.cancelled

    def may_call_brave(self) -> bool:
        return self.brave_calls_left > 0 and self.seconds_left > 3 and not self.cancelled

    def may_test(self, count: int = 1) -> bool:
        return self.tests_left >= count and self.seconds_left > 0 and not self.cancelled

    # ---- consumption ----------------------------------------------------
    def spend_candidates(self, n: int) -> None:
        self.candidates_generated += n
        if self.candidates_left <= 0:
            self._exhaust(StopReason.CANDIDATE_BUDGET)

    def spend_tests(self, n: int = 1) -> None:
        self.tests_run += n
        if self.tests_left <= 0:
            self._exhaust(StopReason.TEST_BUDGET)

    def spend_llm(self, n: int = 1) -> None:
        self.llm_calls += n
        if self.llm_calls_left <= 0:
            self._exhaust(StopReason.LLM_BUDGET)

    def spend_vlm(self, n: int = 1) -> None:
        self.vlm_calls += n
        if self.vlm_calls_left <= 0:
            self._exhaust(StopReason.VLM_BUDGET)

    def spend_brave(self, n: int = 1) -> None:
        self.brave_calls += n

    def spend_plot(self, n: int = 1) -> None:
        self.plots_rendered += n

    def cancel(self) -> None:
        self.cancelled = True

    def _exhaust(self, reason: StopReason) -> StopReason:
        if reason not in self.exhausted:
            self.exhausted.append(reason)
        return reason

    # ---- reporting ------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "elapsed_seconds": round(self.seconds_elapsed, 2),
            "seconds_left": round(self.seconds_left, 2),
            "candidates_generated": self.candidates_generated,
            "candidates_left": self.candidates_left,
            "tests_run": self.tests_run,
            "tests_left": self.tests_left,
            "llm_calls": self.llm_calls,
            "llm_calls_left": self.llm_calls_left,
            "vlm_calls": self.vlm_calls,
            "vlm_calls_left": self.vlm_calls_left,
            "brave_calls": self.brave_calls,
            "plots_rendered": self.plots_rendered,
            "exhausted": [r.value for r in self.exhausted],
            "cancelled": self.cancelled,
            "limits": {
                "max_depth": self.budget.max_depth,
                "beam_width": self.budget.beam_width,
                "max_candidates": self.budget.max_candidates,
                "max_tests": self.budget.max_tests,
                "max_llm_calls": self.budget.max_llm_calls,
                "max_vlm_calls": self.budget.max_vlm_calls,
                "wall_clock_seconds": self.budget.wall_clock_seconds,
            },
        }


@dataclass
class ImprovementTracker:
    """Marginal-improvement stopping for a single branch."""

    min_improvement: float
    patience: int
    best_score: float = 0.0
    rounds_without_improvement: int = 0
    history: list[float] = field(default_factory=list)

    def observe(self, score: float) -> bool:
        """Record a round's best score.  Returns True if it was an improvement."""
        self.history.append(score)
        if score > self.best_score + self.min_improvement:
            self.best_score = score
            self.rounds_without_improvement = 0
            return True
        self.best_score = max(self.best_score, score)
        self.rounds_without_improvement += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.rounds_without_improvement >= self.patience

    def to_dict(self) -> dict:
        return {
            "best_score": round(self.best_score, 4),
            "rounds_without_improvement": self.rounds_without_improvement,
            "patience": self.patience,
            "min_improvement": self.min_improvement,
            "history": [round(h, 4) for h in self.history],
        }
