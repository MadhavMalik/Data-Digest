"""Search state: branches, tested pairs, and the running record of a run.

State is explicit and serializable so an in-progress analysis can be inspected
over HTTP while it runs, and so the final report can say exactly what was
tried, what survived, and where each branch stopped.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum

from signal_engine.search.budgets import ImprovementTracker, StopReason
from signal_engine.search.scorer import CandidateScore
from signal_engine.statistics.correlation import RelationshipResult


class BranchStatus(str, Enum):
    ACTIVE = "active"
    STOPPED = "stopped"
    EXHAUSTED = "exhausted"
    PROMOTED = "promoted"


@dataclass
class TestedPair:
    """One completed statistical test."""

    x: str
    y: str
    expression_hash: str
    result: RelationshipResult
    score: CandidateScore
    depth: int
    branch_id: str = ""
    hypothesis_id: str = ""
    stratification: str = ""
    is_mechanical: bool = False

    @property
    def key(self) -> tuple[str, str, str]:
        """Order-independent identity, so A~B and B~A are one test."""
        a, b = sorted([self.x, self.y])
        return (a, b, self.stratification)

    def to_dict(self) -> dict:
        return {
            "x": self.x,
            "y": self.y,
            "expression_hash": self.expression_hash,
            "depth": self.depth,
            "branch_id": self.branch_id,
            "hypothesis_id": self.hypothesis_id,
            "stratification": self.stratification,
            "is_mechanical": self.is_mechanical,
            "result": self.result.to_dict(),
            "score": self.score.to_dict(),
        }

    def to_compact_text(self) -> str:
        line = self.result.to_compact_text()
        if self.is_mechanical:
            line += " [MECHANICAL: definitional relationship]"
        return line


@dataclass
class Branch:
    """One line of enquiry, e.g. 'distance -> fare'."""

    branch_id: str = field(default_factory=lambda: f"br_{uuid.uuid4().hex[:8]}")
    label: str = ""
    hypothesis_id: str = ""
    seed_features: list[str] = field(default_factory=list)
    target: str = ""
    depth: int = 0
    status: BranchStatus = BranchStatus.ACTIVE
    stop_reason: StopReason | None = None
    stop_detail: str = ""
    improvement: ImprovementTracker | None = None
    best_score: float = 0.0
    tests: list[TestedPair] = field(default_factory=list)
    frontier_hashes: set[str] = field(default_factory=set)

    def record(self, test: TestedPair) -> None:
        self.tests.append(test)
        self.best_score = max(self.best_score, test.score.total)

    def stop(self, reason: StopReason, detail: str = "") -> None:
        self.status = BranchStatus.STOPPED
        self.stop_reason = reason
        self.stop_detail = detail

    @property
    def is_active(self) -> bool:
        return self.status is BranchStatus.ACTIVE

    def to_dict(self) -> dict:
        return {
            "branch_id": self.branch_id,
            "label": self.label,
            "hypothesis_id": self.hypothesis_id,
            "seed_features": self.seed_features,
            "target": self.target,
            "depth": self.depth,
            "status": self.status.value,
            "stop_reason": self.stop_reason.value if self.stop_reason else None,
            "stop_detail": self.stop_detail,
            "best_score": round(self.best_score, 4),
            "tests_run": len(self.tests),
            "improvement": self.improvement.to_dict() if self.improvement else None,
        }


@dataclass
class SearchState:
    """Everything the search loop accumulates."""

    analysis_id: str
    question: str
    dataset_id: str = ""
    dataset_fingerprint: str = ""

    branches: dict[str, Branch] = field(default_factory=dict)
    tested: dict[tuple[str, str, str], TestedPair] = field(default_factory=dict)
    seen_expression_hashes: set[str] = field(default_factory=set)
    evidence_ids: list[str] = field(default_factory=list)
    rounds_completed: int = 0
    stop_reason: StopReason | None = None
    notes: list[str] = field(default_factory=list)
    hypotheses: list[dict] = field(default_factory=list)
    dropped_hypothesis_columns: list[str] = field(default_factory=list)

    # ---- mutation -------------------------------------------------------
    def add_branch(self, branch: Branch) -> Branch:
        self.branches[branch.branch_id] = branch
        return branch

    def already_tested(self, x: str, y: str, stratification: str = "") -> bool:
        a, b = sorted([x, y])
        return (a, b, stratification) in self.tested

    def record_test(self, test: TestedPair) -> bool:
        """Record a test; returns False if this pair was already covered."""
        if test.key in self.tested:
            return False
        self.tested[test.key] = test
        self.seen_expression_hashes.add(test.expression_hash)
        branch = self.branches.get(test.branch_id)
        if branch is not None:
            branch.record(test)
        return True

    def note(self, message: str) -> None:
        if message not in self.notes:
            self.notes.append(message)

    # ---- views ----------------------------------------------------------
    def active_branches(self) -> list[Branch]:
        return [b for b in self.branches.values() if b.is_active]

    def all_tests(self) -> list[TestedPair]:
        return list(self.tested.values())

    def ranked_tests(self, *, limit: int | None = None, meaningful_only: bool = True) -> list[TestedPair]:
        tests = [
            t
            for t in self.tested.values()
            if t.result.skipped_reason is None and (not meaningful_only or t.result.is_meaningful)
        ]
        tests.sort(key=lambda t: -t.score.total)
        return tests[:limit] if limit else tests

    def tested_pair_labels(self, limit: int = 60) -> list[str]:
        return [f"{a}~{b}" for (a, b, _s) in list(self.tested)[:limit]]

    def to_dict(self) -> dict:
        return {
            "analysis_id": self.analysis_id,
            "question": self.question,
            "dataset_id": self.dataset_id,
            "rounds_completed": self.rounds_completed,
            "stop_reason": self.stop_reason.value if self.stop_reason else None,
            "branches": [b.to_dict() for b in self.branches.values()],
            "tests_run": len(self.tested),
            "meaningful_findings": len(self.ranked_tests()),
            "evidence_ids": self.evidence_ids,
            "notes": self.notes,
            "hypotheses": self.hypotheses,
            "dropped_hypothesis_columns": self.dropped_hypothesis_columns,
        }

    def summary_lines(self, limit: int = 20) -> list[str]:
        """Compact result lines fed back to the planner next round."""
        return [t.to_compact_text() for t in self.ranked_tests(limit=limit, meaningful_only=False)]
