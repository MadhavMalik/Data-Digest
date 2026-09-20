"""The bounded hypothesis search loop."""

from signal_engine.search.budgets import BudgetTracker, StopReason
from signal_engine.search.orchestrator import AnalysisRequest, AnalysisResult, SignalEngine
from signal_engine.search.scorer import CandidateScore, score_candidate
from signal_engine.search.state import Branch, SearchState, TestedPair

__all__ = [
    "AnalysisRequest",
    "AnalysisResult",
    "Branch",
    "BudgetTracker",
    "CandidateScore",
    "SearchState",
    "SignalEngine",
    "StopReason",
    "TestedPair",
    "score_candidate",
]
