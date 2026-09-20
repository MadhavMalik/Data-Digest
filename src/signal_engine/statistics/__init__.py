"""The deterministic statistical truth engine.

Nothing in this package calls an LLM.  The model proposes; this decides.
"""

from signal_engine.statistics.correlation import (
    RelationshipResult,
    categorical_association,
    grouped_comparison,
    pairwise_relationship,
)
from signal_engine.statistics.covariance import (
    CovarianceModel,
    build_covariance_model,
    linear_combination_correlation,
)
from signal_engine.statistics.multiple_testing import benjamini_hochberg
from signal_engine.statistics.mutual_information import mutual_information
from signal_engine.statistics.stability import stability_score

__all__ = [
    "CovarianceModel",
    "RelationshipResult",
    "benjamini_hochberg",
    "build_covariance_model",
    "categorical_association",
    "grouped_comparison",
    "linear_combination_correlation",
    "mutual_information",
    "pairwise_relationship",
    "stability_score",
]
