"""Shared fixtures.

Unit tests never touch the network and never require credentials.  Anything
that needs a live service is marked `integration` and skips cleanly when the
service is not configured.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from signal_engine.config import (  # noqa: E402
    ElasticConfig,
    LLMConfig,
    Paths,  # noqa: E402
    SearchBudget,
    Settings,
    StatsConfig,
)
from signal_engine.ingestion.base import register_local_dataset  # noqa: E402
from signal_engine.ingestion.tlc import (  # noqa: E402
    YELLOW_ACCOUNTING_IDENTITIES,
    YELLOW_DATASET_NOTES,
    YELLOW_TAXI_DICTIONARY,
)
from signal_engine.profiling.profiler import profile_dataset  # noqa: E402

RNG_SEED = 20260920
TLC_PATH = REPO_ROOT / "data" / "raw" / "yellow_tripdata_2026-01.parquet"


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(RNG_SEED)


@pytest.fixture
def tmp_paths(tmp_path: Path) -> Paths:
    paths = Paths(
        root=tmp_path,
        data=tmp_path / "data",
        raw=tmp_path / "data" / "raw",
        processed=tmp_path / "data" / "processed",
        metadata=tmp_path / "data" / "metadata",
        artifacts=tmp_path / "artifacts",
        cache=tmp_path / ".cache",
    )
    return paths.ensure()


@pytest.fixture
def offline_settings(tmp_paths: Paths) -> Settings:
    """Settings with every external service deliberately unconfigured."""
    return Settings(
        llm=LLMConfig(base_url="", api_key="", model="meta-llama/Llama-4-Scout-17B-16E-Instruct"),
        elastic=ElasticConfig(),
        budget=SearchBudget(
            max_depth=2,
            beam_width=4,
            max_candidates=500,
            max_tests=200,
            max_llm_calls=3,
            max_vlm_calls=3,
            wall_clock_seconds=120,
            no_improvement_rounds=2,
        ),
        stats=StatsConfig(mi_sample_rows=20_000, stability_folds=4, plot_max_scatter_points=2_000),
        paths=tmp_paths,
    )


# ---------------------------------------------------------------------------
# Synthetic datasets with KNOWN ground truth
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def synthetic_frame() -> pl.DataFrame:
    """A dataset where every relationship is known by construction."""
    rng = np.random.default_rng(RNG_SEED)
    n = 20_000

    x = rng.normal(0.0, 1.0, n)
    symmetric = rng.uniform(-3.0, 3.0, n)

    return pl.DataFrame(
        {
            # A: strong positive, Y = 3X + noise
            "x": x,
            "y_positive": 3.0 * x + rng.normal(0.0, 0.5, n),
            # B: strong negative, Y = -2X + noise
            "y_negative": -2.0 * x + rng.normal(0.0, 0.5, n),
            # C: independent
            "y_independent": rng.normal(0.0, 1.0, n),
            # D: nonlinear, Y = X^2 with X symmetric about 0
            "x_symmetric": symmetric,
            "y_quadratic": symmetric**2 + rng.normal(0.0, 0.1, n),
            # Dimensional columns for unit tests
            "fare_amount": np.abs(rng.normal(20.0, 8.0, n)) + 3.0,
            "trip_distance": np.abs(rng.normal(3.0, 2.0, n)) + 0.3,
            "passenger_count": rng.integers(1, 5, n),
            "zone_id": rng.integers(1, 264, n),
        }
    )


@pytest.fixture(scope="session")
def synthetic_parquet(synthetic_frame: pl.DataFrame, tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("synthetic") / "synthetic.parquet"
    synthetic_frame.write_parquet(path)
    return path


@pytest.fixture(scope="session")
def synthetic_handle(synthetic_parquet: Path):
    return register_local_dataset(
        synthetic_parquet, dataset_id="synthetic", description="Synthetic test dataset."
    )


@pytest.fixture(scope="session")
def synthetic_profile(synthetic_handle):
    return profile_dataset(synthetic_handle, use_cache=False)


# ---------------------------------------------------------------------------
# Real NYC TLC data (skipped when not downloaded)
# ---------------------------------------------------------------------------

requires_tlc = pytest.mark.skipif(
    not TLC_PATH.exists(),
    reason=(
        "NYC TLC data not present. Run: "
        "python scripts/fetch_tlc.py --vehicle yellow --year 2026 --month 1"
    ),
)


@pytest.fixture(scope="session")
def tlc_handle():
    if not TLC_PATH.exists():
        pytest.skip("TLC data not downloaded")
    return register_local_dataset(
        TLC_PATH,
        dataset_id="nyc_tlc_yellow_2026_01",
        description="NYC TLC yellow taxi trip records, 2026-01.",
    )


@pytest.fixture(scope="session")
def tlc_profile(tlc_handle):
    return profile_dataset(
        tlc_handle,
        dictionary=YELLOW_TAXI_DICTIONARY,
        accounting_identities=YELLOW_ACCOUNTING_IDENTITIES,
        dataset_notes=YELLOW_DATASET_NOTES,
        use_cache=False,
    )


@pytest.fixture(scope="session")
def tlc_sample(tlc_handle) -> pl.DataFrame:
    """A deterministic 200k-row slice of the real file.

    A fixed head slice, not a random sample: the tests assert on exact values,
    so the sample must be identical on every run and on every machine.
    """
    return pl.scan_parquet(tlc_handle.path).head(200_000).collect()


# ---------------------------------------------------------------------------
# Service availability
# ---------------------------------------------------------------------------

requires_elastic = pytest.mark.skipif(
    not (
        os.environ.get("ELASTIC_API_KEY")
        and (os.environ.get("ELASTIC_CLOUD_ID") or os.environ.get("ELASTIC_URL"))
    ),
    reason="Elasticsearch credentials not configured",
)

requires_llm = pytest.mark.skipif(
    not (os.environ.get("LLM_BASE_URL") and os.environ.get("LLM_API_KEY")),
    reason="LLM provider not configured",
)
