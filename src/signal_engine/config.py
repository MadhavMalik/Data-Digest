"""Central configuration.

Every secret comes from the environment (or a gitignored `.env`).  Nothing in
this module ever prints, logs, or serializes a credential: `redacted_dump()` is
the only way config is allowed to reach telemetry or an API response.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

_PLACEHOLDER_TOKENS = {"", "__placeholder__", "placeholder", "changeme", "none", "null"}


def _is_real(value: str | None) -> bool:
    """A credential counts as real only if it is set and not a placeholder."""
    return value is not None and value.strip().lower() not in _PLACEHOLDER_TOKENS


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


def _env_int(key: str, default: int) -> int:
    raw = _env(key)
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    raw = _env(key)
    try:
        return float(raw) if raw else default
    except ValueError:
        return default


def _env_bool(key: str, default: bool) -> bool:
    raw = _env(key).lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class LLMConfig:
    base_url: str = ""
    api_key: str = ""
    model: str = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
    vlm_model: str = ""
    temperature: float = 0.2
    max_tokens: int = 2048
    timeout_seconds: int = 90
    max_concurrency: int = 4
    max_retries: int = 3

    @property
    def available(self) -> bool:
        """True when a live provider is configured.  Everything still runs
        without it — the engine falls back to a deterministic planner."""
        return _is_real(self.base_url) and _is_real(self.api_key)

    @property
    def effective_vlm_model(self) -> str:
        return self.vlm_model or self.model


@dataclass(frozen=True)
class ElasticConfig:
    deployment_type: str = "cloud-hosted"
    cloud_id: str = ""
    url: str = ""
    api_key: str = ""
    index_prefix: str = "hackmit_signal"
    inference_endpoint_id: str = ""
    embedding_dims: int = 384
    request_timeout: int = 30
    verify_certs: bool = True

    @property
    def available(self) -> bool:
        has_endpoint = _is_real(self.cloud_id) or _is_real(self.url)
        return has_endpoint and _is_real(self.api_key)

    def index_name(self, suffix: str) -> str:
        return f"{self.index_prefix}_{suffix}"


@dataclass(frozen=True)
class BraveConfig:
    api_key: str = ""
    base_url: str = "https://api.search.brave.com/res/v1"
    max_results: int = 5

    @property
    def available(self) -> bool:
        return _is_real(self.api_key)


@dataclass(frozen=True)
class CompressionConfig:
    provider: str = "noop"
    api_key: str = ""
    base_url: str = ""
    model: str = ""
    target_ratio: float = 0.5

    @property
    def available(self) -> bool:
        return self.provider != "noop" and _is_real(self.api_key) and _is_real(self.base_url)


@dataclass(frozen=True)
class SearchBudget:
    max_depth: int = 3
    beam_width: int = 8
    max_candidates: int = 20_000
    max_tests: int = 4_000
    max_llm_calls: int = 25
    max_vlm_calls: int = 15
    max_brave_calls: int = 5
    wall_clock_seconds: int = 900
    no_improvement_rounds: int = 3
    min_improvement: float = 0.02


@dataclass(frozen=True)
class StatsConfig:
    sample_rows: int = 2_000_000
    mi_sample_rows: int = 200_000
    stability_folds: int = 5
    min_sample_size: int = 100
    plot_max_scatter_points: int = 20_000


@dataclass(frozen=True)
class Paths:
    root: Path
    data: Path
    raw: Path
    processed: Path
    metadata: Path
    artifacts: Path
    cache: Path

    def ensure(self) -> Paths:
        for p in (self.data, self.raw, self.processed, self.metadata, self.artifacts, self.cache):
            p.mkdir(parents=True, exist_ok=True)
        return self


@dataclass(frozen=True)
class Settings:
    llm: LLMConfig = field(default_factory=LLMConfig)
    elastic: ElasticConfig = field(default_factory=ElasticConfig)
    brave: BraveConfig = field(default_factory=BraveConfig)
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    budget: SearchBudget = field(default_factory=SearchBudget)
    stats: StatsConfig = field(default_factory=StatsConfig)
    paths: Paths = field(default_factory=lambda: _default_paths())

    # ---- safe reporting -------------------------------------------------
    def redacted_dump(self) -> dict:
        """Config safe to log, return over HTTP, or put in a report.

        Only *availability booleans* and non-secret settings escape.  No key
        material, no URLs that might embed a token.
        """
        return {
            "llm": {
                "model": self.llm.model,
                "vlm_model": self.llm.effective_vlm_model,
                "configured": self.llm.available,
                "temperature": self.llm.temperature,
                "max_tokens": self.llm.max_tokens,
            },
            "elastic": {
                "configured": self.elastic.available,
                "deployment_type": self.elastic.deployment_type,
                "index_prefix": self.elastic.index_prefix,
                "auth": "api_key" if _is_real(self.elastic.api_key) else "none",
                "endpoint_kind": (
                    "cloud_id"
                    if _is_real(self.elastic.cloud_id)
                    else ("url" if _is_real(self.elastic.url) else "none")
                ),
                "managed_inference": bool(self.elastic.inference_endpoint_id),
                "embedding_dims": self.elastic.embedding_dims,
            },
            "brave": {"configured": self.brave.available, "max_results": self.brave.max_results},
            "compression": {
                "provider": self.compression.provider,
                "configured": self.compression.available,
                "target_ratio": self.compression.target_ratio,
            },
            "budget": self.budget.__dict__,
            "stats": self.stats.__dict__,
        }

    def missing_credentials(self) -> list[str]:
        """Human-readable list of what is still needed for full functionality."""
        missing: list[str] = []
        if not self.llm.available:
            missing.append("LLM_BASE_URL + LLM_API_KEY (hypothesis generation and graph interpretation)")
        if not self.elastic.available:
            missing.append(
                "ELASTIC_API_KEY + (ELASTIC_CLOUD_ID or ELASTIC_URL) (persistent evidence memory)"
            )
        if not self.brave.available:
            missing.append("BRAVE_SEARCH_API_KEY (optional external grounding)")
        return missing


def _default_paths(root: Path | None = None) -> Paths:
    root = root or Path(os.environ.get("SIGNAL_ENGINE_ROOT", Path.cwd()))
    data = root / _env("DATA_DIR", "data")
    return Paths(
        root=root,
        data=data,
        raw=data / "raw",
        processed=data / "processed",
        metadata=data / "metadata",
        artifacts=root / _env("ARTIFACTS_DIR", "artifacts"),
        cache=root / _env("CACHE_DIR", ".cache"),
    )


def load_settings(root: Path | None = None, *, dotenv: bool = True) -> Settings:
    """Build Settings from the environment.  Safe to call when nothing is set."""
    if dotenv:
        # `override=False`: a real exported env var always wins over .env.
        load_dotenv(dotenv_path=(root or Path.cwd()) / ".env", override=False)

    return Settings(
        llm=LLMConfig(
            base_url=_env("LLM_BASE_URL"),
            api_key=_env("LLM_API_KEY"),
            model=_env("LLM_MODEL", "meta-llama/Llama-4-Scout-17B-16E-Instruct"),
            vlm_model=_env("VLM_MODEL"),
            temperature=_env_float("LLM_TEMPERATURE", 0.2),
            max_tokens=_env_int("LLM_MAX_TOKENS", 2048),
            timeout_seconds=_env_int("LLM_TIMEOUT_SECONDS", 90),
            max_concurrency=_env_int("LLM_MAX_CONCURRENCY", 4),
            max_retries=_env_int("LLM_MAX_RETRIES", 3),
        ),
        elastic=ElasticConfig(
            deployment_type=_env("ELASTIC_DEPLOYMENT_TYPE", "cloud-hosted"),
            cloud_id=_env("ELASTIC_CLOUD_ID"),
            url=_env("ELASTIC_URL"),
            api_key=_env("ELASTIC_API_KEY"),
            index_prefix=_env("ELASTIC_INDEX_PREFIX", "hackmit_signal"),
            inference_endpoint_id=_env("ELASTIC_INFERENCE_ENDPOINT_ID"),
            embedding_dims=_env_int("ELASTIC_EMBEDDING_DIMS", 384),
            request_timeout=_env_int("ELASTIC_REQUEST_TIMEOUT", 30),
            verify_certs=_env_bool("ELASTIC_VERIFY_CERTS", True),
        ),
        brave=BraveConfig(
            api_key=_env("BRAVE_SEARCH_API_KEY"),
            base_url=_env("BRAVE_SEARCH_BASE_URL", "https://api.search.brave.com/res/v1"),
            max_results=_env_int("BRAVE_MAX_RESULTS", 5),
        ),
        compression=CompressionConfig(
            provider=_env("CONTEXT_COMPRESSOR", "noop").lower(),
            api_key=_env("TOKEN_COMPANY_API_KEY"),
            base_url=_env("TOKEN_COMPANY_BASE_URL"),
            model=_env("TOKEN_COMPANY_MODEL"),
            target_ratio=_env_float("COMPRESSION_TARGET_RATIO", 0.5),
        ),
        budget=SearchBudget(
            max_depth=_env_int("SEARCH_MAX_DEPTH", 3),
            beam_width=_env_int("SEARCH_BEAM_WIDTH", 8),
            max_candidates=_env_int("SEARCH_MAX_CANDIDATES", 20_000),
            max_tests=_env_int("SEARCH_MAX_TESTS", 4_000),
            max_llm_calls=_env_int("SEARCH_MAX_LLM_CALLS", 25),
            max_vlm_calls=_env_int("SEARCH_MAX_VLM_CALLS", 15),
            max_brave_calls=_env_int("SEARCH_MAX_BRAVE_CALLS", 5),
            wall_clock_seconds=_env_int("SEARCH_WALL_CLOCK_SECONDS", 900),
            no_improvement_rounds=_env_int("SEARCH_NO_IMPROVEMENT_ROUNDS", 3),
            min_improvement=_env_float("SEARCH_MIN_IMPROVEMENT", 0.02),
        ),
        stats=StatsConfig(
            sample_rows=_env_int("STATS_SAMPLE_ROWS", 2_000_000),
            mi_sample_rows=_env_int("STATS_MI_SAMPLE_ROWS", 200_000),
            stability_folds=_env_int("STATS_STABILITY_FOLDS", 5),
            min_sample_size=_env_int("STATS_MIN_SAMPLE_SIZE", 100),
            plot_max_scatter_points=_env_int("PLOT_MAX_SCATTER_POINTS", 20_000),
        ),
        paths=_default_paths(root),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton (cached)."""
    return load_settings()


def reset_settings_cache() -> None:
    get_settings.cache_clear()
