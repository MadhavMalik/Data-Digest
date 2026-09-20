"""Disk-backed cache for model responses.

Two users asking the same question in slightly different words should not pay
twice for identical reasoning.  The key covers everything that can change the
answer:

    provider + model + prompt-template version + normalized messages
    + temperature + max_tokens + json_mode

Temperature is part of the key, and caching is DISABLED above a temperature
threshold: caching a sampled response would silently turn a stochastic call
into a deterministic one, which is a correctness change disguised as an
optimization.

Images contribute their content hash, not their bytes, so a multimodal request
keys cleanly without bloating the index.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from signal_engine.llm.base import ChatMessage, LLMResponse, ModelUsage

# Above this temperature, responses are treated as intentionally varied and
# are not cached.
CACHEABLE_TEMPERATURE = 0.35
PROMPT_VERSION = "v1"


@dataclass
class LLMCacheStats:
    hits: int = 0
    misses: int = 0
    stores: int = 0
    skipped_nondeterministic: int = 0
    tokens_saved: int = 0
    seconds_saved: float = 0.0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def to_dict(self) -> dict:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "stores": self.stores,
            "skipped_nondeterministic": self.skipped_nondeterministic,
            "hit_rate": round(self.hit_rate, 4),
            "tokens_saved": self.tokens_saved,
            "seconds_saved": round(self.seconds_saved, 2),
        }


@dataclass
class LLMCache:
    directory: Path | None = None
    stats: LLMCacheStats = field(default_factory=LLMCacheStats)
    prompt_version: str = PROMPT_VERSION
    enabled: bool = True

    def __post_init__(self) -> None:
        if self.directory is not None:
            self.directory = Path(self.directory)
            self.directory.mkdir(parents=True, exist_ok=True)

    # ---- keys -----------------------------------------------------------
    def key(
        self,
        messages: list[ChatMessage],
        *,
        provider: str,
        model: str,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> str:
        payload = {
            "prompt_version": self.prompt_version,
            "provider": provider,
            "model": model,
            "temperature": round(temperature, 4),
            "max_tokens": max_tokens,
            "json_mode": json_mode,
            "messages": [
                {
                    "role": m.role,
                    "content": _normalize(m.content),
                    "images": [hashlib.sha256(i.encode()).hexdigest()[:16] for i in m.images],
                }
                for m in messages
            ],
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:32]

    def cacheable(self, temperature: float) -> bool:
        return self.enabled and temperature <= CACHEABLE_TEMPERATURE

    def _path(self, key: str) -> Path | None:
        return (self.directory / f"{key}.json") if self.directory else None

    # ---- access ---------------------------------------------------------
    def get(self, key: str) -> LLMResponse | None:
        path = self._path(key)
        if path is None or not path.exists():
            self.stats.misses += 1
            return None
        try:
            data = json.loads(path.read_text())
        except Exception:  # noqa: BLE001 - corrupt entry is a miss
            path.unlink(missing_ok=True)
            self.stats.misses += 1
            return None

        self.stats.hits += 1
        # `to_dict()` emits the derived `total_tokens`, which the constructor
        # does not take; keep only real fields so a cached entry written by
        # any version still loads.
        raw_usage = data.get("usage", {}) or {}
        usage = ModelUsage(
            prompt_tokens=int(raw_usage.get("prompt_tokens", 0)),
            completion_tokens=int(raw_usage.get("completion_tokens", 0)),
            estimated=bool(raw_usage.get("estimated", False)),
        )
        self.stats.tokens_saved += usage.total_tokens
        self.stats.seconds_saved += float(data.get("latency_seconds", 0.0))

        return LLMResponse(
            content=data["content"],
            model=data.get("model", ""),
            usage=usage,
            latency_seconds=0.0,
            cache_hit=True,
            finish_reason=data.get("finish_reason", ""),
            provider=data.get("provider", ""),
        )

    def put(self, key: str, response: LLMResponse) -> None:
        path = self._path(key)
        if path is None:
            return
        payload = {
            "content": response.content,
            "model": response.model,
            "provider": response.provider,
            "usage": response.usage.to_dict(),
            "latency_seconds": response.latency_seconds,
            "finish_reason": response.finish_reason,
            "cached_at": time.time(),
        }
        tmp = path.with_suffix(".json.part")
        try:
            tmp.write_text(json.dumps(payload))
            tmp.replace(path)
            self.stats.stores += 1
        except Exception:  # noqa: BLE001 - best-effort
            tmp.unlink(missing_ok=True)


def _normalize(text: str) -> str:
    """Collapse whitespace so cosmetically different prompts share a key."""
    return " ".join(text.split())
