"""Brave Search — optional external grounding.

Used SPARINGLY and only where it earns its latency and cost:

  * a relationship is statistically strong AND the model could not explain it
  * the user explicitly asked for external context

Running a web search per candidate would be slow, expensive, and mostly
useless — the engine tests thousands of candidates and almost none of them
need the open web.  `should_ground` encodes the gate.

Results are recorded as provenance on the evidence object and are never
treated as proof of causation.  A news article explaining a correlation is a
hypothesis with a citation, not an identification strategy.

The key is read from the environment, sent only as the `X-Subscription-Token`
header, and never logged.  `__PLACEHOLDER__` counts as unset, so the app starts
and runs normally without a key.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from signal_engine.config import BraveConfig


@dataclass
class BraveResult:
    title: str
    url: str
    description: str
    age: str = ""

    def to_dict(self) -> dict:
        return {"title": self.title, "url": self.url, "description": self.description, "age": self.age}

    def to_compact_text(self, max_chars: int = 220) -> str:
        return f"{self.title} — {self.description[:max_chars]} ({self.url})"


@dataclass
class BraveSearchClient:
    """Rate-limited, cached Brave Search client with an explicit disabled state."""

    config: BraveConfig = field(default_factory=BraveConfig)
    cache_dir: Path | None = None
    max_concurrency: int = 2
    call_count: int = 0
    cache_hits: int = 0
    failures: int = 0
    _client: httpx.AsyncClient | None = field(default=None, repr=False)
    _semaphore: asyncio.Semaphore | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.cache_dir is not None:
            self.cache_dir = Path(self.cache_dir) / "brave"
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def enabled(self) -> bool:
        return self.config.available

    @property
    def status(self) -> str:
        return "enabled" if self.enabled else "disabled (no BRAVE_SEARCH_API_KEY configured)"

    # ---- gating ---------------------------------------------------------
    @staticmethod
    def should_ground(
        *,
        effect: float,
        explanation_status: str,
        user_requested: bool = False,
        min_effect: float = 0.35,
    ) -> tuple[bool, str]:
        """Decide whether a finding justifies a web search."""
        if user_requested:
            return True, "user explicitly requested external context"
        if explanation_status not in {"unresolved", "needs_more_evidence"}:
            return False, f"already {explanation_status}; no external grounding needed"
        if effect < min_effect:
            return False, f"effect {effect:.3f} below the {min_effect} grounding threshold"
        return True, f"strong ({effect:.3f}) but unexplained relationship"

    # ---- search ---------------------------------------------------------
    async def search(self, query: str, *, count: int | None = None) -> list[BraveResult]:
        """Search, or return [] when disabled.  Never raises."""
        if not self.enabled:
            return []

        count = count or self.config.max_results
        cache_key = hashlib.sha256(f"{query.lower().strip()}|{count}".encode()).hexdigest()[:24]
        cached = self._cache_get(cache_key)
        if cached is not None:
            self.cache_hits += 1
            return cached

        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrency)
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.config.base_url.rstrip("/"),
                timeout=httpx.Timeout(20.0),
                headers={
                    "Accept": "application/json",
                    "Accept-Encoding": "gzip",
                    "X-Subscription-Token": self.config.api_key,
                },
            )

        async with self._semaphore:
            try:
                resp = await self._client.get(
                    "/web/search", params={"q": query, "count": count, "safesearch": "moderate"}
                )
                if resp.status_code == 429:
                    self.failures += 1
                    return []
                resp.raise_for_status()
                payload = resp.json()
            except Exception:  # noqa: BLE001 - grounding is optional, never fatal
                self.failures += 1
                return []

        self.call_count += 1
        results = _parse(payload)
        self._cache_put(cache_key, results)
        return results

    async def ground_relationship(
        self, *, x_name: str, y_name: str, domain_hint: str = "", direction: str = ""
    ) -> tuple[list[BraveResult], str]:
        """Build a targeted query for an unexplained relationship."""
        terms = [x_name.replace("_", " "), y_name.replace("_", " ")]
        if domain_hint:
            terms.append(domain_hint)
        if direction in {"positive", "negative"}:
            terms.append("relationship explanation")
        query = " ".join(terms)
        return await self.search(query), query

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # ---- cache ----------------------------------------------------------
    def _cache_path(self, key: str) -> Path | None:
        return (self.cache_dir / f"{key}.json") if self.cache_dir else None

    def _cache_get(self, key: str) -> list[BraveResult] | None:
        path = self._cache_path(key)
        if path is None or not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
        except Exception:  # noqa: BLE001
            return None
        return [BraveResult(**r) for r in data.get("results", [])]

    def _cache_put(self, key: str, results: list[BraveResult]) -> None:
        path = self._cache_path(key)
        if path is None:
            return
        try:
            path.write_text(
                json.dumps({"cached_at": time.time(), "results": [r.to_dict() for r in results]})
            )
        except Exception:  # noqa: BLE001
            pass

    def telemetry(self) -> dict:
        return {
            "enabled": self.enabled,
            "status": self.status,
            "calls": self.call_count,
            "cache_hits": self.cache_hits,
            "failures": self.failures,
        }


def _parse(payload: dict) -> list[BraveResult]:
    out: list[BraveResult] = []
    for item in (payload.get("web") or {}).get("results", []):
        out.append(
            BraveResult(
                title=str(item.get("title", ""))[:200],
                url=str(item.get("url", "")),
                description=_strip_tags(str(item.get("description", ""))),
                age=str(item.get("age", "")),
            )
        )
    return out


def _strip_tags(text: str) -> str:
    import re

    return re.sub(r"<[^>]+>", "", text).strip()
