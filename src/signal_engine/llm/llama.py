"""Llama 4 provider over any OpenAI-compatible endpoint.

Default model: `meta-llama/Llama-4-Scout-17B-16E-Instruct`.  Scout is natively
multimodal, so the same deployment serves both the hypothesis stage (text) and
the graph-interpretation stage (image + text).  That is why `VLM_MODEL`
defaults to `LLM_MODEL` rather than requiring a second deployment.

Reliability behaviour, all of it deliberate:
  * bounded concurrency via a semaphore, so a wide search fan-out cannot
    stampede the provider
  * retry with exponential backoff + jitter on 429/5xx, honouring `Retry-After`
  * no retry on 4xx other than 429 — a malformed request will fail identically
  * every failure path raises `LLMUnavailable`, which the orchestrator treats
    as "fall back to the deterministic planner", never as a crash
"""

from __future__ import annotations

import asyncio
import random
import time

import httpx

from signal_engine.config import LLMConfig
from signal_engine.llm.base import (
    ChatMessage,
    LLMProvider,
    LLMResponse,
    LLMUnavailable,
    ModelUsage,
    estimate_tokens,
)
from signal_engine.llm.cache import LLMCache

RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


class LlamaProvider(LLMProvider):
    """OpenAI-compatible chat-completions client."""

    name = "llama-openai-compatible"

    def __init__(
        self,
        config: LLMConfig,
        *,
        cache: LLMCache | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.config = config
        self.cache = cache or LLMCache()
        self._client = client
        self._owns_client = client is None
        self._semaphore = asyncio.Semaphore(max(1, config.max_concurrency))
        self.call_count = 0
        self.failure_count = 0

    @property
    def available(self) -> bool:
        return self.config.available

    # ---- lifecycle ------------------------------------------------------
    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.config.base_url.rstrip("/"),
                timeout=httpx.Timeout(self.config.timeout_seconds),
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json",
                },
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    # ---- completion -----------------------------------------------------
    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
        model: str | None = None,
    ) -> LLMResponse:
        if not self.available:
            raise LLMUnavailable(
                "no LLM provider configured; set LLM_BASE_URL and LLM_API_KEY"
            )

        temperature = self.config.temperature if temperature is None else temperature
        max_tokens = max_tokens or self.config.max_tokens
        model = model or self.config.model

        cache_key = None
        if self.cache.cacheable(temperature):
            cache_key = self.cache.key(
                messages,
                provider=self.name,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                json_mode=json_mode,
            )
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached
        else:
            self.cache.stats.skipped_nondeterministic += 1

        payload: dict = {
            "model": model,
            "messages": [m.to_payload() for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        response = await self._post_with_retry(payload, model)

        if cache_key is not None:
            self.cache.put(cache_key, response)
        return response

    async def _post_with_retry(self, payload: dict, model: str) -> LLMResponse:
        client = await self._get_client()
        last_error: Exception | None = None

        async with self._semaphore:
            for attempt in range(1, self.config.max_retries + 1):
                started = time.time()
                try:
                    resp = await client.post("/chat/completions", json=payload)
                except (httpx.TimeoutException, httpx.TransportError) as exc:
                    last_error = exc
                    await self._backoff(attempt, None)
                    continue

                if resp.status_code in RETRYABLE_STATUS:
                    last_error = RuntimeError(f"provider returned HTTP {resp.status_code}")
                    if attempt < self.config.max_retries:
                        await self._backoff(attempt, resp.headers.get("Retry-After"))
                        continue
                    break

                if resp.status_code >= 400:
                    # Non-retryable.  The body may echo request content, so it
                    # is truncated and never logged with headers attached.
                    self.failure_count += 1
                    raise LLMUnavailable(_explain_http_error(resp))

                self.call_count += 1
                return _parse_response(resp.json(), model, time.time() - started, self.name)

        self.failure_count += 1
        raise LLMUnavailable(f"LLM request failed after {self.config.max_retries} attempts: {last_error}")

    async def _backoff(self, attempt: int, retry_after: str | None) -> None:
        if retry_after:
            try:
                await asyncio.sleep(min(30.0, float(retry_after)))
                return
            except (TypeError, ValueError):
                pass
        # Full jitter: avoids a thundering herd when many branches retry together.
        delay = min(20.0, (2**attempt) * 0.5)
        await asyncio.sleep(random.uniform(0, delay))


# Provider errors that are configuration problems, not transient faults.  These
# get an actionable message, because "HTTP 402" in a degradation list tells an
# operator nothing about what to go and fix.
_ACTIONABLE_STATUS = {
    401: "the API key was rejected (check LLM_API_KEY)",
    402: (
        "the provider requires billing to be configured on this account. The key "
        "authenticates, but inference is refused until a payment method is added"
    ),
    403: "the key is valid but not authorized for this model or endpoint",
    404: "the endpoint or model was not found (check LLM_BASE_URL and LLM_MODEL)",
}


def _explain_http_error(resp) -> str:
    """Turn a provider rejection into something an operator can act on."""
    hint = _ACTIONABLE_STATUS.get(resp.status_code)

    detail = ""
    try:
        payload = resp.json()
        error = payload.get("error") or {}
        detail = str(error.get("message") or error.get("code") or "")[:200]
    except Exception:  # noqa: BLE001 - fall back to raw text
        detail = resp.text[:200]

    parts = [f"provider rejected the request (HTTP {resp.status_code})"]
    if hint:
        parts.append(hint)
    if detail:
        parts.append(f"provider said: {detail}")
    return "; ".join(parts)


def _parse_response(data: dict, model: str, latency: float, provider: str) -> LLMResponse:
    try:
        choice = data["choices"][0]
        content = choice["message"]["content"] or ""
        finish = choice.get("finish_reason", "")
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMUnavailable(f"malformed provider response: {exc}") from exc

    raw_usage = data.get("usage") or {}
    if raw_usage.get("prompt_tokens") is not None:
        usage = ModelUsage(
            prompt_tokens=int(raw_usage.get("prompt_tokens", 0)),
            completion_tokens=int(raw_usage.get("completion_tokens", 0)),
            estimated=False,
        )
    else:
        usage = ModelUsage(completion_tokens=estimate_tokens(content), estimated=True)

    return LLMResponse(
        content=content,
        model=data.get("model", model),
        usage=usage,
        latency_seconds=latency,
        finish_reason=finish,
        provider=provider,
    )


# ---------------------------------------------------------------------------
# Offline provider
# ---------------------------------------------------------------------------


class UnavailableProvider(LLMProvider):
    """Stand-in used when no credentials are configured.

    Every call raises `LLMUnavailable`, which is a normal, handled branch: the
    orchestrator falls back to its deterministic planner and the run completes
    with real statistics and real plots, just without model-authored prose.
    """

    name = "unavailable"

    def __init__(self, reason: str = "no LLM credentials configured") -> None:
        self.reason = reason
        self.call_count = 0
        self.failure_count = 0
        self.cache = LLMCache(enabled=False)

    @property
    def available(self) -> bool:
        return False

    async def complete(self, messages, **kwargs) -> LLMResponse:  # noqa: ANN001
        raise LLMUnavailable(self.reason)

    async def aclose(self) -> None:
        return None


def build_provider(config: LLMConfig, *, cache_dir=None) -> LLMProvider:
    """Return a live provider when configured, otherwise the offline stand-in."""
    if not config.available:
        missing = []
        if not config.base_url:
            missing.append("LLM_BASE_URL")
        if not config.api_key:
            missing.append("LLM_API_KEY")
        return UnavailableProvider(
            f"LLM disabled; missing {', '.join(missing) or 'credentials'}"
        )

    cache = LLMCache(directory=(cache_dir / "llm") if cache_dir else None)
    return LlamaProvider(config, cache=cache)
