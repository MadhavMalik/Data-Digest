"""Provider-agnostic LLM interface.

The engine talks to this, never to a vendor SDK.  Llama 4 Scout may be hosted
by Together, Groq, Fireworks, Bedrock, or a local vLLM server; switching
providers must be an env-var change, not an architecture change.

Also here: robust JSON extraction.  Models wrap JSON in prose, in ```json
fences, or emit trailing commas.  `extract_json` handles all three before
Pydantic validation, which converts "the model formatted it slightly wrong"
from a hard failure into a non-event.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

Role = Literal["system", "user", "assistant"]


class LLMUnavailable(RuntimeError):
    """No provider is configured, or the provider is unreachable.

    Callers treat this as a normal branch: the engine degrades to its
    deterministic planner rather than failing the analysis.
    """


class LLMBadResponse(ValueError):
    """The provider replied, but the content could not be used."""


@dataclass
class ChatMessage:
    role: Role
    content: str
    images: list[str] = field(default_factory=list)  # base64 data URIs

    def to_payload(self) -> dict:
        if not self.images:
            return {"role": self.role, "content": self.content}
        parts: list[dict] = [{"type": "text", "text": self.content}]
        for img in self.images:
            parts.append({"type": "image_url", "image_url": {"url": img}})
        return {"role": self.role, "content": parts}


@dataclass
class ModelUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated: bool = False

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "estimated": self.estimated,
        }


@dataclass
class LLMResponse:
    content: str
    model: str
    usage: ModelUsage = field(default_factory=ModelUsage)
    latency_seconds: float = 0.0
    cache_hit: bool = False
    finish_reason: str = ""
    provider: str = ""

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "provider": self.provider,
            "usage": self.usage.to_dict(),
            "latency_seconds": round(self.latency_seconds, 3),
            "cache_hit": self.cache_hit,
            "finish_reason": self.finish_reason,
            "content_chars": len(self.content),
        }


class LLMProvider(ABC):
    """What the engine requires of any model backend."""

    name: str = "abstract"

    @property
    @abstractmethod
    def available(self) -> bool: ...

    @abstractmethod
    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
        model: str | None = None,
    ) -> LLMResponse: ...

    async def complete_structured(
        self,
        messages: list[ChatMessage],
        schema: type[T],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
        retries: int = 1,
    ) -> tuple[T, LLMResponse]:
        """Complete and validate against `schema`, retrying once with the
        validation error fed back to the model."""
        convo = list(messages)
        last_error: Exception | None = None

        for attempt in range(retries + 1):
            response = await self.complete(
                convo,
                temperature=temperature,
                max_tokens=max_tokens,
                json_mode=True,
                model=model,
            )
            try:
                payload = extract_json(response.content)
                return schema.model_validate(payload), response
            except (LLMBadResponse, ValidationError, ValueError) as exc:
                last_error = exc
                if attempt >= retries:
                    break
                convo = convo + [
                    ChatMessage("assistant", response.content[:2000]),
                    ChatMessage(
                        "user",
                        "That response could not be parsed into the required schema.\n"
                        f"Error: {str(exc)[:500]}\n"
                        "Reply with ONLY the corrected JSON object. No prose, no code fences.",
                    ),
                ]

        raise LLMBadResponse(f"could not obtain valid {schema.__name__}: {last_error}")


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> Any:
    """Pull a JSON object/array out of a model response.

    Tries, in order: the whole string; fenced blocks; the first balanced
    brace/bracket span; then the same with trailing commas removed.
    """
    if not text or not text.strip():
        raise LLMBadResponse("empty model response")

    candidates: list[str] = [text.strip()]
    candidates.extend(m.strip() for m in _FENCE_RE.findall(text))

    span = _balanced_span(text)
    if span:
        candidates.append(span)

    for candidate in candidates:
        for attempt in (candidate, _strip_trailing_commas(candidate)):
            try:
                return json.loads(attempt)
            except (json.JSONDecodeError, TypeError):
                continue

    raise LLMBadResponse(f"no valid JSON found in response (first 200 chars: {text[:200]!r})")


def _balanced_span(text: str) -> str | None:
    """Find the first balanced {...} or [...] region, ignoring braces in strings."""
    start = None
    opener = closer = ""
    for i, ch in enumerate(text):
        if ch in "{[":
            start = i
            opener, closer = ch, ("}" if ch == "{" else "]")
            break
    if start is None:
        return None

    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _strip_trailing_commas(text: str) -> str:
    return re.sub(r",(\s*[}\]])", r"\1", text)


def estimate_tokens(text: str) -> int:
    """Provider-independent token estimate (~4 chars/token for English + JSON).

    Used only when a provider omits usage data, and always flagged
    `estimated=True` so the metrics panel never presents a guess as measured.
    """
    if not text:
        return 0
    return max(1, len(text) // 4)
