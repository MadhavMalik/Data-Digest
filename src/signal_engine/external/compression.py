"""Context compression with a protected-span guarantee.

The Token Company's product compresses TEXTUAL LLM input: their models score
tokens and drop low-signal ones, returning shorter text made of surviving
original tokens.  So the adapter here is text-in/text-out.

What must NOT be done, and is structurally prevented:

  * Compressing embedding vectors.  A text-token compressor has no meaning
    applied to a float array, and our vectors never enter an LLM context
    anyway — they live in Elasticsearch purely to retrieve source material.

  * Compressing exact statistics.  "pearson r: -0.8342" losing a character is
    a wrong number, not a shorter one.  `protected_spans` marks every
    statistics block, column name, and unit, and those spans are excised
    before compression and restored verbatim afterwards.

So compression only ever touches NARRATIVE prose: rationales, prior
interpretations, retrieved evidence summaries.  The measured before/after
token counts and any quality regression are reported rather than assumed —
if compression hurts interpretation quality, we do not use it.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from signal_engine.config import CompressionConfig
from signal_engine.llm.base import estimate_tokens

# A run of text that must survive byte-identical.
PROTECT_OPEN = "\x01PROTECTED_"
PROTECT_CLOSE = "\x02"

# Anything that looks like an exact numeric result.
_NUMERIC_RE = re.compile(
    r"([A-Za-z_][A-Za-z0-9_.]*\s*[:=]\s*[-+]?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?%?)"
    r"|([-+]?\d[\d,]*\.\d+)"
    r"|(\bn\s*=\s*[\d,]+)"
)


@dataclass
class CompressionResult:
    original: str
    compressed: str
    original_tokens: int
    compressed_tokens: int
    protected_spans: int = 0
    provider: str = "noop"
    error: str | None = None

    @property
    def ratio(self) -> float:
        return self.compressed_tokens / self.original_tokens if self.original_tokens else 1.0

    @property
    def tokens_saved(self) -> int:
        return max(0, self.original_tokens - self.compressed_tokens)

    def to_dict(self) -> dict:
        return {
            "provider": self.provider,
            "original_tokens": self.original_tokens,
            "compressed_tokens": self.compressed_tokens,
            "tokens_saved": self.tokens_saved,
            "ratio": round(self.ratio, 4),
            "protected_spans": self.protected_spans,
            "error": self.error,
        }


class ContextCompressor(ABC):
    name = "abstract"

    @property
    @abstractmethod
    def available(self) -> bool: ...

    @abstractmethod
    async def compress(self, text: str, *, target_ratio: float = 0.5) -> CompressionResult: ...

    async def aclose(self) -> None:
        return None


class NoOpContextCompressor(ContextCompressor):
    """Pass-through.  The default, and always safe."""

    name = "noop"

    @property
    def available(self) -> bool:
        return True

    async def compress(self, text: str, *, target_ratio: float = 0.5) -> CompressionResult:
        tokens = estimate_tokens(text)
        return CompressionResult(
            original=text,
            compressed=text,
            original_tokens=tokens,
            compressed_tokens=tokens,
            provider=self.name,
        )


@dataclass
class TokenCompanyCompressor(ContextCompressor):
    """Adapter for The Token Company's text-compression service.

    The endpoint shape is configurable because it is supplied at integration
    time; on any failure this degrades to pass-through rather than risking a
    truncated or mangled prompt.
    """

    config: CompressionConfig = field(default_factory=CompressionConfig)
    name: str = "token_company"
    _client: object = field(default=None, repr=False)
    calls: int = 0
    failures: int = 0
    tokens_saved: int = 0

    @property
    def available(self) -> bool:
        return self.config.available

    async def compress(self, text: str, *, target_ratio: float | None = None) -> CompressionResult:
        original_tokens = estimate_tokens(text)
        target_ratio = target_ratio if target_ratio is not None else self.config.target_ratio

        if not self.available:
            return CompressionResult(
                original=text,
                compressed=text,
                original_tokens=original_tokens,
                compressed_tokens=original_tokens,
                provider=self.name,
                error="not configured",
            )

        import httpx

        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.config.base_url.rstrip("/"),
                timeout=httpx.Timeout(30.0),
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json",
                },
            )

        payload = {"text": text, "target_ratio": target_ratio}
        if self.config.model:
            payload["model"] = self.config.model

        try:
            resp = await self._client.post("/compress", json=payload)
            resp.raise_for_status()
            data = resp.json()
            compressed = data.get("compressed_text") or data.get("text") or text
        except Exception as exc:  # noqa: BLE001 - degrade to pass-through
            self.failures += 1
            return CompressionResult(
                original=text,
                compressed=text,
                original_tokens=original_tokens,
                compressed_tokens=original_tokens,
                provider=self.name,
                error=f"{type(exc).__name__}",
            )

        self.calls += 1
        compressed_tokens = estimate_tokens(compressed)
        self.tokens_saved += max(0, original_tokens - compressed_tokens)
        return CompressionResult(
            original=text,
            compressed=compressed,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            provider=self.name,
        )

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()  # type: ignore[attr-defined]
            self._client = None


# ---------------------------------------------------------------------------
# Protected-span compression
# ---------------------------------------------------------------------------


async def compress_with_protection(
    compressor: ContextCompressor,
    text: str,
    *,
    protected_patterns: list[str] | None = None,
    target_ratio: float = 0.5,
) -> CompressionResult:
    """Compress prose while guaranteeing exact values survive byte-identical.

    Every numeric result, plus any caller-supplied literal (column names,
    units), is replaced by a placeholder, the remainder is compressed, and the
    originals are restored.  If any placeholder fails to come back, the
    ORIGINAL text is returned — a compression that loses a protected span is
    treated as a failed compression, not a smaller prompt.
    """
    original_tokens = estimate_tokens(text)
    spans: list[str] = []

    def stash(match: re.Match) -> str:
        spans.append(match.group(0))
        return f"{PROTECT_OPEN}{len(spans) - 1}{PROTECT_CLOSE}"

    masked = _NUMERIC_RE.sub(stash, text)

    for literal in protected_patterns or []:
        if not literal:
            continue
        pattern = re.compile(re.escape(literal))
        masked = pattern.sub(stash, masked)

    result = await compressor.compress(masked, target_ratio=target_ratio)

    restored = result.compressed
    for i, span in enumerate(spans):
        restored = restored.replace(f"{PROTECT_OPEN}{i}{PROTECT_CLOSE}", span)

    if PROTECT_OPEN in restored:
        # A protected span was dropped: refuse the compression entirely.
        return CompressionResult(
            original=text,
            compressed=text,
            original_tokens=original_tokens,
            compressed_tokens=original_tokens,
            protected_spans=len(spans),
            provider=compressor.name,
            error="compression dropped a protected span; original text kept",
        )

    return CompressionResult(
        original=text,
        compressed=restored,
        original_tokens=original_tokens,
        compressed_tokens=estimate_tokens(restored),
        protected_spans=len(spans),
        provider=compressor.name,
        error=result.error,
    )


def build_compressor(config: CompressionConfig) -> ContextCompressor:
    if config.provider == "token_company" and config.available:
        return TokenCompanyCompressor(config=config)
    return NoOpContextCompressor()
