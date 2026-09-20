"""The evidence-store interface and the local embedding fallback.

`EvidenceStore` is deliberately small.  Elasticsearch is the production
implementation and the sponsor story, but the engine must run its full loop —
including retrieval-driven reinterpretation — with no Elastic credentials at
all, or the tests would require a cloud account and the demo would have a
single point of failure.

Embeddings are behind `Embedder` for the same reason.  The default is a local,
deterministic hashing embedder: not semantically strong, but real vectors that
exercise the whole vector path.  When a managed inference endpoint is
configured, Elasticsearch generates the embeddings instead.
"""

from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Protocol

from signal_engine.evidence.schemas import EvidenceObject, ExplanationStatus


@dataclass
class RetrievalQuery:
    """A hybrid-retrieval request against evidence memory."""

    text: str = ""
    feature_names: list[str] = field(default_factory=list)
    dataset_id: str | None = None
    dataset_fingerprint: str | None = None
    analysis_id: str | None = None
    statuses: list[ExplanationStatus] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    exclude_ids: list[str] = field(default_factory=list)
    min_effect: float | None = None
    limit: int = 8
    # Weights for reciprocal-rank fusion of the lexical and vector streams.
    lexical_weight: float = 1.0
    vector_weight: float = 1.0

    def to_dict(self) -> dict:
        return {
            "text": self.text[:200],
            "feature_names": self.feature_names,
            "dataset_id": self.dataset_id,
            "statuses": [s.value for s in self.statuses],
            "tags": self.tags,
            "limit": self.limit,
        }


@dataclass
class RetrievalResult:
    evidence: EvidenceObject
    score: float
    lexical_rank: int | None = None
    vector_rank: int | None = None
    matched_on: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "evidence_id": self.evidence.evidence_id,
            "score": round(self.score, 5),
            "lexical_rank": self.lexical_rank,
            "vector_rank": self.vector_rank,
            "matched_on": self.matched_on,
            "summary": self.evidence.to_compact_text(),
        }


class EvidenceStore(ABC):
    """Persist and retrieve evidence."""

    name: str = "abstract"

    @property
    @abstractmethod
    def available(self) -> bool: ...

    @abstractmethod
    async def ensure_ready(self) -> None:
        """Create indices/structures if needed.  Must be idempotent."""

    @abstractmethod
    async def put(self, evidence: EvidenceObject) -> str: ...

    @abstractmethod
    async def put_many(self, evidences: list[EvidenceObject]) -> list[str]: ...

    @abstractmethod
    async def get(self, evidence_id: str) -> EvidenceObject | None: ...

    @abstractmethod
    async def search(self, query: RetrievalQuery) -> list[RetrievalResult]: ...

    @abstractmethod
    async def update_status(
        self,
        evidence_id: str,
        status: ExplanationStatus,
        *,
        explanation: str = "",
        confidence: float | None = None,
        resolved_by: list[str] | None = None,
    ) -> bool: ...

    @abstractmethod
    async def count(self, *, analysis_id: str | None = None) -> int: ...

    async def close(self) -> None:
        return None


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


class Embedder(Protocol):
    dims: int

    def embed(self, text: str) -> list[float]: ...

    def embed_many(self, texts: list[str]) -> list[list[float]]: ...


_TOKEN_RE = re.compile(r"[a-z0-9_]+")

_STOPWORDS = frozenset(
    ["a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "of", "in", "on", "at", "to", "for", "with", "by", "from", "as", "and", "or", "not", "this", "that", "these", "those", "it", "its", "their", "there", "here", "what", "which", "who", "whom", "how", "when", "where", "why", "do", "does", "did", "done", "has", "have", "had", "having", "will", "would", "can", "could", "should", "may", "might", "must"]
)


class HashingEmbedder:
    """Deterministic local embeddings — the hashing-trick + n-gram baseline.

    Honest about what it is: this captures lexical overlap, not deep semantics.
    A query about "airport fees" finds evidence mentioning "airport" and "fee";
    it will not connect "taxi" to "cab" the way a trained model would.

    It exists so the vector path is exercised in tests and in credential-free
    demos.  With a managed inference endpoint configured, Elasticsearch does
    the embedding instead and this is bypassed.  Character n-grams are included
    so `fare_per_mile` and `fare_amount` land near each other despite sharing
    no whole token.
    """

    def __init__(self, dims: int = 384, seed: int = 17) -> None:
        self.dims = dims
        self.seed = seed

    def _tokens(self, text: str) -> list[str]:
        words = [w for w in _TOKEN_RE.findall(text.lower()) if w not in _STOPWORDS and len(w) > 1]
        grams: list[str] = []
        for w in words:
            grams.append(w)
            # Sub-token pieces: fare_per_mile -> fare, per, mile
            grams.extend(p for p in w.split("_") if len(p) > 2)
            for i in range(len(w) - 3):
                grams.append(f"#{w[i:i + 4]}")
        return grams

    def _bucket(self, token: str) -> tuple[int, float]:
        digest = hashlib.blake2b(
            token.encode(), digest_size=8, salt=str(self.seed).encode()[:8]
        ).digest()
        value = int.from_bytes(digest, "big")
        return value % self.dims, 1.0 if (value >> 63) & 1 else -1.0

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dims
        tokens = self._tokens(text)
        if not tokens:
            return vec
        # Sub-linear term weighting so a repeated word cannot dominate.
        counts: dict[str, int] = {}
        for t in tokens:
            counts[t] = counts.get(t, 0) + 1
        for token, count in counts.items():
            idx, sign = self._bucket(token)
            vec[idx] += sign * (1.0 + math.log(count))

        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    return max(-1.0, min(1.0, dot))  # both sides are pre-normalized


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    *,
    weights: list[float] | None = None,
    k: int = 60,
) -> dict[str, float]:
    """Merge ranked ID lists with Reciprocal Rank Fusion.

    RRF is the recommended way to combine a lexical (BM25) stream with a vector
    stream: the two produce scores on incomparable scales, and RRF uses only
    the RANKS, so no normalization or tuning is needed.

        score(d) = sum_i  w_i / (k + rank_i(d))
    """
    weights = weights or [1.0] * len(ranked_lists)
    scores: dict[str, float] = {}
    for ranked, weight in zip(ranked_lists, weights):
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (k + rank)
    return scores


def bm25_scores(
    query_tokens: list[str],
    documents: dict[str, list[str]],
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> dict[str, float]:
    """Plain BM25 over a small in-memory corpus (local store only)."""
    if not documents:
        return {}
    n_docs = len(documents)
    lengths = {doc_id: len(toks) for doc_id, toks in documents.items()}
    avg_len = sum(lengths.values()) / n_docs if n_docs else 0.0

    doc_freq: dict[str, int] = {}
    term_freq: dict[str, dict[str, int]] = {}
    for doc_id, tokens in documents.items():
        counts: dict[str, int] = {}
        for t in tokens:
            counts[t] = counts.get(t, 0) + 1
        term_freq[doc_id] = counts
        for t in counts:
            doc_freq[t] = doc_freq.get(t, 0) + 1

    scores: dict[str, float] = {}
    for doc_id in documents:
        score = 0.0
        dl = lengths[doc_id] or 1
        for term in query_tokens:
            tf = term_freq[doc_id].get(term, 0)
            if tf == 0:
                continue
            df = doc_freq.get(term, 0)
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / (avg_len or 1)))
        if score > 0:
            scores[doc_id] = score
    return scores


def tokenize_for_bm25(text: str) -> list[str]:
    return [w for w in _TOKEN_RE.findall(text.lower()) if w not in _STOPWORDS and len(w) > 1]
