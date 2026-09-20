"""In-process evidence store with optional JSONL persistence.

This is the unit-test store AND the local-demo fallback, so it implements the
same hybrid retrieval the Elasticsearch store does — BM25 plus cosine
similarity, merged with Reciprocal Rank Fusion. Matching the production
semantics matters: if the fallback ranked differently, tests would validate a
retrieval behaviour the demo never uses.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from signal_engine.evidence.base import (
    Embedder,
    EvidenceStore,
    HashingEmbedder,
    RetrievalQuery,
    RetrievalResult,
    bm25_scores,
    cosine_similarity,
    reciprocal_rank_fusion,
    tokenize_for_bm25,
)
from signal_engine.evidence.schemas import EvidenceObject, ExplanationStatus


@dataclass
class MemoryEvidenceStore(EvidenceStore):
    """Dict-backed evidence store with hybrid retrieval."""

    name: str = "memory"
    persist_path: Path | None = None
    embedder: Embedder = field(default_factory=HashingEmbedder)
    _items: dict[str, EvidenceObject] = field(default_factory=dict)
    _tokens: dict[str, list[str]] = field(default_factory=dict)
    stats: dict = field(default_factory=lambda: {"writes": 0, "reads": 0, "searches": 0})

    def __post_init__(self) -> None:
        if self.persist_path is not None:
            self.persist_path = Path(self.persist_path)
            if self.persist_path.exists():
                self._load()

    @property
    def available(self) -> bool:
        return True

    async def ensure_ready(self) -> None:
        if self.persist_path is not None:
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- writes ---------------------------------------------------------
    async def put(self, evidence: EvidenceObject) -> str:
        if evidence.embedding is None:
            evidence.embedding = self.embedder.embed(evidence.semantic_search_text)
        self._items[evidence.evidence_id] = evidence
        self._tokens[evidence.evidence_id] = tokenize_for_bm25(evidence.semantic_search_text)
        self.stats["writes"] += 1
        self._append(evidence)
        return evidence.evidence_id

    async def put_many(self, evidences: list[EvidenceObject]) -> list[str]:
        return [await self.put(e) for e in evidences]

    async def update_status(
        self,
        evidence_id: str,
        status: ExplanationStatus,
        *,
        explanation: str = "",
        confidence: float | None = None,
        resolved_by: list[str] | None = None,
    ) -> bool:
        item = self._items.get(evidence_id)
        if item is None:
            return False
        if status is ExplanationStatus.EXPLAINED and resolved_by:
            item.mark_resolved(
                by_ids=resolved_by,
                explanation=explanation or item.textual_summary,
                confidence=confidence if confidence is not None else item.explanation_confidence,
            )
        else:
            item.explanation_status = status
            if explanation:
                item.textual_summary = explanation
            if confidence is not None:
                item.explanation_confidence = confidence
            item.semantic_search_text = item.build_search_text()
        item.embedding = self.embedder.embed(item.semantic_search_text)
        self._tokens[evidence_id] = tokenize_for_bm25(item.semantic_search_text)
        self._rewrite()
        return True

    # ---- reads ----------------------------------------------------------
    async def get(self, evidence_id: str) -> EvidenceObject | None:
        self.stats["reads"] += 1
        return self._items.get(evidence_id)

    async def count(self, *, analysis_id: str | None = None) -> int:
        if analysis_id is None:
            return len(self._items)
        return sum(1 for e in self._items.values() if e.analysis_id == analysis_id)

    async def all(self) -> list[EvidenceObject]:
        return list(self._items.values())

    async def search(self, query: RetrievalQuery) -> list[RetrievalResult]:
        """Structured filters, then BM25 + vector, fused with RRF."""
        self.stats["searches"] += 1
        candidates = [e for e in self._items.values() if self._passes_filters(e, query)]
        if not candidates:
            return []

        candidate_ids = {e.evidence_id for e in candidates}
        query_text = self._query_text(query)

        lexical = bm25_scores(
            tokenize_for_bm25(query_text),
            {cid: self._tokens.get(cid, []) for cid in candidate_ids},
        )
        lexical_ranked = [d for d, _ in sorted(lexical.items(), key=lambda kv: -kv[1])]

        query_vec = self.embedder.embed(query_text)
        vector: dict[str, float] = {}
        for e in candidates:
            if e.embedding:
                sim = cosine_similarity(query_vec, e.embedding)
                if sim > 0:
                    vector[e.evidence_id] = sim
        vector_ranked = [d for d, _ in sorted(vector.items(), key=lambda kv: -kv[1])]

        fused = reciprocal_rank_fusion(
            [lexical_ranked, vector_ranked],
            weights=[query.lexical_weight, query.vector_weight],
        )

        # A filter-only query ("every unresolved finding for this dataset") has
        # no text for either stream to rank on. The structured filters are
        # still a complete answer, so fall back to returning them by effect
        # size rather than reporting nothing.
        if not fused:
            for candidate in candidates:
                fused[candidate.evidence_id] = candidate.statistical_metrics.effect

        # Direct feature-name overlap is a strong, cheap signal that neither
        # text stream captures reliably; it gets an explicit boost.
        wanted = {f.lower() for f in query.feature_names}
        for e in candidates:
            if wanted & {f.lower() for f in e.feature_names}:
                fused[e.evidence_id] = fused.get(e.evidence_id, 0.0) + 0.02

        lex_pos = {d: i + 1 for i, d in enumerate(lexical_ranked)}
        vec_pos = {d: i + 1 for i, d in enumerate(vector_ranked)}

        results: list[RetrievalResult] = []
        for doc_id, score in sorted(fused.items(), key=lambda kv: -kv[1])[: query.limit]:
            evidence = self._items[doc_id]
            matched = []
            if doc_id in lex_pos:
                matched.append("lexical")
            if doc_id in vec_pos:
                matched.append("vector")
            if wanted & {f.lower() for f in evidence.feature_names}:
                matched.append("feature_overlap")
            results.append(
                RetrievalResult(
                    evidence=evidence,
                    score=score,
                    lexical_rank=lex_pos.get(doc_id),
                    vector_rank=vec_pos.get(doc_id),
                    matched_on=matched,
                )
            )
        return results

    # ---- helpers --------------------------------------------------------
    @staticmethod
    def _query_text(query: RetrievalQuery) -> str:
        return " ".join([query.text, " ".join(query.feature_names), " ".join(query.tags)]).strip()

    @staticmethod
    def _passes_filters(evidence: EvidenceObject, query: RetrievalQuery) -> bool:
        if query.exclude_ids and evidence.evidence_id in query.exclude_ids:
            return False
        if query.dataset_id and evidence.dataset_id != query.dataset_id:
            return False
        if query.dataset_fingerprint and evidence.dataset_fingerprint != query.dataset_fingerprint:
            return False
        if query.analysis_id and evidence.analysis_id != query.analysis_id:
            return False
        if query.statuses and evidence.explanation_status not in query.statuses:
            return False
        if query.tags and not (set(query.tags) & set(evidence.tags)):
            return False
        if query.min_effect is not None and evidence.statistical_metrics.effect < query.min_effect:
            return False
        return True

    # ---- persistence ----------------------------------------------------
    def _append(self, evidence: EvidenceObject) -> None:
        if self.persist_path is None:
            return
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        with self.persist_path.open("a") as fh:
            fh.write(json.dumps(evidence.to_dict(include_embedding=False)) + "\n")

    def _rewrite(self) -> None:
        if self.persist_path is None:
            return
        tmp = self.persist_path.with_suffix(".jsonl.part")
        with tmp.open("w") as fh:
            for e in self._items.values():
                fh.write(json.dumps(e.to_dict(include_embedding=False)) + "\n")
        tmp.replace(self.persist_path)

    def _load(self) -> None:
        if self.persist_path is None or not self.persist_path.exists():
            return
        for line in self.persist_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                evidence = EvidenceObject.from_dict(json.loads(line))
            except Exception:  # noqa: BLE001 - skip a corrupt line, keep the rest
                continue
            evidence.embedding = self.embedder.embed(evidence.semantic_search_text)
            self._items[evidence.evidence_id] = evidence
            self._tokens[evidence.evidence_id] = tokenize_for_bm25(evidence.semantic_search_text)

    def clear(self) -> None:
        self._items.clear()
        self._tokens.clear()
