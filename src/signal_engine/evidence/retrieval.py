"""EvidenceMemory — the service the search loop talks to.

Wraps a store with the behaviour that makes evidence memory a *product
feature* rather than a database:

  * write-behind queueing, so an Elasticsearch outage never loses a finding
    or stalls the analysis
  * retrieval tuned for the two questions the loop actually asks:
      "what do we already know about these features?"
      "which open questions might this new finding answer?"
  * joint reinterpretation: when a new finding relates to an UNRESOLVED one,
    ask the model to reconsider them together, and promote the old record if
    the combination explains it
"""

from __future__ import annotations

from dataclasses import dataclass, field

from signal_engine.evidence.base import EvidenceStore, RetrievalQuery, RetrievalResult
from signal_engine.evidence.memory import MemoryEvidenceStore
from signal_engine.evidence.schemas import EvidenceObject, ExplanationStatus
from signal_engine.llm.base import LLMProvider, LLMUnavailable
from signal_engine.llm.prompts import build_joint_reinterpretation_prompt
from signal_engine.llm.schemas import ConclusionStatus, JointReinterpretation


@dataclass
class MemoryStats:
    stored: int = 0
    retrieved: int = 0
    searches: int = 0
    queued_for_retry: int = 0
    write_failures: int = 0
    reinterpretations: int = 0
    resolutions: int = 0

    def to_dict(self) -> dict:
        return {
            "stored": self.stored,
            "retrieved": self.retrieved,
            "searches": self.searches,
            "queued_for_retry": self.queued_for_retry,
            "write_failures": self.write_failures,
            "reinterpretations": self.reinterpretations,
            "resolutions": self.resolutions,
        }


@dataclass
class EvidenceMemory:
    """The evidence-memory service."""

    store: EvidenceStore = field(default_factory=MemoryEvidenceStore)
    fallback: EvidenceStore | None = None
    stats: MemoryStats = field(default_factory=MemoryStats)
    pending: list[EvidenceObject] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    async def ensure_ready(self) -> None:
        try:
            await self.store.ensure_ready()
        except Exception as exc:  # noqa: BLE001
            self._degrade(f"store unavailable at startup: {type(exc).__name__}")

    # ---- writes ---------------------------------------------------------
    async def remember(self, evidence: EvidenceObject) -> str:
        """Store a finding.  Never raises; a failed write is queued, not lost."""
        try:
            evidence_id = await self.store.put(evidence)
            self.stats.stored += 1
            return evidence_id
        except Exception as exc:  # noqa: BLE001
            self.stats.write_failures += 1
            self.pending.append(evidence)
            self.stats.queued_for_retry = len(self.pending)
            self._degrade(f"write failed ({type(exc).__name__}); queued for retry")
            if self.fallback is not None:
                try:
                    return await self.fallback.put(evidence)
                except Exception:  # noqa: BLE001
                    pass
            return evidence.evidence_id

    async def flush_pending(self) -> int:
        """Retry queued writes.  Returns how many were persisted."""
        if not self.pending:
            return 0
        try:
            await self.store.put_many(self.pending)
        except Exception:  # noqa: BLE001
            return 0
        count = len(self.pending)
        self.stats.stored += count
        self.pending.clear()
        self.stats.queued_for_retry = 0
        return count

    # ---- reads ----------------------------------------------------------
    async def search(self, query: RetrievalQuery) -> list[RetrievalResult]:
        self.stats.searches += 1
        try:
            results = await self.store.search(query)
        except Exception as exc:  # noqa: BLE001
            self._degrade(f"search failed ({type(exc).__name__})")
            return []
        self.stats.retrieved += len(results)
        return results

    async def related_to(
        self,
        *,
        feature_names: list[str],
        text: str = "",
        dataset_id: str | None = None,
        exclude_ids: list[str] | None = None,
        limit: int = 5,
    ) -> list[RetrievalResult]:
        """"What do we already know about these features?" """
        return await self.search(
            RetrievalQuery(
                text=text,
                feature_names=feature_names,
                dataset_id=dataset_id,
                exclude_ids=exclude_ids or [],
                limit=limit,
            )
        )

    async def open_questions(
        self,
        *,
        feature_names: list[str],
        text: str = "",
        dataset_id: str | None = None,
        exclude_ids: list[str] | None = None,
        limit: int = 4,
    ) -> list[RetrievalResult]:
        """"Which unresolved findings might this new one bear on?" """
        return await self.search(
            RetrievalQuery(
                text=text,
                feature_names=feature_names,
                dataset_id=dataset_id,
                statuses=[ExplanationStatus.UNRESOLVED, ExplanationStatus.NEEDS_MORE_EVIDENCE],
                exclude_ids=exclude_ids or [],
                limit=limit,
            )
        )

    # ---- joint reinterpretation ----------------------------------------
    async def reinterpret_with_history(
        self,
        provider: LLMProvider,
        *,
        question: str,
        new_evidence: EvidenceObject,
        prior: list[EvidenceObject],
    ) -> JointReinterpretation | None:
        """Reconsider a new finding together with related prior findings.

        This is the payoff of the whole design: an observation that could not
        be explained on its own becomes explainable once a related observation
        arrives, exactly as a researcher's understanding accumulates.
        """
        if not prior or not provider.available:
            return None

        messages = build_joint_reinterpretation_prompt(
            question=question,
            new_evidence_text=new_evidence.to_compact_text(max_chars=700),
            prior_evidence_texts=[p.to_compact_text(max_chars=500) for p in prior],
        )
        try:
            joint, _ = await provider.complete_structured(messages, JointReinterpretation)
        except (LLMUnavailable, Exception):  # noqa: BLE001 - optional enrichment
            return None

        self.stats.reinterpretations += 1

        # Only promote records the model actually named AND that we retrieved,
        # so a hallucinated evidence_id cannot mutate the store.
        valid_ids = {p.evidence_id for p in prior}
        to_resolve = [i for i in joint.resolves_evidence_ids if i in valid_ids]

        if to_resolve and joint.conclusion_status is ConclusionStatus.EXPLAINED:
            for evidence_id in to_resolve:
                ok = await self._resolve(
                    evidence_id,
                    explanation=joint.combined_explanation,
                    confidence=joint.confidence,
                    resolved_by=[new_evidence.evidence_id],
                )
                if ok:
                    self.stats.resolutions += 1
                    new_evidence.supports_ids = list(
                        dict.fromkeys(new_evidence.supports_ids + [evidence_id])
                    )

        related = [i for i in joint.supporting_evidence_ids if i in valid_ids]
        new_evidence.related_evidence_ids = list(
            dict.fromkeys(new_evidence.related_evidence_ids + related)
        )
        new_evidence.contradicts_ids = list(
            dict.fromkeys(
                new_evidence.contradicts_ids
                + [i for i in joint.contradicting_evidence_ids if i in valid_ids]
            )
        )
        return joint

    async def _resolve(
        self, evidence_id: str, *, explanation: str, confidence: float, resolved_by: list[str]
    ) -> bool:
        try:
            return await self.store.update_status(
                evidence_id,
                ExplanationStatus.EXPLAINED,
                explanation=explanation,
                confidence=confidence,
                resolved_by=resolved_by,
            )
        except Exception as exc:  # noqa: BLE001
            self._degrade(f"status update failed ({type(exc).__name__})")
            return False

    # ---- housekeeping ---------------------------------------------------
    def _degrade(self, message: str) -> None:
        """Record a degradation.  Never silent, never fatal."""
        if message not in self.errors:
            self.errors.append(message)

    async def close(self) -> None:
        await self.flush_pending()
        try:
            await self.store.close()
        except Exception:  # noqa: BLE001
            pass

    def telemetry(self) -> dict:
        return {
            "store": self.store.name,
            "available": self.store.available,
            **self.stats.to_dict(),
            "degradations": self.errors,
            "store_stats": getattr(self.store, "stats", {}),
        }


def build_memory(settings, *, cache_dir=None) -> EvidenceMemory:
    """Build the evidence memory: Elasticsearch when configured, local otherwise."""
    local = MemoryEvidenceStore(
        persist_path=(cache_dir / "evidence.jsonl") if cache_dir else None
    )
    if not settings.elastic.available:
        return EvidenceMemory(store=local)

    try:
        from signal_engine.evidence.elastic import ElasticsearchEvidenceStore

        store = ElasticsearchEvidenceStore(config=settings.elastic)
        return EvidenceMemory(store=store, fallback=local)
    except Exception:  # noqa: BLE001 - missing package or bad config: stay local
        return EvidenceMemory(store=local)
