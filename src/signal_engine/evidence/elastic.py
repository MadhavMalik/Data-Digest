"""Elasticsearch evidence store — hybrid retrieval over persistent memory.

What Elasticsearch is for here, and what it is NOT for:

  IS      persistent semantic memory of findings across runs and datasets;
          hybrid retrieval (BM25 + dense-vector kNN, merged with RRF) over
          those findings; structured filtering by dataset, status, and tags.

  IS NOT  the statistics engine.  It never computes a correlation.  Pushing
          the numerical work into Elasticsearch would be slower than Polars
          and NumPy on a local file and would couple correctness to a network
          service.

Retrieval strategy, in order of preference:
  1. native `rrf` retriever (Elasticsearch 8.15+/serverless) — the server
     fuses the lexical and vector streams
  2. two separate queries fused client-side with the same RRF formula
  3. BM25 only, if the vector field is unavailable

The fallback ladder is what stops a version difference from taking the demo
down; the resolved mode is recorded in telemetry so we can see which ran.

Security: API-key auth only, TLS verification always on, and no credential is
ever logged, echoed in an error, or returned by the API.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from signal_engine.config import ElasticConfig
from signal_engine.evidence.base import (
    Embedder,
    EvidenceStore,
    HashingEmbedder,
    RetrievalQuery,
    RetrievalResult,
    reciprocal_rank_fusion,
)
from signal_engine.evidence.schemas import EvidenceObject, ExplanationStatus

RRF_K = 60


def _evidence_mapping(dims: int, *, inference_endpoint: str = "") -> dict:
    """Index mapping.

    `semantic_search_text` is indexed as `text` for BM25 and mirrored into a
    `dense_vector` for kNN.  `dense_vector` uses `int8_hnsw` quantization:
    roughly 4x less memory than float32 with negligible recall loss at this
    corpus size, and Elasticsearch rescores from the full-precision values.
    """
    properties: dict[str, Any] = {
        "evidence_id": {"type": "keyword"},
        "analysis_id": {"type": "keyword"},
        "hypothesis_id": {"type": "keyword"},
        "dataset_id": {"type": "keyword"},
        "dataset_fingerprint": {"type": "keyword"},
        "user_question": {"type": "text"},
        "feature_names": {"type": "keyword"},
        "canonical_expression": {"type": "keyword"},
        "expression_hash": {"type": "keyword"},
        "transformation_description": {"type": "text"},
        "units": {"type": "object", "enabled": False},
        "statistical_metrics": {
            "properties": {
                "n": {"type": "long"},
                "pearson_r": {"type": "float"},
                "pearson_p": {"type": "float"},
                "spearman_rho": {"type": "float"},
                "mutual_information": {"type": "float"},
                "eta": {"type": "float"},
                "r_squared": {"type": "float"},
                "slope": {"type": "float"},
                "covariance": {"type": "float"},
                "effect": {"type": "float"},
                "direction": {"type": "keyword"},
                "strength": {"type": "keyword"},
                "stability": {"type": "float"},
                "q_value": {"type": "float"},
                "method": {"type": "keyword"},
            }
        },
        "sample_size": {"type": "long"},
        "filters": {"type": "keyword"},
        "stratification": {"type": "keyword"},
        "plot_type": {"type": "keyword"},
        "plot_uri": {"type": "keyword"},
        "plot_description": {"type": "text"},
        "textual_summary": {"type": "text"},
        "vlm_interpretation": {"type": "object", "enabled": False},
        "explanation_status": {"type": "keyword"},
        "explanation_confidence": {"type": "float"},
        "critic_report": {"type": "object", "enabled": False},
        "warnings": {"type": "text"},
        "parent_evidence_ids": {"type": "keyword"},
        "related_evidence_ids": {"type": "keyword"},
        "supports_ids": {"type": "keyword"},
        "contradicts_ids": {"type": "keyword"},
        "resolved_by_ids": {"type": "keyword"},
        "created_at": {"type": "date"},
        "updated_at": {"type": "date"},
        "tags": {"type": "keyword"},
        "provenance": {"type": "text"},
        "semantic_search_text": {"type": "text", "analyzer": "english"},
    }

    if inference_endpoint:
        # Let Elasticsearch generate and search embeddings itself.
        properties["semantic_body"] = {
            "type": "semantic_text",
            "inference_id": inference_endpoint,
        }
    else:
        properties["embedding"] = {
            "type": "dense_vector",
            "dims": dims,
            "index": True,
            "similarity": "cosine",
            "index_options": {"type": "int8_hnsw", "m": 16, "ef_construction": 100},
        }

    return {"mappings": {"properties": properties}}


@dataclass
class ElasticsearchEvidenceStore(EvidenceStore):
    """Production evidence store."""

    config: ElasticConfig = field(default_factory=ElasticConfig)
    embedder: Embedder = field(default_factory=HashingEmbedder)
    index_suffix: str = "evidence"
    name: str = "elasticsearch"
    _client: Any = field(default=None, repr=False)
    _ready: bool = False
    # Writes since the last refresh.  Elasticsearch indexing is near-real-time,
    # so a document written moments ago is not yet searchable.  Within a single
    # analysis the engine writes evidence and then immediately searches for
    # related evidence, so without a refresh it would never see its own recent
    # findings.  We refresh lazily -- once, before a search, only when dirty --
    # rather than on every write, which would be far more expensive.
    _dirty: bool = False
    stats: dict = field(
        default_factory=lambda: {
            "writes": 0,
            "reads": 0,
            "searches": 0,
            "errors": 0,
            "retrieval_mode": "unknown",
            "last_search_ms": 0.0,
        }
    )

    # ---- lifecycle ------------------------------------------------------
    @property
    def index(self) -> str:
        return self.config.index_name(self.index_suffix)

    @property
    def available(self) -> bool:
        return self.config.available

    def _build_client(self):
        try:
            from elasticsearch import AsyncElasticsearch
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "the `elasticsearch` package is required for the Elastic evidence store; "
                "install the `elastic` extra"
            ) from exc

        kwargs: dict = {
            "api_key": self.config.api_key,
            "request_timeout": self.config.request_timeout,
            # TLS verification stays ON. Disabling it would make the API key
            # interceptable, which is a far worse outcome than a setup error.
            "verify_certs": self.config.verify_certs,
        }
        if self.config.cloud_id:
            kwargs["cloud_id"] = self.config.cloud_id
        else:
            kwargs["hosts"] = [self.config.url]
        return AsyncElasticsearch(**kwargs)

    async def client(self):
        if self._client is None:
            if not self.available:
                raise RuntimeError(
                    "Elasticsearch is not configured; set ELASTIC_API_KEY and "
                    "ELASTIC_CLOUD_ID or ELASTIC_URL"
                )
            self._client = self._build_client()
        return self._client

    async def ensure_ready(self) -> None:
        if self._ready:
            return
        es = await self.client()
        exists = await es.indices.exists(index=self.index)
        if not exists:
            body = _evidence_mapping(
                self.config.embedding_dims,
                inference_endpoint=self.config.inference_endpoint_id,
            )
            await es.indices.create(index=self.index, **body)
        self._ready = True

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None
            self._ready = False

    async def ping(self) -> dict:
        """Connectivity probe.  Returns a status dict, never raises, and never
        includes credential material in the message."""
        if not self.available:
            return {"ok": False, "reason": "not configured"}
        try:
            es = await self.client()
            info = await es.info()
            return {
                "ok": True,
                "cluster_name": info.get("cluster_name", ""),
                "version": info.get("version", {}).get("number", ""),
                "index": self.index,
            }
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "reason": f"{type(exc).__name__}: {_safe(exc)}"}

    # ---- writes ---------------------------------------------------------
    # Fields mapped as `date`: Elasticsearch rejects "" for these, so an unset
    # timestamp must be omitted from the document rather than sent as empty.
    _DATE_FIELDS = ("created_at", "updated_at")

    def _document(self, evidence: EvidenceObject) -> dict:
        doc = evidence.to_dict(include_embedding=False)
        doc.pop("embedding", None)
        for field_name in self._DATE_FIELDS:
            if not doc.get(field_name):
                doc.pop(field_name, None)
        if self.config.inference_endpoint_id:
            doc["semantic_body"] = evidence.semantic_search_text
        else:
            doc["embedding"] = evidence.embedding or self.embedder.embed(
                evidence.semantic_search_text
            )
        return doc

    async def put(self, evidence: EvidenceObject) -> str:
        await self.ensure_ready()
        es = await self.client()
        await es.index(index=self.index, id=evidence.evidence_id, document=self._document(evidence))
        self.stats["writes"] += 1
        self._dirty = True
        return evidence.evidence_id

    async def put_many(self, evidences: list[EvidenceObject]) -> list[str]:
        if not evidences:
            return []
        await self.ensure_ready()
        es = await self.client()

        operations: list[dict] = []
        for e in evidences:
            operations.append({"index": {"_index": self.index, "_id": e.evidence_id}})
            operations.append(self._document(e))

        result = await es.bulk(operations=operations, refresh=False)
        if result.get("errors"):
            self.stats["errors"] += sum(
                1 for item in result.get("items", []) if item.get("index", {}).get("error")
            )
        self.stats["writes"] += len(evidences)
        self._dirty = True
        return [e.evidence_id for e in evidences]

    async def update_status(
        self,
        evidence_id: str,
        status: ExplanationStatus,
        *,
        explanation: str = "",
        confidence: float | None = None,
        resolved_by: list[str] | None = None,
    ) -> bool:
        existing = await self.get(evidence_id)
        if existing is None:
            return False
        if status is ExplanationStatus.EXPLAINED and resolved_by:
            existing.mark_resolved(
                by_ids=resolved_by,
                explanation=explanation or existing.textual_summary,
                confidence=confidence if confidence is not None else existing.explanation_confidence,
            )
        else:
            existing.explanation_status = status
            if explanation:
                existing.textual_summary = explanation
            if confidence is not None:
                existing.explanation_confidence = confidence
            existing.semantic_search_text = existing.build_search_text()
        existing.embedding = self.embedder.embed(existing.semantic_search_text)
        await self.put(existing)
        return True

    # ---- reads ----------------------------------------------------------
    async def get(self, evidence_id: str) -> EvidenceObject | None:
        await self.ensure_ready()
        es = await self.client()
        self.stats["reads"] += 1
        try:
            resp = await es.get(index=self.index, id=evidence_id)
        except Exception:  # noqa: BLE001 - NotFoundError and transport errors alike
            return None
        return EvidenceObject.from_dict(resp["_source"])

    async def count(self, *, analysis_id: str | None = None) -> int:
        await self.ensure_ready()
        es = await self.client()
        query = {"match_all": {}} if analysis_id is None else {"term": {"analysis_id": analysis_id}}
        resp = await es.count(index=self.index, query=query)
        return int(resp["count"])

    async def refresh(self) -> None:
        """Force a refresh so just-written docs are searchable (tests/demo)."""
        es = await self.client()
        await es.indices.refresh(index=self.index)
        self._dirty = False

    # ---- hybrid search --------------------------------------------------
    def _filters(self, query: RetrievalQuery) -> list[dict]:
        filters: list[dict] = []
        if query.dataset_id:
            filters.append({"term": {"dataset_id": query.dataset_id}})
        if query.dataset_fingerprint:
            filters.append({"term": {"dataset_fingerprint": query.dataset_fingerprint}})
        if query.analysis_id:
            filters.append({"term": {"analysis_id": query.analysis_id}})
        if query.statuses:
            filters.append({"terms": {"explanation_status": [s.value for s in query.statuses]}})
        if query.tags:
            filters.append({"terms": {"tags": query.tags}})
        if query.min_effect is not None:
            filters.append({"range": {"statistical_metrics.effect": {"gte": query.min_effect}}})
        if query.exclude_ids:
            # The exclusion must live in the FILTER list, not only in the
            # lexical query's must_not: the kNN retriever carries its own
            # filter, so an id excluded from one stream would otherwise come
            # straight back through the other.
            filters.append({"bool": {"must_not": [{"ids": {"values": query.exclude_ids}}]}})
        return filters

    async def search(self, query: RetrievalQuery) -> list[RetrievalResult]:
        import time

        await self.ensure_ready()
        es = await self.client()
        self.stats["searches"] += 1
        started = time.time()

        if self._dirty:
            try:
                await es.indices.refresh(index=self.index)
                self.stats["refreshes"] = self.stats.get("refreshes", 0) + 1
            except Exception:  # noqa: BLE001 - a failed refresh only costs recency
                pass
            self._dirty = False

        text = " ".join([query.text, " ".join(query.feature_names)]).strip()
        # `filters` already carries the exclusion, so both retrieval streams
        # honour it; `must_not` stays for the lexical query's own clarity.
        filters = self._filters(query)
        must_not = [{"ids": {"values": query.exclude_ids}}] if query.exclude_ids else []

        try:
            results = await self._search_native_rrf(es, query, text, filters, must_not)
            self.stats["retrieval_mode"] = "native_rrf"
        except Exception:  # noqa: BLE001 - older cluster, or no rrf licence
            try:
                results = await self._search_client_side_rrf(es, query, text, filters, must_not)
                self.stats["retrieval_mode"] = "client_side_rrf"
            except Exception as exc:  # noqa: BLE001
                self.stats["errors"] += 1
                self.stats["retrieval_mode"] = f"failed: {type(exc).__name__}"
                return []

        self.stats["last_search_ms"] = round((time.time() - started) * 1000, 2)
        return results

    def _lexical_query(self, text: str, filters: list[dict], must_not: list[dict]) -> dict:
        should: list[dict] = []
        if text:
            should.append(
                {
                    "multi_match": {
                        "query": text,
                        "fields": [
                            "semantic_search_text^2",
                            "textual_summary^1.5",
                            "transformation_description",
                            "feature_names^3",
                            "plot_description",
                            "user_question",
                        ],
                        "type": "best_fields",
                    }
                }
            )
        return {
            "bool": {
                "should": should or [{"match_all": {}}],
                "filter": filters,
                "must_not": must_not,
                "minimum_should_match": 1 if should else 0,
            }
        }

    async def _search_native_rrf(self, es, query, text, filters, must_not) -> list[RetrievalResult]:
        """Server-side RRF via the `retriever` API."""
        standard = {"standard": {"query": self._lexical_query(text, filters, must_not)}}

        if self.config.inference_endpoint_id:
            second = {
                "standard": {
                    "query": {
                        "bool": {
                            "must": [{"semantic": {"field": "semantic_body", "query": text}}],
                            "filter": filters,
                            "must_not": must_not,
                        }
                    }
                }
            }
        else:
            second = {
                "knn": {
                    "field": "embedding",
                    "query_vector": self.embedder.embed(text),
                    "k": max(query.limit * 4, 20),
                    "num_candidates": max(query.limit * 20, 100),
                    "filter": filters,
                }
            }

        body = {
            "retriever": {
                "rrf": {
                    "retrievers": [standard, second],
                    "rank_window_size": max(query.limit * 5, 50),
                    "rank_constant": RRF_K,
                }
            },
            "size": query.limit,
        }
        resp = await es.search(index=self.index, **body)
        return _to_results(resp, matched=["native_rrf"])

    async def _search_client_side_rrf(
        self, es, query, text, filters, must_not
    ) -> list[RetrievalResult]:
        """Run both streams separately and fuse the RANKS locally.

        Identical fusion maths to the server-side path, so ranking behaviour
        does not change with the cluster version.
        """
        window = max(query.limit * 5, 30)

        lexical_task = es.search(
            index=self.index, query=self._lexical_query(text, filters, must_not), size=window
        )
        vector_task = es.search(
            index=self.index,
            knn={
                "field": "embedding",
                "query_vector": self.embedder.embed(text),
                "k": window,
                "num_candidates": window * 4,
                "filter": filters,
            },
            size=window,
        )

        lexical_resp, vector_resp = await asyncio.gather(
            lexical_task, vector_task, return_exceptions=True
        )

        lexical_hits = _hits(lexical_resp)
        vector_hits = _hits(vector_resp)
        if not lexical_hits and not vector_hits:
            raise RuntimeError("both retrieval streams failed")

        lexical_ids = [h["_id"] for h in lexical_hits]
        vector_ids = [h["_id"] for h in vector_hits]

        fused = reciprocal_rank_fusion(
            [lexical_ids, vector_ids],
            weights=[query.lexical_weight, query.vector_weight],
            k=RRF_K,
        )

        sources = {h["_id"]: h["_source"] for h in lexical_hits + vector_hits}
        lex_pos = {d: i + 1 for i, d in enumerate(lexical_ids)}
        vec_pos = {d: i + 1 for i, d in enumerate(vector_ids)}

        out: list[RetrievalResult] = []
        for doc_id, score in sorted(fused.items(), key=lambda kv: -kv[1])[: query.limit]:
            matched = []
            if doc_id in lex_pos:
                matched.append("lexical")
            if doc_id in vec_pos:
                matched.append("vector")
            out.append(
                RetrievalResult(
                    evidence=EvidenceObject.from_dict(sources[doc_id]),
                    score=score,
                    lexical_rank=lex_pos.get(doc_id),
                    vector_rank=vec_pos.get(doc_id),
                    matched_on=matched,
                )
            )
        return out

    # ---- maintenance ----------------------------------------------------
    async def delete_index(self) -> bool:
        """Delete this store's index.

        Guarded: refuses unless the index name contains `test`, so a
        misconfigured test run can never drop a production index.
        """
        if "test" not in self.index.lower():
            raise RuntimeError(
                f"refusing to delete index {self.index!r}: only indices whose name contains "
                f"'test' may be deleted through this method"
            )
        es = await self.client()
        await es.indices.delete(index=self.index, ignore_unavailable=True)
        self._ready = False
        return True


def _hits(resp) -> list[dict]:
    if isinstance(resp, Exception) or resp is None:
        return []
    try:
        return list(resp["hits"]["hits"])
    except (KeyError, TypeError):
        return []


def _to_results(resp, *, matched: list[str]) -> list[RetrievalResult]:
    out: list[RetrievalResult] = []
    for rank, hit in enumerate(_hits(resp), start=1):
        out.append(
            RetrievalResult(
                evidence=EvidenceObject.from_dict(hit["_source"]),
                score=float(hit.get("_score") or 0.0),
                lexical_rank=rank,
                matched_on=list(matched),
            )
        )
    return out


def _safe(exc: Exception) -> str:
    """Error text with anything key-shaped stripped.

    Transport errors can echo request headers; this makes it structurally
    impossible for an API key to reach a log or an HTTP response.
    """
    import re

    text = str(exc)[:300]
    text = re.sub(r"(?i)(api[_-]?key|authorization|bearer)\s*[=:]\s*\S+", r"\1=***", text)
    return re.sub(r"\b[A-Za-z0-9+/=_-]{32,}\b", "***", text)
