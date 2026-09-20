"""Live Elasticsearch integration tests.

Spec section 15.8.  These run ONLY when credentials are configured, and they
use a `_test_` index prefix throughout.  `delete_index()` refuses any index
whose name does not contain "test", so a misconfigured run cannot drop the
real evidence index.

They are native async tests: the Elasticsearch client binds its connection
pool to the event loop it was created on, so the fixture and the test body
must share one loop.
"""

from __future__ import annotations

import uuid

import pytest
import pytest_asyncio

from signal_engine.config import load_settings
from signal_engine.evidence.base import RetrievalQuery
from signal_engine.evidence.schemas import (
    EvidenceObject,
    ExplanationStatus,
    StatisticalMetrics,
)
from tests.conftest import requires_elastic

pytestmark = [pytest.mark.integration, requires_elastic]


@pytest_asyncio.fixture
async def store():
    """A store bound to a throwaway, uniquely-named test index."""
    from signal_engine.evidence.elastic import ElasticsearchEvidenceStore

    settings = load_settings()
    config = settings.elastic.__class__(
        **{
            **settings.elastic.__dict__,
            "index_prefix": f"{settings.elastic.index_prefix}_test_{uuid.uuid4().hex[:8]}",
        }
    )
    s = ElasticsearchEvidenceStore(config=config)
    await s.ensure_ready()
    yield s
    try:
        await s.delete_index()
    finally:
        await s.close()


@pytest_asyncio.fixture
async def populated(store):
    await store.put_many(
        [
            make_evidence(
                feature_names=["is_airport_pickup", "fare_per_mile"],
                textual_summary="Airport pickups show an unusually high fare per mile.",
                explanation_status=ExplanationStatus.UNRESOLVED,
                tags=["surprising"],
            ),
            make_evidence(
                feature_names=["is_airport_pickup", "airport_fee"],
                textual_summary="Airport pickups carry a fixed airport surcharge.",
                explanation_status=ExplanationStatus.EXPLAINED,
            ),
            make_evidence(
                feature_names=["passenger_count", "VendorID"],
                textual_summary="Passenger count barely varies by vendor.",
                explanation_status=ExplanationStatus.EXPLAINED,
            ),
        ]
    )
    await store.refresh()
    return store


def make_evidence(**kwargs) -> EvidenceObject:
    defaults = {
        "dataset_id": "nyc_test",
        "dataset_fingerprint": "fp_test",
        "analysis_id": "an_test",
        "feature_names": ["trip_distance", "fare_amount"],
        "statistical_metrics": StatisticalMetrics(
            n=1000, pearson_r=0.8, effect=0.8, direction="positive", strength="very strong"
        ),
    }
    defaults.update(kwargs)
    return EvidenceObject(**defaults)


class TestConnection:
    async def test_ping_succeeds(self, store):
        info = await store.ping()
        assert info["ok"] is True
        assert info["version"]

    async def test_index_is_created_idempotently(self, store):
        await store.ensure_ready()
        await store.ensure_ready()
        assert await store.count() == 0


class TestWriteAndRead:
    async def test_write_then_read_exact(self, store):
        evidence = make_evidence()
        await store.put(evidence)
        await store.refresh()

        fetched = await store.get(evidence.evidence_id)
        assert fetched is not None
        assert fetched.evidence_id == evidence.evidence_id
        assert fetched.statistical_metrics.pearson_r == pytest.approx(0.8)
        assert fetched.feature_names == evidence.feature_names
        assert fetched.explanation_status is evidence.explanation_status

    async def test_missing_document_returns_none(self, store):
        assert await store.get("ev_does_not_exist") is None

    async def test_bulk_write(self, store):
        items = [make_evidence(feature_names=[f"col_{i}", "fare_amount"]) for i in range(12)]
        ids = await store.put_many(items)
        await store.refresh()
        assert len(ids) == 12
        assert await store.count() == 12

    async def test_count_filters_by_analysis(self, store):
        await store.put(make_evidence(analysis_id="an_a"))
        await store.put(make_evidence(analysis_id="an_b"))
        await store.refresh()
        assert await store.count(analysis_id="an_a") == 1

    async def test_embedding_is_generated_on_write(self, store):
        evidence = make_evidence()
        assert evidence.embedding is None
        document = store._document(evidence)
        assert len(document["embedding"]) == store.config.embedding_dims


class TestHybridSearch:
    async def test_semantic_search_finds_related_evidence(self, populated):
        results = await populated.search(
            RetrievalQuery(text="airport surcharge fee premium", limit=5)
        )
        assert results
        assert "is_airport_pickup" in set(results[0].evidence.feature_names)

    async def test_feature_name_retrieval(self, populated):
        results = await populated.search(
            RetrievalQuery(feature_names=["is_airport_pickup"], limit=5)
        )
        assert results
        assert all("is_airport_pickup" in r.evidence.feature_names for r in results[:2])

    async def test_status_filter(self, populated):
        results = await populated.search(
            RetrievalQuery(text="airport", statuses=[ExplanationStatus.UNRESOLVED], limit=10)
        )
        assert results
        assert all(
            r.evidence.explanation_status is ExplanationStatus.UNRESOLVED for r in results
        )

    async def test_tag_filter(self, populated):
        results = await populated.search(
            RetrievalQuery(text="airport", tags=["surprising"], limit=10)
        )
        assert results
        assert all("surprising" in r.evidence.tags for r in results)

    async def test_dataset_filter_isolates(self, populated):
        await populated.put(
            make_evidence(dataset_id="other_dataset", feature_names=["airport_fee"])
        )
        await populated.refresh()
        results = await populated.search(
            RetrievalQuery(text="airport", dataset_id="nyc_test", limit=10)
        )
        assert results
        assert all(r.evidence.dataset_id == "nyc_test" for r in results)

    async def test_exclusion_list(self, populated):
        first = await populated.search(RetrievalQuery(text="airport", limit=5))
        excluded = first[0].evidence.evidence_id
        again = await populated.search(
            RetrievalQuery(text="airport", exclude_ids=[excluded], limit=5)
        )
        assert excluded not in [r.evidence.evidence_id for r in again]

    async def test_retrieval_mode_is_recorded(self, populated):
        await populated.search(RetrievalQuery(text="airport", limit=3))
        mode = populated.stats["retrieval_mode"]
        assert mode in {"native_rrf", "client_side_rrf"}, f"unexpected retrieval mode: {mode}"

    async def test_min_effect_filter(self, populated):
        await populated.put(
            make_evidence(
                feature_names=["weak_signal", "fare_amount"],
                statistical_metrics=StatisticalMetrics(n=100, effect=0.02),
            )
        )
        await populated.refresh()
        results = await populated.search(
            RetrievalQuery(text="fare airport", min_effect=0.5, limit=10)
        )
        assert all(r.evidence.statistical_metrics.effect >= 0.5 for r in results)


class TestStatusLifecycle:
    async def test_unresolved_is_promoted_to_explained(self, store):
        """The core product loop, exercised against the real store."""
        unresolved = make_evidence(
            feature_names=["is_airport_pickup", "fare_per_mile"],
            textual_summary="Airport pickups show a high fare per mile.",
            explanation_status=ExplanationStatus.UNRESOLVED,
        )
        await store.put(unresolved)
        await store.refresh()

        resolver = make_evidence(feature_names=["is_airport_pickup", "airport_fee"])
        ok = await store.update_status(
            unresolved.evidence_id,
            ExplanationStatus.EXPLAINED,
            explanation="The premium may be partly explained by fixed airport fees and tolls.",
            confidence=0.75,
            resolved_by=[resolver.evidence_id],
        )
        assert ok
        await store.refresh()

        updated = await store.get(unresolved.evidence_id)
        assert updated.explanation_status is ExplanationStatus.EXPLAINED
        assert resolver.evidence_id in updated.resolved_by_ids
        assert "fixed airport fees" in updated.textual_summary
        assert updated.updated_at

    async def test_update_of_a_missing_document_returns_false(self, store):
        assert await store.update_status("ev_nope", ExplanationStatus.EXPLAINED) is False


class TestSafety:
    async def test_delete_refuses_a_non_test_index(self):
        """The guard that protects a production index from a bad test run."""
        from signal_engine.evidence.elastic import ElasticsearchEvidenceStore

        settings = load_settings()
        config = settings.elastic.__class__(
            **{**settings.elastic.__dict__, "index_prefix": "hackmit_signal_production"}
        )
        s = ElasticsearchEvidenceStore(config=config)
        with pytest.raises(RuntimeError, match="refusing to delete"):
            await s.delete_index()

    def test_errors_never_leak_credentials(self):
        """Transport errors can echo request headers; the scrubber must redact
        anything key-shaped before it reaches a log or an HTTP response.

        The token below is synthetic. Never put real credential material in a
        test fixture, even a fragment of one -- test files are committed.
        """
        from signal_engine.evidence.elastic import _safe

        fake_token = "AAAAfakeAAAAtestAAAAtokenAAAAnotArealKeyAAAA1234"
        message = _safe(Exception(f"connection failed: Authorization: Bearer {fake_token}"))
        assert fake_token not in message
        assert "***" in message

        keyed = _safe(Exception(f"request rejected: api_key={fake_token}"))
        assert fake_token not in keyed
