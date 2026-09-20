"""LLM contract tests: the engine must survive everything a model can do wrong.

Spec section 15.9.  Every test here uses a mock provider — no network, no keys.
The contract is that NONE of these inputs may crash the engine or allow
unvalidated model output to reach the data layer.
"""

from __future__ import annotations

import asyncio

import pytest

from signal_engine.llm.base import (
    ChatMessage,
    LLMBadResponse,
    LLMProvider,
    LLMResponse,
    LLMUnavailable,
    ModelUsage,
    extract_json,
)
from signal_engine.llm.cache import LLMCache
from signal_engine.llm.llama import UnavailableProvider
from signal_engine.llm.schemas import (
    ConclusionStatus,
    Direction,
    Hypothesis,
    HypothesisBatch,
    Priority,
    VisualInterpretation,
)


class MockProvider(LLMProvider):
    """A provider that returns scripted content."""

    name = "mock"

    def __init__(self, responses: list[str], *, fail_with: Exception | None = None) -> None:
        self.responses = responses
        self.fail_with = fail_with
        self.calls = 0
        self.received: list[list[ChatMessage]] = []

    @property
    def available(self) -> bool:
        return True

    async def complete(self, messages, **kwargs) -> LLMResponse:  # noqa: ANN001
        self.received.append(messages)
        if self.fail_with is not None:
            raise self.fail_with
        content = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return LLMResponse(
            content=content,
            model="mock-model",
            usage=ModelUsage(prompt_tokens=100, completion_tokens=50),
            latency_seconds=0.01,
        )


class TestJSONExtraction:
    """Models wrap JSON in prose, fences, and trailing commas."""

    def test_bare_json(self):
        assert extract_json('{"a": 1}') == {"a": 1}

    def test_fenced_json(self):
        assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}

    def test_unlabelled_fence(self):
        assert extract_json('```\n{"a": 1}\n```') == {"a": 1}

    def test_json_surrounded_by_prose(self):
        text = 'Here is my analysis:\n{"a": 1, "b": [2, 3]}\nHope that helps!'
        assert extract_json(text) == {"a": 1, "b": [2, 3]}

    def test_trailing_commas_are_tolerated(self):
        assert extract_json('{"a": 1, "b": 2,}') == {"a": 1, "b": 2}

    def test_braces_inside_strings_do_not_confuse_the_scanner(self):
        assert extract_json('{"note": "use {curly} braces", "n": 1}')["n"] == 1

    def test_escaped_quotes_are_handled(self):
        assert extract_json('{"note": "he said \\"hi\\"", "n": 2}')["n"] == 2

    def test_array_at_the_top_level(self):
        assert extract_json("[1, 2, 3]") == [1, 2, 3]

    def test_empty_response_raises(self):
        with pytest.raises(LLMBadResponse):
            extract_json("")

    def test_non_json_raises(self):
        with pytest.raises(LLMBadResponse):
            extract_json("I'm sorry, I cannot help with that request.")


class TestSchemaTolerance:
    """Accept permissively, produce strictly."""

    def test_uppercase_enum_is_coerced(self):
        h = Hypothesis(base_features=["a"], priority="HIGH", expected_direction="POSITIVE")
        assert h.priority is Priority.HIGH
        assert h.expected_direction is Direction.POSITIVE

    def test_unknown_enum_falls_back_to_a_default(self):
        h = Hypothesis(base_features=["a"], priority="extremely urgent")
        assert h.priority is Priority.MEDIUM

    def test_numeric_priority_is_bucketed(self):
        assert Hypothesis(base_features=["a"], priority=0.9).priority is Priority.HIGH
        assert Hypothesis(base_features=["a"], priority=0.1).priority is Priority.LOW

    def test_string_where_a_list_belongs(self):
        h = Hypothesis(base_features="fare_amount")
        assert h.base_features == ["fare_amount"]

    def test_out_of_range_floats_are_clamped(self):
        h = Hypothesis(base_features=["a"], estimated_information_gain=99.0)
        assert h.estimated_information_gain == 1.0

    def test_confidence_on_a_0_100_scale_is_normalized(self):
        assert VisualInterpretation(confidence=85).confidence == pytest.approx(0.85)

    def test_hypothesis_with_no_features_is_rejected(self):
        with pytest.raises(Exception):
            Hypothesis(base_features=[], transformation_candidates=[])

    def test_batch_accepts_a_single_object(self):
        batch = HypothesisBatch(hypotheses={"base_features": ["a"]})
        assert len(batch.hypotheses) == 1

    def test_batch_accepts_none(self):
        assert HypothesisBatch(hypotheses=None).hypotheses == []

    def test_interpretation_defaults_to_unresolved(self):
        assert VisualInterpretation().conclusion_status is ConclusionStatus.UNRESOLVED


class TestHallucinatedColumns:
    """The gate that stops a made-up column from reaching the dataframe."""

    def test_unknown_columns_are_dropped_and_reported(self):
        h = Hypothesis(
            base_features=["trip_distance", "passenger_mood", "quantum_flux"],
            target_features=["fare_amount"],
        )
        clean, dropped = h.resolve_columns({"trip_distance", "fare_amount"})
        assert clean.base_features == ["trip_distance"]
        assert set(dropped) == {"passenger_mood", "quantum_flux"}

    def test_case_mismatch_is_resolved_not_dropped(self):
        h = Hypothesis(base_features=["airport_fee"])
        clean, dropped = h.resolve_columns({"Airport_fee"})
        assert clean.base_features == ["Airport_fee"]
        assert not dropped

    def test_every_field_is_cleaned(self):
        h = Hypothesis(
            base_features=["real"],
            target_features=["fake1"],
            proposed_controls=["fake2"],
            proposed_stratifications=["real"],
        )
        clean, dropped = h.resolve_columns({"real"})
        assert clean.target_features == []
        assert clean.proposed_controls == []
        assert clean.proposed_stratifications == ["real"]
        assert set(dropped) == {"fake1", "fake2"}


class TestStructuredCompletion:
    def test_valid_response_parses(self):
        provider = MockProvider(['{"hypotheses": [{"base_features": ["a"]}]}'])
        batch, response = asyncio.run(
            provider.complete_structured([ChatMessage("user", "go")], HypothesisBatch)
        )
        assert len(batch.hypotheses) == 1
        assert response.usage.total_tokens == 150

    def test_invalid_json_is_retried_with_the_error_fed_back(self):
        provider = MockProvider(
            ["this is not json at all", '{"hypotheses": [{"base_features": ["a"]}]}']
        )
        batch, _ = asyncio.run(
            provider.complete_structured([ChatMessage("user", "go")], HypothesisBatch, retries=1)
        )
        assert len(batch.hypotheses) == 1
        assert provider.calls == 2
        # The retry must actually tell the model what went wrong.
        assert "could not be parsed" in provider.received[1][-1].content

    def test_persistent_bad_json_raises_a_handled_error(self):
        provider = MockProvider(["nope", "still nope", "nope again"])
        with pytest.raises(LLMBadResponse):
            asyncio.run(
                provider.complete_structured([ChatMessage("user", "go")], HypothesisBatch, retries=1)
            )

    def test_provider_failure_surfaces_as_llm_unavailable(self):
        provider = MockProvider([], fail_with=LLMUnavailable("rate limited"))
        with pytest.raises(LLMUnavailable):
            asyncio.run(provider.complete_structured([ChatMessage("user", "go")], HypothesisBatch))

    def test_timeout_surfaces_cleanly(self):
        provider = MockProvider([], fail_with=asyncio.TimeoutError())
        with pytest.raises(asyncio.TimeoutError):
            asyncio.run(provider.complete([ChatMessage("user", "go")]))


class TestUnavailableProvider:
    def test_reports_itself_unavailable(self):
        assert UnavailableProvider().available is False

    def test_every_call_raises_the_handled_exception(self):
        provider = UnavailableProvider("no credentials")
        with pytest.raises(LLMUnavailable, match="no credentials"):
            asyncio.run(provider.complete([ChatMessage("user", "hi")]))


class TestLLMCache:
    def test_identical_requests_share_a_key(self, tmp_path):
        cache = LLMCache(directory=tmp_path)
        messages = [ChatMessage("user", "What drives fares?")]
        args = {"provider": "p", "model": "m", "temperature": 0.2, "max_tokens": 100, "json_mode": True}
        assert cache.key(messages, **args) == cache.key(messages, **args)

    def test_whitespace_differences_share_a_key(self, tmp_path):
        """Two users phrasing the same prompt differently should reuse work."""
        cache = LLMCache(directory=tmp_path)
        args = {"provider": "p", "model": "m", "temperature": 0.2, "max_tokens": 100, "json_mode": True}
        a = cache.key([ChatMessage("user", "What  drives\n\nfares?")], **args)
        b = cache.key([ChatMessage("user", "What drives fares?")], **args)
        assert a == b

    def test_different_models_do_not_share_a_key(self, tmp_path):
        cache = LLMCache(directory=tmp_path)
        messages = [ChatMessage("user", "q")]
        base = {"provider": "p", "temperature": 0.2, "max_tokens": 100, "json_mode": True}
        assert cache.key(messages, model="a", **base) != cache.key(messages, model="b", **base)

    def test_prompt_version_invalidates_the_cache(self, tmp_path):
        messages = [ChatMessage("user", "q")]
        args = {"provider": "p", "model": "m", "temperature": 0.2, "max_tokens": 100, "json_mode": True}
        assert LLMCache(directory=tmp_path, prompt_version="v1").key(messages, **args) != LLMCache(
            directory=tmp_path, prompt_version="v2"
        ).key(messages, **args)

    def test_high_temperature_is_not_cached(self):
        """Caching a sampled response would silently make it deterministic."""
        cache = LLMCache()
        assert cache.cacheable(0.2) is True
        assert cache.cacheable(0.9) is False

    def test_round_trip(self, tmp_path):
        cache = LLMCache(directory=tmp_path)
        key = "abc"
        cache.put(
            key,
            LLMResponse(
                content='{"ok": true}',
                model="m",
                usage=ModelUsage(prompt_tokens=10, completion_tokens=5),
                latency_seconds=1.5,
            ),
        )
        hit = cache.get(key)
        assert hit is not None
        assert hit.cache_hit is True
        assert hit.content == '{"ok": true}'
        assert cache.stats.tokens_saved == 15
        assert cache.stats.seconds_saved == 1.5

    def test_images_key_by_content_hash(self, tmp_path):
        cache = LLMCache(directory=tmp_path)
        args = {"provider": "p", "model": "m", "temperature": 0.2, "max_tokens": 100, "json_mode": True}
        a = ChatMessage("user", "look", images=["data:image/png;base64,AAAA"])
        b = ChatMessage("user", "look", images=["data:image/png;base64,BBBB"])
        assert cache.key([a], **args) != cache.key([b], **args)

    def test_corrupt_entry_is_a_miss(self, tmp_path):
        cache = LLMCache(directory=tmp_path)
        (tmp_path / "bad.json").write_text("{not json")
        assert cache.get("bad") is None


class TestMultimodalPayload:
    def test_text_only_message_is_a_plain_string(self):
        assert ChatMessage("user", "hello").to_payload()["content"] == "hello"

    def test_image_message_becomes_a_content_array(self):
        payload = ChatMessage("user", "look", images=["data:image/png;base64,AAA"]).to_payload()
        assert isinstance(payload["content"], list)
        assert payload["content"][0]["type"] == "text"
        assert payload["content"][1]["type"] == "image_url"
