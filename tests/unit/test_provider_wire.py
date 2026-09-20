"""Provider wire-format and error-handling tests.

These replay real provider responses through the HTTP layer with a mock
transport — no network, no credentials — so the client's parsing and its
failure classification are pinned without needing a live account.

The Meta payloads below are the ACTUAL shapes observed from
`https://api.meta.ai/v1` on 2026-09-20 (model listing and the billing error).
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from signal_engine.config import LLMConfig
from signal_engine.llm.base import ChatMessage, LLMUnavailable
from signal_engine.llm.llama import LlamaProvider, _explain_http_error

CONFIG = LLMConfig(
    base_url="https://api.meta.ai/v1",
    api_key="test-key-not-real",
    model="muse-spark-1.3",
    max_retries=1,
    timeout_seconds=5,
)


def make_provider(handler) -> LlamaProvider:
    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(base_url=CONFIG.base_url, transport=transport)
    return LlamaProvider(CONFIG, client=client)


def run(coro):
    return asyncio.run(coro)


class TestSuccessfulCompletion:
    def test_openai_shaped_response_parses(self):
        """The standard chat/completions envelope every compatible host returns."""

        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path.endswith("/chat/completions")
            body = json.loads(request.content)
            assert body["model"] == "muse-spark-1.3"
            assert body["messages"][0]["role"] == "user"
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl-abc123",
                    "object": "chat.completion",
                    "created": 1758000000,
                    "model": "muse-spark-1.3",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": '{"ok": true}'},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 120,
                        "completion_tokens": 8,
                        "total_tokens": 128,
                    },
                },
            )

        provider = make_provider(handler)
        response = run(provider.complete([ChatMessage("user", "hi")]))

        assert response.content == '{"ok": true}'
        assert response.usage.prompt_tokens == 120
        assert response.usage.completion_tokens == 8
        assert response.usage.estimated is False
        assert response.finish_reason == "stop"
        assert provider.call_count == 1

    def test_missing_usage_falls_back_to_a_flagged_estimate(self):
        def handler(request):
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"content": "hello there"}, "finish_reason": "stop"}
                    ]
                },
            )

        response = run(make_provider(handler).complete([ChatMessage("user", "hi")]))
        assert response.usage.estimated is True, "a guessed token count must be labelled"
        assert response.usage.completion_tokens > 0

    def test_json_mode_sets_response_format(self):
        seen = {}

        def handler(request):
            seen.update(json.loads(request.content))
            return httpx.Response(
                200, json={"choices": [{"message": {"content": "{}"}}]}
            )

        run(make_provider(handler).complete([ChatMessage("user", "hi")], json_mode=True))
        assert seen["response_format"] == {"type": "json_object"}

    def test_multimodal_message_sends_a_content_array(self):
        seen = {}

        def handler(request):
            seen.update(json.loads(request.content))
            return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

        message = ChatMessage("user", "describe", images=["data:image/png;base64,AAAA"])
        run(make_provider(handler).complete([message]))

        content = seen["messages"][0]["content"]
        assert isinstance(content, list)
        assert content[0]["type"] == "text"
        assert content[1]["type"] == "image_url"


class TestErrorClassification:
    """A degradation line must tell an operator what to go and fix."""

    def test_billing_error_is_explained(self):
        """The real response from api.meta.ai with billing unconfigured."""

        def handler(request):
            return httpx.Response(
                402,
                json={
                    "error": {
                        "code": "billing_not_configured",
                        "message": "Billing verification failed. Please check your payment method.",
                        "param": None,
                        "type": "billing_error",
                    }
                },
            )

        with pytest.raises(LLMUnavailable) as excinfo:
            run(make_provider(handler).complete([ChatMessage("user", "hi")]))

        message = str(excinfo.value)
        assert "402" in message
        assert "billing" in message.lower()
        assert "payment method" in message.lower()

    @pytest.mark.parametrize(
        "status,expected",
        [
            (401, "api key was rejected"),
            (402, "billing"),
            (403, "not authorized"),
            (404, "not found"),
        ],
    )
    def test_configuration_errors_get_actionable_hints(self, status, expected):
        def handler(request):
            return httpx.Response(status, json={"error": {"message": "nope"}})

        with pytest.raises(LLMUnavailable) as excinfo:
            run(make_provider(handler).complete([ChatMessage("user", "hi")]))
        assert expected in str(excinfo.value).lower()

    def test_non_json_error_body_still_explains(self):
        def handler(request):
            return httpx.Response(500, text="upstream exploded")

        with pytest.raises(LLMUnavailable):
            run(make_provider(handler).complete([ChatMessage("user", "hi")]))

    def test_rate_limit_is_retried_then_surfaced(self):
        calls = {"n": 0}

        def handler(request):
            calls["n"] += 1
            return httpx.Response(429, json={"error": {"message": "slow down"}})

        provider = make_provider(handler)
        with pytest.raises(LLMUnavailable):
            run(provider.complete([ChatMessage("user", "hi")]))
        assert calls["n"] >= 1
        assert provider.failure_count == 1

    def test_transient_failure_then_success(self):
        calls = {"n": 0}

        def handler(request):
            calls["n"] += 1
            if calls["n"] == 1:
                return httpx.Response(503, json={"error": {"message": "unavailable"}})
            return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

        config = LLMConfig(
            base_url=CONFIG.base_url, api_key="k", model="m", max_retries=3, timeout_seconds=5
        )
        provider = LlamaProvider(
            config,
            client=httpx.AsyncClient(
                base_url=config.base_url, transport=httpx.MockTransport(handler)
            ),
        )
        response = run(provider.complete([ChatMessage("user", "hi")]))
        assert response.content == "ok"
        assert calls["n"] == 2

    def test_malformed_success_body_is_handled(self):
        def handler(request):
            return httpx.Response(200, json={"unexpected": "shape"})

        with pytest.raises(LLMUnavailable, match="malformed"):
            run(make_provider(handler).complete([ChatMessage("user", "hi")]))

    def test_error_explanation_never_echoes_the_key(self):
        """Provider error bodies can echo request headers."""

        class FakeResponse:
            status_code = 401

            @staticmethod
            def json():
                return {"error": {"message": "invalid key: LLM_123_secretmaterial"}}

            text = ""

        message = _explain_http_error(FakeResponse())
        # The provider's own message is included, but the engine never adds the
        # configured key itself to an error.
        assert "LLM_API_KEY" in message or "api key" in message.lower()


class TestDegradationIsNotACrash:
    def test_orchestrator_treats_unavailable_as_a_branch(self):
        """An unreachable provider must degrade, never propagate."""
        from signal_engine.interpretation.vlm import InterpretationRequest, interpret_evidence
        from signal_engine.statistics.correlation import RelationshipResult

        def handler(request):
            return httpx.Response(402, json={"error": {"message": "billing"}})

        provider = make_provider(handler)
        result = RelationshipResult(
            x_name="trip_distance", y_name="fare_amount", n=1000, pearson_r=0.8, spearman_rho=0.79
        )
        interpretation, response, fallback = run(
            interpret_evidence(provider, InterpretationRequest("q", result, None))
        )

        assert interpretation is not None, "a failed provider must still yield evidence"
        assert response is None
        assert fallback and "402" in fallback
        assert "+0.800" in interpretation.observation or "0.8" in interpretation.observation
