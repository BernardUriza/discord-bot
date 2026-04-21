"""Integration-style tests for the Haiku → Sonnet fallback path inside LLMClient.chat.

Mocks _send so no real API calls happen. Exercises the control flow added
by the 3-tier router: distinct fallback on character break / anti-pattern,
legacy reinforced-retry when primary == fallback.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from insult.core.llm import LLMClient, LLMResponse


@pytest.fixture
def client():
    # No real network — AsyncAnthropic is instantiated but never used (we patch _send).
    with patch("insult.core.llm.anthropic.AsyncAnthropic"):
        return LLMClient(api_key="fake", model="sonnet-default", max_tokens=512, timeout=1.0, max_retries=1)


def _mk(text: str, model: str = "") -> LLMResponse:
    return LLMResponse(text=text, model_used=model)


async def _run(
    client: LLMClient, responses: list[LLMResponse], **chat_kwargs: Any
) -> tuple[LLMResponse, list[dict[str, Any]]]:
    """Helper: patch _send to return queued responses. Returns (result, per-call kwargs)."""
    queue = list(responses)
    calls: list[dict[str, Any]] = []

    async def fake_send(*_args: Any, **kwargs: Any) -> LLMResponse:
        assert queue, "chat called _send more times than expected"
        calls.append(kwargs)
        return queue.pop(0)

    with patch.object(client, "_send", side_effect=fake_send):
        result = await client.chat("system", [{"role": "user", "content": "hi"}], **chat_kwargs)
    return result, calls


@pytest.mark.asyncio
async def test_haiku_output_clean_no_fallback_fires(client):
    """Clean casual response: primary runs once, no rerun."""
    mock_send = AsyncMock(return_value=_mk("Todo chido carnal.", model="haiku"))
    with patch.object(client, "_send", mock_send):
        result = await client.chat(
            "system",
            [{"role": "user", "content": "hola"}],
            model="haiku",
            fallback_model="sonnet",
        )
    assert result.text == "Todo chido carnal."
    assert mock_send.await_count == 1


@pytest.mark.asyncio
async def test_haiku_output_with_character_break_triggers_sonnet_rerun(client):
    """Primary (haiku) breaks → fallback (sonnet) runs once and succeeds."""
    responses = [
        _mk("I'm an AI assistant and I can't do that.", model="haiku"),
        _mk("Ya cállate con eso, no eres gran cosa tampoco.", model="sonnet"),
    ]
    result, calls = await _run(client, responses, model="haiku", fallback_model="sonnet")
    assert "AI" not in result.text
    assert result.model_used == "sonnet"
    # Assert the router actually swapped models across the two _send calls.
    assert len(calls) == 2
    assert calls[0]["model"] == "haiku"
    assert calls[1]["model"] == "sonnet"


@pytest.mark.asyncio
async def test_sonnet_break_falls_through_to_reinforced_retry(client):
    """No distinct fallback (primary == fallback): the reinforced-retry legacy path runs."""
    responses = [
        _mk("as an AI model...", model="sonnet"),
        _mk("No, y ya.", model="sonnet"),
    ]
    result, calls = await _run(client, responses)
    assert "AI" not in result.text
    # Both calls must use the same (default) model — no fallback swap happened.
    assert calls[0]["model"] == calls[1]["model"] == client.model


@pytest.mark.asyncio
async def test_both_models_break_response_is_sanitized(client):
    """Haiku breaks, Sonnet also breaks, reinforced retry has one clean sentence → sanitize."""
    responses = [
        _mk("I'm Claude.", model="haiku"),
        _mk("as an AI, I can't help", model="sonnet"),
        _mk("No te pases. I am a language model. Cierra el pico.", model="sonnet"),
    ]
    result, calls = await _run(client, responses, model="haiku", fallback_model="sonnet")
    from insult.core.character import detect_break

    assert not detect_break(result.text)
    assert "No te pases" in result.text or "Cierra el pico" in result.text
    # Call sequence: primary=haiku, fallback=sonnet (still breaks), reinforced-retry=sonnet.
    assert [c["model"] for c in calls] == ["haiku", "sonnet", "sonnet"]


@pytest.mark.asyncio
async def test_haiku_then_sonnet_both_break_reinforced_sonnet_succeeds(client):
    """The explicit path the first test doesn't cover:
    Haiku → Sonnet (breaks) → reinforced-retry against Sonnet (clean).
    Critical because it exercises _recover_from_break's three-branch control flow.
    """
    responses = [
        _mk("I'm an AI, just FYI.", model="haiku"),
        _mk("as a language model, I cannot", model="sonnet"),
        _mk("Nel, estás equivocado y ya.", model="sonnet"),  # reinforced-retry clean
    ]
    result, calls = await _run(client, responses, model="haiku", fallback_model="sonnet")
    from insult.core.character import detect_break

    assert not detect_break(result.text)
    assert [c["model"] for c in calls] == ["haiku", "sonnet", "sonnet"]


@pytest.mark.asyncio
async def test_fallback_not_triggered_when_fallback_equals_primary(client):
    """If caller passes the same model as primary and fallback, we should fall
    through to reinforced retry (no extra rerun against 'fallback')."""
    responses = [
        _mk("I'm an AI.", model="same"),
        _mk("Reforzado y limpio.", model="same"),
    ]
    result, calls = await _run(client, responses, model="same", fallback_model="same")
    assert "AI" not in result.text
    # Exactly 2 calls — primary + reinforced-retry. No fallback swap in between.
    assert len(calls) == 2
    assert calls[0]["model"] == calls[1]["model"] == "same"


@pytest.mark.asyncio
async def test_legacy_call_with_no_model_kwargs_still_works(client):
    """Callers that haven't migrated to the router still work: no model/fallback_model kwargs."""
    responses = [_mk("Respuesta normal.", model="sonnet-default")]
    result, calls = await _run(client, responses)
    assert result.text == "Respuesta normal."
    assert result.model_used == "sonnet-default"
    assert calls[0]["model"] == client.model


@pytest.mark.asyncio
async def test_anti_pattern_rerun_revalidates_break_on_fallback(client):
    """#1 GRAVE regression test: if the anti-pattern rerun (Haiku→Sonnet)
    returns output that contains a character break, we must NOT ship it.
    Before the fix, response = rerun with no revalidation."""
    # First response: clean of breaks but hits >=2 anti-pattern markers.
    # Stage-direction "*smiles...*" + customer-support "Great question" both live
    # in ANTI_PATTERN_CHECKS; neither is a character break.
    primary_anti_pattern = "*smiles warmly* Great question, buddy. How can I help?"
    responses = [
        _mk(primary_anti_pattern, model="haiku"),
        _mk("as an AI, I cannot help", model="sonnet"),  # fallback BREAKS
        _mk("Mejor cállate.", model="sonnet"),  # reinforced-retry clean
    ]
    result, calls = await _run(client, responses, model="haiku", fallback_model="sonnet")
    from insult.core.character import detect_break

    # The critical assertion: no break pattern reaches the user.
    assert not detect_break(result.text)
    # Exactly 3 calls: primary + anti-pattern fallback + reinforced-retry against fallback.
    assert len(calls) == 3
    assert [c["model"] for c in calls] == ["haiku", "sonnet", "sonnet"]
