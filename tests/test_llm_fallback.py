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


async def _run(client: LLMClient, responses: list[LLMResponse], **chat_kwargs: Any) -> LLMResponse:
    """Helper: patch _send to return the given responses in order, then call chat."""
    queue = list(responses)

    async def fake_send(*_args: Any, **_kwargs: Any) -> LLMResponse:
        assert queue, "chat called _send more times than expected"
        return queue.pop(0)

    with patch.object(client, "_send", side_effect=fake_send):
        result = await client.chat("system", [{"role": "user", "content": "hi"}], **chat_kwargs)
    return result


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
        _mk("I'm an AI assistant and I can't do that.", model="haiku"),  # primary break
        _mk("Ya cállate con eso, no eres gran cosa tampoco.", model="sonnet"),  # fallback clean
    ]
    result = await _run(client, responses, model="haiku", fallback_model="sonnet")
    assert "AI" not in result.text
    assert result.model_used == "sonnet"


@pytest.mark.asyncio
async def test_sonnet_break_falls_through_to_reinforced_retry(client):
    """No distinct fallback (primary == fallback): the reinforced-retry legacy path runs."""
    responses = [
        _mk("as an AI model...", model="sonnet"),  # break
        _mk("No, y ya.", model="sonnet"),  # reinforced retry — clean
    ]
    # No model override → uses self.model and self.model as fallback (identical).
    result = await _run(client, responses)
    assert "AI" not in result.text


@pytest.mark.asyncio
async def test_both_models_break_response_is_sanitized(client):
    """Haiku breaks, Sonnet also breaks, reinforced retry also breaks → sanitize.

    `sanitize()` removes full sentences that contain break patterns. As long
    as the reinforced retry has at least one clean sentence, the result must
    not leak any break pattern.
    """
    responses = [
        _mk("I'm Claude.", model="haiku"),
        _mk("as an AI, I can't help", model="sonnet"),
        _mk("No te pases. I am a language model. Cierra el pico.", model="sonnet"),
    ]
    result = await _run(client, responses, model="haiku", fallback_model="sonnet")
    from insult.core.character import detect_break

    assert not detect_break(result.text)
    assert "No te pases" in result.text or "Cierra el pico" in result.text


@pytest.mark.asyncio
async def test_fallback_not_triggered_when_fallback_equals_primary(client):
    """If caller passes the same model as primary and fallback, we should not rerun
    against the 'fallback' — we should fall through to reinforced retry."""
    responses = [
        _mk("I'm an AI.", model="same"),
        _mk("Reforzado y limpio.", model="same"),
    ]
    result = await _run(client, responses, model="same", fallback_model="same")
    assert "AI" not in result.text


@pytest.mark.asyncio
async def test_legacy_call_with_no_model_kwargs_still_works(client):
    """Callers that haven't migrated to the router still work: no model/fallback_model kwargs."""
    responses = [_mk("Respuesta normal.", model="sonnet-default")]
    result = await _run(client, responses)
    assert result.text == "Respuesta normal."
    assert result.model_used == "sonnet-default"
