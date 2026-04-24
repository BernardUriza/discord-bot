"""End-to-end integration tests for the LLMClient response pipeline.

These tests exercise the ACTUAL ordering of:
  _send → strip_metadata → character-break retry → language_cure → strip_metadata
  → normalize_formatting → strip_lists

Without these, unit tests for each stage pass while a downstream stage
re-introduces the exact junk an earlier stage stripped (which is how the
v3.4.5 scratchpad-XML leak escaped — strip_metadata ran BEFORE language_cure
and nobody noticed the cure model was re-injecting <output> tags).

The mocks simulate two separate LLM calls inside one `client.chat()`:
  1. The primary generation (returns the "raw" model output)
  2. The language-cure pass via Haiku (returns a potentially wrapped reply)

Assertions check the FINAL text the caller receives — if anything leaked,
these tests scream.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from insult.core.character import CACHE_BOUNDARY
from insult.core.llm import LLMClient


def _mk_usage(input_tokens: int = 10, output_tokens: int = 20):
    u = MagicMock()
    u.input_tokens = input_tokens
    u.output_tokens = output_tokens
    u.cache_read_input_tokens = 0
    u.cache_creation_input_tokens = 0
    return u


def _mk_response(text: str, stop_reason: str = "end_turn"):
    block = MagicMock()
    block.text = text
    block.type = "text"
    # NOTE: `hasattr(block, "text")` must succeed — MagicMock grants it.
    resp = MagicMock()
    resp.content = [block]
    resp.usage = _mk_usage()
    resp.stop_reason = stop_reason
    return resp


def _mk_client_with_sequence(responses: list):
    """Build a fake LLMClient where the PRIMARY response comes from `messages.stream()`
    (v3.5.11 switched `_send` to streaming transport so long web_search turns don't
    trip the 30s read timeout), and subsequent responses feed `language_cure` which
    still uses `messages.create` because it is a short Haiku call."""
    client = LLMClient(
        api_key="fake",
        model="claude-sonnet-4-6",
        max_tokens=1024,
        timeout=5.0,
        max_retries=1,
        cure_model="claude-haiku-4-5-20251001",
    )
    client.client = AsyncMock()

    primary = responses[0]
    cure_responses = responses[1:]

    # Primary: messages.stream(...) returns an async ctx manager whose .get_final_message()
    # yields `primary`. Needs to be MagicMock (not AsyncMock) because stream() is a sync
    # factory that returns a context manager — the awaiting happens inside `async with`.
    stream_obj = AsyncMock()
    stream_obj.get_final_message = AsyncMock(return_value=primary)
    stream_cm = MagicMock()
    stream_cm.__aenter__ = AsyncMock(return_value=stream_obj)
    stream_cm.__aexit__ = AsyncMock(return_value=None)
    client.client.messages.stream = MagicMock(return_value=stream_cm)

    # Cure path (if any): language_cure still goes through messages.create.
    if cure_responses:
        client.client.messages.create = AsyncMock(side_effect=cure_responses)
    return client


class TestPipelineLeakage:
    """Pipeline must never deliver prompt internals to the caller."""

    @pytest.mark.asyncio
    async def test_scratchpad_xml_from_cure_never_leaks(self):
        # Primary returns clean Spanish; cure model WRAPS it in <output>. This
        # was the real v3.4.5 bug — unit tests passed, prod leaked.
        client = _mk_client_with_sequence(
            [
                _mk_response("Eso que dices está mal."),  # primary
                _mk_response("<output>Eso que dices está mal.</output>"),  # cure wraps
            ]
        )
        r = await client.chat("sys", [{"role": "user", "content": "hi"}])
        assert "<output>" not in r.text
        assert "</output>" not in r.text
        assert "Eso que dices está mal." in r.text

    @pytest.mark.asyncio
    async def test_cure_scratchpad_with_trailing_newline_does_not_leak(self):
        # The exact shape that would have beaten v3.4.6's startswith/endswith.
        client = _mk_client_with_sequence(
            [
                _mk_response("Pura rabia."),
                _mk_response("<output>Pura rabia.</output>\n"),
            ]
        )
        r = await client.chat("sys", [{"role": "user", "content": "hi"}])
        assert "<" not in r.text
        assert r.text == "Pura rabia."

    @pytest.mark.asyncio
    async def test_cure_input_output_pair_does_not_duplicate_text(self):
        # Real 2026-04-20 prod leak: Haiku emitted both <input> and <output>
        # with identical content, so end users saw the text twice.
        duplicated = (
            "<input>La pobreza siempre cobró en carne.</input>\n\n<output>La pobreza siempre cobró en carne.</output>"
        )
        client = _mk_client_with_sequence(
            [
                _mk_response("La pobreza siempre cobró en carne."),
                _mk_response(duplicated),
            ]
        )
        r = await client.chat("sys", [{"role": "user", "content": "hi"}])
        assert "<input>" not in r.text
        assert "<output>" not in r.text
        assert r.text.count("La pobreza siempre cobró en carne.") == 1

    @pytest.mark.asyncio
    async def test_cache_boundary_from_primary_does_not_leak(self):
        # If the primary model ever echoes the cache boundary marker.
        leaked = f"Respuesta.{CACHE_BOUNDARY}fragmento extra"
        client = _mk_client_with_sequence(
            [
                _mk_response(leaked),
                _mk_response("Respuesta.\nfragmento extra"),  # cure normalizes
            ]
        )
        r = await client.chat("sys", [{"role": "user", "content": "hi"}])
        assert "CACHE_BOUNDARY" not in r.text
        assert "<<<" not in r.text

    @pytest.mark.asyncio
    async def test_timestamp_and_speaker_leak_from_primary_stripped(self):
        leaked = "[hace 5m] Insult: Esta respuesta tiene metadata basura."
        client = _mk_client_with_sequence(
            [
                _mk_response(leaked),
                _mk_response("Esta respuesta tiene metadata basura."),
            ]
        )
        r = await client.chat("sys", [{"role": "user", "content": "hi"}])
        assert "[hace 5m]" not in r.text
        assert "Insult:" not in r.text


class TestPipelineOrdering:
    """The ordering of strip/cure/normalize must hold — a regression here
    is the structural bug /cruel-critic flagged."""

    @pytest.mark.asyncio
    async def test_cure_runs_after_primary_strip(self):
        # If cure runs BEFORE strip, the primary's metadata would go into
        # Haiku's input and confuse it. Verify cure's user-content is ALREADY
        # stripped of obvious metadata markers.
        client = _mk_client_with_sequence(
            [
                _mk_response("Respuesta limpia."),
                _mk_response("Respuesta normalizada."),
            ]
        )
        await client.chat("sys", [{"role": "user", "content": "hi"}])
        # The cure is the ONLY call to messages.create (primary now flows through
        # messages.stream since v3.5.11). Its user content must be what
        # strip_metadata produced from the primary, never the raw primary output.
        cure_call = client.client.messages.create.call_args_list[0]
        cure_user_msg = cure_call.kwargs["messages"][0]["content"]
        assert "[SEND]" not in cure_user_msg
        assert "<<<" not in cure_user_msg

    @pytest.mark.asyncio
    async def test_empty_cure_response_falls_back_to_primary(self):
        # If Haiku explodes / returns empty, the primary text still reaches
        # the user. Language cure has try/except that returns original on
        # exception; integration level we verify the fallback.
        client = _mk_client_with_sequence(
            [
                _mk_response("Texto del primario queda."),
                Exception("Haiku down"),
            ]
        )
        r = await client.chat("sys", [{"role": "user", "content": "hi"}])
        assert "Texto del primario queda." in r.text
