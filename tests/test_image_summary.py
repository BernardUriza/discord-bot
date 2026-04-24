"""Tests for insult.core.image_summary — the Haiku-vision-backed memory aid.

The module is deliberately best-effort: any failure returns None so the caller
can fall back to plain text storage. These tests lock in that contract."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from insult.core.image_summary import summarize_images


def _mk_image_block(data: str = "fakebase64") -> dict:
    """A minimal Claude API-shaped image block."""
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": data},
    }


def _mk_client_with_text_response(text: str) -> MagicMock:
    """Mock AsyncAnthropic client whose messages.create returns a response
    with a single text content block."""
    block = SimpleNamespace(type="text", text=text)
    response = SimpleNamespace(content=[block])
    client = MagicMock()
    client.messages.create = AsyncMock(return_value=response)
    return client


@pytest.mark.asyncio
async def test_returns_description_for_valid_images():
    """Happy path: one image → Haiku returns a line → function returns it verbatim."""
    client = _mk_client_with_text_response("1. Gráfica de barras trimestrales.")
    result = await summarize_images(
        [_mk_image_block()],
        client=client,
        model="claude-haiku-4-5-20251001",
    )
    assert result == "1. Gráfica de barras trimestrales."
    # Haiku was actually called — not silently short-circuited.
    assert client.messages.create.await_count == 1
    call_kwargs = client.messages.create.await_args.kwargs
    assert call_kwargs["model"] == "claude-haiku-4-5-20251001"
    # The image block was included in the content payload sent to Claude.
    sent_content = call_kwargs["messages"][0]["content"]
    assert any(c.get("type") == "image" for c in sent_content)


@pytest.mark.asyncio
async def test_empty_list_returns_none_without_api_call():
    """No images → no Haiku call, no tokens burned, return None."""
    client = _mk_client_with_text_response("should not be called")
    result = await summarize_images(
        [],
        client=client,
        model="claude-haiku-4-5-20251001",
    )
    assert result is None
    assert client.messages.create.await_count == 0


@pytest.mark.asyncio
async def test_api_error_returns_none_without_raising():
    """Haiku call explodes → function swallows and returns None so the
    caller can fall back to plain text. No crash bubbles to _respond."""
    client = MagicMock()
    client.messages.create = AsyncMock(side_effect=RuntimeError("kaboom"))
    result = await summarize_images(
        [_mk_image_block()],
        client=client,
        model="claude-haiku-4-5-20251001",
    )
    assert result is None


@pytest.mark.asyncio
async def test_empty_text_response_returns_none():
    """Haiku returns an empty string → treat as 'no useful description', return None."""
    client = _mk_client_with_text_response("   ")
    result = await summarize_images(
        [_mk_image_block()],
        client=client,
        model="claude-haiku-4-5-20251001",
    )
    assert result is None
