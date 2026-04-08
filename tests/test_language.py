"""Tests for language cure — post-generation normalization."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from insult.core.language import language_cure


@pytest.fixture
def mock_client():
    """Mock Anthropic client that returns cured text."""
    client = AsyncMock()
    return client


def _mock_response(text: str):
    """Create a mock API response with given text."""
    block = MagicMock()
    block.text = text
    response = MagicMock()
    response.content = [block]
    return response


class TestLanguageCure:
    @pytest.mark.asyncio
    async def test_translates_english_to_spanish(self, mock_client):
        mock_client.messages.create = AsyncMock(return_value=_mock_response("Pero Bernard, este video es INTENSO."))
        result = await language_cure(mock_client, "haiku", "But Bernard, this video is INTENSE.")
        assert result == "Pero Bernard, este video es INTENSO."

    @pytest.mark.asyncio
    async def test_returns_unchanged_spanish(self, mock_client):
        original = "¿En serio no sabes qué son las Corporate Wars?"
        mock_client.messages.create = AsyncMock(return_value=_mock_response(original))
        result = await language_cure(mock_client, "haiku", original)
        assert result == original

    @pytest.mark.asyncio
    async def test_preserves_react_markers(self, mock_client):
        cured = "Eso es un patrón clásico [REACT:💀]"
        mock_client.messages.create = AsyncMock(return_value=_mock_response(cured))
        result = await language_cure(mock_client, "haiku", "That's a classic pattern [REACT:💀]")
        assert "[REACT:💀]" in result

    @pytest.mark.asyncio
    async def test_strips_output_tags(self, mock_client):
        mock_client.messages.create = AsyncMock(return_value=_mock_response("<output>Texto curado aquí</output>"))
        result = await language_cure(mock_client, "haiku", "Cured text here in English")
        assert result == "Texto curado aquí"

    @pytest.mark.asyncio
    async def test_returns_original_on_error(self, mock_client):
        mock_client.messages.create = AsyncMock(side_effect=Exception("API down"))
        original = "This should come back unchanged"
        result = await language_cure(mock_client, "haiku", original)
        assert result == original

    @pytest.mark.asyncio
    async def test_returns_original_on_empty_response(self, mock_client):
        mock_client.messages.create = AsyncMock(return_value=_mock_response(""))
        original = "Some mixed text here amigo"
        result = await language_cure(mock_client, "haiku", original)
        assert result == original

    @pytest.mark.asyncio
    async def test_returns_original_on_length_mismatch(self, mock_client):
        # Haiku returns something way too short
        mock_client.messages.create = AsyncMock(return_value=_mock_response("Si"))
        original = "This is a very long English sentence that should be translated properly"
        result = await language_cure(mock_client, "haiku", original)
        assert result == original

    @pytest.mark.asyncio
    async def test_skips_short_text(self, mock_client):
        result = await language_cure(mock_client, "haiku", "hola")
        assert result == "hola"
        mock_client.messages.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_empty_text(self, mock_client):
        result = await language_cure(mock_client, "haiku", "")
        assert result == ""

    @pytest.mark.asyncio
    async def test_wraps_input_in_xml_tags(self, mock_client):
        mock_client.messages.create = AsyncMock(return_value=_mock_response("texto"))
        await language_cure(mock_client, "haiku", "some text here right now")
        # Verify the input was wrapped in <input> tags
        call_args = mock_client.messages.create.call_args
        user_msg = call_args.kwargs["messages"][0]["content"]
        assert user_msg.startswith("<input>")
        assert user_msg.endswith("</input>")
