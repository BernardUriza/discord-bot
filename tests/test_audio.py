"""Tests for insult.core.audio — YouTube clip extraction + Freesound search."""

import io
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from insult.core.audio import (
    AUDIO_COOLDOWN_SECONDS,
    is_throttled,
    search_and_clip_youtube,
    search_freesound,
)


class TestIsThrottled:
    def test_not_throttled_initially(self):
        import insult.core.audio as mod

        mod._last_audio_time = 0.0
        assert not is_throttled()

    @patch("insult.core.audio.time")
    def test_throttled_within_cooldown(self, mock_time):
        import insult.core.audio as mod

        mock_time.monotonic.return_value = 100.0
        mod._last_audio_time = 100.0 - AUDIO_COOLDOWN_SECONDS + 1
        assert is_throttled()

    @patch("insult.core.audio.time")
    def test_not_throttled_after_cooldown(self, mock_time):
        import insult.core.audio as mod

        mock_time.monotonic.return_value = 100.0
        mod._last_audio_time = 100.0 - AUDIO_COOLDOWN_SECONDS - 1
        assert not is_throttled()


class TestSearchAndClipYoutube:
    @pytest.fixture(autouse=True)
    def reset_throttle(self):
        import insult.core.audio as mod

        mod._last_audio_time = 0.0

    async def test_empty_query_returns_none(self):
        result = await search_and_clip_youtube("")
        assert result is None

    @patch("insult.core.audio.is_throttled", return_value=True)
    async def test_throttled_returns_none(self, _mock):
        result = await search_and_clip_youtube("test song")
        assert result is None

    @patch("insult.core.audio.asyncio.create_subprocess_exec")
    async def test_ytdlp_failure_returns_none(self, mock_exec):
        mock_proc = AsyncMock()
        mock_proc.returncode = 1
        mock_proc.communicate = AsyncMock(return_value=(b"", b"error"))
        mock_exec.return_value = mock_proc

        result = await search_and_clip_youtube("test song")
        assert result is None

    @patch("insult.core.audio.asyncio.create_subprocess_exec")
    async def test_ytdlp_not_installed_returns_none(self, mock_exec):
        mock_exec.side_effect = FileNotFoundError("yt-dlp not found")
        result = await search_and_clip_youtube("test song")
        assert result is None


class TestSearchFreesound:
    @pytest.fixture(autouse=True)
    def reset_throttle(self):
        import insult.core.audio as mod

        mod._last_audio_time = 0.0

    async def test_no_api_key_returns_none(self):
        result = await search_freesound("bruh", api_key=None)
        assert result is None

    @patch("insult.core.audio.is_throttled", return_value=True)
    async def test_throttled_returns_none(self, _mock):
        result = await search_freesound("bruh", api_key="test_key")
        assert result is None

    async def test_successful_search(self):
        with patch("insult.core.audio.aiohttp.ClientSession") as mock_cls:
            # Mock search response
            search_resp = AsyncMock()
            search_resp.status = 200
            search_resp.json = AsyncMock(
                return_value={
                    "results": [
                        {
                            "id": 1,
                            "name": "bruh",
                            "previews": {"preview-hq-mp3": "https://example.com/bruh.mp3"},
                        }
                    ]
                }
            )

            # Mock download response
            download_resp = AsyncMock()
            download_resp.status = 200
            download_resp.read = AsyncMock(return_value=b"audio-data-here")

            mock_session = AsyncMock()
            # get() returns different responses for different calls
            ctx1 = AsyncMock()
            ctx1.__aenter__ = AsyncMock(return_value=search_resp)
            ctx1.__aexit__ = AsyncMock(return_value=False)
            ctx2 = AsyncMock()
            ctx2.__aenter__ = AsyncMock(return_value=download_resp)
            ctx2.__aexit__ = AsyncMock(return_value=False)
            mock_session.get = MagicMock(side_effect=[ctx1, ctx2])

            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_session

            result = await search_freesound("bruh", api_key="test_key")
            assert result is not None
            assert isinstance(result, io.BytesIO)

    async def test_no_results_returns_none(self):
        with patch("insult.core.audio.aiohttp.ClientSession") as mock_cls:
            search_resp = AsyncMock()
            search_resp.status = 200
            search_resp.json = AsyncMock(return_value={"results": []})

            ctx = AsyncMock()
            ctx.__aenter__ = AsyncMock(return_value=search_resp)
            ctx.__aexit__ = AsyncMock(return_value=False)

            mock_session = AsyncMock()
            mock_session.get = MagicMock(return_value=ctx)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_session

            result = await search_freesound("nonexistent_sound_xyz", api_key="test_key")
            assert result is None


class TestAudioToolSchema:
    def test_tool_exists(self):
        from insult.core.actions import AUDIO_TOOLS

        assert len(AUDIO_TOOLS) == 1
        assert AUDIO_TOOLS[0]["name"] == "play_audio"

    def test_query_required(self):
        from insult.core.actions import AUDIO_TOOLS

        schema = AUDIO_TOOLS[0]["input_schema"]
        assert "query" in schema["properties"]
        assert schema["required"] == ["query"]

    def test_source_enum(self):
        from insult.core.actions import AUDIO_TOOLS

        schema = AUDIO_TOOLS[0]["input_schema"]
        assert set(schema["properties"]["source"]["enum"]) == {"youtube", "meme"}

    def test_conforms_to_anthropic_schema(self):
        from insult.core.actions import AUDIO_TOOLS

        for tool in AUDIO_TOOLS:
            assert "name" in tool
            assert "input_schema" in tool
            assert tool["input_schema"]["type"] == "object"
            assert "strict" not in tool
