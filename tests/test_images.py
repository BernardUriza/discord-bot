"""Tests for insult.core.images — Pollinations image generation + throttling."""

import io
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from insult.core.images import (
    IMAGE_COOLDOWN_SECONDS,
    MAX_PROMPT_LENGTH,
    generate_image,
    is_throttled,
)


class TestIsThrottled:
    def test_not_throttled_initially(self):
        import insult.core.images as mod

        mod._last_generation_time = 0.0
        assert not is_throttled()

    @patch("insult.core.images.time")
    def test_throttled_within_cooldown(self, mock_time):
        import insult.core.images as mod

        mock_time.monotonic.return_value = 100.0
        mod._last_generation_time = 100.0 - IMAGE_COOLDOWN_SECONDS + 1
        assert is_throttled()

    @patch("insult.core.images.time")
    def test_not_throttled_after_cooldown(self, mock_time):
        import insult.core.images as mod

        mock_time.monotonic.return_value = 100.0
        mod._last_generation_time = 100.0 - IMAGE_COOLDOWN_SECONDS - 1
        assert not is_throttled()


class TestGenerateImage:
    @pytest.fixture(autouse=True)
    def reset_throttle(self):
        import insult.core.images as mod

        mod._last_generation_time = 0.0

    async def test_empty_prompt_returns_none(self):
        result = await generate_image("")
        assert result is None

    async def test_prompt_truncated(self):
        long_prompt = "a" * 1000
        with patch("insult.core.images.aiohttp.ClientSession") as mock_session_cls:
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_resp.read = AsyncMock(return_value=b"x" * 5000)

            mock_ctx = AsyncMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
            mock_ctx.__aexit__ = AsyncMock(return_value=False)

            mock_session = AsyncMock()
            mock_session.get = MagicMock(return_value=mock_ctx)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)

            mock_session_cls.return_value = mock_session

            result = await generate_image(long_prompt)
            assert result is not None
            # Verify prompt was truncated
            assert len(long_prompt[:MAX_PROMPT_LENGTH]) == MAX_PROMPT_LENGTH

    async def test_returns_none_on_http_error(self):
        with patch("insult.core.images.aiohttp.ClientSession") as mock_session_cls:
            mock_resp = AsyncMock()
            mock_resp.status = 500

            mock_ctx = AsyncMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
            mock_ctx.__aexit__ = AsyncMock(return_value=False)

            mock_session = AsyncMock()
            mock_session.get = MagicMock(return_value=mock_ctx)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)

            mock_session_cls.return_value = mock_session

            result = await generate_image("test prompt")
            assert result is None

    async def test_returns_none_on_small_response(self):
        with patch("insult.core.images.aiohttp.ClientSession") as mock_session_cls:
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_resp.read = AsyncMock(return_value=b"tiny")  # < 1000 bytes

            mock_ctx = AsyncMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
            mock_ctx.__aexit__ = AsyncMock(return_value=False)

            mock_session = AsyncMock()
            mock_session.get = MagicMock(return_value=mock_ctx)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)

            mock_session_cls.return_value = mock_session

            result = await generate_image("test prompt")
            assert result is None

    async def test_successful_generation(self):
        image_data = b"x" * 5000  # valid-sized response
        with patch("insult.core.images.aiohttp.ClientSession") as mock_session_cls:
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_resp.read = AsyncMock(return_value=image_data)

            mock_ctx = AsyncMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
            mock_ctx.__aexit__ = AsyncMock(return_value=False)

            mock_session = AsyncMock()
            mock_session.get = MagicMock(return_value=mock_ctx)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)

            mock_session_cls.return_value = mock_session

            result = await generate_image("a beautiful sunset")
            assert result is not None
            assert isinstance(result, io.BytesIO)
            assert result.read() == image_data

    @patch("insult.core.images.is_throttled", return_value=True)
    async def test_throttled_returns_none(self, _mock):
        result = await generate_image("test")
        assert result is None

    async def test_passes_model_and_dimensions(self):
        with patch("insult.core.images.aiohttp.ClientSession") as mock_session_cls:
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_resp.read = AsyncMock(return_value=b"x" * 5000)

            mock_ctx = AsyncMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
            mock_ctx.__aexit__ = AsyncMock(return_value=False)

            mock_session = AsyncMock()
            mock_session.get = MagicMock(return_value=mock_ctx)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)

            mock_session_cls.return_value = mock_session

            await generate_image("test", model="turbo", width=512, height=512)
            call_kwargs = mock_session.get.call_args
            params = call_kwargs[1]["params"]
            assert params["model"] == "turbo"
            assert params["width"] == 512
            assert params["height"] == 512
            assert params["safe"] == "true"
