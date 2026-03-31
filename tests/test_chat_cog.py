"""Tests for insult.cogs.chat — ChatCog with mocked dependencies.

We call the underlying callback directly (cog.chat.callback) to bypass
discord.py's command decorator machinery which expects a real Context.
"""

from unittest.mock import AsyncMock

import pytest

from insult.cogs.chat import MAX_MESSAGE_LENGTH, ChatCog


class TestChatCog:
    @pytest.fixture
    def cog(self, mock_container):
        return ChatCog(mock_container)

    async def _call_chat(self, cog, ctx, message: str):
        """Call the chat command's underlying callback, bypassing the decorator."""
        await cog.chat.callback(cog, ctx, message=message)

    async def test_rejects_too_long_message(self, cog, mock_ctx):
        long_msg = "x" * (MAX_MESSAGE_LENGTH + 1)
        await self._call_chat(cog, mock_ctx, long_msg)
        mock_ctx.send.assert_called_once()
        cog.llm.chat.assert_not_called()

    async def test_stores_user_message(self, cog, mock_ctx):
        cog.llm.chat = AsyncMock(return_value="respuesta")
        await self._call_chat(cog, mock_ctx, "hola wey")
        cog.memory.store.assert_any_call(
            str(mock_ctx.channel.id),
            str(mock_ctx.author.id),
            mock_ctx.author.display_name,
            "user",
            "hola wey",
        )

    async def test_updates_user_profile(self, cog, mock_ctx):
        cog.llm.chat = AsyncMock(return_value="respuesta")
        await self._call_chat(cog, mock_ctx, "hola wey")
        cog.memory.update_profile.assert_called_once_with(str(mock_ctx.author.id), "hola wey")

    async def test_sends_llm_response(self, cog, mock_ctx):
        cog.llm.chat = AsyncMock(return_value="Eres un pendejo.")
        await self._call_chat(cog, mock_ctx, "ayudame")
        calls = mock_ctx.send.call_args_list
        assert any("Eres un pendejo." in str(c) for c in calls)

    async def test_stores_assistant_response(self, cog, mock_ctx):
        cog.llm.chat = AsyncMock(return_value="respuesta del bot")
        await self._call_chat(cog, mock_ctx, "hola")
        cog.memory.store.assert_any_call(
            str(mock_ctx.channel.id),
            str(cog.bot.user.id),
            cog.bot.user.name,
            "assistant",
            "respuesta del bot",
        )

    async def test_handles_llm_error_gracefully(self, cog, mock_ctx):
        cog.llm.chat = AsyncMock(side_effect=RuntimeError("something broke"))
        await self._call_chat(cog, mock_ctx, "hola")
        # Should send an in-character error response, not crash
        mock_ctx.send.assert_called()

    async def test_handles_context_failure(self, cog, mock_ctx):
        cog.memory.get_recent = AsyncMock(side_effect=Exception("DB error"))
        await self._call_chat(cog, mock_ctx, "hola")
        mock_ctx.send.assert_called()
        cog.llm.chat.assert_not_called()

    async def test_handles_profile_failure_gracefully(self, cog, mock_ctx):
        """Profile update failure should NOT block the chat flow."""
        cog.memory.update_profile = AsyncMock(side_effect=Exception("DB error"))
        cog.llm.chat = AsyncMock(return_value="sigo funcionando")
        await self._call_chat(cog, mock_ctx, "hola")
        cog.llm.chat.assert_called_once()

    async def test_chunks_long_responses(self, cog, mock_ctx):
        long_response = "A" * 4000  # > 1990 chars = 3 chunks
        cog.llm.chat = AsyncMock(return_value=long_response)
        await self._call_chat(cog, mock_ctx, "hola")
        # store call + chunk sends (4000/1990 = 3 chunks)
        send_calls = mock_ctx.send.call_args_list
        assert len(send_calls) >= 2
