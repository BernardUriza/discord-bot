"""Shared fixtures for all tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from insult.core.llm import LLMResponse
from insult.core.style import UserStyleProfile

# --- Mock Container ---


@pytest.fixture
def mock_memory():
    """Mocked MemoryStore with async methods."""
    mem = AsyncMock()
    mem.store = AsyncMock()
    mem.get_recent = AsyncMock(return_value=[])
    mem.search = AsyncMock(return_value=[])
    mem.get_stats = AsyncMock(return_value={"total_messages": 0, "unique_users": 0, "unique_channels": 0})
    mem.get_profile = AsyncMock(return_value=UserStyleProfile())
    mem.update_profile = AsyncMock(return_value=UserStyleProfile())
    mem.build_context = MagicMock(return_value=[])
    mem.connect = AsyncMock()
    mem.close = AsyncMock()
    mem.get_channel_summaries = AsyncMock(return_value=[])
    mem.get_channel_activity_since = AsyncMock(return_value=[])
    mem.get_recent_for_summary = AsyncMock(return_value=[])
    mem.upsert_channel_summary = AsyncMock()
    # Reminder methods
    mem.save_reminder = AsyncMock(return_value=1)
    mem.get_pending_reminders = AsyncMock(return_value=[])
    mem.mark_reminder_delivered = AsyncMock()
    mem.update_reminder_time = AsyncMock()
    mem.get_channel_reminders = AsyncMock(return_value=[])
    mem.delete_reminder = AsyncMock(return_value=True)
    mem.get_channel_participants = AsyncMock(return_value=[])
    # Phase 1 (v3.0.0): disclosure, arcs, stances, contradictions
    mem.store_disclosure = AsyncMock()
    mem.get_arc = AsyncMock(return_value=None)
    mem.upsert_arc = AsyncMock()
    mem.store_stance = AsyncMock()
    mem.get_stances = AsyncMock(return_value=[])
    mem.store_contradiction = AsyncMock()
    # Guild config (v3.3.0)
    mem.get_guild_config = AsyncMock(return_value=None)
    mem.save_guild_config = AsyncMock()
    return mem


@pytest.fixture
def mock_llm():
    """Mocked LLMClient."""
    llm = AsyncMock()
    llm.chat = AsyncMock(return_value=LLMResponse(text="Test response from Insult"))
    return llm


@pytest.fixture
def mock_settings():
    """Mocked Settings object."""
    s = MagicMock()
    s.system_prompt = "You are Insult."
    s.memory_recent_limit = 50
    s.memory_relevant_limit = 5
    s.command_prefix = "!"
    s.llm_model = "claude-sonnet-4-20250514"
    s.llm_max_tokens = 1024
    s.discord_token = "fake-token"  # noqa: S105
    s.anthropic_api_key = "fake-key"
    return s


@pytest.fixture
def mock_bot():
    """Mocked discord Bot."""
    bot = MagicMock()
    bot.user = MagicMock()
    bot.user.id = 999999
    bot.user.name = "Insult"
    bot.latency = 0.05
    bot.guilds = []
    return bot


@pytest.fixture
def mock_container(mock_settings, mock_memory, mock_llm, mock_bot):
    """Full mocked DI container."""
    container = MagicMock()
    container.settings = mock_settings
    container.memory = mock_memory
    container.llm = mock_llm
    container.bot = mock_bot
    return container


@pytest.fixture
def mock_ctx():
    """Mocked discord.py Context with message.channel async methods."""
    ctx = MagicMock()
    ctx.send = AsyncMock()
    ctx.author.id = 123456
    ctx.author.display_name = "TestUser"
    ctx.channel.id = 789

    # mock_ctx.message simulates a discord.Message
    msg = MagicMock()
    msg.author = ctx.author
    msg.channel = MagicMock()
    msg.channel.id = 789
    msg.channel.send = AsyncMock()
    msg.channel.typing = MagicMock(return_value=AsyncMock())
    msg.channel.typing.return_value.__aenter__ = AsyncMock()
    msg.channel.typing.return_value.__aexit__ = AsyncMock(return_value=False)
    msg.attachments = []
    msg.add_reaction = AsyncMock()
    ctx.message = msg

    # Also set up ctx.typing for backwards compat
    ctx.typing = MagicMock(return_value=AsyncMock())
    ctx.typing.return_value.__aenter__ = AsyncMock()
    ctx.typing.return_value.__aexit__ = AsyncMock(return_value=False)
    return ctx
