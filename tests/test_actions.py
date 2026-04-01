"""Tests for insult.core.actions — tool definitions, sanitization, and channel creation execution."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from insult.core.actions import (
    CHANNEL_TOOLS,
    ToolCall,
    execute_create_channel,
    sanitize_channel_name,
)

# --- CHANNEL_TOOLS schema ---


class TestChannelTools:
    def test_tool_exists(self):
        assert len(CHANNEL_TOOLS) == 1
        assert CHANNEL_TOOLS[0]["name"] == "create_channel"

    def test_strict_enabled(self):
        assert CHANNEL_TOOLS[0]["strict"] is True

    def test_required_fields(self):
        schema = CHANNEL_TOOLS[0]["input_schema"]
        assert "name" in schema["properties"]
        assert "channel_type" in schema["properties"]
        assert schema["required"] == ["name", "channel_type"]
        assert schema["additionalProperties"] is False

    def test_valid_channel_types(self):
        schema = CHANNEL_TOOLS[0]["input_schema"]
        assert set(schema["properties"]["channel_type"]["enum"]) == {"private", "topic", "category"}


# --- sanitize_channel_name ---


class TestSanitizeChannelName:
    def test_basic_name(self):
        assert sanitize_channel_name("mi canal") == "mi-canal"

    def test_uppercase(self):
        assert sanitize_channel_name("ADHD Focus") == "adhd-focus"

    def test_special_chars(self):
        assert sanitize_channel_name("café & más!") == "caf-ms"

    def test_multiple_hyphens(self):
        assert sanitize_channel_name("a---b") == "a-b"

    def test_empty_fallback(self):
        assert sanitize_channel_name("!!!") == "nuevo-canal"

    def test_max_length(self):
        result = sanitize_channel_name("a" * 200)
        assert len(result) <= 100

    def test_leading_trailing_hyphens(self):
        assert sanitize_channel_name("-test-") == "test"


# --- execute_create_channel ---


class TestExecuteCreateChannel:
    @pytest.fixture
    def mock_guild(self):
        guild = MagicMock()
        guild.name = "TestGuild"
        guild.channels = [MagicMock() for _ in range(10)]

        bot_member = MagicMock()
        bot_member.guild_permissions = MagicMock()
        bot_member.guild_permissions.manage_channels = True
        guild.me = bot_member
        guild.default_role = MagicMock()

        created_channel = MagicMock()
        created_channel.mention = "#test-channel"
        guild.create_text_channel = AsyncMock(return_value=created_channel)
        guild.create_category = AsyncMock(return_value=created_channel)

        return guild

    @pytest.fixture
    def mock_user(self):
        user = MagicMock()
        user.display_name = "TestUser"
        return user

    @patch("insult.core.actions.asyncio.sleep", new_callable=AsyncMock)
    async def test_creates_private_channel(self, _sleep, mock_guild, mock_user):
        tool_call = ToolCall(id="tc_1", name="create_channel", input={"name": "mi espacio", "channel_type": "private"})
        result = await execute_create_channel(mock_guild, tool_call, mock_user)
        assert result is not None
        mock_guild.create_text_channel.assert_called_once()
        call_kwargs = mock_guild.create_text_channel.call_args
        assert "overwrites" in call_kwargs.kwargs or len(call_kwargs.args) > 1

    @patch("insult.core.actions.asyncio.sleep", new_callable=AsyncMock)
    async def test_creates_topic_channel(self, _sleep, mock_guild, mock_user):
        tool_call = ToolCall(id="tc_2", name="create_channel", input={"name": "adhd-focus", "channel_type": "topic"})
        result = await execute_create_channel(mock_guild, tool_call, mock_user)
        assert result is not None
        mock_guild.create_text_channel.assert_called_once_with("adhd-focus")

    @patch("insult.core.actions.asyncio.sleep", new_callable=AsyncMock)
    async def test_creates_category(self, _sleep, mock_guild, mock_user):
        tool_call = ToolCall(id="tc_3", name="create_channel", input={"name": "bienestar", "channel_type": "category"})
        result = await execute_create_channel(mock_guild, tool_call, mock_user)
        assert result is not None
        mock_guild.create_category.assert_called_once_with("bienestar")

    @patch("insult.core.actions.asyncio.sleep", new_callable=AsyncMock)
    async def test_missing_permissions_returns_none(self, _sleep, mock_guild, mock_user):
        mock_guild.me.guild_permissions.manage_channels = False
        tool_call = ToolCall(id="tc_4", name="create_channel", input={"name": "test", "channel_type": "private"})
        result = await execute_create_channel(mock_guild, tool_call, mock_user)
        assert result is None
        mock_guild.create_text_channel.assert_not_called()

    @patch("insult.core.actions.asyncio.sleep", new_callable=AsyncMock)
    async def test_channel_limit_returns_none(self, _sleep, mock_guild, mock_user):
        mock_guild.channels = [MagicMock() for _ in range(460)]
        tool_call = ToolCall(id="tc_5", name="create_channel", input={"name": "test", "channel_type": "topic"})
        result = await execute_create_channel(mock_guild, tool_call, mock_user)
        assert result is None

    @patch("insult.core.actions.asyncio.sleep", new_callable=AsyncMock)
    async def test_defaults_to_private_on_invalid_type(self, _sleep, mock_guild, mock_user):
        tool_call = ToolCall(id="tc_6", name="create_channel", input={"name": "test", "channel_type": "invalid"})
        result = await execute_create_channel(mock_guild, tool_call, mock_user)
        assert result is not None
        call_kwargs = mock_guild.create_text_channel.call_args
        assert "overwrites" in call_kwargs.kwargs or len(call_kwargs.args) > 1

    @patch("insult.core.actions.asyncio.sleep", new_callable=AsyncMock)
    async def test_sanitizes_channel_name(self, _sleep, mock_guild, mock_user):
        tool_call = ToolCall(
            id="tc_7", name="create_channel", input={"name": "Mi Canal Especial!!!", "channel_type": "topic"}
        )
        await execute_create_channel(mock_guild, tool_call, mock_user)
        mock_guild.create_text_channel.assert_called_once_with("mi-canal-especial")
