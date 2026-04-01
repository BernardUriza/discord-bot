"""Tests for insult.core.actions — action parsing, stripping, sanitization, execution."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from insult.core.actions import (
    MAX_ACTIONS_PER_RESPONSE,
    BotAction,
    execute_create_channel,
    parse_actions,
    sanitize_channel_name,
    strip_actions,
)

# --- parse_actions ---


class TestParseActions:
    def test_single_action(self):
        result = parse_actions("Ok[ACTION:create_channel|name=privado|type=private]")
        assert len(result) == 1
        assert result[0].action_type == "create_channel"
        assert result[0].params == {"name": "privado", "type": "private"}

    def test_action_with_for_param(self):
        result = parse_actions("[ACTION:create_channel|name=espacio-bern|type=private|for=bernard2389]")
        assert result[0].params["for"] == "bernard2389"

    def test_no_action_marker(self):
        assert parse_actions("Respuesta normal sin acciones.") == []

    def test_case_insensitive(self):
        result = parse_actions("[action:create_channel|name=test]")
        assert len(result) == 1

    def test_max_actions_enforced(self):
        response = "[ACTION:create_channel|name=a]" * 5
        assert len(parse_actions(response)) == MAX_ACTIONS_PER_RESPONSE

    def test_empty_action(self):
        assert parse_actions("[ACTION:]") == []

    def test_action_type_only_no_params(self):
        result = parse_actions("[ACTION:create_channel]")
        assert len(result) == 1
        assert result[0].action_type == "create_channel"
        assert result[0].params == {}

    def test_action_with_spaces(self):
        result = parse_actions("[ACTION: create_channel | name = mi canal | type = topic ]")
        assert result[0].params["name"] == "mi canal"
        assert result[0].params["type"] == "topic"

    def test_multiple_actions(self):
        response = "Listo[ACTION:create_channel|name=a|type=private][ACTION:create_channel|name=b|type=topic]"
        result = parse_actions(response)
        assert len(result) == 2
        assert result[0].params["name"] == "a"
        assert result[1].params["name"] == "b"


# --- strip_actions ---


class TestStripActions:
    def test_strips_action(self):
        assert strip_actions("Hola[ACTION:create_channel|name=x]") == "Hola"

    def test_strips_multiple(self):
        result = strip_actions("[ACTION:a|x=1]Hola[ACTION:b|y=2]")
        assert "ACTION" not in result
        assert "Hola" in result

    def test_no_action_unchanged(self):
        assert strip_actions("Normal text") == "Normal text"


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
        guild.channels = [MagicMock() for _ in range(10)]  # 10 existing channels

        # Bot member with permissions
        bot_member = MagicMock()
        bot_member.guild_permissions = MagicMock()
        bot_member.guild_permissions.manage_channels = True
        guild.me = bot_member

        # Default role
        guild.default_role = MagicMock()

        # Channel creation mocks
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
    async def test_creates_private_channel(self, mock_sleep, mock_guild, mock_user):
        action = BotAction(action_type="create_channel", params={"name": "mi espacio", "type": "private"})
        result = await execute_create_channel(mock_guild, action, mock_user)
        assert result is not None
        mock_guild.create_text_channel.assert_called_once()
        # Verify overwrites were passed (private channel)
        call_kwargs = mock_guild.create_text_channel.call_args
        assert "overwrites" in call_kwargs.kwargs or len(call_kwargs.args) > 1

    @patch("insult.core.actions.asyncio.sleep", new_callable=AsyncMock)
    async def test_creates_topic_channel(self, mock_sleep, mock_guild, mock_user):
        action = BotAction(action_type="create_channel", params={"name": "adhd-focus", "type": "topic"})
        result = await execute_create_channel(mock_guild, action, mock_user)
        assert result is not None
        mock_guild.create_text_channel.assert_called_once_with("adhd-focus")

    @patch("insult.core.actions.asyncio.sleep", new_callable=AsyncMock)
    async def test_creates_category(self, mock_sleep, mock_guild, mock_user):
        action = BotAction(action_type="create_channel", params={"name": "bienestar", "type": "category"})
        result = await execute_create_channel(mock_guild, action, mock_user)
        assert result is not None
        mock_guild.create_category.assert_called_once_with("bienestar")

    @patch("insult.core.actions.asyncio.sleep", new_callable=AsyncMock)
    async def test_missing_permissions_returns_none(self, mock_sleep, mock_guild, mock_user):
        mock_guild.me.guild_permissions.manage_channels = False
        action = BotAction(action_type="create_channel", params={"name": "test"})
        result = await execute_create_channel(mock_guild, action, mock_user)
        assert result is None
        mock_guild.create_text_channel.assert_not_called()

    @patch("insult.core.actions.asyncio.sleep", new_callable=AsyncMock)
    async def test_channel_limit_returns_none(self, mock_sleep, mock_guild, mock_user):
        mock_guild.channels = [MagicMock() for _ in range(460)]  # Over limit
        action = BotAction(action_type="create_channel", params={"name": "test"})
        result = await execute_create_channel(mock_guild, action, mock_user)
        assert result is None

    @patch("insult.core.actions.asyncio.sleep", new_callable=AsyncMock)
    async def test_defaults_to_private(self, mock_sleep, mock_guild, mock_user):
        action = BotAction(action_type="create_channel", params={"name": "test"})
        result = await execute_create_channel(mock_guild, action, mock_user)
        assert result is not None
        # Should have called with overwrites (private is default)
        call_kwargs = mock_guild.create_text_channel.call_args
        assert "overwrites" in call_kwargs.kwargs or len(call_kwargs.args) > 1

    @patch("insult.core.actions.asyncio.sleep", new_callable=AsyncMock)
    async def test_sanitizes_channel_name(self, mock_sleep, mock_guild, mock_user):
        action = BotAction(action_type="create_channel", params={"name": "Mi Canal Especial!!!", "type": "topic"})
        await execute_create_channel(mock_guild, action, mock_user)
        mock_guild.create_text_channel.assert_called_once_with("mi-canal-especial")
