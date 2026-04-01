"""Generic action system — LLM outputs [ACTION:type|key=value] markers that trigger bot actions.

Currently supports:
- create_channel: Create text channels, private channels, or categories

Extensible for future actions (invite_user, archive_channel, etc.)
"""

import asyncio
import re
from dataclasses import dataclass

import discord
import structlog

log = structlog.get_logger()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ACTION_PATTERN = re.compile(r"\[ACTION:([^\]]+)\]", re.IGNORECASE)
MAX_ACTIONS_PER_RESPONSE = 2  # Safety: prevent LLM from creating 50 channels
VALID_CHANNEL_TYPES = {"private", "topic", "category"}
CHANNEL_NAME_MAX_LEN = 100
CHANNEL_LIMIT_BUFFER = 50  # Don't create if guild has >= 500 - buffer channels


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class BotAction:
    """A parsed action from the LLM response."""

    action_type: str
    params: dict[str, str]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_actions(response: str) -> list[BotAction]:
    """Extract [ACTION:type|key=val|key=val] markers from LLM response.

    Returns at most MAX_ACTIONS_PER_RESPONSE actions.
    """
    matches = ACTION_PATTERN.findall(response)
    actions: list[BotAction] = []

    for raw in matches[:MAX_ACTIONS_PER_RESPONSE]:
        parts = [p.strip() for p in raw.split("|") if p.strip()]
        if not parts:
            continue

        action_type = parts[0].lower()
        params: dict[str, str] = {}
        for part in parts[1:]:
            if "=" in part:
                key, _, value = part.partition("=")
                params[key.strip().lower()] = value.strip()

        actions.append(BotAction(action_type=action_type, params=params))

    return actions


def strip_actions(response: str) -> str:
    """Remove [ACTION:...] markers from the response text."""
    return ACTION_PATTERN.sub("", response).strip()


# ---------------------------------------------------------------------------
# Channel name sanitization
# ---------------------------------------------------------------------------


def sanitize_channel_name(name: str) -> str:
    """Sanitize a string into a valid Discord channel name.

    Rules: lowercase, hyphens instead of spaces, strip invalid chars, max 100 chars.
    """
    name = name.lower().strip()
    name = name.replace(" ", "-")
    name = re.sub(r"[^a-z0-9\-]", "", name)
    name = re.sub(r"-{2,}", "-", name)  # collapse multiple hyphens
    name = name.strip("-")
    return name[:CHANNEL_NAME_MAX_LEN] or "nuevo-canal"


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


async def execute_create_channel(
    guild: discord.Guild,
    action: BotAction,
    requesting_user: discord.Member,
) -> discord.abc.GuildChannel | None:
    """Create a Discord channel based on action params.

    Params:
        name: Channel name (will be sanitized)
        type: "private" (user + bot only), "topic" (public), "category"
        for: Optional display name of the user to grant access (defaults to requesting_user)

    Returns the created channel, or None on failure.
    """
    # Validate bot permissions
    bot_member = guild.me
    if not bot_member.guild_permissions.manage_channels:
        log.warning("action_missing_permission", permission="manage_channels", guild=guild.name)
        return None

    # Check guild channel limit
    if len(guild.channels) >= (500 - CHANNEL_LIMIT_BUFFER):
        log.warning("action_channel_limit", current=len(guild.channels), guild=guild.name)
        return None

    raw_name = action.params.get("name", "nuevo-canal")
    channel_name = sanitize_channel_name(raw_name)
    channel_type = action.params.get("type", "private").lower()

    if channel_type not in VALID_CHANNEL_TYPES:
        log.warning("action_invalid_channel_type", type=channel_type)
        channel_type = "private"  # safe default

    try:
        if channel_type == "category":
            channel = await guild.create_category(channel_name)
            log.info("action_category_created", name=channel_name, guild=guild.name)

        elif channel_type == "private":
            overwrites = {
                guild.default_role: discord.PermissionOverwrite(read_messages=False),
                bot_member: discord.PermissionOverwrite(read_messages=True, send_messages=True),
                requesting_user: discord.PermissionOverwrite(read_messages=True, send_messages=True),
            }
            channel = await guild.create_text_channel(channel_name, overwrites=overwrites)
            log.info(
                "action_private_channel_created",
                name=channel_name,
                for_user=requesting_user.display_name,
                guild=guild.name,
            )

        else:  # topic (public)
            channel = await guild.create_text_channel(channel_name)
            log.info("action_topic_channel_created", name=channel_name, guild=guild.name)

        # Small delay for Discord permission propagation
        await asyncio.sleep(0.3)
        return channel

    except discord.Forbidden:
        log.error("action_channel_forbidden", name=channel_name, guild=guild.name)
        return None
    except discord.HTTPException as e:
        log.error("action_channel_http_error", name=channel_name, error=str(e))
        return None
