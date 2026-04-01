"""Action system — Claude tool_use for real Discord actions (channel creation, etc.).

Uses Claude's native tool_use with strict:true for guaranteed schema compliance.
Text markers ([SEND], [REACT:]) remain for cosmetic features.
Side-effect actions use tool_use because reliability is critical.
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

MAX_ACTIONS_PER_RESPONSE = 2  # Safety: prevent LLM from creating 50 channels
VALID_CHANNEL_TYPES = {"private", "topic", "category"}
CHANNEL_NAME_MAX_LEN = 100
CHANNEL_LIMIT_BUFFER = 50  # Don't create if guild has >= 500 - buffer channels

# ---------------------------------------------------------------------------
# Tool definitions for Claude API (strict: true = guaranteed schema compliance)
# ---------------------------------------------------------------------------

CHANNEL_TOOLS = [
    {
        "name": "create_channel",
        "description": (
            "ACTUALLY create a real Discord channel in the server. You MUST call this tool whenever "
            "a user asks you to create, make, or set up a channel. Do NOT just say you created it — "
            "you must call this tool for the channel to actually exist. Without calling this tool, "
            "no channel is created and you would be lying. "
            "Types: 'private' (only requesting user + you can see it), "
            "'topic' (visible to everyone in server), 'category' (channel grouping)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Channel name: lowercase, hyphens, no spaces. E.g. 'ciencia-y-mates', 'espacio-privado'",
                },
                "channel_type": {
                    "type": "string",
                    "enum": ["private", "topic", "category"],
                    "description": "private=only user+bot can see, topic=visible to all, category=channel grouping",
                },
            },
            "required": ["name", "channel_type"],
            "additionalProperties": False,
        },
    },
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ToolCall:
    """A tool_use block from the Claude API response."""

    id: str
    name: str
    input: dict


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
    tool_call: ToolCall,
    requesting_user: discord.Member,
) -> discord.abc.GuildChannel | None:
    """Create a Discord channel based on tool_call input.

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

    raw_name = tool_call.input.get("name", "nuevo-canal")
    channel_name = sanitize_channel_name(raw_name)
    channel_type = tool_call.input.get("channel_type", "private")

    if channel_type not in VALID_CHANNEL_TYPES:
        log.warning("action_invalid_channel_type", type=channel_type)
        channel_type = "private"

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
