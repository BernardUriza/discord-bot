"""Emoji reaction system — parsing, stripping, and async execution.

Handles [REACT:emoji1,emoji2] markers in LLM responses:
- parse_reactions(): extract emoji list from response text
- strip_reactions(): remove [REACT:] markers from response text
- add_reactions(): async background task to add emojis to Discord messages
"""

import asyncio
import random
import re

import discord
import structlog

log = structlog.get_logger()

# [REACT:💀,🔥] parsed from LLM response
REACTION_PATTERN = re.compile(r"\[REACT:([^\]]*)\]", re.IGNORECASE)
MAX_REACTIONS = 3
REACTION_DELAY_MIN = 0.5  # seconds before first reaction (human-like pause)
REACTION_DELAY_MAX = 2.0
REACTION_INTERVAL = 0.35  # seconds between multiple reactions (rate limit safety)


def parse_reactions(response: str) -> list[str]:
    """Extract emoji reactions from LLM response.

    Parses [REACT:emoji1,emoji2] markers and returns a list of emoji strings.
    Returns at most MAX_REACTIONS emojis. Returns empty list if no marker found.
    """
    match = REACTION_PATTERN.search(response)
    if not match:
        return []

    raw = match.group(1).strip()
    if not raw:
        return []

    # Split by comma, strip whitespace, filter empty
    emojis = [e.strip() for e in raw.split(",") if e.strip()]
    return emojis[:MAX_REACTIONS]


def strip_reactions(response: str) -> str:
    """Remove [REACT:...] markers from the response text."""
    return REACTION_PATTERN.sub("", response).strip()


async def add_reactions(message: discord.Message, emojis: list[str]) -> None:
    """Add emoji reactions to a Discord message with human-like delay.

    Designed to run as a background task via asyncio.create_task().
    """
    try:
        # Initial delay — humans don't react instantly
        await asyncio.sleep(random.uniform(REACTION_DELAY_MIN, REACTION_DELAY_MAX))

        for i, emoji in enumerate(emojis):
            try:
                await message.add_reaction(emoji)
                log.info("reaction_added", emoji=emoji, message_id=message.id)
            except (discord.HTTPException, discord.NotFound) as e:
                log.warning("reaction_failed", emoji=emoji, error=str(e))
                break  # Don't try remaining if one fails
            # Small delay between multiple reactions (rate limit safety)
            if i < len(emojis) - 1:
                await asyncio.sleep(REACTION_INTERVAL)
    except Exception:
        log.exception("reaction_task_failed", message_id=message.id)
