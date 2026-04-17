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
MAX_REACTIONS = 8
REACTION_DELAY_MIN = 0.5  # seconds before first reaction (human-like pause)
REACTION_DELAY_MAX = 2.0
REACTION_INTERVAL = 0.35  # seconds between multiple reactions (rate limit safety)

# Unicode emoji grapheme — matches one emoji (base + optional ZWJ sequences, skin tones, variation selectors).
# Used to split tokens where the LLM concatenated emojis without commas: "🦷🪬🫧" → ["🦷","🪬","🫧"].
_EMOJI_GRAPHEME = re.compile(
    r"(?:"
    r"[\U0001F1E6-\U0001F1FF]{2}"  # regional indicators (flags)
    r"|[\U0001F000-\U0001FFFF\u2600-\u27BF\u2300-\u23FF\u2B00-\u2BFF]"  # base emoji
    r"(?:[\U0001F3FB-\U0001F3FF])?"  # optional skin tone
    r"(?:\uFE0F)?"  # optional variation selector
    r"(?:\u200D"  # optional ZWJ sequences
    r"[\U0001F000-\U0001FFFF\u2600-\u27BF][\U0001F3FB-\U0001F3FF]?\uFE0F?)*"
    r")"
)
_MAX_EMOJI_LEN = 16  # safety cap per token — anything longer is garbage


def _split_emoji_token(token: str) -> list[str]:
    """Split a token that may contain multiple concatenated unicode emojis.

    Custom Discord emojis (<:name:id>) and single short emojis pass through unchanged.
    """
    if token.startswith("<") and token.endswith(">"):
        return [token]
    matches = _EMOJI_GRAPHEME.findall(token)
    if len(matches) >= 2:
        return matches
    return [token]


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

    tokens = [e.strip() for e in raw.split(",") if e.strip()]
    emojis: list[str] = []
    for tok in tokens:
        for piece in _split_emoji_token(tok):
            if piece and len(piece) <= _MAX_EMOJI_LEN:
                emojis.append(piece)
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
