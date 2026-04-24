"""Voice cog — TTS via Azure OpenAI, sent as audio file attachment.

React with 🔊 on any message to have the bot generate a voice clip
and send it as an MP3 file in the channel.
"""

from __future__ import annotations

import io
import re
from typing import TYPE_CHECKING

import discord
import structlog
from discord.ext import commands
from openai import AsyncAzureOpenAI

if TYPE_CHECKING:
    from insult.app import Container

log = structlog.get_logger()

SPEAK_EMOJI = "🔊"

# Strips the version-tag suffix that delivery.py appends to the final
# chunk of a response ("\n-# ᵛ³·⁵·XX"). We strip it before the memory
# substring lookup because memory stores the pre-chunk text without it.
_VERSION_TAG_RE = re.compile(r"\n-#\s*ᵛ.+$", re.UNICODE)


async def resolve_full_response(
    memory,
    channel_id: str,
    chunk_text: str,
    *,
    lookback: int = 10,
) -> str | None:
    """Given a chunk of bot text a user reacted on, return the full pre-chunk
    response from memory if it can be located.

    Long bot responses get split by `delivery.py` into Discord messages of
    ≤1990 chars each. When a user reacts 🔊 on one of those chunks,
    `message.content` is only that slice — so TTS speaks ~2 min and stops.
    `turn.py` persists the full response (before chunking) in memory with
    role="assistant"; we locate it via substring match on the chunk text.

    Returns None when no assistant entry in the last `lookback` messages
    contains the chunk — caller should fall back to the chunk text.
    """
    needle = _VERSION_TAG_RE.sub("", chunk_text).strip()
    if not needle:
        return None

    try:
        recent = await memory.get_recent(channel_id, limit=lookback)
    except Exception:
        log.exception("tts_memory_lookup_failed", channel_id=channel_id)
        return None

    # `get_recent` returns oldest-first; walk newest-first so we prefer
    # the most recent turn when the same substring repeats across history.
    for entry in reversed(recent):
        if entry.get("role") != "assistant":
            continue
        content = entry.get("content", "")
        if needle in content:
            return content

    return None


class VoiceCog(commands.Cog):
    def __init__(self, container: Container):
        self.settings = container.settings
        self.bot = container.bot
        self.memory = container.memory
        self._tts_client: AsyncAzureOpenAI | None = None

    def _get_tts_client(self) -> AsyncAzureOpenAI | None:
        """Lazy-init Azure OpenAI client for TTS."""
        if self._tts_client:
            return self._tts_client
        endpoint = self.settings.azure_openai_endpoint
        key = self.settings.azure_openai_key.get_secret_value()
        if not endpoint or not key:
            return None
        self._tts_client = AsyncAzureOpenAI(
            azure_endpoint=endpoint,
            api_key=key,
            api_version="2024-12-01-preview",
        )
        return self._tts_client

    @commands.Cog.listener()
    async def on_raw_reaction_add(self, payload: discord.RawReactionActionEvent):
        """When someone reacts with 🔊, generate TTS and send as audio file."""
        if str(payload.emoji) != SPEAK_EMOJI:
            return
        if payload.user_id == self.bot.user.id:
            return

        guild = self.bot.get_guild(payload.guild_id)
        if not guild:
            return

        channel = guild.get_channel(payload.channel_id)
        if not channel:
            return

        try:
            message = await channel.fetch_message(payload.message_id)
        except discord.NotFound:
            return

        text = message.content.strip()
        if not text:
            return

        # Multi-chunk reassembly. delivery.py splits long responses into
        # Discord messages of ≤1990 chars (~2:05 of speech). When a user
        # reacts 🔊 on a bot-authored chunk, look up the full pre-chunk
        # response in memory so TTS doesn't cut off mid-thought.
        original_chunk_len = len(text)
        reassembled = False
        if message.author.id == self.bot.user.id:
            full = await resolve_full_response(self.memory, str(channel.id), message.content)
            if full and full != message.content:
                text = full.strip()
                reassembled = True

        client = self._get_tts_client()
        if not client:
            log.warning("tts_not_configured")
            return

        # Generate TTS audio
        try:
            async with channel.typing():
                tts_response = await client.audio.speech.create(
                    model=self.settings.azure_openai_tts_deployment,
                    voice=self.settings.tts_voice,
                    input=text[:4096],
                    response_format="mp3",
                )
                audio_bytes = tts_response.content
                log.info(
                    "tts_generated",
                    text_len=len(text),
                    chunk_len=original_chunk_len,
                    reassembled=reassembled,
                    audio_bytes=len(audio_bytes),
                )
        except Exception:
            log.exception("tts_generation_failed")
            await channel.send("Se me trabo la voz. Intentale otra vez.")
            return

        # Send as audio file attachment
        audio_file = discord.File(io.BytesIO(audio_bytes), filename="insult.mp3")
        await channel.send(file=audio_file)
        log.info("tts_sent", text_len=len(text), channel=channel.name)
