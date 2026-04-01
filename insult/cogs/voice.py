"""Voice cog — TTS via Azure OpenAI, sent as audio file attachment.

React with 🔊 on any message to have the bot generate a voice clip
and send it as an MP3 file in the channel.
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

import discord
import structlog
from discord.ext import commands
from openai import AsyncAzureOpenAI

if TYPE_CHECKING:
    from insult.app import Container

log = structlog.get_logger()

SPEAK_EMOJI = "🔊"


class VoiceCog(commands.Cog):
    def __init__(self, container: Container):
        self.settings = container.settings
        self.bot = container.bot
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
                log.info("tts_generated", text_len=len(text), audio_bytes=len(audio_bytes))
        except Exception:
            log.exception("tts_generation_failed")
            await channel.send("Se me trabo la voz. Intentale otra vez.")
            return

        # Send as audio file attachment
        audio_file = discord.File(io.BytesIO(audio_bytes), filename="insult.mp3")
        await channel.send(file=audio_file)
        log.info("tts_sent", text_len=len(text), channel=channel.name)
