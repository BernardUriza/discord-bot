"""Voice cog — TTS playback via Azure OpenAI.

React with 🔊 on any message to have the bot join your voice channel
and read the message aloud using Azure OpenAI TTS.
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
VOICE_DISCONNECT_AFTER = 30.0  # seconds idle before auto-disconnect


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
        """When someone reacts with 🔊, TTS the message in their voice channel."""
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

        # Find the user's voice channel
        member = guild.get_member(payload.user_id)
        if not member or not member.voice or not member.voice.channel:
            await channel.send("Metete a un canal de voz primero, genio.")
            return

        voice_channel = member.voice.channel

        client = self._get_tts_client()
        if not client:
            log.warning("tts_not_configured")
            await channel.send("No tengo voz configurada. Dile al admin.")
            return

        # Generate TTS audio
        try:
            tts_response = await client.audio.speech.create(
                model=self.settings.azure_openai_tts_deployment,
                voice=self.settings.tts_voice,
                input=text[:4096],  # Azure TTS limit
            )
            audio_bytes = tts_response.content
            log.info("tts_generated", text_len=len(text), audio_bytes=len(audio_bytes))
        except Exception:
            log.exception("tts_generation_failed")
            await channel.send("Se me trabo la voz. Intentale otra vez.")
            return

        # Connect to voice channel and play
        try:
            vc = guild.voice_client
            if vc and vc.is_connected():
                if vc.channel != voice_channel:
                    await vc.move_to(voice_channel)
            else:
                vc = await voice_channel.connect()

            audio_source = discord.FFmpegPCMAudio(io.BytesIO(audio_bytes), pipe=True)

            if vc.is_playing():
                vc.stop()

            vc.play(
                audio_source,
                after=lambda e: log.info("tts_playback_done")
                if not e
                else log.error("tts_playback_error", error=str(e)),
            )
            log.info("tts_playing", channel=voice_channel.name, text_len=len(text))

        except discord.ClientException as e:
            log.error("voice_connection_failed", error=str(e))
            await channel.send("No me puedo conectar al canal de voz.")
        except Exception:
            log.exception("voice_playback_failed")
            await channel.send("Algo salio mal con la voz. Intentale de nuevo.")
