"""Chat cog — responds to all messages, no prefix needed."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import discord
import structlog
from discord.ext import commands

from insult.core.attachments import process_attachments
from insult.core.character import build_adaptive_prompt
from insult.core.errors import classify_error, get_error_response

if TYPE_CHECKING:
    from insult.app import Container

log = structlog.get_logger()

MAX_MESSAGE_LENGTH = 4000
COOLDOWN_SECONDS = 15


class ChatCog(commands.Cog):
    def __init__(self, container: Container):
        self.memory = container.memory
        self.llm = container.llm
        self.settings = container.settings
        self.bot = container.bot
        self._cooldowns: dict[int, float] = {}

    def _check_cooldown(self, user_id: int) -> float:
        """Returns 0 if ready, or seconds remaining."""
        now = time.monotonic()
        last = self._cooldowns.get(user_id, 0)
        remaining = COOLDOWN_SECONDS - (now - last)
        if remaining <= 0:
            self._cooldowns[user_id] = now
            return 0
        return remaining

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        """Respond to every message — no !chat prefix needed."""
        # Ignore bots (including ourselves)
        if message.author.bot:
            return

        # Ignore other commands (!ping, !memoria, !buscar, !perfil, !chat)
        if message.content.startswith(self.settings.command_prefix):
            return

        # Ignore empty messages (e.g. only attachments with no text, stickers)
        text = message.content.strip()
        if not text and not message.attachments:
            return

        if len(text) > MAX_MESSAGE_LENGTH:
            await message.channel.send(get_error_response("too_long"))
            return

        # Per-user cooldown
        remaining = self._check_cooldown(message.author.id)
        if remaining > 0:
            await message.channel.send(f"Calmate, espera {remaining:.0f}s. Cual es la urgencia?")
            return

        await self._respond(message, text)

    @commands.command(name="chat")
    @commands.cooldown(1, COOLDOWN_SECONDS, commands.BucketType.user)
    async def chat(self, ctx: commands.Context, *, message: str):
        """Fallback: !chat still works for explicit invocation."""
        await self._respond(ctx.message, message)

    async def _respond(self, message: discord.Message, text: str):
        """Core response logic — shared by on_message and !chat command."""
        channel_id = str(message.channel.id)
        user_id = str(message.author.id)
        user_name = message.author.display_name

        # Process attachments
        attachment_blocks = []
        if message.attachments:
            blocks, errors = await process_attachments(message.attachments)
            attachment_blocks = blocks
            for err in errors:
                await message.channel.send(err)

        try:
            await self.memory.store(channel_id, user_id, user_name, "user", text)
        except Exception:
            log.exception("chat_store_user_failed", channel_id=channel_id)

        try:
            profile = await self.memory.update_profile(user_id, text)
        except Exception:
            log.exception("chat_profile_update_failed", user_id=user_id)
            profile = None

        try:
            recent = await self.memory.get_recent(channel_id, self.settings.memory_recent_limit, user_id=user_id)
            relevant = await self.memory.search(channel_id, text, self.settings.memory_relevant_limit, user_id=user_id)
            context = self.memory.build_context(recent, relevant)
        except Exception:
            log.exception("chat_context_failed", channel_id=channel_id)
            await message.channel.send(get_error_response("context_failed"))
            return

        if attachment_blocks and context:
            last_msg = context[-1]
            if last_msg["role"] == "user":
                text_block = {"type": "text", "text": last_msg["content"]}
                context[-1] = {"role": "user", "content": [text_block, *attachment_blocks]}

        system_prompt = build_adaptive_prompt(self.settings.system_prompt, profile, len(context))
        if profile and profile.is_confident:
            log.info(
                "style_adapted",
                user_id=user_id,
                language=profile.detected_language,
                formality=round(profile.formality, 2),
                technical=round(profile.technical_level, 2),
                verbosity=round(profile.avg_word_count, 1),
            )

        try:
            async with message.channel.typing():
                response = await self.llm.chat(system_prompt, context)
        except Exception as e:
            log.exception("chat_llm_failed", channel_id=channel_id, error_type=type(e).__name__)
            await message.channel.send(get_error_response(classify_error(e)))
            return

        try:
            await self.memory.store(
                channel_id, str(self.bot.user.id), self.bot.user.name, "assistant", response, for_user_id=user_id
            )
        except Exception:
            log.exception("chat_store_response_failed", channel_id=channel_id)

        for chunk in [response[i : i + 1990] for i in range(0, len(response), 1990)]:
            await message.channel.send(chunk)
