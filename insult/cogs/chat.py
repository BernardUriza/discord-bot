"""!chat command — core conversation with longitudinal memory."""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog
from discord.ext import commands

from insult.core.attachments import process_attachments
from insult.core.character import build_adaptive_prompt
from insult.core.errors import classify_error, get_error_response

if TYPE_CHECKING:
    from insult.app import Container

log = structlog.get_logger()

MAX_MESSAGE_LENGTH = 4000


class ChatCog(commands.Cog):
    def __init__(self, container: Container):
        self.memory = container.memory
        self.llm = container.llm
        self.settings = container.settings
        self.bot = container.bot

    @commands.command(name="chat")
    @commands.cooldown(1, 15, commands.BucketType.user)
    async def chat(self, ctx: commands.Context, *, message: str):
        """Habla con la IA. Usa memoria longitudinal del canal."""
        if len(message) > MAX_MESSAGE_LENGTH:
            await ctx.send(get_error_response("too_long"))
            return

        channel_id = str(ctx.channel.id)
        user_id = str(ctx.author.id)
        user_name = ctx.author.display_name

        # Process attachments (images, text files, PDFs)
        attachment_blocks = []
        if ctx.message.attachments:
            blocks, errors = await process_attachments(ctx.message.attachments)
            attachment_blocks = blocks
            for err in errors:
                await ctx.send(err)

        try:
            await self.memory.store(channel_id, user_id, user_name, "user", message)
        except Exception:
            log.exception("chat_store_user_failed", channel_id=channel_id)

        # Update user style profile (EMA)
        try:
            profile = await self.memory.update_profile(user_id, message)
        except Exception:
            log.exception("chat_profile_update_failed", user_id=user_id)
            profile = None

        try:
            recent = await self.memory.get_recent(channel_id, self.settings.memory_recent_limit, user_id=user_id)
            relevant = await self.memory.search(channel_id, message, self.settings.memory_relevant_limit, user_id=user_id)
            context = self.memory.build_context(recent, relevant)
        except Exception:
            log.exception("chat_context_failed", channel_id=channel_id)
            await ctx.send(get_error_response("context_failed"))
            return

        # Inject attachment content blocks into the last user message
        if attachment_blocks and context:
            last_msg = context[-1]
            if last_msg["role"] == "user":
                # Convert text-only message to multimodal content blocks
                text_block = {"type": "text", "text": last_msg["content"]}
                context[-1] = {"role": "user", "content": [text_block, *attachment_blocks]}

        # Build adaptive system prompt: base persona + user style + identity reinforcement
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
            async with ctx.typing():
                response = await self.llm.chat(system_prompt, context)
        except Exception as e:
            log.exception("chat_llm_failed", channel_id=channel_id, error_type=type(e).__name__)
            await ctx.send(get_error_response(classify_error(e)))
            return

        try:
            await self.memory.store(
                channel_id, str(self.bot.user.id), self.bot.user.name, "assistant", response, for_user_id=user_id
            )
        except Exception:
            log.exception("chat_store_response_failed", channel_id=channel_id)

        for chunk in [response[i : i + 1990] for i in range(0, len(response), 1990)]:
            await ctx.send(chunk)
